"""
pose.py — full-body pose trajectories rendered as moving landmarks
==================================================================

A dancer's movement becomes a query.  `PoseTracker` runs MediaPipe's
PoseLandmarker on each frame of a video (file or camera) and a
`PoseTrajectory` keeps the running time series of the 33 body landmarks.
The renderer draws that trajectory back as an image: the current skeleton
plus *temporal traces*, fading trails left by selected joints over the last
second or two.  `render_frame` gives one instant (for an image embedder),
`render_clip` resamples a window at a fixed frame rate and returns a
(T, H, W, 3) array (for a video embedder), and `write_video` saves frames as
an mp4.

Like traversal.py, the module is transport-agnostic: it knows nothing about
WebSockets or the browser.  `PoseStream` bundles a tracker and a trajectory
for one live camera fed JPEG frames; `web/backend/app/routes/realtime.py`
wires it into the live graph socket as a query source.

Core loop
---------
    traj = track_video("dance.mp4")              # PoseTrajectory, one row per frame
    img  = render_frame(traj, t=4.0)             # pose at 4 s with 1 s of traces
    clip = render_clip(traj, seconds=2.0, fps=16, style=PoseStyle(size=224, square=True))
    write_video(iter_frames(traj, fps=25), "dance_pose.mp4", fps=25)
    traj.save("dance_pose.npz")

From the shell:
    python -m delyrism.pose dance.mp4 -o dance_pose.mp4 --trace 1.5 --fit body

Coordinates
-----------
Landmarks are stored as MediaPipe returns them: x, y normalised to the source
frame's width and height, z (depth, same scale as x), visibility, and world
coordinates in metres with the origin between the hips.  A missed detection
is a row of NaN, so the timeline stays continuous and traces break where the
dancer left the frame.  The renderer projects with one of three fits:

    camera   the source framing, letterboxed (travel across the stage shows)
    window   the bounding box of the rendered window (travel shows, body fills the frame)
    body     each sample re-centred on its own torso and scaled by torso length
             (travel removed, traces show gesture relative to the body)
"""
from __future__ import annotations

import argparse
import bisect
import os
import tempfile
import threading
import time
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np


# ----------------------------------------------------------------------------
#  landmark topology
# ----------------------------------------------------------------------------

N_LANDMARKS = 33

# MediaPipe's PoseLandmarksConnections.POSE_LANDMARKS, copied so a saved
# trajectory renders without mediapipe installed.
CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)

FACE = frozenset(range(1, 11))          # eyes, ears, mouth; the nose (0) stands for the head
LEFT = frozenset({1, 2, 3, 7, 9} | set(range(11, 33, 2)))     # the subject's left
RIGHT = frozenset({4, 5, 6, 8, 10} | set(range(12, 33, 2)))

JOINT_SETS = {
    "extremities": (0, 15, 16, 27, 28),             # head, wrists, ankles
    "hands": (15, 16),
    "feet": (27, 28),
    "limbs": (13, 14, 15, 16, 25, 26, 27, 28),      # elbows, wrists, knees, ankles
    "all": tuple(i for i in range(N_LANDMARKS) if i not in FACE),
}

# column layout of one stored row: x y z visibility wx wy wz
_X, _Y, _Z, _V = 0, 1, 2, 3
_COLS = 7


def resolve_joints(joints: Union[str, Sequence[int]]) -> Tuple[int, ...]:
    if isinstance(joints, str):
        if joints not in JOINT_SETS:
            raise ValueError(f"unknown joint set '{joints}'; choose from {sorted(JOINT_SETS)}")
        return JOINT_SETS[joints]
    return tuple(int(j) for j in joints)


# ----------------------------------------------------------------------------
#  tracking
# ----------------------------------------------------------------------------

@dataclass
class PoseSample:
    xy: np.ndarray            # (33, 2) image coords, 0..1 of width / height
    z: np.ndarray             # (33,)   depth relative to the hips, same scale as x
    visibility: np.ndarray    # (33,)   0..1, likelihood the landmark is in view and unoccluded
    world: np.ndarray         # (33, 3) metres, origin between the hips


_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")


def model_path(variant: str = "full") -> str:
    """Path to the PoseLandmarker .task file, downloaded on first use into
    $DELYRISM_CACHE/mediapipe (default ~/.cache/delyrism/mediapipe)."""
    if variant not in ("lite", "full", "heavy"):
        raise ValueError("variant must be lite | full | heavy")
    root = os.environ.get("DELYRISM_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "delyrism")
    path = os.path.join(root, "mediapipe", f"pose_landmarker_{variant}.task")
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        print(f"[pose] downloading pose_landmarker_{variant}.task ...")
        urllib.request.urlretrieve(_MODEL_URL.format(v=variant), tmp)
        os.replace(tmp, path)
    return path


class PoseTracker:
    """MediaPipe PoseLandmarker in VIDEO mode: one person, tracked across
    successive frames (MediaPipe smooths landmarks between frames in this
    mode).  Not thread-safe; give each stream its own tracker.

    variant: lite (fastest) | full | heavy (most accurate)."""

    def __init__(self, variant: str = "full", model_asset_path: Optional[str] = None,
                 min_detection: float = 0.5, min_presence: float = 0.5, min_tracking: float = 0.5):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:
            raise ImportError("pose tracking needs mediapipe: pip install mediapipe") from e
        self._mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path or model_path(variant)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection,
            min_pose_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._last_ms = -1

    def detect(self, frame_rgb: np.ndarray, t: float) -> Optional[PoseSample]:
        """Landmarks for one RGB uint8 frame at time `t` (seconds), or None
        when no person is found."""
        # VIDEO mode requires strictly increasing timestamps
        ms = max(int(round(t * 1000.0)), self._last_ms + 1)
        self._last_ms = ms
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(frame_rgb, dtype=np.uint8))
        res = self._landmarker.detect_for_video(img, ms)
        if not res.pose_landmarks:
            return None
        a = np.array([[p.x, p.y, p.z, 1.0 if p.visibility is None else p.visibility]
                      for p in res.pose_landmarks[0]], dtype=np.float32)
        if res.pose_world_landmarks:
            world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]], dtype=np.float32)
        else:
            world = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
        return PoseSample(xy=a[:, :2], z=a[:, 2], visibility=a[:, 3], world=world)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class LandmarkFilter:
    """One Euro filter over a pose sample (Casiez et al., CHI 2012): a
    low-pass whose cutoff rises with speed, so a still body stops jittering
    while a fast gesture keeps its sharpness.

    `smooth` in [0, 1] sets the resting cutoff on a log scale, from 10 Hz
    (0, almost raw) through 1 Hz (0.5) to 0.1 Hz (1, heavy).  `beta` is how
    fast the cutoff opens with speed, in Hz per (frame-heights per second).
    A missed detection resets the filter, so a dancer re-entering the frame
    starts where they are instead of sliding in from where they left."""

    def __init__(self, smooth: float = 0.5, beta: float = 8.0, d_cutoff: float = 1.0):
        self.smooth = smooth
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._t: Optional[float] = None
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None

    @staticmethod
    def _alpha(dt: float, cutoff) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff, dtype=np.float32))
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, sample: Optional[PoseSample]) -> Optional[PoseSample]:
        if sample is None or self.smooth <= 0:
            self.reset()
            return sample
        x = np.concatenate([sample.xy, sample.z[:, None], sample.world], axis=1)      # (33, 6)
        if self._x is None or self._t is None or t <= self._t:
            self._t, self._x, self._dx = t, x, np.zeros_like(x)
            return sample
        dt = t - self._t
        dx = (x - self._x) / dt
        dx_hat = self._dx + self._alpha(dt, self.d_cutoff) * (dx - self._dx)
        min_cutoff = 10.0 ** (1.0 - 2.0 * float(np.clip(self.smooth, 0.0, 1.0)))
        # one cutoff per landmark, from its image-plane speed
        speed = np.linalg.norm(dx_hat[:, :2], axis=1, keepdims=True)
        x_hat = self._x + self._alpha(dt, min_cutoff + self.beta * speed) * (x - self._x)
        self._t, self._x, self._dx = t, x_hat, dx_hat
        return PoseSample(xy=x_hat[:, :2].astype(np.float32), z=x_hat[:, 2].astype(np.float32),
                          visibility=sample.visibility, world=x_hat[:, 3:].astype(np.float32))


# ----------------------------------------------------------------------------
#  trajectory
# ----------------------------------------------------------------------------

class PoseTrajectory:
    """Running time series of pose samples.

    With `max_seconds` set, samples further than that behind the newest one
    are dropped (live use); None keeps everything (files).  `aspect` is the
    source frame's width / height, so renders stay undistorted.  Appends and
    reads share a lock: live, the pose thread appends while the embedder
    thread renders."""

    def __init__(self, max_seconds: Optional[float] = 30.0, aspect: float = 4 / 3):
        self.max_seconds = max_seconds
        self.aspect = float(aspect)
        self._lock = threading.Lock()
        self._t: List[float] = []
        self._rows: List[np.ndarray] = []           # each (33, 7)
        self._stacked: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # -------- writing --------
    def append(self, t: float, sample: Optional[PoseSample]) -> None:
        """Add a sample at time `t` (seconds).  None records a missed detection."""
        if sample is None:
            row = np.full((N_LANDMARKS, _COLS), np.nan, dtype=np.float32)
        else:
            row = np.concatenate([sample.xy, sample.z[:, None], sample.visibility[:, None],
                                  sample.world], axis=1).astype(np.float32)
        with self._lock:
            if self._t and t <= self._t[-1]:
                t = self._t[-1] + 1e-3
            self._t.append(float(t))
            self._rows.append(row)
            self._stacked = None
            if self.max_seconds is not None:
                cut = bisect.bisect_left(self._t, t - self.max_seconds)
                if cut:
                    del self._t[:cut]
                    del self._rows[:cut]

    def clear(self) -> None:
        with self._lock:
            self._t.clear()
            self._rows.clear()
            self._stacked = None

    # -------- reading --------
    def __len__(self) -> int:
        return len(self._t)

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Snapshot (t, L): t (T,) seconds, L (T, 33, 7) as x y z visibility wx wy wz.
        Cached until the next append; treat the arrays as read-only."""
        with self._lock:
            if self._stacked is None:
                if not self._t:
                    return np.zeros(0), np.zeros((0, N_LANDMARKS, _COLS), dtype=np.float32)
                self._stacked = (np.asarray(self._t), np.stack(self._rows))
            return self._stacked

    @property
    def t(self) -> np.ndarray:
        return self.arrays()[0]

    @property
    def xy(self) -> np.ndarray:
        return self.arrays()[1][..., :2]

    @property
    def visibility(self) -> np.ndarray:
        return self.arrays()[1][..., _V]

    @property
    def world(self) -> np.ndarray:
        return self.arrays()[1][..., 4:]

    @property
    def span(self) -> Tuple[float, float]:
        with self._lock:
            return (self._t[0], self._t[-1]) if self._t else (0.0, 0.0)

    def present(self) -> bool:
        """True when the newest sample holds a detected person."""
        with self._lock:
            return bool(self._rows) and not np.isnan(self._rows[-1][0, 0])

    @classmethod
    def from_arrays(cls, t: np.ndarray, L: np.ndarray, aspect: float,
                    max_seconds: Optional[float] = None) -> "PoseTrajectory":
        traj = cls(max_seconds=max_seconds, aspect=aspect)
        traj._t = [float(x) for x in t]
        traj._rows = [np.asarray(r, dtype=np.float32) for r in L]
        traj._stacked = None
        return traj

    def window(self, seconds: float, end: Optional[float] = None) -> "PoseTrajectory":
        """The samples in [end − seconds, end]; end defaults to the newest sample."""
        t, L = self.arrays()
        if end is None:
            end = float(t[-1]) if len(t) else 0.0
        keep = (t >= end - seconds) & (t <= end)
        return PoseTrajectory.from_arrays(t[keep], L[keep], self.aspect)

    def resample(self, fps: float, start: Optional[float] = None, end: Optional[float] = None,
                 max_gap: float = 0.5) -> "PoseTrajectory":
        """Uniform timeline at `fps`, linearly interpolated between the two
        neighbouring samples.  A target time beside a missed detection, or
        inside a stretch longer than `max_gap` seconds with no samples at
        all, stays NaN, so interpolation never bridges a lost person."""
        t, L = self.arrays()
        if len(t) == 0:
            return PoseTrajectory(max_seconds=None, aspect=self.aspect)
        start = float(t[0]) if start is None else float(start)
        end = float(t[-1]) if end is None else float(end)
        n = max(1, int(np.floor((end - start) * fps + 1e-6)) + 1)
        ts = start + np.arange(n) / fps
        if len(t) == 1:
            out = np.repeat(L[:1], n, axis=0)
            out[np.abs(ts - t[0]) > max_gap] = np.nan
            return PoseTrajectory.from_arrays(ts, out, self.aspect)

        i = np.clip(np.searchsorted(t, ts, side="right") - 1, 0, len(t) - 2)
        dt = t[i + 1] - t[i]
        w = np.clip((ts - t[i]) / np.where(dt > 0, dt, 1.0), 0.0, 1.0)[:, None, None]
        out = (1.0 - w) * L[i] + w * L[i + 1]           # NaN on either side propagates
        exact_lo, exact_hi = (w[:, 0, 0] < 1e-6), (w[:, 0, 0] > 1 - 1e-6)
        out[exact_lo] = L[i][exact_lo]
        out[exact_hi] = L[i + 1][exact_hi]
        out[(dt > max_gap) & ~exact_lo & ~exact_hi] = np.nan
        out[(ts < t[0] - max_gap) | (ts > t[-1] + max_gap)] = np.nan
        return PoseTrajectory.from_arrays(ts, out.astype(np.float32), self.aspect)

    # -------- persistence --------
    def save(self, path) -> None:
        """Write t, landmarks and aspect as compressed npz (a path or a binary file object)."""
        t, L = self.arrays()
        np.savez_compressed(path, t=t, landmarks=L, aspect=np.float32(self.aspect))

    @classmethod
    def load(cls, path: str) -> "PoseTrajectory":
        z = np.load(path)
        return cls.from_arrays(z["t"], z["landmarks"], float(z["aspect"]))


# ----------------------------------------------------------------------------
#  rendering
# ----------------------------------------------------------------------------

@dataclass
class PoseStyle:
    size: int = 512                 # long side in px; the short side follows the source aspect
    square: bool = False            # size × size canvas (what most image / video encoders take)
    fit: str = "camera"             # camera | window | body (see module docstring)
    trace_seconds: float = 1.0      # how far back the traces reach; 0 draws the skeleton only
    trace_joints: Union[str, Sequence[int]] = "extremities"
    trace_width: float = 3.0        # px at the newest end; older segments thin out
    fade: float = 1.5               # exponent on the trace fade (higher → shorter visible tail)
    skeleton_width: float = 2.0
    joint_radius: float = 3.0
    face: bool = False              # draw eyes, ears and mouth
    color: str = "sides"            # sides (subject's left warm, right cool) | mono
    background: Tuple[int, int, int] = (0, 0, 0)
    min_visibility: float = 0.5
    margin: float = 0.08            # padding fraction for the window and body fits

    def canvas_size(self, aspect: float) -> Tuple[int, int]:
        if self.square:
            return self.size, self.size
        if aspect >= 1.0:
            return self.size, max(2, int(round(self.size / aspect)))
        return max(2, int(round(self.size * aspect))), self.size


_WARM, _COOL, _WHITE = (255, 168, 64), (72, 188, 255), (236, 236, 236)


def _joint_color(j: int, style: PoseStyle) -> Tuple[int, int, int]:
    if style.color == "mono":
        return _WHITE
    return _WARM if j in LEFT else _COOL if j in RIGHT else _WHITE


def _bone_color(a: int, b: int, style: PoseStyle) -> Tuple[int, int, int]:
    if style.color == "mono":
        return _WHITE
    if a in LEFT and b in LEFT:
        return _WARM
    if a in RIGHT and b in RIGHT:
        return _COOL
    return _WHITE


def _project(L: np.ndarray, aspect: float, W: int, H: int, style: PoseStyle) -> np.ndarray:
    """(T, 33, ≥4) landmarks → (T, 33, 2) pixel coordinates under `style.fit`.
    Landmarks below `min_visibility` come back NaN.  The window and body fits
    are computed over all T rows at once, so frames rendered from one
    projection share a stable framing."""
    uv = np.stack([L[..., _X] * aspect, L[..., _Y]], axis=-1)      # isotropic units, height = 1
    uv[L[..., _V] < style.min_visibility] = np.nan

    if style.fit == "camera":
        lo, ext = np.array([0.0, 0.0]), np.array([aspect, 1.0])
    elif style.fit == "body":
        with warnings.catch_warnings():                             # rows with no torso in view → NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            centre = np.nanmean(uv[:, [11, 12, 23, 24]], axis=1, keepdims=True)   # survives hips out of frame
            hips = np.nanmean(uv[:, [23, 24]], axis=1)
            shoulders = np.nanmean(uv[:, [11, 12]], axis=1)
        uv = uv - centre
        torso = np.linalg.norm(shoulders - hips, axis=-1)
        scale = float(np.nanmedian(torso)) if np.isfinite(torso).any() else 0.25
        half = 2.3 * max(scale, 1e-3) * (1.0 + style.margin)      # torso centre to feet ≈ 2.3 torsos
        lo, ext = np.array([-half, -half]), np.array([2 * half, 2 * half])
    elif style.fit == "window":
        if np.isfinite(uv).any():
            lo, hi = np.nanmin(uv, axis=(0, 1)), np.nanmax(uv, axis=(0, 1))
        else:
            lo, hi = np.array([0.0, 0.0]), np.array([aspect, 1.0])
        pad = style.margin * max(float(np.max(hi - lo)), 1e-3)
        lo, ext = lo - pad, (hi - lo) + 2 * pad
    else:
        raise ValueError("fit must be camera | window | body")

    k = min(W / max(ext[0], 1e-6), H / max(ext[1], 1e-6))          # uniform scale, letterboxed
    off = np.array([(W - k * ext[0]) / 2.0, (H - k * ext[1]) / 2.0])
    return (uv - lo) * k + off


_SHIFT = 4                                  # cv2 sub-pixel bits: smooth traces at small sizes


def _fixed(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel coords → (cv2 fixed-point int coords, finite mask), once per sequence."""
    ok = np.isfinite(P).all(axis=-1)
    Q = np.zeros(P.shape, dtype=np.int64)
    Q[ok] = np.round(P[ok] * (1 << _SHIFT)).astype(np.int64)
    return Q, ok


def _draw(img: np.ndarray, Q: np.ndarray, ok: np.ndarray, ts: np.ndarray, k: int, style: PoseStyle,
          joints: Tuple[int, ...], bg: Tuple[int, int, int]) -> None:
    """Draw frame k of a projected sequence (Q, ok from `_fixed`) onto img:
    traces over the samples within trace_seconds before ts[k], then the
    skeleton and joints of sample k."""
    import cv2
    aa = cv2.LINE_AA
    bg_arr = np.asarray(bg, dtype=np.float32)

    if style.trace_seconds > 0 and k > 0:
        i0 = max(int(np.searchsorted(ts, ts[k] - style.trace_seconds, side="left")), 1)
        idx = np.arange(i0, k + 1)
        fade = np.clip(1.0 - (ts[k] - ts[idx]) / style.trace_seconds, 0.0, 1.0) ** style.fade
        widths = np.maximum(1, np.round(style.trace_width * (0.35 + 0.65 * fade))).astype(int)
        for j in joints:
            base = np.asarray(_joint_color(j, style), dtype=np.float32)
            cols = (bg_arr + (base - bg_arr) * fade[:, None]).astype(int)
            seg = ok[idx - 1, j] & ok[idx, j] & (fade > 0.01)
            for n in np.flatnonzero(seg):
                i = idx[n]
                cv2.line(img, (int(Q[i - 1, j, 0]), int(Q[i - 1, j, 1])), (int(Q[i, j, 0]), int(Q[i, j, 1])),
                         tuple(int(c) for c in cols[n]), int(widths[n]), aa, _SHIFT)

    cur, cur_ok = Q[k], ok[k]
    hide = FACE if not style.face else frozenset()
    sw = max(1, int(round(style.skeleton_width)))
    for a, b in CONNECTIONS:
        if a in hide or b in hide or not (cur_ok[a] and cur_ok[b]):
            continue
        cv2.line(img, (int(cur[a, 0]), int(cur[a, 1])), (int(cur[b, 0]), int(cur[b, 1])),
                 _bone_color(a, b, style), sw, aa, _SHIFT)
    r = int(round(style.joint_radius * (1 << _SHIFT)))
    if r > 0:
        for j in np.flatnonzero(cur_ok):
            if j not in hide:
                cv2.circle(img, (int(cur[j, 0]), int(cur[j, 1])), r, _joint_color(int(j), style), -1, aa, _SHIFT)


def _composite(layer: np.ndarray, under: np.ndarray, dim: float = 0.45) -> np.ndarray:
    import cv2
    H, W = layer.shape[:2]
    if under.shape[:2] != (H, W):
        under = cv2.resize(under, (W, H), interpolation=cv2.INTER_AREA)
    return np.maximum((under.astype(np.float32) * dim).astype(np.uint8), layer)


def render_frame(traj: PoseTrajectory, t: Optional[float] = None, style: Optional[PoseStyle] = None,
                 under: Optional[np.ndarray] = None) -> np.ndarray:
    """One instant of the trajectory as an RGB uint8 image (H, W, 3): the
    pose at time `t` (default: newest sample) with traces over the preceding
    `style.trace_seconds`.  `under` (an RGB frame) is dimmed and drawn
    beneath, for checking tracking against the source video."""
    style = style or PoseStyle()
    W, H = style.canvas_size(traj.aspect)
    t_all, L_all = traj.arrays()
    bg = (0, 0, 0) if under is not None else style.background
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:] = bg
    if len(t_all):
        if t is None:
            t = float(t_all[-1])
        k_end = int(np.searchsorted(t_all, t, side="right"))
        k0 = int(np.searchsorted(t_all, t - style.trace_seconds, side="left"))
        ts, L = t_all[k0:k_end], L_all[k0:k_end]
        if len(ts):
            Q, ok = _fixed(_project(L, traj.aspect, W, H, style))
            _draw(img, Q, ok, ts, len(ts) - 1, style, resolve_joints(style.trace_joints), bg)
    return _composite(img, under) if under is not None else img


def iter_frames(traj: PoseTrajectory, fps: float, start: Optional[float] = None,
                end: Optional[float] = None, style: Optional[PoseStyle] = None) -> Iterator[np.ndarray]:
    """Render [start, end] at a fixed `fps`, one RGB frame at a time.  The
    trajectory is resampled first (live samples arrive at uneven intervals)
    and projected once, so the window and body fits hold still across frames."""
    style = style or PoseStyle()
    t_all = traj.t
    if len(t_all) == 0:
        return
    start = float(t_all[0]) if start is None else float(start)
    end = float(t_all[-1]) if end is None else float(end)
    lead = start - style.trace_seconds
    res = traj.resample(fps, start=lead, end=end)
    ts, L = res.arrays()
    W, H = style.canvas_size(traj.aspect)
    Q, ok = _fixed(_project(L, traj.aspect, W, H, style))
    joints = resolve_joints(style.trace_joints)
    first = int(np.searchsorted(ts, start - 1e-9, side="left"))
    for k in range(first, len(ts)):
        img = np.empty((H, W, 3), dtype=np.uint8)
        img[:] = style.background
        _draw(img, Q, ok, ts, k, style, joints, style.background)
        yield img


def render_clip(traj: PoseTrajectory, seconds: float = 2.0, fps: float = 16.0,
                end: Optional[float] = None, style: Optional[PoseStyle] = None) -> np.ndarray:
    """The last `seconds` before `end` (default: newest sample) as a
    (round(seconds·fps), H, W, 3) uint8 array, the input shape of a video
    encoder.  The last frame falls on `end`."""
    style = style or PoseStyle()
    if end is None:
        end = traj.span[1]
    n = max(1, int(round(seconds * fps)))
    frames = list(iter_frames(traj, fps, start=end - (n - 1) / fps, end=end, style=style))[:n]
    if not frames:
        W, H = style.canvas_size(traj.aspect)
        return np.zeros((0, H, W, 3), dtype=np.uint8)
    return np.stack(frames)


def write_video(frames: Iterable[np.ndarray], path: str, fps: float, codec: str = "mp4v") -> int:
    """Write RGB frames to a video file; returns the number of frames written.
    The default mp4v codec plays in VLC and desktop players; browsers want H.264."""
    import cv2
    writer, n = None, 0
    try:
        for f in frames:
            if writer is None:
                h, w = f.shape[:2]
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), float(fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"could not open a {codec} video writer for {path}")
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            n += 1
    finally:
        if writer is not None:
            writer.release()
    return n


def encode_mp4(frames: Iterable[np.ndarray], fps: float, codec: str = "mp4v") -> bytes:
    """Frames → mp4 bytes (through a temp file, which is what cv2 can write)."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        write_video(frames, path, fps, codec)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


# ----------------------------------------------------------------------------
#  video files and live streams
# ----------------------------------------------------------------------------

def iter_video(path: str, max_side: Optional[int] = None, stride: int = 1,
               start: float = 0.0, end: Optional[float] = None) -> Iterator[Tuple[float, np.ndarray]]:
    """(t seconds, RGB frame) for every `stride`-th frame of a video file
    between `start` and `end` seconds.  t is the time in the source file."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = int(round(start * fps))
    if i > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    i0 = i
    try:
        while end is None or i / fps <= end:
            ok, bgr = cap.read()
            if not ok:
                break
            if (i - i0) % stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if max_side and max(rgb.shape[:2]) > max_side:
                    k = max_side / max(rgb.shape[:2])
                    rgb = cv2.resize(rgb, (round(rgb.shape[1] * k), round(rgb.shape[0] * k)),
                                     interpolation=cv2.INTER_AREA)
                yield i / fps, rgb
            i += 1
    finally:
        cap.release()


def track_video(path: str, tracker: Optional[PoseTracker] = None, max_side: Optional[int] = 960,
                stride: int = 1, start: float = 0.0, end: Optional[float] = None,
                variant: str = "full", smooth: float = 0.0, progress: bool = False) -> PoseTrajectory:
    """Track a video file (or its [start, end] seconds) into an unbounded
    PoseTrajectory.  `smooth` > 0 runs the landmarks through a LandmarkFilter."""
    own = tracker is None
    tracker = tracker or PoseTracker(variant=variant)
    filt = LandmarkFilter(smooth=smooth)
    traj: Optional[PoseTrajectory] = None
    try:
        for n, (t, rgb) in enumerate(iter_video(path, max_side=max_side, stride=stride, start=start, end=end)):
            if traj is None:
                traj = PoseTrajectory(max_seconds=None, aspect=rgb.shape[1] / rgb.shape[0])
            traj.append(t, filt(t, tracker.detect(rgb, t)))
            if progress and n % 100 == 0:
                print(f"[pose] {t:7.2f} s")
    finally:
        if own:
            tracker.close()
    return traj or PoseTrajectory(max_seconds=None)


class PoseStream:
    """Tracker + running trajectory for one live camera fed JPEG frames.
    Timestamps come from the arrival clock.  The tracker loads on the
    first frame, so building a stream costs nothing until it is used."""

    def __init__(self, max_seconds: float = 30.0, variant: str = "full", smooth: float = 0.5):
        self.variant = variant
        self.trajectory = PoseTrajectory(max_seconds=max_seconds)
        self.filter = LandmarkFilter(smooth=smooth)  # set .filter.smooth to retune live
        self._tracker: Optional[PoseTracker] = None
        self._lock = threading.Lock()               # close() waits for a detect in flight
        self._t0 = time.perf_counter()

    def push_jpeg(self, jpeg: bytes) -> bool:
        """Decode, track and append one frame; True when a person was found."""
        import cv2
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return False
        return self.push_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def push_rgb(self, rgb: np.ndarray) -> bool:
        with self._lock:
            if self._tracker is None:
                self._tracker = PoseTracker(variant=self.variant)
            t = time.perf_counter() - self._t0
            self.trajectory.aspect = rgb.shape[1] / rgb.shape[0]
            sample = self.filter(t, self._tracker.detect(rgb, t))
            self.trajectory.append(t, sample)
            return sample is not None

    def close(self) -> None:
        with self._lock:
            if self._tracker is not None:
                self._tracker.close()
                self._tracker = None


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m delyrism.pose",
                                 description="Track full-body pose in a video and render it as moving landmarks with traces.")
    ap.add_argument("input", help="video file, or a .npz trajectory saved earlier")
    ap.add_argument("-o", "--out", help="output video (default: <input>_pose.mp4)")
    ap.add_argument("--npz", help="also save the trajectory here")
    ap.add_argument("--fps", type=float, help="output frame rate (default: the source rate, or 25)")
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--square", action="store_true")
    ap.add_argument("--fit", default="camera", choices=["camera", "window", "body"])
    ap.add_argument("--trace", type=float, default=1.0, help="trace length in seconds")
    ap.add_argument("--joints", default="extremities", help=f"one of {sorted(JOINT_SETS)} or comma-separated indices")
    ap.add_argument("--mono", action="store_true", help="white skeleton instead of left/right colours")
    ap.add_argument("--overlay", action="store_true", help="draw over the dimmed source video")
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--max-side", type=int, default=960, help="downscale frames before tracking")
    ap.add_argument("--smooth", type=float, default=0.5, help="landmark smoothing 0 (raw) .. 1 (heavy), One Euro filter")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the input")
    ap.add_argument("--end", type=float, help="stop at this many seconds into the input")
    a = ap.parse_args(argv)

    joints: Union[str, Sequence[int]] = a.joints if a.joints in JOINT_SETS else [int(x) for x in a.joints.split(",")]
    style = PoseStyle(size=a.size, square=a.square, fit=a.fit, trace_seconds=a.trace,
                      trace_joints=joints, color="mono" if a.mono else "sides")
    out = a.out or os.path.splitext(a.input)[0] + "_pose.mp4"

    if a.input.endswith(".npz"):
        if a.overlay:
            ap.error("--overlay needs the source video, not a .npz")
        traj, src_fps = PoseTrajectory.load(a.input), None
    else:
        import cv2
        cap = cv2.VideoCapture(a.input)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or None
        cap.release()
        t0 = time.perf_counter()
        traj = track_video(a.input, max_side=a.max_side, start=a.start, end=a.end, variant=a.model,
                           smooth=a.smooth, progress=True)
        found = int(np.isfinite(traj.arrays()[1][:, 0, 0]).sum())
        print(f"[pose] tracked {len(traj)} frames in {time.perf_counter() - t0:.1f} s, "
              f"person found in {found} ({100.0 * found / max(len(traj), 1):.0f} %)")
    if a.npz:
        traj.save(a.npz)
        print(f"[pose] trajectory -> {a.npz}")

    fps = a.fps or src_fps or 25.0
    if a.overlay:
        frames = (render_frame(traj, t=t, style=style, under=rgb)
                  for t, rgb in iter_video(a.input, max_side=a.max_side, start=a.start, end=a.end))
    else:
        t0, t1 = traj.span
        frames = iter_frames(traj, fps, start=max(t0, a.start), end=t1 if a.end is None else min(t1, a.end), style=style)
    n = write_video(frames, out, fps)
    print(f"[pose] {n} frames -> {out}")


if __name__ == "__main__":
    main()
