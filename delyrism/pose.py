"""
pose.py — body and hand trajectories rendered as moving landmarks
=================================================================

A dancer's movement becomes a query.  A tracker runs MediaPipe on each frame
of a video (file or camera) and a `PoseTrajectory` keeps the running time
series of the landmarks.  Three trackers share one pipeline:

    body    PoseTracker       PoseLandmarker, 33 points from head to feet
    hands   HandTracker       HandLandmarker, 21 points per hand with fingers,
                              no body needed in frame
    face    FaceTracker       FaceLandmarker, 478 points, drawn as contours
    both    BodyHandsTracker  body and hands together, 75 points, each hand
                              pinned to the nearest body wrist

MediaPipe's Python build has no GPU path on Windows, so the live query
tracks in the browser (tasks-vision on the GPU) and hands `PoseStream`
the landmarks; these Python trackers serve video files and the CLI.

Each has a `Skeleton` (points, bones, sides, trace joint sets).

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
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np


# ----------------------------------------------------------------------------
#  landmark layouts
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Skeleton:
    """Landmark layout of one tracker: how many points, which connect, which
    side of the subject each belongs to, and the joint sets traces follow."""
    name: str
    n: int
    connections: Tuple[Tuple[int, int], ...]
    left: frozenset                         # the subject's left
    right: frozenset
    face: frozenset                         # drawn only with PoseStyle.face
    skip: frozenset                         # never drawn (superseded by a finer tracker)
    small: frozenset                        # joints drawn at reduced radius
    joint_sets: Dict[str, Tuple[int, ...]]
    default_joints: str
    centre: Tuple[int, ...]                 # the `body` fit centres each sample on these
    span: Tuple[Tuple[int, int], ...]       # … and scales by the median length of these pairs
    reach: float                            # half the `body` canvas, in those lengths
    dots: Optional[frozenset] = None        # joints drawn as dots (None: every drawn joint)


# MediaPipe's PoseLandmarksConnections.POSE_LANDMARKS and
# HandLandmarksConnections.HAND_CONNECTIONS, copied so a saved trajectory
# renders without mediapipe installed.
CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),                 # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),                 # index
    (5, 9), (9, 10), (10, 11), (11, 12),            # middle
    (9, 13), (13, 14), (14, 15), (15, 16),          # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky and palm
)
FINGERTIPS = (4, 8, 12, 16, 20)

# MediaPipe's FaceLandmarksConnections: FACE_LANDMARKS_CONTOURS (face oval, lips,
# eyes, eyebrows) plus both iris rings, and the points on the subject's left
# and right eye, brow and iris (MediaPipe's "left" is the subject's left).
FACE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321,
    375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269,
    270), (270, 409), (409, 291), (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317),
    (317, 402), (402, 318), (318, 324), (324, 308), (78, 191), (191, 80), (80, 81), (81, 82), (82,
    13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308), (263, 249), (249, 390), (390,
    373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), (263, 466), (466, 388), (388,
    387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362), (276, 283), (283, 282), (282,
    295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336), (33, 7), (7, 163), (163, 144),
    (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), (33, 246), (246, 161), (161, 160),
    (160, 159), (159, 158), (158, 157), (157, 173), (173, 133), (46, 53), (53, 52), (52, 65), (65,
    55), (70, 63), (63, 105), (105, 66), (66, 107), (10, 338), (338, 297), (297, 332), (332, 284),
    (284, 251), (251, 389), (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234),
    (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10), (474,
    475), (475, 476), (476, 477), (477, 474), (469, 470), (470, 471), (471, 472), (472, 469),
)
_FACE_LEFT: Tuple[int, ...] = (
    249, 263, 276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 362, 373, 374, 380, 381, 382, 384,
    385, 386, 387, 388, 390, 398, 466, 474, 475, 476, 477, 473,
)
_FACE_RIGHT: Tuple[int, ...] = (
    7, 33, 46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 133, 144, 145, 153, 154, 155, 157, 158, 159,
    160, 161, 163, 173, 246, 469, 470, 471, 472, 468,
)
HAND_N = 21

N_LANDMARKS = 33
FACE = frozenset(range(1, 11))          # eyes, ears, mouth; the nose (0) stands for the head
LEFT = frozenset({1, 2, 3, 7, 9} | set(range(11, 33, 2)))
RIGHT = frozenset({4, 5, 6, 8, 10} | set(range(12, 33, 2)))
_COARSE_HANDS = frozenset(range(17, 23))     # the body model's pinky / index / thumb knuckles


def _hand(offset: int) -> Tuple[Tuple[int, int], ...]:
    return tuple((a + offset, b + offset) for a, b in HAND_CONNECTIONS)


BODY = Skeleton(
    name="body", n=N_LANDMARKS, connections=CONNECTIONS, left=LEFT, right=RIGHT,
    face=FACE, skip=frozenset(), small=frozenset(),
    joint_sets={
        "extremities": (0, 15, 16, 27, 28),             # head, wrists, ankles
        "wrists": (15, 16),
        "feet": (27, 28),
        "limbs": (13, 14, 15, 16, 25, 26, 27, 28),      # elbows, wrists, knees, ankles
        "all": tuple(i for i in range(N_LANDMARKS) if i not in FACE),
    },
    default_joints="extremities",
    centre=(11, 12, 23, 24), span=((11, 23), (12, 24)), reach=2.3,   # torso centre to feet ≈ 2.3 torsos
)

# the subject's left hand at 0–20, their right at 21–41
HANDS = Skeleton(
    name="hands", n=2 * HAND_N, connections=_hand(0) + _hand(HAND_N),
    left=frozenset(range(HAND_N)), right=frozenset(range(HAND_N, 2 * HAND_N)),
    face=frozenset(), skip=frozenset(), small=frozenset(),
    joint_sets={
        "fingertips": FINGERTIPS + tuple(f + HAND_N for f in FINGERTIPS),
        "index": (8, 8 + HAND_N),
        "wrists": (0, HAND_N),
        "all": tuple(range(2 * HAND_N)),
    },
    default_joints="fingertips",
    centre=(0, 9, HAND_N, 9 + HAND_N), span=((0, 9), (HAND_N, 9 + HAND_N)), reach=4.0,   # wrist → middle knuckle
)

# body at 0–32, the subject's left hand at 33–53, their right at 54–74
_LH, _RH = N_LANDMARKS, N_LANDMARKS + HAND_N
BOTH = Skeleton(
    name="both", n=N_LANDMARKS + 2 * HAND_N,
    connections=tuple(c for c in CONNECTIONS if not set(c) & _COARSE_HANDS) + _hand(_LH) + _hand(_RH),
    left=LEFT | frozenset(range(_LH, _RH)), right=RIGHT | frozenset(range(_RH, _RH + HAND_N)),
    face=FACE, skip=_COARSE_HANDS, small=frozenset(range(_LH, _RH + HAND_N)),
    joint_sets={
        "extremities": (0, 15, 16, 27, 28),
        "fingertips": tuple(f + _LH for f in FINGERTIPS) + tuple(f + _RH for f in FINGERTIPS),
        "limbs": (13, 14, 15, 16, 25, 26, 27, 28),
        "all": tuple(i for i in range(N_LANDMARKS + 2 * HAND_N) if i not in FACE | _COARSE_HANDS),
    },
    default_joints="extremities",
    centre=(11, 12, 23, 24), span=((11, 23), (12, 24)), reach=2.3,
)

FACE_N = 478
_FACE_POINTS = tuple(sorted({i for c in FACE_CONNECTIONS for i in c} | {468, 473}))
FACEMESH = Skeleton(
    name="face", n=FACE_N, connections=FACE_CONNECTIONS,
    left=frozenset(_FACE_LEFT), right=frozenset(_FACE_RIGHT),
    face=frozenset(), skip=frozenset(), small=frozenset(),
    joint_sets={
        "features": (1, 152, 61, 291, 105, 334),        # nose tip, chin, mouth corners, brow peaks
        "mouth": (61, 291, 13, 14),                     # corners, upper and lower lip
        "eyes": (468, 473),                             # iris centres (gaze)
        "nose": (1,),
        "all": _FACE_POINTS,
    },
    default_joints="features",
    centre=(1, 33, 263), span=((33, 263),), reach=1.7,  # outer eye corners; the face spans ≈ 2 of them
    dots=frozenset({468, 473}),                         # contours only, plus the irises
)

SKELETONS: Dict[str, Skeleton] = {"body": BODY, "hands": HANDS, "face": FACEMESH, "both": BOTH}
JOINT_SETS = BODY.joint_sets

# column layout of one stored row: x y z visibility wx wy wz
_X, _Y, _Z, _V = 0, 1, 2, 3
_COLS = 7


def resolve_joints(joints: Union[str, Sequence[int]], skeleton: Skeleton = BODY) -> Tuple[int, ...]:
    """Joint indices for a set name or an explicit list.  A name the
    skeleton lacks (say "fingertips" on a body-only trajectory) falls back
    to the skeleton's default set."""
    if isinstance(joints, str):
        return skeleton.joint_sets.get(joints, skeleton.joint_sets[skeleton.default_joints])
    return tuple(int(j) for j in joints if 0 <= int(j) < skeleton.n)


# ----------------------------------------------------------------------------
#  tracking
# ----------------------------------------------------------------------------

@dataclass
class PoseSample:
    xy: np.ndarray            # (N, 2) image coords, 0..1 of width / height
    z: np.ndarray             # (N,)   depth (body: relative to the hips; hands: to the wrist)
    visibility: np.ndarray    # (N,)   0..1, likelihood the landmark is in view and unoccluded
    world: np.ndarray         # (N, 3) metres (body: origin between the hips; hands: hand centre)


_MODEL_URLS = {
    "pose": ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"),
    "hand": ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task"),
    "face": ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task"),
}


def model_path(variant: str = "full", task: str = "pose") -> str:
    """Path to a MediaPipe .task file, downloaded on first use into
    $DELYRISM_CACHE/mediapipe (default ~/.cache/delyrism/mediapipe).
    task: pose (variant lite | full | heavy) | hand | face."""
    if task == "pose" and variant not in ("lite", "full", "heavy"):
        raise ValueError("variant must be lite | full | heavy")
    name = f"pose_landmarker_{variant}.task" if task == "pose" else f"{task}_landmarker.task"
    root = os.environ.get("DELYRISM_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "delyrism")
    path = os.path.join(root, "mediapipe", name)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        print(f"[pose] downloading {name} ...")
        urllib.request.urlretrieve(_MODEL_URLS[task].format(v=variant), tmp)
        os.replace(tmp, path)
    return path


def _mediapipe():
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as e:
        raise ImportError("pose tracking needs mediapipe: pip install mediapipe") from e
    return mp, BaseOptions, vision


def _mp_image(mp, frame_rgb: np.ndarray):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb, dtype=np.uint8))


class PoseTracker:
    """MediaPipe PoseLandmarker in VIDEO mode: one person, tracked across
    successive frames (MediaPipe smooths landmarks between frames in this
    mode).  Not thread-safe; give each stream its own tracker.

    variant: lite (fastest) | full | heavy (most accurate)."""

    skeleton = BODY

    def __init__(self, variant: str = "full", model_asset_path: Optional[str] = None,
                 min_detection: float = 0.5, min_presence: float = 0.5, min_tracking: float = 0.5):
        mp, BaseOptions, vision = _mediapipe()
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
        res = self._landmarker.detect_for_video(_mp_image(self._mp, frame_rgb), ms)
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


class HandTracker:
    """MediaPipe HandLandmarker in VIDEO mode: up to two hands, 21 landmarks
    each (the wrist, then four joints per finger from thumb to pinky).  It
    needs no body in frame, so hands alone in front of the camera track.

    The sample has 42 rows, the subject's left hand then their right, and a
    hand not found is NaN.  On unmirrored frames (camera, video files) the
    model's "Left" label is the subject's left hand: checked against the
    body model's wrists, it agrees in 92 % of clear cases (195 of 211), so
    the swap MediaPipe's docs describe for selfie input is not applied."""

    skeleton = HANDS

    def __init__(self, model_asset_path: Optional[str] = None,
                 min_detection: float = 0.5, min_presence: float = 0.5, min_tracking: float = 0.5):
        mp, BaseOptions, vision = _mediapipe()
        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path or model_path(task="hand")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._last_ms = -1

    def detect_hands(self, frame_rgb: np.ndarray, t: float) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """[(side, (21, 4) x y z score, (21, 3) world)] for each hand found;
        side is the subject's ("left" | "right")."""
        ms = max(int(round(t * 1000.0)), self._last_ms + 1)
        self._last_ms = ms
        res = self._landmarker.detect_for_video(_mp_image(self._mp, frame_rgb), ms)
        hands = []
        for i, lms in enumerate(res.hand_landmarks or []):
            cat = res.handedness[i][0] if res.handedness and res.handedness[i] else None
            side = "left" if cat is not None and cat.category_name == "Left" else "right"
            score = float(cat.score) if cat is not None else 1.0
            a = np.array([[p.x, p.y, p.z, 1.0] for p in lms], dtype=np.float32)
            w = res.hand_world_landmarks[i] if res.hand_world_landmarks else None
            world = (np.array([[p.x, p.y, p.z] for p in w], dtype=np.float32) if w
                     else np.full((HAND_N, 3), np.nan, dtype=np.float32))
            hands.append((side, a, world, score))
        # two hands called the same side: keep the surer call, give the other the free slot
        if len(hands) == 2 and hands[0][0] == hands[1][0]:
            keep = 0 if hands[0][3] >= hands[1][3] else 1
            hands[1 - keep] = ("left" if hands[keep][0] == "right" else "right",) + hands[1 - keep][1:]
        return [(s, a, w) for s, a, w, _ in hands]

    def detect(self, frame_rgb: np.ndarray, t: float) -> Optional[PoseSample]:
        hands = self.detect_hands(frame_rgb, t)
        if not hands:
            return None
        return _assemble(HANDS, None, {s: (a, w) for s, a, w in hands}, offset=0)

    def close(self) -> None:
        self._landmarker.close()


class BodyHandsTracker:
    """Body and hands together: PoseLandmarker for the 33 body points and
    HandLandmarker for the fingers, 75 rows in all.  Each hand found goes
    to the body wrist it lies closest to, which holds steadier than the hand
    model's own left / right call, and a hand farther than 0.6 torso lengths
    from both visible wrists (someone else's) is dropped.  With no body or
    no visible wrist in frame, the hand model's call is used."""

    skeleton = BOTH
    REACH = 0.6         # torso lengths between a hand's wrist and the body's

    def __init__(self, variant: str = "full"):
        self._body = PoseTracker(variant=variant)
        self._hands = HandTracker()

    def detect(self, frame_rgb: np.ndarray, t: float) -> Optional[PoseSample]:
        body = self._body.detect(frame_rgb, t)
        hands = self._hands.detect_hands(frame_rgb, t)
        if body is None and not hands:
            return None
        wrists = {} if body is None else {
            s: body.xy[i] for s, i in (("left", 15), ("right", 16)) if body.visibility[i] >= 0.3}
        if wrists and hands:
            torso = np.nanmean([np.linalg.norm(body.xy[11] - body.xy[23]), np.linalg.norm(body.xy[12] - body.xy[24])])
            gate = max(0.05, self.REACH * float(torso))
            d = lambda a, s: float(np.linalg.norm(a[0, :2] - wrists[s])) if s in wrists else np.inf
            if len(hands) == 2:
                (_, a0, w0), (_, a1, w1) = hands
                straight = d(a0, "left") + d(a1, "right")
                crossed = d(a0, "right") + d(a1, "left")
                pairs = [("left", a0, w0), ("right", a1, w1)] if straight <= crossed else \
                        [("right", a0, w0), ("left", a1, w1)]
                if not np.isfinite(min(straight, crossed)):        # one wrist hidden: pair each hand on its own
                    pairs = [(min(wrists, key=lambda s: d(a, s)), a, w) for _, a, w in hands]
            else:
                pairs = [(min(wrists, key=lambda s: d(a, s)), a, w) for _, a, w in hands]
            hands = [(s, a, w) for s, a, w in pairs if d(a, s) <= gate]
        return _assemble(BOTH, body, {s: (a, w) for s, a, w in hands}, offset=N_LANDMARKS)

    def close(self) -> None:
        self._body.close()
        self._hands.close()


def _assemble(skeleton: Skeleton, body: Optional[PoseSample],
              hands: Dict[str, Tuple[np.ndarray, np.ndarray]], offset: int) -> PoseSample:
    """Pack a body sample and per-side hands into one sample of `skeleton`."""
    a = np.full((skeleton.n, 4), np.nan, dtype=np.float32)
    world = np.full((skeleton.n, 3), np.nan, dtype=np.float32)
    if body is not None:
        a[:N_LANDMARKS] = np.concatenate([body.xy, body.z[:, None], body.visibility[:, None]], axis=1)
        world[:N_LANDMARKS] = body.world
    for side, (ha, hw) in hands.items():
        s = offset + (0 if side == "left" else HAND_N)
        a[s:s + HAND_N], world[s:s + HAND_N] = ha, hw
    return PoseSample(xy=a[:, :2], z=a[:, 2], visibility=a[:, 3], world=world)


class FaceTracker:
    """MediaPipe FaceLandmarker in VIDEO mode: one face, 478 points (the
    face mesh with both irises).  Drawn as contours: face oval, lips, eyes,
    eyebrows and irises."""

    skeleton = FACEMESH

    def __init__(self, model_asset_path: Optional[str] = None,
                 min_detection: float = 0.5, min_presence: float = 0.5, min_tracking: float = 0.5):
        mp, BaseOptions, vision = _mediapipe()
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path or model_path(task="face")),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection,
            min_face_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._last_ms = -1

    def detect(self, frame_rgb: np.ndarray, t: float) -> Optional[PoseSample]:
        ms = max(int(round(t * 1000.0)), self._last_ms + 1)
        self._last_ms = ms
        res = self._landmarker.detect_for_video(_mp_image(self._mp, frame_rgb), ms)
        if not res.face_landmarks:
            return None
        a = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float32)[:FACE_N]
        return PoseSample(xy=a[:, :2], z=a[:, 2], visibility=np.ones(len(a), dtype=np.float32),
                          world=np.full((len(a), 3), np.nan, dtype=np.float32))

    def close(self) -> None:
        self._landmarker.close()


def make_tracker(tracker: str = "body", variant: str = "full"):
    """body (PoseLandmarker, 33 points) | hands (HandLandmarker, 2 × 21) |
    face (FaceLandmarker, 478) | both (body + hands, 75)."""
    if tracker == "hands":
        return HandTracker()
    if tracker == "face":
        return FaceTracker()
    if tracker == "both":
        return BodyHandsTracker(variant=variant)
    if tracker == "body":
        return PoseTracker(variant=variant)
    raise ValueError("tracker must be body | hands | face | both")


class LandmarkFilter:
    """One Euro filter over a pose sample (Casiez et al., CHI 2012): a
    low-pass whose cutoff rises with speed, so a still body stops jittering
    while a fast gesture keeps its sharpness.

    `smooth` in [0, 1] sets the resting cutoff on a log scale, from 10 Hz
    (0, almost raw) through 1 Hz (0.5) to 0.1 Hz (1, heavy).  `beta` is how
    fast the cutoff opens with speed, in Hz per (frame-heights per second).
    A missed detection resets the filter, and so does each landmark that
    goes missing on its own (one hand leaving the frame), so whatever
    re-enters starts where it is instead of sliding in from where it left."""

    JUMP = 0.25         # frame-heights in one step: an identity switch, never a movement

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
        x = np.concatenate([sample.xy, sample.z[:, None], sample.world], axis=1)      # (N, 6)
        if self._x is None or self._t is None or t <= self._t or self._x.shape != x.shape:
            self._t, self._x, self._dx = t, x, np.zeros_like(x)
            return sample
        dt = t - self._t
        prev = self._x
        fresh = np.isnan(prev) & ~np.isnan(x)            # (re)appeared: start where it is
        # a jump no limb makes in one frame is the tracker switching hands or
        # people: restart that landmark instead of gliding across the frame
        jumped = np.linalg.norm(x[:, :2] - prev[:, :2], axis=1, keepdims=True) > self.JUMP
        fresh |= jumped & ~np.isnan(x)
        dx = (x - prev) / dt
        dx_hat = np.nan_to_num(self._dx + self._alpha(dt, self.d_cutoff) * (dx - self._dx))
        min_cutoff = 10.0 ** (1.0 - 2.0 * float(np.clip(self.smooth, 0.0, 1.0)))
        # one cutoff per landmark, from its image-plane speed
        speed = np.linalg.norm(dx_hat[:, :2], axis=1, keepdims=True)
        x_hat = prev + self._alpha(dt, min_cutoff + self.beta * speed) * (x - prev)
        x_hat = np.where(fresh, x, x_hat)                # NaN where the landmark is gone
        dx_hat = np.where(fresh, 0.0, dx_hat)
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
    source frame's width / height, so renders stay undistorted.  `skeleton`
    is the landmark layout of the tracker that fills it (BODY, HANDS or
    BOTH).  Appends and reads share a lock: live, the pose thread appends
    while the embedder thread renders."""

    def __init__(self, max_seconds: Optional[float] = 30.0, aspect: float = 4 / 3,
                 skeleton: Skeleton = BODY):
        self.max_seconds = max_seconds
        self.aspect = float(aspect)
        self.skeleton = skeleton
        self._lock = threading.Lock()
        self._t: List[float] = []
        self._rows: List[np.ndarray] = []           # each (skeleton.n, 7)
        self._stacked: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # -------- writing --------
    def append(self, t: float, sample: Optional[PoseSample]) -> None:
        """Add a sample at time `t` (seconds).  None records a missed detection."""
        if sample is None:
            row = np.full((self.skeleton.n, _COLS), np.nan, dtype=np.float32)
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
        """Snapshot (t, L): t (T,) seconds, L (T, N, 7) as x y z visibility wx wy wz.
        Cached until the next append; treat the arrays as read-only."""
        with self._lock:
            if self._stacked is None:
                if not self._t:
                    return np.zeros(0), np.zeros((0, self.skeleton.n, _COLS), dtype=np.float32)
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
        """True when the newest sample holds anything tracked (a body or a hand)."""
        with self._lock:
            return bool(self._rows) and bool(np.isfinite(self._rows[-1][:, _X]).any())

    @classmethod
    def from_arrays(cls, t: np.ndarray, L: np.ndarray, aspect: float,
                    max_seconds: Optional[float] = None, skeleton: Optional[Skeleton] = None) -> "PoseTrajectory":
        if skeleton is None:
            skeleton = next((s for s in SKELETONS.values() if L.ndim == 3 and s.n == L.shape[1]), BODY)
        traj = cls(max_seconds=max_seconds, aspect=aspect, skeleton=skeleton)
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
        return PoseTrajectory.from_arrays(t[keep], L[keep], self.aspect, skeleton=self.skeleton)

    def resample(self, fps: float, start: Optional[float] = None, end: Optional[float] = None,
                 max_gap: float = 0.5) -> "PoseTrajectory":
        """Uniform timeline at `fps`, linearly interpolated between the two
        neighbouring samples.  A target time beside a missed detection, or
        inside a stretch longer than `max_gap` seconds with no samples at
        all, stays NaN, so interpolation never bridges a lost person."""
        t, L = self.arrays()
        if len(t) == 0:
            return PoseTrajectory(max_seconds=None, aspect=self.aspect, skeleton=self.skeleton)
        start = float(t[0]) if start is None else float(start)
        end = float(t[-1]) if end is None else float(end)
        n = max(1, int(np.floor((end - start) * fps + 1e-6)) + 1)
        ts = start + np.arange(n) / fps
        if len(t) == 1:
            out = np.repeat(L[:1], n, axis=0)
            out[np.abs(ts - t[0]) > max_gap] = np.nan
            return PoseTrajectory.from_arrays(ts, out, self.aspect, skeleton=self.skeleton)

        i = np.clip(np.searchsorted(t, ts, side="right") - 1, 0, len(t) - 2)
        dt = t[i + 1] - t[i]
        w = np.clip((ts - t[i]) / np.where(dt > 0, dt, 1.0), 0.0, 1.0)[:, None, None]
        out = (1.0 - w) * L[i] + w * L[i + 1]           # NaN on either side propagates
        exact_lo, exact_hi = (w[:, 0, 0] < 1e-6), (w[:, 0, 0] > 1 - 1e-6)
        out[exact_lo] = L[i][exact_lo]
        out[exact_hi] = L[i + 1][exact_hi]
        out[(dt > max_gap) & ~exact_lo & ~exact_hi] = np.nan
        out[(ts < t[0] - max_gap) | (ts > t[-1] + max_gap)] = np.nan
        return PoseTrajectory.from_arrays(ts, out.astype(np.float32), self.aspect, skeleton=self.skeleton)

    # -------- persistence --------
    def save(self, path) -> None:
        """Write t, landmarks, aspect and the skeleton name as compressed npz
        (a path or a binary file object)."""
        t, L = self.arrays()
        np.savez_compressed(path, t=t, landmarks=L, aspect=np.float32(self.aspect),
                            skeleton=np.array(self.skeleton.name))

    @classmethod
    def load(cls, path: str) -> "PoseTrajectory":
        z = np.load(path)
        sk = SKELETONS.get(str(z["skeleton"])) if "skeleton" in z.files else None
        return cls.from_arrays(z["t"], z["landmarks"], float(z["aspect"]), skeleton=sk)


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


def _joint_color(j: int, sk: Skeleton, style: PoseStyle) -> Tuple[int, int, int]:
    if style.color == "mono":
        return _WHITE
    return _WARM if j in sk.left else _COOL if j in sk.right else _WHITE


def _bone_color(a: int, b: int, sk: Skeleton, style: PoseStyle) -> Tuple[int, int, int]:
    if style.color == "mono":
        return _WHITE
    if a in sk.left and b in sk.left:
        return _WARM
    if a in sk.right and b in sk.right:
        return _COOL
    return _WHITE


def _project(L: np.ndarray, aspect: float, W: int, H: int, style: PoseStyle, sk: Skeleton = BODY) -> np.ndarray:
    """(T, N, ≥4) landmarks → (T, N, 2) pixel coordinates under `style.fit`.
    Landmarks below `min_visibility` come back NaN.  The window and body fits
    are computed over all T rows at once, so frames rendered from one
    projection share a stable framing."""
    uv = np.stack([L[..., _X] * aspect, L[..., _Y]], axis=-1)      # isotropic units, height = 1
    uv[L[..., _V] < style.min_visibility] = np.nan

    if style.fit == "camera":
        lo, ext = np.array([0.0, 0.0]), np.array([aspect, 1.0])
    elif style.fit == "body":
        # each sample re-centred on its own torso (hands: on the hands) and
        # scaled by the median torso (hand) length over the sequence
        with warnings.catch_warnings():                             # rows with nothing in view → NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            centre = np.nanmean(uv[:, list(sk.centre)], axis=1, keepdims=True)
        uv = uv - centre
        lengths = np.stack([np.linalg.norm(uv[:, a] - uv[:, b], axis=-1) for a, b in sk.span])
        scale = float(np.nanmedian(lengths)) if np.isfinite(lengths).any() else 0.25
        half = sk.reach * max(scale, 1e-3) * (1.0 + style.margin)
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
          joints: Tuple[int, ...], bg: Tuple[int, int, int], sk: Skeleton = BODY) -> None:
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
        # a step longer than a quarter of the canvas is the tracker switching
        # hands or people; the trail breaks there instead of streaking across
        max_step = 0.25 * max(img.shape[:2]) * (1 << _SHIFT)
        for j in joints:
            base = np.asarray(_joint_color(j, sk, style), dtype=np.float32)
            cols = (bg_arr + (base - bg_arr) * fade[:, None]).astype(int)
            step = np.linalg.norm((Q[idx, j] - Q[idx - 1, j]).astype(np.float64), axis=-1)
            seg = ok[idx - 1, j] & ok[idx, j] & (fade > 0.01) & (step <= max_step)
            for n in np.flatnonzero(seg):
                i = idx[n]
                cv2.line(img, (int(Q[i - 1, j, 0]), int(Q[i - 1, j, 1])), (int(Q[i, j, 0]), int(Q[i, j, 1])),
                         tuple(int(c) for c in cols[n]), int(widths[n]), aa, _SHIFT)

    cur, cur_ok = Q[k], ok[k]
    hide = sk.skip | (sk.face if not style.face else frozenset())
    sw = max(1, int(round(style.skeleton_width)))
    for a, b in sk.connections:
        if a in hide or b in hide or not (cur_ok[a] and cur_ok[b]):
            continue
        cv2.line(img, (int(cur[a, 0]), int(cur[a, 1])), (int(cur[b, 0]), int(cur[b, 1])),
                 _bone_color(a, b, sk, style), sw, aa, _SHIFT)
    r = int(round(style.joint_radius * (1 << _SHIFT)))
    if r > 0:
        r_small = max(1 << _SHIFT, int(round(r * 0.55)))
        for j in np.flatnonzero(cur_ok):
            if j not in hide and (sk.dots is None or j in sk.dots):
                cv2.circle(img, (int(cur[j, 0]), int(cur[j, 1])), r_small if j in sk.small else r,
                           _joint_color(int(j), sk, style), -1, aa, _SHIFT)


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
            sk = traj.skeleton
            Q, ok = _fixed(_project(L, traj.aspect, W, H, style, sk))
            _draw(img, Q, ok, ts, len(ts) - 1, style, resolve_joints(style.trace_joints, sk), bg, sk)
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
    sk = traj.skeleton
    Q, ok = _fixed(_project(L, traj.aspect, W, H, style, sk))
    joints = resolve_joints(style.trace_joints, sk)
    first = int(np.searchsorted(ts, start - 1e-9, side="left"))
    for k in range(first, len(ts)):
        img = np.empty((H, W, 3), dtype=np.uint8)
        img[:] = style.background
        _draw(img, Q, ok, ts, k, style, joints, style.background, sk)
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


def track_video(path: str, tracker=None, max_side: Optional[int] = 960,
                stride: int = 1, start: float = 0.0, end: Optional[float] = None,
                variant: str = "full", smooth: float = 0.0, progress: bool = False) -> PoseTrajectory:
    """Track a video file (or its [start, end] seconds) into an unbounded
    PoseTrajectory.  `tracker` is body | hands | both, or a tracker object
    to reuse.  `smooth` > 0 runs the landmarks through a LandmarkFilter."""
    own = tracker is None or isinstance(tracker, str)
    if own:
        tracker = make_tracker(tracker or "body", variant=variant)
    filt = LandmarkFilter(smooth=smooth)
    traj: Optional[PoseTrajectory] = None
    try:
        for n, (t, rgb) in enumerate(iter_video(path, max_side=max_side, stride=stride, start=start, end=end)):
            if traj is None:
                traj = PoseTrajectory(max_seconds=None, aspect=rgb.shape[1] / rgb.shape[0],
                                      skeleton=tracker.skeleton)
            traj.append(t, filt(t, tracker.detect(rgb, t)))
            if progress and n % 100 == 0:
                print(f"[pose] {t:7.2f} s")
    finally:
        if own:
            tracker.close()
    return traj or PoseTrajectory(max_seconds=None, skeleton=tracker.skeleton)


class PoseStream:
    """Tracker + running trajectory for one live camera fed JPEG frames.
    `tracker` is body | hands | both.  Timestamps come from the arrival
    clock.  The tracker loads on the first frame, so building a stream
    costs nothing until it is used."""

    def __init__(self, max_seconds: float = 30.0, variant: str = "full", smooth: float = 0.5,
                 tracker: str = "body"):
        self.variant = variant
        self.kind = tracker
        self.trajectory = PoseTrajectory(max_seconds=max_seconds, skeleton=SKELETONS[tracker])
        self.filter = LandmarkFilter(smooth=smooth)  # set .filter.smooth to retune live
        self._tracker = None
        self._lock = threading.Lock()               # close() waits for a detect in flight
        self._t0 = time.perf_counter()

    def push_jpeg(self, jpeg: bytes) -> bool:
        """Decode, track and append one frame; True when anything was found."""
        import cv2
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return False
        return self.push_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def push_rgb(self, rgb: np.ndarray) -> bool:
        with self._lock:
            if self._tracker is None:
                self._tracker = make_tracker(self.kind, variant=self.variant)
            t = time.perf_counter() - self._t0
            self.trajectory.aspect = rgb.shape[1] / rgb.shape[0]
            sample = self.filter(t, self._tracker.detect(rgb, t))
            self.trajectory.append(t, sample)
            return sample is not None

    def push_landmarks(self, rows: np.ndarray, t: float, aspect: float) -> bool:
        """Append landmarks tracked elsewhere (the browser, on the GPU).
        `rows` is (N, 7) in this stream's layout, empty when nothing was
        found; `t` is the tracker's clock in seconds."""
        with self._lock:
            self.trajectory.aspect = float(aspect)
            sample = None
            if len(rows) == self.trajectory.skeleton.n:
                r = np.asarray(rows, dtype=np.float32)
                sample = PoseSample(xy=r[:, :2], z=r[:, 2], visibility=r[:, 3], world=r[:, 4:7])
            sample = self.filter(t, sample)
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
    ap.add_argument("--tracker", default="body", choices=sorted(SKELETONS),
                    help="body (33 points) | hands (21 per hand, with fingers) | face (478) | both")
    ap.add_argument("--joints", help="joints that leave traces: a set name "
                    + "; ".join(f"{k}: {', '.join(s.joint_sets)}" for k, s in SKELETONS.items())
                    + " (default: the tracker's own), or comma-separated indices")
    ap.add_argument("--mono", action="store_true", help="white skeleton instead of left/right colours")
    ap.add_argument("--overlay", action="store_true", help="draw over the dimmed source video")
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--max-side", type=int, default=960, help="downscale frames before tracking")
    ap.add_argument("--smooth", type=float, default=0.5, help="landmark smoothing 0 (raw) .. 1 (heavy), One Euro filter")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the input")
    ap.add_argument("--end", type=float, help="stop at this many seconds into the input")
    a = ap.parse_args(argv)

    joints: Union[str, Sequence[int]] = (
        a.joints if a.joints and not a.joints[0].isdigit()
        else [int(x) for x in a.joints.split(",")] if a.joints
        else SKELETONS[a.tracker].default_joints)
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
        traj = track_video(a.input, tracker=a.tracker, max_side=a.max_side, start=a.start, end=a.end,
                           variant=a.model, smooth=a.smooth, progress=True)
        found = int(np.isfinite(traj.arrays()[1][..., _X]).any(axis=1).sum())
        print(f"[pose] tracked {len(traj)} frames in {time.perf_counter() - t0:.1f} s, "
              f"{a.tracker} found in {found} ({100.0 * found / max(len(traj), 1):.0f} %)")
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
