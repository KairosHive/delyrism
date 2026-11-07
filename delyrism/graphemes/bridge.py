"""
bridge.py — IMU or pose estimation → graphemes → weights over symbols
====================================================================

The grapheme basis (basis.py) reads drawing motion.  This module turns
the live movement sources into a 2D hand path and reads it against a set
of symbols:

    pose keypoints (T, J, 2|3) ──┐
    IMU accel + gyro (T, 6) ─────┼─→ path (T, 2) ─→ grapheme sequence ─→ P(symbol)
    IMU quaternions (T, 4) ──────┘                   └→ P(grapheme)

Paths
    pose        a joint (a wrist) on the body's front plane: shoulder line
                × up, relative to the shoulders, so turning or travelling
                does not draw.  2D keypoints use the image plane.
    IMU raw     orientation from gyro and gravity (complementary filter);
                the forearm, which hangs along gravity at rest, becomes a
                pointer: its yaw and elevation are the path.  Elevation is
                gravity-referenced, and yaw keeps the drawer's left and
                right whatever their facing.
    IMU fused   the same pointer from a segment's quaternion (any fused
                IMU, an Xsens suit).

Graphemes along a path: windows of fixed arc length slide along it, each
written in the orthonormal wavelet basis (atom × direction).  Their mean
is P(grapheme | motion).  Their order is kept as a sequence.

Symbols: each of SYMBOLS10 is drawn procedurally in its common ways
(directions, stroke orders, start points of closed shapes), in air mode
(the travel between strokes is drawn, since a hand never lifts in the
air).  A motion's distance to a symbol is the DTW alignment of grapheme
sequences with the nearest of its drawings; weights are a softmax.  New
symbols come from any drawing (`add_symbol`) or a recorded movement
(`add_symbol_path`), which also personalises an existing one.

    br = SymbolBridge()
    br.weights(path)                                       # {"circle": 0.71, ...}
    br.weights_pose(keypoints, fps=30, joints=[10, 11], up=(0, 1, 0), shoulders=(4, 8))
    br.weights_quat(forearm_quaternions, axis=(0, -1, 0))
    t, W, names = br.stream(path, fps=60, window=3.0, hop=0.5)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import basis as G
from .glyphs import Drawing, _time

# ----------------------------------------------------------------------------
#  ten visual symbols, each with its common ways of being drawn
# ----------------------------------------------------------------------------
# strokes in a unit box, y up; each variant is a list of strokes in order


def _arc(cx, cy, r, a0, a1, n=64):
    t = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


def _poly(*pts, n=24):
    P = np.asarray(pts, float)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        u = np.linspace(0, 1, n)[1:, None]
        out.extend(a + (b - a) * u)
    return np.array(out)


def _rev(strokes):
    return [s[::-1] for s in strokes[::-1]]


def _circle():
    ccw = [_arc(0.5, 0.5, 0.5, np.pi / 2, np.pi / 2 + 2 * np.pi, 96)]
    return [ccw, [ccw[0][::-1]]]


def _cross():
    v, h = _poly((0.5, 1), (0.5, 0)), _poly((0, 0.5), (1, 0.5))
    return [[v, h], [h, v], [v[::-1], h], [v, h[::-1]]]


def _saltire():
    a, b = _poly((0, 1), (1, 0)), _poly((1, 1), (0, 0))
    return [[a, b], [b, a], [a, b[::-1]], [a[::-1], b]]


def _tri_up():
    ccw = [_poly((0.5, 1), (0, 0), (1, 0), (0.5, 1))]
    cw = [_poly((0.5, 1), (1, 0), (0, 0), (0.5, 1))]
    bl = [_poly((0, 0), (0.5, 1), (1, 0), (0, 0))]
    return [ccw, cw, bl]


def _tri_down():
    a = [_poly((0, 1), (1, 1), (0.5, 0), (0, 1))]
    b = [_poly((0, 1), (0.5, 0), (1, 1), (0, 1))]
    return [a, b]


def _square():
    a = [_poly((0, 1), (0, 0), (1, 0), (1, 1), (0, 1))]
    b = [_poly((0, 1), (1, 1), (1, 0), (0, 0), (0, 1))]
    return [a, b]


def _spiral():
    t = np.linspace(0, 2.6 * 2 * np.pi, 220)
    r = 0.04 + 0.46 * t / t[-1]
    out = np.c_[0.5 + r * np.cos(t), 0.5 + r * np.sin(t)]
    mirror = out * [-1, 1] + [1, 0]
    return [[out], [mirror], [out[::-1]], [mirror[::-1]]]


def _wave():
    x = np.linspace(0, 1, 120)
    w = np.c_[x, 0.5 + 0.32 * np.sin(2 * np.pi * 2 * x)]
    return [[w], [w[::-1]]]


def _zigzag():
    z = [_poly((0.65, 1), (0.25, 0.5), (0.75, 0.5), (0.35, 0))]
    m = [z[0] * [-1, 1] + [1, 0]]
    return [z, _rev(z), m, _rev(m)]


def _infinity():
    t = np.linspace(0, 2 * np.pi, 160)
    # lemniscate of Bernoulli, lying across, starting at the centre
    lx = 0.5 + 0.5 * np.cos(t - np.pi / 2) / (1 + np.sin(t - np.pi / 2) ** 2)
    ly = 0.5 + 0.5 * np.sin(t - np.pi / 2) * np.cos(t - np.pi / 2) / (1 + np.sin(t - np.pi / 2) ** 2)
    a = np.c_[lx, ly * 1.6 - 0.3]
    return [[a], [a * [-1, 1] + [1, 0]], [a[::-1]]]


SYMBOLS10: Dict[str, Callable[[], List[List[np.ndarray]]]] = {
    "circle": _circle, "cross": _cross, "x": _saltire, "triangle up": _tri_up, "triangle down": _tri_down,
    "square": _square, "spiral": _spiral, "wave": _wave, "zigzag": _zigzag, "infinity": _infinity,
}
SYMBOL_GLYPHS = {"circle": "○", "cross": "+", "x": "×", "triangle up": "△", "triangle down": "▽",
                 "square": "□", "spiral": "꩜", "wave": "∿", "zigzag": "ϟ", "infinity": "∞"}


def symbol_drawings(name: str, maker=None) -> List[Drawing]:
    """The variants of one symbol as timed Drawings (power-law speed)."""
    maker = maker or SYMBOLS10[name]
    out = []
    for k, strokes in enumerate(maker()):
        S = [np.asarray(s, float) * 100.0 for s in strokes]
        out.append(Drawing("symbols10", name, f"v{k}", _time(S, 100.0)))
    return out


# ----------------------------------------------------------------------------
#  paths
# ----------------------------------------------------------------------------

def _normalise(P2: np.ndarray) -> np.ndarray:
    ok = np.isfinite(P2).all(1)
    if not ok.any():
        return P2
    lo, hi = np.nanmin(P2[ok], 0), np.nanmax(P2[ok], 0)
    s = max(float(np.max(hi - lo)), 1e-6)
    return (P2 - (lo + hi) / 2) / s


def _resample(P: np.ndarray, fps: float, fs: float = G.FS) -> np.ndarray:
    t = np.arange(len(P)) / fps
    g = np.arange(0, t[-1] + 1e-9, 1 / fs)
    out = np.stack([np.interp(g, t[np.isfinite(P[:, k])], P[np.isfinite(P[:, k]), k]) for k in range(P.shape[1])], 1)
    return out


def air_path(d: Drawing, fs: float = G.FS) -> Tuple[np.ndarray, np.ndarray]:
    """A drawing as a hand path in the air: travel between strokes drawn."""
    X, _ = G.motion(d, fs, lift=0.0)
    return X, np.ones(len(X), bool)


def path_from_pose(kp: np.ndarray, fps: float, joints: Sequence[int], up: Optional[Sequence[float]] = None,
                   shoulders: Optional[Tuple[int, int]] = None, image_y_down: bool = True) -> np.ndarray:
    """(T, 2) path of the mean of `joints` on the body's front plane.

    3D keypoints: the plane spanned by the shoulder line (horizontal part)
    and `up`; positions relative to the shoulders' midpoint, so walking
    or turning does not draw.  2D keypoints: image x, y (y flipped up)."""
    P = np.nanmean(kp[:, list(joints), :], axis=1)
    if kp.shape[2] == 2:
        out = P.copy()
        if image_y_down:
            out[:, 1] = -out[:, 1]
        return out
    up = np.asarray(up if up is not None else (0, 1, 0), float)
    up /= np.linalg.norm(up)
    if shoulders is None:
        return np.c_[P @ np.cross(up, [0, 0, 1] if abs(up[2]) < 0.9 else [1, 0, 0]), P @ up]
    L, R = kp[:, shoulders[0], :], kp[:, shoulders[1], :]
    right = R - L
    right -= (right @ up)[:, None] * up
    right /= np.linalg.norm(right, axis=1, keepdims=True) + 1e-9
    right = _smooth_rows(right)
    mid = (L + R) / 2
    rel = P - mid
    return np.c_[(rel * right).sum(1), rel @ up]


def _smooth_rows(V: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    V = gaussian_filter1d(V, sigma, axis=0, mode="nearest")
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)


# ---- IMU

def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _rotm(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def imu_orientation(acc: np.ndarray, gyr: np.ndarray, fps: float, gain: float = 0.02) -> np.ndarray:
    """(T, 3, 3) sensor-to-world rotations (z up) from a complementary
    filter: gyro integration, tilt pulled toward the measured gravity."""
    a0 = acc[: max(3, int(0.2 * fps))].mean(0)
    z = a0 / (np.linalg.norm(a0) + 1e-9)                    # gravity direction in the sensor frame at rest
    # initial rotation taking z (sensor) to world up
    v = np.cross(z, [0, 0, 1.0])
    s, c = np.linalg.norm(v), float(z @ [0, 0, 1.0])
    q = np.array([1.0, 0, 0, 0]) if s < 1e-6 else np.r_[math.cos(math.atan2(s, c) / 2), v / s * math.sin(math.atan2(s, c) / 2)]
    out = []
    dt = 1.0 / fps
    for k in range(len(gyr)):
        w = gyr[k]
        th = np.linalg.norm(w) * dt
        if th > 1e-12:
            ax = w / np.linalg.norm(w)
            q = _quat_mul(q, np.r_[math.cos(th / 2), ax * math.sin(th / 2)])
        Rk = _rotm(q)
        g_world = Rk @ (acc[k] / (np.linalg.norm(acc[k]) + 1e-9))
        corr = np.cross(g_world, [0, 0, 1.0])               # small rotation pulling measured gravity to +z
        n = np.linalg.norm(corr)
        if n > 1e-9 and abs(np.linalg.norm(acc[k]) - 1.0) < 0.3:   # only trust gravity when not accelerating hard
            ang = gain * math.asin(min(1.0, n))
            q = _quat_mul(np.r_[math.cos(ang / 2), corr / n * math.sin(ang / 2)], q)
        q /= np.linalg.norm(q)
        out.append(_rotm(q))
    return np.array(out)


def path_from_imu(acc_g: np.ndarray, gyr_dps: np.ndarray, fps: float, method: str = "pointer",
                  active: slice = slice(None)) -> np.ndarray:
    """(T, 2) drawing path from a wrist IMU (accelerometer in g, gyroscope
    in deg/s), up = against gravity.  Orientation is tracked over the
    whole take; the path covers `active` (the arm raised to draw)."""
    gyr = np.radians(gyr_dps)
    R = imu_orientation(acc_g, gyr, fps)
    a0 = acc_g[: max(3, int(0.3 * fps))].mean(0)
    axis = -a0 / (np.linalg.norm(a0) + 1e-9)                 # the forearm hangs along gravity at rest
    if method == "pointer":
        p = np.einsum("tij,j->ti", R, axis)[active]          # pointing direction in the world (z up)
        yaw = np.unwrap(np.arctan2(p[:, 1], p[:, 0]))
        elev = np.arcsin(np.clip(p[:, 2], -1, 1))
        return np.c_[-yaw * np.cos(elev), elev]              # rightward sweeps turn clockwise seen from above
    fwd = np.einsum("tij,j->ti", R[active], axis).mean(0)
    R, acc_g = R[active], acc_g[active]
    a = np.einsum("tij,tj->ti", R, acc_g) - [0, 0, 1.0]     # world linear acceleration, g
    a *= 9.81
    a -= a.mean(0)
    v = np.cumsum(a, 0) / fps
    t = np.arange(len(v))
    v -= np.outer(t / max(len(v) - 1, 1), v[-1]) + v[0]      # at rest at both ends: remove the linear drift
    p = np.cumsum(v, 0) / fps
    # the drawing plane: the two principal directions, the in-plane one closest to up as y
    C = p - p.mean(0)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    n = Vt[2]
    fwd[2] = 0.0
    if fwd @ n < 0:                                          # the plane's normal points away from the body
        n = -n
    upv = np.array([0, 0, 1.0]) - n[2] * n
    if np.linalg.norm(upv) < 1e-6:
        upv = Vt[1]
    upv /= np.linalg.norm(upv)
    xv = np.cross(n, upv)                                    # the drawer's right
    return np.c_[C @ xv, C @ upv]


# ----------------------------------------------------------------------------
#  the bridge
# ----------------------------------------------------------------------------

def _arc_resample(P: np.ndarray, M: int = 256) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return np.repeat(P[:1], M, 0)
    s = np.linspace(0, d[-1], M)
    return np.stack([np.interp(s, d, P[:, k]) for k in range(2)], 1)


def path_from_quat(q: np.ndarray, axis: Sequence[float] = (0, 0, -1)) -> np.ndarray:
    """(T, 2) pointer path from a segment's orientation quaternions (w, x, y,
    z; segment to world, z up), the output of any fused IMU.  `axis` is the
    segment's long axis in its own frame (Xsens arm segments: -z)."""
    R = np.array([_rotm(qq / (np.linalg.norm(qq) + 1e-12)) for qq in q])
    p = np.einsum("tij,j->ti", R, np.asarray(axis, float))
    yaw = np.unwrap(np.arctan2(p[:, 1], p[:, 0]))
    elev = np.arcsin(np.clip(p[:, 2], -1, 1))
    return np.c_[-yaw * np.cos(elev), elev]


# ----------------------------------------------------------------------------
#  dense graphemes along a path
# ----------------------------------------------------------------------------

def grapheme_sequence(path2d: np.ndarray, basis: G.GraphemeBasis, frac: float = 0.25, n_win: int = 24,
                      cells: int = 1) -> np.ndarray:
    """(n_win, tokens): windows of fixed arc length (frac of the path) at
    n_win evenly spaced places along it, each written in the orthonormal
    basis; each row is that window's distribution over graphemes (atom x
    direction [x cell]).  Arc length removes speed; dense windows remove
    any dependence on where a fragment would be cut."""
    P = np.asarray(path2d, float)
    P = P[np.isfinite(P).all(1)]
    k = len(basis.planar)
    if len(P) < 4 or np.ptp(P, 0).max() < 1e-9:
        return np.full((n_win, cells * cells * k * G.DIRS), 1.0 / (cells * cells * k * G.DIRS))
    P = _normalise(_arc_resample(_normalise(P)))
    M = len(P)
    n = max(8, int(frac * M))
    Wp = basis.W[basis.planar]
    cb = basis.configured(cells=cells)
    g = np.linspace(0, 1, G.L + 1)
    u = np.linspace(0, 1, n)
    rows = []
    for a in np.linspace(0, M - n, n_win).astype(int):
        seg = P[a:a + n]
        D = np.diff(np.stack([np.interp(g, u, seg[:, j]) for j in range(2)], 1), axis=0)
        z = D[:, 0] + 1j * D[:, 1]
        z = z / max(float(np.abs(z).sum()), 1e-9)
        e = (cb._cell_weights(seg.mean(0))[:, None, None] * G._dir_split(Wp @ z)[None]).reshape(-1)
        rows.append(e / max(e.sum(), 1e-12))
    return np.array(rows)


def dtw(A: np.ndarray, B: np.ndarray, band: float = 0.3) -> float:
    """DTW between two grapheme sequences, cost 1 - Bhattacharyya per window pair."""
    C = 1.0 - np.sqrt(A) @ np.sqrt(B).T
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    w = max(int(band * max(n, m)), abs(n - m))
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(m, i + w)
        for j in range(lo, hi + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def _start_points(drawings: Sequence[Drawing], k: int = 4) -> List[Drawing]:
    """A closed one-stroke shape can start anywhere along it."""
    out = []
    for d in drawings:
        out.append(d)
        S = d.strokes[0]
        if len(d.strokes) == 1 and len(S) > 8 and np.linalg.norm(S[0, :2] - S[-1, :2]) < 5:
            for r in range(1, k):
                sh = int(r * len(S) / k)
                pts = np.vstack([S[sh:-1, :2], S[:sh + 1, :2]])
                out.append(Drawing(d.system, d.symbol, f"{d.drawer}s{r}",
                                   [np.c_[pts, np.linspace(S[0, 2], S[-1, 2], len(pts))]]))
    return out


# ----------------------------------------------------------------------------
#  where the drawing is (recordings that start and end at rest)
# ----------------------------------------------------------------------------

def trim_rest_moves(path2d: np.ndarray, steep: float = 0.6, still: float = 0.15) -> slice:
    """Drop the stillness and the near-vertical rise from rest at the start,
    and the near-vertical drop back to rest at the end."""
    from scipy.ndimage import gaussian_filter1d
    Q = gaussian_filter1d(np.asarray(path2d, float), 1.5, axis=0, mode="nearest")
    v = np.gradient(Q, axis=0)
    sp = np.linalg.norm(v, axis=1)
    moving = sp > still * np.percentile(sp, 90)
    up = (v[:, 1] > 0) & (np.abs(v[:, 0]) < steep * np.abs(v[:, 1]))
    down = (v[:, 1] < 0) & (np.abs(v[:, 0]) < steep * np.abs(v[:, 1]))
    i0, i1 = 0, len(Q) - 1
    while i0 < len(Q) - 1 and (not moving[i0] or up[i0]):
        i0 += 1
    while i1 > i0 and (not moving[i1] or down[i1]):
        i1 -= 1
    return slice(i0, i1 + 1) if i1 - i0 > 5 else slice(0, len(Q))


def active_imu(acc_g: np.ndarray, fps: float, angle: float = 40.0) -> slice:
    """The longest stretch with the forearm turned away from its resting
    tilt (the first 0.3 s): the arm raised to draw."""
    from scipy.ndimage import uniform_filter1d
    a = uniform_filter1d(acc_g, max(1, int(0.2 * fps)), axis=0)
    g0 = a[: max(2, int(0.3 * fps))].mean(0)
    cosang = (a @ g0) / (np.linalg.norm(a, axis=1) * np.linalg.norm(g0) + 1e-9)
    return _longest(np.degrees(np.arccos(np.clip(cosang, -1, 1))) > angle)


def _longest(mask: np.ndarray) -> slice:
    best, start = (0, 0), None
    for i, m in enumerate(np.r_[mask, False]):
        if m and start is None:
            start = i
        if not m and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return slice(best[0], best[1]) if best[1] - best[0] >= 3 else slice(0, len(mask))


# ----------------------------------------------------------------------------
#  the bridge
# ----------------------------------------------------------------------------

@dataclass
class SymbolBridge:
    """Movement paths → P(grapheme) and → weights over a set of symbols.

    Each symbol is held as the grapheme sequences of its ways of being
    drawn (and their mean, its distribution over graphemes).  A motion's
    likelihood under a symbol is how closely its grapheme sequence aligns
    with the nearest of them (DTW); weights are a softmax of that."""
    symbols: Optional[Dict[str, Callable]] = None
    basis: Optional[G.GraphemeBasis] = None
    temperature: float = 0.015         # faithful mode, in DTW distance units
    z_temperature: float = 0.35        # distinct mode, in standard deviations
    frac: float = 0.25
    n_win: int = 24
    templates: Optional[Dict[str, List[np.ndarray]]] = None
    background: Optional[Dict[str, Tuple[float, float]]] = None   # per-symbol distance mean, sd over ordinary movement

    def __post_init__(self):
        self.symbols = self.symbols or SYMBOLS10
        self.basis = self.basis or G.default_basis(6)
        if self.templates is None:
            self.templates = {}
            for name, maker in self.symbols.items():
                self.add_symbol(name, symbol_drawings(name, maker))

    def add_symbol(self, name: str, drawings: Sequence[Drawing]) -> None:
        """A new symbol from drawings of it (glyph tracings, a sketch, a
        recorded gesture as a Drawing); each is one way of drawing it."""
        self.templates[name] = [self.sequence(air_path(d)[0][:, :2]) for d in _start_points(drawings)]

    def add_symbol_path(self, name: str, path2d: np.ndarray) -> None:
        """Add one recorded movement path as a way of drawing `name`."""
        self.templates.setdefault(name, []).append(self.sequence(path2d))

    def sequence(self, path2d: np.ndarray) -> np.ndarray:
        return grapheme_sequence(path2d, self.basis, self.frac, self.n_win)

    def graphemes(self, path2d: np.ndarray) -> np.ndarray:
        """P(grapheme | motion): the path's windows pooled (atom x direction)."""
        p = self.sequence(path2d).mean(0)
        return p / p.sum()

    @property
    def profiles(self) -> Dict[str, np.ndarray]:
        """P(grapheme | symbol): each symbol's drawings pooled."""
        return {n: np.mean([t.mean(0) for t in v], 0) for n, v in self.templates.items()}

    def distances(self, path2d: np.ndarray) -> Dict[str, float]:
        q = self.sequence(path2d)
        return {n: min(dtw(q, t) for t in v) for n, v in self.templates.items()}

    def calibrate(self, paths: Sequence[np.ndarray]) -> None:
        """Learn each symbol's distance mean and spread over ordinary movement
        (a session's recent windows, another dataset).  Simple symbols sit
        close to most movement and would otherwise win everywhere."""
        D = [self.distances(p) for p in paths if len(p) > 4]
        self.background = {n: (float(np.mean([d[n] for d in D])), float(np.std([d[n] for d in D]) + 1e-6))
                           for n in self.templates}

    def weights(self, path2d: np.ndarray, temperature: Optional[float] = None,
                mode: Optional[str] = None) -> Dict[str, float]:
        """Softmax over symbols.  mode "faithful": the absolutely nearest
        symbol wins; "distinct" (the default once calibrated): distances are
        standardised per symbol, so each has an equal baseline and a symbol
        wins when the movement is unusually close to it."""
        d = self.distances(path2d)
        names = list(d)
        mode = mode or ("distinct" if self.background else "faithful")
        if mode == "distinct" and self.background:
            x = -np.array([(d[n] - self.background[n][0]) / self.background[n][1] for n in names])
            x = x / (temperature or self.z_temperature)
        else:
            x = -np.array([d[n] for n in names]) / (temperature or self.temperature)
        w = np.exp(x - x.max())
        return dict(zip(names, (w / w.sum()).tolist()))

    def weights_pose(self, kp: np.ndarray, fps: float, joints: Sequence[int], trim: bool = False,
                     **kw) -> Dict[str, float]:
        P = path_from_pose(kp, fps, joints, **kw)
        return self.weights(P[trim_rest_moves(P)] if trim else P)

    def weights_imu(self, acc_g: np.ndarray, gyr_dps: np.ndarray, fps: float, trim: bool = True) -> Dict[str, float]:
        sl = active_imu(acc_g, fps) if trim else slice(None)
        return self.weights(path_from_imu(acc_g, gyr_dps, fps, "pointer", sl))

    def weights_quat(self, q: np.ndarray, axis: Sequence[float] = (0, 0, -1)) -> Dict[str, float]:
        return self.weights(path_from_quat(q, axis))

    def stream(self, path2d: np.ndarray, fps: float, window: float = 3.0, hop: float = 0.5,
               min_travel: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Symbol weights over sliding windows of a long path: (times, (n, S), names).
        A window that travels less than min_travel gets uniform weights."""
        names = list(self.templates)
        n, h = int(window * fps), max(1, int(hop * fps))
        ts, W = [], []
        for a in range(0, max(1, len(path2d) - n + 1), h):
            seg = path2d[a:a + n]
            if np.isfinite(seg).all(1).mean() < 0.6:
                continue
            travel = float(np.nansum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
            if travel <= min_travel:
                W.append(np.full(len(names), 1 / len(names)))
            else:
                w = self.weights(seg)
                W.append(np.array([w[k] for k in names]))
            ts.append((a + n / 2) / fps)
        return np.array(ts), np.array(W), names


# ----------------------------------------------------------------------------
#  invented symbols: an alphabet nobody has seen, for testing
# ----------------------------------------------------------------------------

_SYLLABLES = ["ka", "ru", "vo", "shi", "el", "ta", "mo", "in", "ze", "or", "ly", "an", "thé", "qu", "is",
              "ne", "ar", "ul", "ix", "da", "om", "sa", "ve", "yr"]


def _primitive(rng: np.random.Generator) -> np.ndarray:
    """One stroke shape starting at the origin, about one unit long."""
    kind = rng.choice(["line", "arc", "hook", "s", "loop", "zig", "wave", "corner"])
    if kind == "line":
        return _poly((0, 0), (1, 0))
    if kind == "arc":
        sw = rng.uniform(0.6, 1.5) * np.pi
        P = _arc(0, 0, 0.5, -np.pi / 2, -np.pi / 2 + sw, 60)
    elif kind == "hook":
        A = _poly((0, 0), (0.8, 0))
        B = _arc(0.8, 0.15, 0.15, -np.pi / 2, np.pi * rng.uniform(0.3, 0.9), 24)
        P = np.vstack([A, B[1:]])
    elif kind == "s":
        A = _arc(0, 0.25, 0.25, -np.pi / 2, np.pi / 2, 30)
        B = _arc(0, 0.75, 0.25, -np.pi / 2, -3 * np.pi / 2, 30)
        P = np.vstack([A, B[1:]])
    elif kind == "loop":
        r = rng.uniform(0.2, 0.35)
        P = _arc(0, r, r, -np.pi / 2, 3 * np.pi / 2, 64)
    elif kind == "zig":
        k = int(rng.integers(2, 4))
        pts = [(i / k, (0.35 if i % 2 else 0.0)) for i in range(k + 1)]
        P = _poly(*pts)
    elif kind == "wave":
        x = np.linspace(0, 1, 60)
        P = np.c_[x, 0.15 * np.sin(2 * np.pi * rng.uniform(1.0, 1.6) * x)]
    else:                                                     # corner
        a = rng.uniform(0.35, 0.8) * np.pi
        P = _poly((0, 0), (0.6, 0), (0.6 + 0.6 * np.cos(a), 0.6 * np.sin(a)))
    return P - P[0]


def _invented_glyph(rng: np.random.Generator) -> List[np.ndarray]:
    n = int(rng.integers(2, 5))
    strokes: List[np.ndarray] = []
    for _ in range(n):
        P = _primitive(rng)
        ang = rng.uniform(0, 2 * np.pi)
        Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        P = P @ Rm.T * rng.uniform(0.35, 0.8)
        if strokes and rng.random() < 0.55:                  # grow from an end of an earlier stroke
            src = strokes[int(rng.integers(len(strokes)))]
            anchor = src[-1] if rng.random() < 0.5 else src[0]
        else:
            anchor = rng.uniform(0.15, 0.85, 2)
        strokes.append(P + anchor)
    if rng.random() < 0.4:                                   # mirror symmetry about the vertical axis
        allp = np.vstack(strokes)
        cx = (allp[:, 0].min() + allp[:, 0].max()) / 2
        strokes += [np.c_[2 * cx - s[:, 0], s[:, 1]] for s in strokes[: max(1, 4 - len(strokes) + 1)]]
    allp = np.vstack(strokes)
    lo, hi = allp.min(0), allp.max(0)
    sc = max(float(np.max(hi - lo)), 1e-6)
    return [(s - lo) / sc for s in strokes]


def invented_symbols(n: int = 12, seed: int = 7, pool: int = 150) -> Dict[str, Callable]:
    """n invented glyphs, chosen from `pool` random candidates to lie as far
    apart as possible in grapheme space (farthest-point selection on the DTW
    distance of their air-drawn grapheme sequences).  {name: maker}, each
    maker giving the glyph drawn forward and drawn backward."""
    rng = np.random.default_rng(seed)
    cands = [_invented_glyph(rng) for _ in range(pool)]
    basis = G.default_basis(6)

    def seq(strokes):
        d = Drawing("invented", "c", "v0", _time([s * 100.0 for s in strokes], 100.0))
        return grapheme_sequence(air_path(d)[0][:, :2], basis)

    S = [seq(c) for c in cands]
    complexity = np.array([sum(np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in c) for c in cands])
    chosen = [int(np.argmax(complexity))]
    dmin = np.array([dtw(S[i], S[chosen[0]]) for i in range(pool)])
    while len(chosen) < n:
        i = int(np.argmax(dmin))
        chosen.append(i)
        dmin = np.minimum(dmin, [dtw(S[j], S[i]) for j in range(pool)])
    names, out = set(), {}
    for i in chosen:
        while True:
            nm = "".join(rng.choice(_SYLLABLES, int(rng.integers(2, 4))))
            if nm not in names:
                names.add(nm)
                break
        glyph = cands[i]
        out[nm] = (lambda g: (lambda: [g, _rev(g)]))(glyph)
    return out
