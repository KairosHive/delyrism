"""
motion.py — movement qualities from landmark trajectories, in words
====================================================================

`describe(traj, seconds)` reads the last few seconds of a PoseTrajectory
(body, hands, or body + hands) and measures *how* the movement moves.  The
dimensions follow Laban's Effort factors and Shape, measured with the
kinematic cues used in expressive-movement research (quantity of motion,
impulsivity, path directness, normalised jerk, contraction index,
periodicity; Camurri et al. 2003, Larboulette & Gibet 2015):

    energy      still ↔ vigorous        speed of the two most active effectors
    size        small ↔ broad           largest excursion of an arm or leg
    time        sustained ↔ sudden      peak acceleration against speed (Laban Time)
    weight      light ↔ strong          acceleration magnitude (Laban Weight)
    space       indirect ↔ direct       straightness of each stroke (Laban Space)
    flow        bound ↔ free            normalised jerk of each stroke (Laban Flow)
    shape       contracted ↔ expansive  reach of the extremities from the body centre
    level       low ↔ high              height of the arms, crouch
    symmetry    asymmetric ↔ mirrored   left against mirrored right, and their timing
    rhythm      irregular ↔ periodic    autocorrelation of the speed signal, with tempo

plus trends (opening / closing, rising / sinking, speeding up / settling),
travel, approach, turning, pauses and which part leads.  Hands add
openness, finger articulation and the hands drawing together or apart.

Units are body-relative: lengths in torso lengths (hands: wrist to middle
knuckle), time in seconds, so a dancer far from the camera reads like one
close by.  `to_text` keeps only the salient qualities (far from the middle
of their range) and writes them as one layered description that names no
body part: the pace, Laban's effort action or state and the size of the
gestures, the qualities behind them, the movement in space, and its
timing.  The pace (absolute energy) picks the register of the words, so
slow movement reads "gliding, smooth, serene" and only vigorous movement
can read "explosive" or "torrential".  That text is what queries the space
through its text tower.  The single levels (abstract, qualities, body)
remain available for comparison.

A fixed calibration leaves a real dancer at one end of several qualities
for most windows, so the same words keep returning.  `SessionNorm` keeps
each quality's recent history and re-centres it on the session's own
median and spread, so the words follow what changes within a performance
(the live query uses it by default).  Stillness stays absolute.

    reading = describe(traj, seconds=3.0)
    reading = describe(traj, seconds=3.0, session=norm)     # norm = SessionNorm(), one per dancer
    reading.text            # "Vigorous, sudden and strong movement. ..."
    reading.qualities       # {"energy": Quality(value=0.81, raw=1.9, ...), ...}

From the shell, a description every second of a video:
    python -m delyrism.motion dance.mp4 --tracker body --window 3
"""
from __future__ import annotations

import argparse
import itertools
import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pose import BODY, BOTH, HANDS, HAND_N, N_LANDMARKS, FINGERTIPS, PoseTrajectory, _V, _X, _Y

FPS = 30.0          # the window is resampled to this rate before differentiation


# ----------------------------------------------------------------------------
#  readings
# ----------------------------------------------------------------------------

@dataclass
class Quality:
    key: str
    value: float            # 0 (low pole) … 1 (high pole), NaN when not measurable
    raw: float              # the measure behind it, in `unit`
    unit: str
    low: str                # the low pole, in words
    high: str               # the high pole
    absolute: float = float("nan")   # the fixed-calibration value when `value` is session-relative


@dataclass
class MotionReading:
    tracker: str
    seconds: float
    coverage: float                         # fraction of the window with landmarks
    qualities: Dict[str, Quality] = field(default_factory=dict)
    trends: Dict[str, float] = field(default_factory=dict)     # signed rates, see describe()
    hands: Dict[str, Quality] = field(default_factory=dict)    # hand-specific qualities
    text: str = ""

    def to_dict(self) -> dict:
        clean = lambda q: {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                           for k, v in asdict(q).items()}
        return {
            "tracker": self.tracker, "seconds": self.seconds, "coverage": round(self.coverage, 3),
            "qualities": [clean(q) for q in self.qualities.values()],
            "hands": [clean(q) for q in self.hands.values()],
            "trends": {k: (round(v, 3) if math.isfinite(v) else None) for k, v in self.trends.items()},
            "text": self.text,
        }


# ----------------------------------------------------------------------------
#  numerics
# ----------------------------------------------------------------------------

def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*a, **k)


def _ramp(x: float, lo: float, hi: float, log: bool = False) -> float:
    """Map x from [lo, hi] to [0, 1], clipped; log for ratio-like measures."""
    if x is None or not np.isfinite(x):
        return float("nan")
    if log:
        x, lo, hi = math.log(max(x, 1e-9)), math.log(lo), math.log(hi)
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _smooth(P: np.ndarray, k: int = 5) -> np.ndarray:
    """NaN-aware moving average along time (axis 0)."""
    if len(P) < k:
        return P.copy()
    finite = np.isfinite(P)
    Z = np.where(finite, P, 0.0)
    ker = np.ones(k) / k
    conv = lambda a: np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 0, a)
    num, den = conv(Z), conv(finite.astype(float))
    out = np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan)
    out[~finite] = np.nan
    return out


def _d(P: np.ndarray) -> np.ndarray:
    return np.gradient(P, 1.0 / FPS, axis=0) if len(P) > 2 else np.full_like(P, np.nan)


def _norm(V: np.ndarray) -> np.ndarray:
    return np.linalg.norm(V, axis=-1)


def _slope(y: np.ndarray) -> float:
    """Least-squares slope of y per second, NaN-aware."""
    t = np.arange(len(y)) / FPS
    m = np.isfinite(y)
    if m.sum() < max(6, len(y) // 4):
        return float("nan")
    return float(np.polyfit(t[m], y[m], 1)[0])


def _strokes(spd: np.ndarray) -> List[Tuple[int, int]]:
    """Strokes of one effector from its speed profile: the runs between
    pauses, a pause being speed below a fifth of its 90th-percentile speed.
    Speed dips inside a run stay in the stroke (they shape its smoothness);
    sharp reversals are left out of curvature by `_curvature`."""
    ok = np.isfinite(spd)
    if ok.sum() < 6:
        return []
    th = max(0.06, 0.2 * float(np.nanpercentile(spd, 90)))
    moving = ok & (spd >= th)
    runs, start = [], None
    for i, m in enumerate(np.append(moving, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1)); start = None
    return [(a, b) for a, b in runs if b - a >= 4]


def _curvature(seg: np.ndarray, ds: float = 0.1) -> Tuple[float, float]:
    """(total turning in radians, arc length) of a path resampled at fixed
    arc-length steps `ds`, so tracker jitter at low speed adds no turning.
    Turns sharper than ~140° are reversals (stroke ends), not curvature."""
    steps = _norm(np.diff(seg, axis=0))
    s = np.concatenate([[0.0], np.cumsum(steps)])
    if s[-1] < 2 * ds:
        return 0.0, float(s[-1])
    q = np.arange(0.0, s[-1], ds)
    pts = np.stack([np.interp(q, s, seg[:, 0]), np.interp(q, s, seg[:, 1])], axis=1)
    d = np.diff(pts, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    return float(turn[turn < 2.4].sum()), float(s[-1])


def _sparc(speed: np.ndarray, fc: float = 6.0, amp_th: float = 0.05, pad: int = 4) -> float:
    """Spectral arc length of a speed profile (Balasubramanian et al. 2015):
    about −1.5 for one smooth bell, more negative as the profile breaks into
    sub-movements.  Robust to landmark noise, unlike jerk."""
    n = len(speed)
    nfft = int(2 ** (math.ceil(math.log2(n)) + pad))
    f = np.arange(nfft) * FPS / nfft
    M = np.abs(np.fft.fft(speed, nfft))
    M = M / (M.max() + 1e-12)
    sel = f <= fc
    f, M = f[sel], M[sel]
    idx = np.nonzero(M >= amp_th)[0]
    if len(idx) < 2:
        return -1.5
    f, M = f[idx[0]:idx[-1] + 1], M[idx[0]:idx[-1] + 1]
    if f[-1] - f[0] <= 0:
        return -1.5
    return float(-np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(M) ** 2)))


HOLD = 0.17        # seconds below the pause speed that count as a held stop (a reversal is quicker)
PAUSE = 0.4        # seconds of near-stillness that count as a held pause in the description


def _effector(R: np.ndarray, min_path: float) -> Dict[str, float]:
    """Stroke-level measures of one effector path R (T, 2), body-relative:
    curvature (rad per unit path), smoothness (SPARC per stroke), held stops
    per second, path."""
    spd = _norm(_d(R))
    strokes = _strokes(spd)
    turn, arc, sparcs, weights = 0.0, 0.0, [], []
    for a, b in strokes:
        seg = R[a:b + 1]
        if not np.isfinite(seg).all():
            continue
        t_, s_ = _curvature(seg)
        if s_ < min_path:
            continue
        turn += t_; arc += s_
        sparcs.append(_sparc(spd[a:b + 1])); weights.append(s_)
    holds = sum(1 for (_, b), (a2, _) in zip(strokes[:-1], strokes[1:]) if (a2 - b - 1) / FPS >= HOLD)
    secs = len(R) / FPS
    return {
        "curvature": turn / arc if arc > 0 else float("nan"),
        "sparc": float(np.average(sparcs, weights=weights)) if weights else float("nan"),
        "stopgo": holds / secs,
        "path": arc,
    }


def _effort(v: np.ndarray, a: np.ndarray, scale: float = 1.0) -> Tuple[float, float, float, float]:
    """Laban Time and Weight of the most active point, from its speed v and
    acceleration a (finite samples, body units; `scale` = body units per
    torso, so hands keep the body's thresholds).

    Time is sudden when acceleration comes in bursts (95th percentile over
    median) or when the movement oscillates quickly (rms acceleration over
    rms speed, in rad/s).  Weight is the 90th-percentile acceleration.
    Returns (time value, rad/s, weight value, acceleration)."""
    if len(v) <= 10:
        return float("nan"), float("nan"), float("nan"), float("nan")
    burst = float(np.percentile(a, 95) / (np.median(a) + 0.5 * scale))
    omega = float(np.sqrt(np.mean(a ** 2)) / (np.sqrt(np.mean(v ** 2)) + 0.05 * scale))
    force = float(np.percentile(a, 90))
    time_v = max(_ramp(burst, 2.5, 10.0, log=True), _ramp(omega, 3.0, 14.0, log=True))
    return time_v, omega, _ramp(force, 1.0 * scale, 40.0 * scale, log=True), force


def _weighted(effs: List[Dict[str, float]], key: str) -> float:
    pairs = [(e[key], e["path"]) for e in effs if np.isfinite(e[key]) and e["path"] > 0]
    if not pairs:
        return float("nan")
    v, w = zip(*pairs)
    return float(np.average(v, weights=w))


def _periodicity(s: np.ndarray, seconds: float) -> Tuple[float, float]:
    """(peak autocorrelation, period in seconds) of a signal over lags
    0.3 s … half the window (two cycles must fit)."""
    x = s[np.isfinite(s)]
    if len(x) < FPS * 1.5 or float(np.std(x)) < 1e-3:
        return float("nan"), float("nan")
    t = np.arange(len(x))
    x = x - np.polyval(np.polyfit(t, x, 1), t)           # a drift or a slow sink is no rhythm
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    # two cycles must fit in the window and in the samples actually tracked
    # (a dancer half out of frame leaves fewer samples than the window holds)
    lo, hi = int(0.3 * FPS), min(int(min(2.0, seconds / 2.0) * FPS), len(x) // 2)
    if hi <= lo + 2:
        return float("nan"), float("nan")
    # a period is a local maximum that the curve first dipped well below
    best_r, best_p = 0.0, float("nan")
    for i in range(lo + 1, hi - 1):
        if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > best_r and ac[:i].min() < ac[i] - 0.3:
            best_r, best_p = float(ac[i]), i / FPS
    return best_r, best_p


# ----------------------------------------------------------------------------
#  measurement
# ----------------------------------------------------------------------------

def _window(traj: PoseTrajectory, seconds: float):
    t_end = traj.span[1]
    res = traj.resample(FPS, start=t_end - seconds, end=t_end)
    _, L = res.arrays()
    P = np.stack([L[..., _X] * traj.aspect, L[..., _Y]], axis=-1).astype(np.float64)
    P[L[..., _V] < 0.5] = np.nan
    return P, L


def _q(key, value, raw, unit, low, high) -> Quality:
    return Quality(key, value, float(raw) if raw is not None else float("nan"), unit, low, high)


def _body(P: np.ndarray, L: np.ndarray, seconds: float) -> Tuple[Dict[str, Quality], Dict[str, float]]:
    """Body qualities from (T, ≥33, 2) image positions (isotropic units)."""
    B = P[:, :N_LANDMARKS]
    side_t = np.stack([_norm(B[:, 11] - B[:, 23]), _norm(B[:, 12] - B[:, 24])], axis=1)
    torso = _quiet(np.nanmedian, side_t)
    if not np.isfinite(torso):                    # hips out of frame: shoulder width ≈ 0.78 torso
        torso = _quiet(np.nanmedian, _norm(B[:, 11] - B[:, 12])) / 0.78
    if not np.isfinite(torso) or torso <= 1e-6:
        return {}, {}
    B = _smooth(B / torso)
    centre = _quiet(np.nanmean, B[:, [11, 12, 23, 24]], axis=1)                 # (T, 2)
    R = B - centre[:, None, :]                                                  # body-relative
    arms, legs = [15, 16], [27, 28]
    eff = [0] + arms + legs

    V = _d(B)
    spd = _norm(V)
    s_t = _quiet(np.nanmean, spd[:, eff], axis=1)
    # energy of the two most active effectors: one arm circling slowly while
    # the rest stands is slow movement, not stillness diluted five ways
    eff_mean = np.array([_quiet(np.nanmean, spd[:, j]) for j in eff])
    eff_mean = np.sort(eff_mean[np.isfinite(eff_mean)])[::-1]
    energy_raw = (float("nan") if not len(eff_mean) else float(eff_mean[0]) if len(eff_mean) == 1
                  else 0.65 * float(eff_mean[0]) + 0.35 * float(eff_mean[1]))
    top_t = _quiet(np.nanmax, spd[:, eff], axis=1)                              # fastest effector per frame
    A = _norm(_d(V))
    # Time and Weight from the most active limb (a one-armed punch is strong
    # even when the rest of the body stands still)
    act = {j: _quiet(np.nanmean, spd[:, j]) for j in arms + legs + [0]}
    lead_j = max((j for j in act if np.isfinite(act[j])), key=lambda j: act[j], default=None)
    time_v = omega = weight_v = force = float("nan")
    if lead_j is not None:
        m = np.isfinite(A[:, lead_j]) & np.isfinite(spd[:, lead_j])
        time_v, omega, weight_v, force = _effort(spd[m, lead_j], A[m, lead_j])

    effs = [_effector(R[:, j], min_path=0.15) for j in arms + legs]
    curv = _weighted(effs, "curvature")
    sparc = _weighted(effs, "sparc")
    moving = [e for e in effs if e["path"] > 0.3]
    stopgo = float(np.mean([e["stopgo"] for e in moving])) if moving else float("nan")
    flow_v = _ramp(sparc, -4.5, -1.7)
    if np.isfinite(stopgo) and np.isfinite(flow_v):
        flow_v = min(flow_v, 1.0 - _ramp(stopgo, 0.4, 1.6))      # stop-and-go is staccato, bound

    reach_t = _quiet(np.nanmean, _norm(R[:, arms]), axis=1)
    reach = _quiet(np.nanmean, reach_t)
    shoulder_y = _quiet(np.nanmean, B[:, [11, 12], 1], axis=1)
    height = _quiet(np.nanmean, shoulder_y[:, None] - B[:, arms, 1])           # y grows downward
    legs_ext = _quiet(np.nanmean, B[:, legs, 1] - _quiet(np.nanmean, B[:, [23, 24], 1], axis=1)[:, None])
    # level: arms at rest sit in the middle; raised arms lift it, a crouch lowers it
    level_v = 0.45 + 0.55 * _ramp(height, -0.9, 1.1) if np.isfinite(height) else float("nan")
    if np.isfinite(legs_ext) and legs_ext < 1.5:
        level_v = 0.45 * _ramp(legs_ext, 0.7, 1.5)

    mirror = R[:, 16] * np.array([-1.0, 1.0])
    sym_pose = _quiet(np.nanmean, np.exp(-_norm(R[:, 15] - mirror) / 0.6))
    sl, sr = spd[:, 15], spd[:, 16]
    m = np.isfinite(sl) & np.isfinite(sr)
    sync = float(np.corrcoef(sl[m], sr[m])[0, 1]) if m.sum() > 10 and sl[m].std() > 1e-6 and sr[m].std() > 1e-6 else float("nan")
    symmetry = sym_pose if not np.isfinite(sync) else 0.6 * sym_pose + 0.4 * max(sync, 0.0)

    # rhythm: the speed pulse, or the cycling of the leading limb's position
    # (a constant-speed circle has no speed pulse but repeats in space)
    period_r, period_s = _periodicity(s_t, seconds)
    if lead_j is not None and not (np.isfinite(period_r) and period_r >= 0.5):   # a clear pulse wins
        for axis in (0, 1):
            r_, p_ = _periodicity(R[:, lead_j, axis], seconds)
            if np.isfinite(r_) and (not np.isfinite(period_r) or r_ > period_r):
                period_r, period_s = r_, p_
    # travel: net displacement of the body centre (start and end averaged over 0.3 s)
    k = max(3, int(0.3 * FPS))
    c0, c1 = _quiet(np.nanmean, centre[:k], axis=0), _quiet(np.nanmean, centre[-k:], axis=0)
    travel = float(_norm(c1 - c0)) / seconds if np.isfinite(c0).all() and np.isfinite(c1).all() else float("nan")
    # held pauses: runs of at least PAUSE seconds with every effector below
    # 0.08 torso/s (the turn at the end of a slow sweep is quicker than that)
    still = float("nan")
    if np.isfinite(top_t).any():
        low = np.append(np.isfinite(top_t) & (top_t < 0.08), False)
        held, run = 0, 0
        for x in low:
            if x:
                run += 1
            else:
                held += run if run >= PAUSE * FPS else 0
                run = 0
        still = held / float(len(top_t))
    # size: the largest excursion of an arm or leg around its mean place, in torso lengths
    exc = []
    for j in arms + legs:
        r = R[:, j][np.isfinite(R[:, j, 0])]
        if len(r) > 10:
            exc.append(float(_norm(np.percentile(r, 90, axis=0) - np.percentile(r, 10, axis=0))))
    size = max(exc) if exc else float("nan")

    # turning: yaw of the shoulder line in MediaPipe's world coordinates (metres, hip-centred)
    W = L[:, :N_LANDMARKS, 4:7].astype(np.float64)
    sh = W[:, 11] - W[:, 12]
    yaw = np.unwrap(np.arctan2(sh[:, 2], sh[:, 0])[np.isfinite(sh[:, 0]) & np.isfinite(sh[:, 2])])
    turning = float(np.degrees(np.sum(np.abs(np.diff(yaw)))) / seconds) if len(yaw) > 5 else float("nan")

    groups = {"arms": _quiet(np.nanmean, _norm(_d(R[:, [13, 14, 15, 16]]))),
              "legs": _quiet(np.nanmean, _norm(_d(R[:, [25, 26, 27, 28]]))),
              "head": _quiet(np.nanmean, _norm(_d(R[:, 0]))),
              "torso": _quiet(np.nanmean, _norm(_d(centre)))}
    ranked = sorted(((v, k) for k, v in groups.items() if np.isfinite(v)), reverse=True)
    lead = ranked[0][1] if len(ranked) > 1 and ranked[0][0] > 1.6 * ranked[1][0] and ranked[0][0] > 0.25 else ""

    q = {
        "energy": _q("energy", _ramp(energy_raw, 0.08, 3.0, log=True), energy_raw, "torso/s", "still", "vigorous"),
        "time": _q("time", time_v, omega, "rad/s", "sustained", "sudden"),
        "weight": _q("weight", weight_v, force, "torso/s²", "light", "strong"),
        "space": _q("space", 1.0 - _ramp(curv, 0.4, 3.0, log=True), curv, "rad/torso", "indirect", "direct"),
        "flow": _q("flow", flow_v, sparc, "SPARC", "bound", "free"),
        "shape": _q("shape", _ramp(reach, 0.15, 1.6), reach, "torso", "contracted", "expansive"),
        "size": _q("size", _ramp(size, 0.12, 1.6, log=True), size, "torso", "small", "broad"),
        "level": _q("level", level_v, height, "torso", "low", "high"),
        "symmetry": _q("symmetry", _ramp(symmetry, 0.25, 0.8), symmetry, "", "asymmetric", "symmetric"),
        "rhythm": _q("rhythm", _ramp(period_r, 0.2, 0.7), period_r, "autocorr", "irregular", "rhythmic"),
    }
    tr = {
        "opening": _slope(reach_t),                              # torso/s, + opening
        "rising": -_slope(_quiet(np.nanmean, B[:, [0, 11, 12], 1], axis=1)),   # torso/s, + rising
        "accelerating": _slope(np.log(s_t + 0.05)),              # log-speed/s
        "travel": travel,                                        # torso/s
        "approach": _slope(np.log(_quiet(np.nanmean, side_t, axis=1))),        # + toward the camera
        "turning": turning,                                      # deg/s
        "stillness": still,                                      # fraction of the window
        "crouch": legs_ext,                                      # hip to ankle, torso (≈1.7 standing)
        "tempo_bpm": 60.0 / period_s if np.isfinite(period_s) else float("nan"),
        "lead": {"arms": 1.0, "legs": 2.0, "head": 3.0, "torso": 4.0}.get(lead, 0.0),
    }
    return q, tr


def _hands(P: np.ndarray, off: int) -> Dict[str, Quality]:
    """Hand qualities from the 42 hand rows starting at `off` (left hand, then right)."""
    H = [P[:, off:off + HAND_N], P[:, off + HAND_N:off + 2 * HAND_N]]
    present = [h for h in H if np.isfinite(h[:, 0]).mean() > 0.3]
    if not present:
        return {}
    lens = [_quiet(np.nanmedian, _norm(h[:, 9] - h[:, 0])) for h in present]
    hl = _quiet(np.nanmedian, np.asarray(lens))
    if not np.isfinite(hl) or hl <= 1e-6:
        return {}
    present = [_smooth(h / hl) for h in present]
    tips = list(FINGERTIPS)

    open_t = [_quiet(np.nanmean, _norm(h[:, tips] - h[:, [0]]), axis=1) for h in present]
    openness = _quiet(np.nanmean, np.concatenate(open_t))
    opening = _quiet(np.nanmean, [_slope(o) for o in open_t])

    art, whole, effs = [], [], []
    for h in present:
        Vw = _d(h[:, 0])
        Vt = _d(h[:, tips])
        art.append(_quiet(np.nanmean, _norm(Vt - Vw[:, None, :])))
        whole.append(_quiet(np.nanmean, _norm(Vw)))
        effs.append(_effector(h[:, 0], min_path=0.5))
    articulation = _quiet(np.nanmean, art) / (_quiet(np.nanmean, art) + _quiet(np.nanmean, whole) + 1e-6)
    speed = _quiet(np.nanmean, np.asarray(art) + np.asarray(whole))
    curv, sparc = _weighted(effs, "curvature"), _weighted(effs, "sparc")

    # Time and Weight from the most active point (a wrist or a fingertip),
    # as for the body; a hand length is about a fifth of a torso, so the
    # noise floors and force scale grow by that factor
    best = None
    for h in present:
        for j in [0] + tips:
            v = _norm(_d(h[:, j]))
            a = _norm(_d(_d(h[:, j])))
            m = np.isfinite(v) & np.isfinite(a)
            if m.sum() > 10 and (best is None or float(v[m].mean()) > best[0]):
                best = (float(v[m].mean()), v[m], a[m])
    time_v = omega = weight_v = force = float("nan")
    if best is not None:
        time_v, omega, weight_v, force = _effort(best[1], best[2], scale=5.5)

    q = {
        "time": _q("time", time_v, omega, "rad/s", "sustained", "sudden"),
        "weight": _q("weight", weight_v, force, "hand/s²", "light", "strong"),
        "openness": _q("openness", _ramp(openness, 1.15, 1.85), openness, "hand", "closed", "open"),
        "articulation": _q("articulation", _ramp(articulation, 0.25, 0.6), articulation, "",
                           "moving as one block", "fingers articulating"),
        "hand_speed": _q("hand_speed", _ramp(speed, 0.3, 8.0, log=True), speed, "hand/s", "still", "quick"),
    }
    if np.isfinite(curv):
        q["hand_space"] = _q("hand_space", 1.0 - _ramp(curv, 0.3, 3.0, log=True), curv, "rad/hand", "curving", "direct")
    if np.isfinite(sparc):
        q["hand_flow"] = _q("hand_flow", _ramp(sparc, -4.5, -1.7), sparc, "SPARC", "jerky", "smooth")
    if len(present) == 2:
        gap = _norm(present[0][:, 0] - present[1][:, 0])
        q["together"] = _q("together", 1.0 - _ramp(_quiet(np.nanmean, gap), 1.5, 6.0), _quiet(np.nanmean, gap),
                           "hand", "far apart", "close together")
        q["converging"] = _q("converging", float("nan"), -_slope(gap), "hand/s", "drawing apart", "drawing together")
    q["opening"] = _q("opening", float("nan"), opening, "hand/s", "closing", "opening")
    return q


STILL_SIZE = 0.12  # torso lengths: tracker jitter at rest stays well below, a very slow sweep goes beyond


def _is_still(energy: Optional[Quality], size: Optional[Quality] = None) -> bool:
    """Near stillness, judged on the fixed calibration even when the value
    is session-relative (a quiet stretch of a lively session still moves).
    A low speed with a real excursion (`size`) is very slow movement."""
    if energy is None:
        return False
    v = energy.absolute if np.isfinite(energy.absolute) else energy.value
    if size is not None and np.isfinite(size.raw) and size.raw >= STILL_SIZE:
        return False
    return bool(np.isfinite(v) and v < 1 / 6)


class SessionNorm:
    """Each quality measured against this session's own range.

    The fixed calibration leaves most windows of a real dancer at one end
    of several qualities (always fluid, always low, never rhythmic), so the
    same words come back however the movement changes.  `apply` re-centres
    every quality on its recent history: the session median maps to the
    middle (left unsaid), and distance from it is scaled by the session's
    spread.  The spread has a floor, so a dancer holding one movement does
    not have tracker noise read as contrast, and the re-centred value stays
    within `reach` of the fixed calibration, so a slow movement can read as
    relatively quick or firm but never as sudden or forceful.  Energy and
    hand speed stay absolute: they set the register of the words.
    Stillness is judged on the absolute energy, and a still window adds
    only its energy to the history.

        norm = SessionNorm()                    # one per dancer / tracker
        describe(traj, 3.0, session=norm)       # every refresh
    """

    # intensity stays on the fixed calibration: a slow dancer who speeds up a
    # little is still slow, whatever the session has seen before
    ABSOLUTE = frozenset({"energy", "hand_speed"})

    def __init__(self, memory: float = 120.0, floor: float = 0.12, warmup: int = 5, reach: float = 0.3):
        self.memory = float(memory)     # seconds of history
        self.floor = float(floor)       # smallest spread (0-1 units) counted as contrast
        self.warmup = int(warmup)       # readings before a quality is re-centred
        self.reach = float(reach)       # furthest the session moves a value from its fixed calibration
        self._hist: Dict[str, List[Tuple[float, float]]] = {}

    def reset(self) -> None:
        self._hist.clear()

    @property
    def samples(self) -> int:
        return max((len(h) for h in self._hist.values()), default=0)

    def apply(self, reading: MotionReading, t: float, relative: float = 1.0) -> MotionReading:
        """Re-centre `reading`'s qualities in place (`relative` 0 keeps the
        fixed calibration, 1 is fully session-relative) and record them."""
        still = _is_still(_effort_q(reading).get("energy"), reading.qualities.get("size"))
        for d in (reading.qualities, reading.hands):
            for k, q in list(d.items()):
                if not np.isfinite(q.value) or (still and k not in ("energy", "hand_speed")):
                    continue
                h = self._hist.setdefault(k, [])
                if h and t < h[-1][0] - 1.0:             # the clock restarted: a new session
                    h.clear()
                h.append((t, q.value))
                while h[0][0] < t - self.memory:
                    h.pop(0)
                if still or len(h) < self.warmup or relative <= 0 or k in self.ABSOLUTE:
                    d[k] = Quality(q.key, q.value, q.raw, q.unit, q.low, q.high, absolute=q.value)
                    continue
                H = np.array([v for _, v in h])
                spread = max((np.percentile(H, 84) - np.percentile(H, 16)) / 2.0, self.floor)
                rel = 0.5 + 0.5 * math.tanh((q.value - float(np.median(H))) / (1.5 * spread))
                rel = float(np.clip(rel, q.value - self.reach, q.value + self.reach))
                d[k] = Quality(q.key, float((1.0 - relative) * q.value + relative * rel), q.raw, q.unit,
                               q.low, q.high, absolute=q.value)
        return reading


def describe(traj: PoseTrajectory, seconds: float = 3.0, level: str = "full",
             session: Optional[SessionNorm] = None, relative: float = 1.0) -> MotionReading:
    """Measure the movement qualities of the last `seconds` of a trajectory
    and describe them at `level` (full | abstract | qualities | body, see to_text).
    With a `session`, the qualities are re-expressed against the session's
    own range first (`relative` 0 = fixed calibration … 1 = fully relative)."""
    sk = traj.skeleton
    reading = MotionReading(tracker=sk.name, seconds=seconds, coverage=0.0)
    if len(traj) < 3:
        reading.text = to_text(reading, level)
        return reading
    P, L = _window(traj, seconds)
    reading.coverage = float(np.isfinite(P[..., 0]).any(axis=1).mean())
    if reading.coverage < 0.3:
        reading.text = to_text(reading, level)
        return reading
    if sk.name in ("body", "both"):
        reading.qualities, reading.trends = _body(P, L, seconds)
    if sk.name in ("hands", "both"):
        reading.hands = _hands(P, 0 if sk.name == "hands" else N_LANDMARKS)
    if session is not None:
        session.apply(reading, float(traj.span[1]), relative)
    reading.text = to_text(reading, level)
    return reading


# ----------------------------------------------------------------------------
#  words
# ----------------------------------------------------------------------------
# Each quality has six bands, from very low to very high; the two middle
# bands (0.4–0.6) are left out of the text, so only what stands out is said.

_BANDS = {
    "energy":   ("near stillness", "quiet, gentle movement", "calm movement",
                 "active movement", "lively, energetic movement", "vigorous, explosive movement"),
    "time":     ("very sustained, lingering", "sustained, unhurried", "even-paced",
                 "quickening", "quick, sudden", "very quick, sudden, urgent"),
    "weight":   ("feather-light, delicate", "light, buoyant", "soft",
                 "firm", "strong, forceful", "heavy, powerful, forceful"),
    "space":    ("wandering, meandering paths", "indirect, curving paths", "loosely curving paths",
                 "fairly direct paths", "direct, focused paths", "straight, pinpoint, cutting paths"),
    "flow":     ("tightly bound, jerky, staccato", "bound, held, controlled", "somewhat held",
                 "fairly smooth", "free, fluid, flowing", "very fluid, continuous, unbroken"),
    "shape":    ("arms folded tightly inward, the body closed", "arms drawn in close, the body contracted",
                 "arms near the body", "arms reaching out", "arms open wide, the body expansive",
                 "arms fully extended, spreading wide open"),
    "level":    ("crouched very low to the ground", "low, knees bent", "slightly lowered",
                 "arms lifting to chest height", "arms raised high", "arms reaching high above the head"),
    "symmetry": ("strongly asymmetrical, one side acting alone", "asymmetrical, one side leading",
                 "slightly uneven", "fairly even", "symmetrical, both sides together",
                 "mirrored, perfectly symmetrical"),
    "rhythm":   ("irregular, unpredictable timing", "uneven timing", "loose timing",
                 "a hint of pulse", "rhythmic, pulsing", "strongly rhythmic, steady repeating pulse"),
    "size":     ("tiny, minute gestures", "small, contained gestures", "modest gestures",
                 "fairly broad gestures", "broad, ample gestures", "large, sweeping gestures"),
}


def _band(v: float) -> int:
    if not np.isfinite(v):
        return -1
    return min(5, int(v * 6))


def _salient(q: Optional[Quality]) -> Optional[str]:
    """The quality's words when it stands out (outside 0.4–0.6), else None."""
    if q is None or not np.isfinite(q.value) or 0.4 <= q.value <= 0.6:
        return None
    return _BANDS[q.key][_band(q.value)]


def _join(parts: Sequence[str]) -> str:
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _sentence(s: str) -> str:
    s = s.strip()
    return (s[0].upper() + s[1:] + ("" if s.endswith(".") else ".")) if s else ""


def _body_text(q: Dict[str, Quality], tr: Dict[str, float]) -> List[str]:
    out: List[str] = []
    energy = q.get("energy")
    if _is_still(energy, q.get("size")):
        lines = ["Near stillness, the body barely moving"]
        if tr.get("stillness", 0) > 0.8:
            lines[0] += ", held and quiet"
        return [_sentence(lines[0])]

    # 1 · energy with Time and Weight
    head = _BANDS["energy"][_band(energy.value)] if energy is not None and np.isfinite(energy.value) else "movement"
    effort = _join([_salient(q.get("time")), _salient(q.get("weight"))])
    out.append(_sentence(f"{head}, {effort}" if effort else head))

    # 2 · Space and Flow
    space, flow = _salient(q.get("space")), _salient(q.get("flow"))
    if space and flow:
        out.append(_sentence(f"it follows {space} with a {flow} flow"))
    elif space:
        out.append(_sentence(f"it follows {space}"))
    elif flow:
        out.append(_sentence(f"the flow is {flow}"))

    # 3 · Shape and level, with their trends
    shape_parts = [s for s in (_salient(q.get("size")),) if s]
    shape = _salient(q.get("shape"))
    if shape:
        shape_parts.append(shape)
    if tr.get("opening", 0) > 0.25:
        shape_parts.append("opening outward")
    elif tr.get("opening", 0) < -0.25:
        shape_parts.append("folding inward")
    level = _salient(q.get("level"))
    if level:
        shape_parts.append(level)
    if tr.get("rising", 0) > 0.2:
        shape_parts.append("rising")
    elif tr.get("rising", 0) < -0.2:
        shape_parts.append("sinking toward the ground")
    if shape_parts:
        out.append(_sentence(_join(shape_parts)))

    # 4 · symmetry, rhythm, travel, turning, tempo, lead, pauses
    rest = []
    sym = _salient(q.get("symmetry"))
    if sym:
        rest.append(sym)
    rh = q.get("rhythm")
    rhythm = _salient(rh)
    bpm = tr.get("tempo_bpm", float("nan"))
    if rhythm and rh is not None and rh.value > 0.6 and np.isfinite(bpm):
        rest.append(f"{rhythm} at about {int(round(bpm / 5.0) * 5)} beats per minute")
    elif rhythm:
        rest.append(rhythm)
    if tr.get("turning", 0) > 45:
        rest.append("turning and spiralling")
    if tr.get("travel", 0) > 0.25:
        rest.append("travelling across the space")
    if tr.get("approach", 0) > 0.12:
        rest.append("coming toward the viewer")
    elif tr.get("approach", 0) < -0.12:
        rest.append("receding from the viewer")
    if tr.get("accelerating", 0) > 0.35:
        rest.append("gathering speed")
    elif tr.get("accelerating", 0) < -0.35:
        rest.append("slowing and settling")
    lead = {1.0: "led by the arms", 2.0: "led by the legs and feet", 3.0: "led by the head", 4.0: "moving from the torso"}
    if tr.get("lead", 0) in lead:
        rest.append(lead[tr["lead"]])
    if 0.2 < tr.get("stillness", 0) < 0.8:
        rest.append("with held pauses")
    if rest:
        out.append(_sentence(_join(rest)))
    return out


def _hands_text(h: Dict[str, Quality]) -> List[str]:
    """Body level: the hands named once, then what they do."""
    if not h:
        return []
    parts = []
    sp = h.get("hand_speed")
    if sp is not None and np.isfinite(sp.value):
        parts.append("still" if sp.value < 0.15 else "slow" if sp.value < 0.35
                     else "quick and busy" if sp.value > 0.75 else "")
    op = h.get("openness")
    if op is not None and np.isfinite(op.value):
        parts.append("palms open and spread" if op.value > 0.65 else "fingers curled closed" if op.value < 0.35 else "")
    opening = h.get("opening")
    if opening is not None and np.isfinite(opening.raw):
        parts.append("unfolding" if opening.raw > 0.4 else "closing into fists" if opening.raw < -0.4 else "")
    ar = h.get("articulation")
    if ar is not None and np.isfinite(ar.value):
        parts.append("fingers fluttering one by one" if ar.value > 0.65
                     else "moving as one shape" if ar.value < 0.3 else "")
    hs, hf = h.get("hand_space"), h.get("hand_flow")
    if hs is not None and np.isfinite(hs.value) and (hs.value < 0.35 or hs.value > 0.65):
        parts.append("tracing curving, looping paths" if hs.value < 0.35 else "pointing in straight, precise strokes")
    if hf is not None and np.isfinite(hf.value) and (hf.value < 0.35 or hf.value > 0.65):
        parts.append("smooth" if hf.value > 0.65 else "jerky and tense")
    tg, cv = h.get("together"), h.get("converging")
    if cv is not None and np.isfinite(cv.raw) and abs(cv.raw) > 0.5:
        parts.append("drawing together" if cv.raw > 0 else "drawing apart")
    elif tg is not None and np.isfinite(tg.value) and (tg.value > 0.7 or tg.value < 0.2):
        parts.append("held close together" if tg.value > 0.7 else "far apart")
    parts = [p for p in parts if p]
    return [_sentence("hands " + _join(parts))] if parts else []


# ---- shared helpers for the qualities and abstract levels -------------------

LEVELS = ("full", "abstract", "qualities", "body")


def _effort_q(reading: MotionReading) -> Dict[str, Quality]:
    """The effort qualities to describe: the body's, else the hands' (their
    paths and smoothness stand in for Space and Flow, their speed for energy)."""
    if reading.qualities:
        return reading.qualities
    h = reading.hands
    out = {k: h[k] for k in ("time", "weight") if k in h}
    for src, dst in (("hand_space", "space"), ("hand_flow", "flow"), ("hand_speed", "energy")):
        if src in h:
            out[dst] = h[src]
    return out


def _pole(q: Optional[Quality]) -> int:
    """-1 (low pole), +1 (high pole), 0 (middle or unmeasured)."""
    if q is None or not np.isfinite(q.value):
        return 0
    return -1 if q.value < 0.4 else 1 if q.value > 0.6 else 0


def _strength(q: Optional[Quality]) -> float:
    return abs(q.value - 0.5) if q is not None and np.isfinite(q.value) else 0.0


def _trend_words(tr: Dict[str, float], words: Dict[str, Tuple[str, str]]) -> List[str]:
    """Words for the trends that pass their thresholds; words[k] = (negative, positive)."""
    rules = (("opening", 0.25), ("rising", 0.2), ("accelerating", 0.35), ("approach", 0.12))
    out = []
    for k, th in rules:
        v = tr.get(k, float("nan"))
        if k in words and np.isfinite(v) and abs(v) > th:
            out.append(words[k][1] if v > 0 else words[k][0])
    if tr.get("turning", 0) > 45 and "turning" in words:
        out.append(words["turning"][1])
    if tr.get("travel", 0) > 0.25 and "travel" in words:
        out.append(words["travel"][1])
    return out


# ---- qualities: adjectives only, no body parts -----------------------------

_Q_WORDS = {      # very low, low, high, very high
    "energy": ("still", "quiet", "lively", "vigorous"),
    "time": ("lingering", "sustained", "quick", "sudden"),
    "weight": ("delicate", "light", "firm", "forceful"),
    "space": ("wandering", "indirect", "direct", "pinpoint"),
    "flow": ("staccato", "bound", "fluid", "streaming"),
    "shape": ("enclosed", "contracted", "open", "expansive"),
    "level": ("grounded", "low", "lifted", "soaring"),
    "symmetry": ("lopsided", "asymmetrical", "symmetrical", "mirrored"),
    "rhythm": ("erratic", "irregular", "rhythmic", "pulsing"),
    "articulation": ("unified", "whole", "articulate", "intricate"),
    "openness": ("clenched", "closed", "open", "unfurled"),
    "together": ("scattered", "apart", "close", "joined"),
    "size": ("tiny", "small", "broad", "sweeping"),
}
_Q_GENTLE = {     # the high poles of Time and Weight in the slow register
    "time": ("lingering", "sustained", "quickening", "brisk"),
    "weight": ("delicate", "light", "firm", "weighted"),
}
_Q_TRENDS = {"opening": ("closing", "opening"), "rising": ("sinking", "rising"),
             "accelerating": ("subsiding", "quickening"), "approach": ("retreating", "advancing"),
             "turning": ("", "turning"), "travel": ("", "travelling")}


def _qword(q: Optional[Quality], reg: int = 1) -> Optional[str]:
    """One adjective for a quality that stands out; `reg` 0 softens the
    high poles of Time and Weight (a slow movement is never "sudden")."""
    if q is None or q.key not in _Q_WORDS or not np.isfinite(q.value) or 0.4 <= q.value <= 0.6:
        return None
    w = _Q_GENTLE[q.key] if reg == 0 and q.key in _Q_GENTLE else _Q_WORDS[q.key]
    v = q.value
    return w[0] if v < 1 / 6 else w[1] if v < 0.4 else w[3] if v > 5 / 6 else w[2]


def _qualities_text(reading: MotionReading) -> str:
    eff = _effort_q(reading)
    if _is_still(eff.get("energy"), reading.qualities.get("size")):
        return "Still, motionless, at rest."            # the other measures are tracker noise here
    reg = _register(reading)
    first = [w for w in [_pace(reading)] + [_qword(eff.get(k), reg) for k in ("time", "weight", "space", "flow")] if w]
    second = [w for w in (_qword(reading.qualities.get(k)) for k in ("size", "shape", "level", "symmetry")) if w]
    second += _trend_words(reading.trends, _Q_TRENDS)
    rh = reading.qualities.get("rhythm")
    bpm = reading.trends.get("tempo_bpm", float("nan"))
    if rh is not None and np.isfinite(rh.value) and rh.value > 0.6 and np.isfinite(bpm):
        second.append(f"pulsing at about {int(round(bpm / 5.0) * 5)} beats per minute")
    elif _qword(rh) and (rh.value > 0.6 or reg > 0):       # slow movement too long to repeat is not erratic
        second.append(_qword(rh))
    second += [w for w in (_qword(reading.hands.get(k)) for k in ("articulation", "openness", "together")) if w]
    cv = reading.hands.get("converging")
    if cv is not None and np.isfinite(cv.raw) and abs(cv.raw) > 0.5:
        second.append("converging" if cv.raw > 0 else "diverging")
    return " ".join(_sentence(_join(s)) for s in (first, second) if s)


# ---- abstract: Laban effort actions, effort states, spatial imagery ---------
# Laban's eight basic effort actions combine Time, Weight and Space; two
# factors together make an effort state (Near, Awake, Mobile, Stable, Dream,
# Remote).  The words carry those qualities without naming a body part.

# Every table has three registers, chosen by the absolute speed (see
# _register): 0 slow, 1 moderate, 2 vigorous.  Laban's factors are relative
# within a register, so a slow movement that is quicker or firmer than its
# neighbours reads "flickering" or "weighted", never "explosive".

_ACTIONS = {      # (time, weight, space) poles → the effort action's words, per register
    (1, 1, 1): ("firm, deliberate, pointed", "thrusting, decisive, sharp",
                "thrusting, decisive, explosive"),                                   # punch
    (1, 1, -1): ("swaying, weighted, swinging", "slashing, sweeping, swinging",
                 "slashing, sweeping, tempestuous"),                                 # slash
    (1, -1, 1): ("tapping, precise, light", "dabbing, crisp, sparkling", "dabbing, crisp, sparkling"),   # dab
    (1, -1, -1): ("flickering, fluttering, light", "flicking, flitting, playful",
                  "flicking, flitting, playful"),                                    # flick
    (-1, 1, 1): ("pressing, slow, deliberate", "pressing, insistent, steady",
                 "pressing, insistent, unyielding"),                                 # press
    (-1, 1, -1): ("twisting, kneading, slow", "wringing, twisting, effortful",
                  "wringing, twisting, anguished"),                                  # wring
    (-1, -1, 1): ("gliding, smooth, serene", "gliding, poised, effortless", "gliding, serene, effortless"),  # glide
    (-1, -1, -1): ("floating, drifting, weightless", "floating, drifting, buoyant",
                   "floating, drifting, weightless"),                                # float
}
_STATES = {       # (factor, factor) → {(pole, pole): words per register}
    ("time", "weight"): {
        (1, 1): ("punctuated, grounded, firm", "pounding, percussive, earthy", "pounding, percussive, earthy"),
        (1, -1): ("fluttering, nimble, light", "skittering, nimble, sparkling", "skittering, nimble, sparkling"),
        (-1, 1): ("slow, heavy, grounded", "ponderous, grinding, heavy", "ponderous, grinding, heavy"),
        (-1, -1): ("lulling, tender, gentle", "lulling, tender, gentle", "lulling, tender, gentle")},
    ("time", "space"): {
        (1, 1): ("alert, attentive, precise", "alert, piercing, vigilant", "alert, piercing, vigilant"),
        (1, -1): ("flickering, shifting, searching", "darting, startled, scattered", "darting, startled, scattered"),
        (-1, 1): ("deliberate, focused, steady", "deliberate, focused, unwavering", "deliberate, focused, unwavering"),
        (-1, -1): ("meandering, contemplative, roaming", "meandering, contemplative, roaming",
                   "meandering, contemplative, roaming")},
    ("time", "flow"): {
        (1, 1): ("spontaneous, loose, playful", "impulsive, spilling, spontaneous", "impulsive, spilling, reckless"),
        (1, -1): ("hesitant, halting, careful", "jolting, nervous, abrupt", "jolting, nervous, abrupt"),
        (-1, 1): ("unfurling, streaming, continuous", "unfurling, streaming, continuous",
                  "unfurling, streaming, continuous"),
        (-1, -1): ("cautious, careful, guarded", "cautious, careful, guarded", "cautious, careful, guarded")},
    ("weight", "space"): {
        (1, 1): ("firm, rooted, steady", "firm, resolute, unshakable", "firm, resolute, unshakable"),
        (1, -1): ("swaying, rolling, heavy", "churning, swaying, heaving", "churning, turbulent, heaving"),
        (-1, 1): ("poised, precise, delicate", "poised, precise, delicate", "poised, precise, delicate"),
        (-1, -1): ("airy, wafting, wandering", "airy, wafting, wandering", "airy, wafting, wandering")},
    ("weight", "flow"): {
        (1, 1): ("weighty, flowing, released", "surging, released, powerful", "surging, unleashed, torrential"),
        (1, -1): ("tense, held, contained", "tense, straining, clenched", "tense, straining, clenched"),
        (-1, 1): ("dreamy, buoyant, weightless", "dreamy, buoyant, weightless", "dreamy, buoyant, weightless"),
        (-1, -1): ("fragile, hesitant, tentative", "fragile, hesitant, tentative", "fragile, hesitant, tentative")},
    ("space", "flow"): {
        (1, 1): ("clear, flowing, unerring", "clear, streaming, unerring", "clear, streaming, unerring"),
        (1, -1): ("careful, exact, contained", "rigid, mechanical, exact", "rigid, mechanical, exact"),
        (-1, 1): ("swirling, curling, unfolding", "swirling, billowing, unfolding", "swirling, billowing, unfolding"),
        (-1, -1): ("knotted, tangled, winding", "knotted, tangled, convoluted", "knotted, tangled, convoluted")},
}
_SINGLE = {       # factor → (low pole, high pole) per register
    "time": (("unhurried, lingering", "quickening, alert"), ("unhurried, lingering", "quick, brisk"),
             ("unhurried, lingering", "sudden, urgent")),
    "weight": (("light, delicate", "weighted, firm"), ("light, delicate", "firm, solid"),
               ("light, delicate", "forceful, heavy")),
    "space": (("roundabout, wandering", "direct, focused"),) * 3,
    "flow": (("held, careful", "free, flowing"), ("held, restrained", "free, released"),
             ("held, restrained", "free, released")),
}
_FLOW_MOD = (("contained, careful", "flowing, continuous"), ("contained, restrained", "free, flowing"),
             ("contained, restrained", "released, pouring"))
# the pace itself, from the absolute energy (still is handled before)
_PACE = ((0.28, "very slow, almost suspended"), (0.45, "slow, unhurried"), (0.62, "calm, moderate"),
         (0.78, "lively, animated"), (0.92, "vigorous, energetic"), (2.0, "intense, vigorous"))
REGISTERS = (0.45, 0.78)   # absolute energy where the words turn moderate, then vigorous
_SIZE = ("tiny, minute", "small, contained", "", "", "broad, ample", "large, sweeping")


def _abs(q: Optional[Quality]) -> float:
    """The fixed-calibration value of a quality (its session-free reading)."""
    if q is None:
        return float("nan")
    return q.absolute if np.isfinite(q.absolute) else q.value


def _register(reading: MotionReading) -> int:
    """0 slow, 1 moderate, 2 vigorous, from the absolute energy."""
    e = _abs(_effort_q(reading).get("energy"))
    if not np.isfinite(e):
        return 1
    return 0 if e < REGISTERS[0] else 1 if e < REGISTERS[1] else 2


def _pace(reading: MotionReading) -> Optional[str]:
    e = _abs(_effort_q(reading).get("energy"))
    if not np.isfinite(e):
        return None
    return next(w for th, w in _PACE if e < th)


def _size(reading: MotionReading) -> Optional[str]:
    q = reading.qualities.get("size")
    if q is None or not np.isfinite(q.value) or 0.4 <= q.value <= 0.6:
        return None
    w = _SIZE[_band(q.value)]
    return f"{w} gestures" if w else None
_A_TRENDS = {"opening": ("enclosing", "unfolding"), "rising": ("descending", "ascending"),
             "accelerating": ("subsiding", "gathering momentum"), "approach": ("withdrawing", "advancing"),
             "turning": ("", "spiralling"), "travel": ("", "journeying through space")}


def _effort_parts(reading: MotionReading) -> Tuple[List[str], set]:
    """The pace, then the Laban effort action (three factors), state (two)
    or single factor that stands out, worded for the pace's register, then
    the size of the gestures ("slow, unhurried, gliding, smooth, serene,
    broad, ample gestures"), with the factors those words already carry."""
    eff = _effort_q(reading)
    reg = _register(reading)
    poles = {f: _pole(eff.get(f)) for f in ("time", "weight", "space", "flow")}
    on = [f for f in ("time", "weight", "space", "flow") if poles[f]]
    words: List[str] = [w for w in (_pace(reading),) if w]
    used: set = set()
    if poles["time"] and poles["weight"] and poles["space"]:
        words.append(_ACTIONS[(poles["time"], poles["weight"], poles["space"])][reg])
        used |= {"time", "weight", "space"}
    elif len(on) >= 2:
        pair = max(_STATES, key=lambda p: (poles[p[0]] != 0 and poles[p[1]] != 0,
                                           _strength(eff.get(p[0])) + _strength(eff.get(p[1]))))
        words.append(_STATES[pair][(poles[pair[0]], poles[pair[1]])][reg])
        used |= set(pair)
    elif len(on) == 1:
        words.append(_SINGLE[on[0]][reg][poles[on[0]] > 0])
        used.add(on[0])
    if poles["flow"] and "flow" not in used and len(on) > 1:
        words.append(_FLOW_MOD[reg][poles["flow"] > 0])
        used.add("flow")
    size = _size(reading)
    if size:
        words.append(size)
    return words, used


def _effort_words(reading: MotionReading) -> List[str]:
    return _effort_parts(reading)[0]


def _still(reading: MotionReading) -> bool:
    return _is_still(_effort_q(reading).get("energy"), reading.qualities.get("size"))


def _abstract_text(reading: MotionReading) -> str:
    if _still(reading):
        return "Stillness, suspended, a held breath."
    words = _effort_words(reading)

    space_words: List[str] = []
    shape, level = reading.qualities.get("shape"), reading.qualities.get("level")
    if _pole(shape):
        space_words.append("radiating outward" if _pole(shape) > 0 else "gathering inward")
    if _pole(level):
        space_words.append("reaching skyward" if _pole(level) > 0 else "close to the earth")
    space_words += _trend_words(reading.trends, _A_TRENDS)
    h = reading.hands
    if _pole(h.get("articulation")):
        space_words.append("intricate, filigreed" if _pole(h["articulation"]) > 0 else "whole, unified")
    if _pole(h.get("openness")):
        space_words.append("offering, unfurling" if _pole(h["openness"]) > 0 else "gripping, clenched")
    cv = h.get("converging")
    if cv is not None and np.isfinite(cv.raw) and abs(cv.raw) > 0.5:
        space_words.append("gathering together" if cv.raw > 0 else "parting, releasing")

    time_words: List[str] = []
    rh = reading.qualities.get("rhythm")
    bpm = reading.trends.get("tempo_bpm", float("nan"))
    if rh is not None and np.isfinite(rh.value) and rh.value > 0.6:
        time_words.append("a slow, steady pulse" if not np.isfinite(bpm) or bpm < 70
                          else "a steady pulse" if bpm < 110 else "a racing pulse")
    elif rh is not None and np.isfinite(rh.value) and rh.value < 0.25 and _register(reading) > 0:
        time_words.append("erratic, unpredictable")
    sym = reading.qualities.get("symmetry")
    if _pole(sym):
        time_words.append("balanced, mirrored" if _pole(sym) > 0 else "off-balance, lopsided")
    if 0.2 < reading.trends.get("stillness", 0) < 0.8:
        time_words.append("broken by stillness")

    return " ".join(_sentence(", ".join(s)) for s in (words, space_words, time_words) if s)


# ---- full: every level at once, no body parts (the default) ----------------

_F_SHAPE = {"shape": ("drawing inward, gathered close", "opening wide, radiating outward"),
            "level": ("low, close to the earth", "reaching upward, lifted high")}
_F_TRENDS = {"opening": ("closing in", "unfolding outward"), "rising": ("sinking, descending", "rising, ascending"),
             "approach": ("withdrawing", "advancing"), "turning": ("", "turning, spiralling"),
             "travel": ("", "travelling through space")}


def _full_text(reading: MotionReading) -> str:
    """A layered description: the effort action or state, the measured
    qualities, the movement in space, its timing, and the fine detail of
    the hands when they are tracked.  No body part is named."""
    if _still(reading):
        return "Stillness, suspended, a held breath. Quiet, at rest."
    eff = _effort_q(reading)
    reg = _register(reading)

    # 1 · pace, effort action or state, size ("Slow, unhurried, gliding, smooth, serene, broad, ample gestures.")
    s1, used = _effort_parts(reading)
    said = {w.strip() for phrase in s1 for w in phrase.split(",")}

    # 2 · the qualities the effort words left unsaid, as plain adjectives
    s2 = [w for w in (_qword(eff.get(k), reg) for k in ("time", "weight", "space", "flow") if k not in used)
          if w and w not in said]

    # 3 · the movement in space, then fine detail
    s3 = [_F_SHAPE[k][_pole(reading.qualities.get(k)) > 0] for k in ("shape", "level") if _pole(reading.qualities.get(k))]
    s3 += _trend_words(reading.trends, _F_TRENDS)
    h = reading.hands
    if _pole(h.get("articulation")):
        s3.append("intricate, articulate detail" if _pole(h["articulation"]) > 0 else "moving as one, unified")
    if _pole(h.get("openness")):
        s3.append("open, offering" if _pole(h["openness"]) > 0 else "closed, gripping")
    op = h.get("opening")
    if op is not None and np.isfinite(op.raw) and abs(op.raw) > 0.4:
        s3.append("unfurling" if op.raw > 0 else "curling closed")
    cv = h.get("converging")
    if cv is not None and np.isfinite(cv.raw) and abs(cv.raw) > 0.5:
        s3.append("gathering together" if cv.raw > 0 else "parting, releasing")

    # 4 · timing
    s4: List[str] = []
    rh = reading.qualities.get("rhythm")
    bpm = reading.trends.get("tempo_bpm", float("nan"))
    if rh is not None and np.isfinite(rh.value) and rh.value > 0.6:
        s4.append(f"pulsing at about {int(round(bpm / 5.0) * 5)} beats per minute" if np.isfinite(bpm) else "pulsing, cyclical")
    elif rh is not None and np.isfinite(rh.value) and rh.value < 0.25 and reg > 0:
        s4.append("erratic, unpredictable timing")
    acc = reading.trends.get("accelerating", float("nan"))
    if np.isfinite(acc) and abs(acc) > 0.35:
        s4.append("gathering momentum" if acc > 0 else "slowing, settling")
    sym = reading.qualities.get("symmetry")
    if _pole(sym):
        s4.append("balanced, mirrored" if _pole(sym) > 0 else "off-balance, one-sided")
    if 0.2 < reading.trends.get("stillness", 0) < 0.8:
        s4.append("broken by held pauses")

    return " ".join(_sentence(", ".join(x)) for x in (s1, s2, s3, s4) if x)


def to_text(reading: MotionReading, level: str = "full") -> str:
    """The qualities that stand out, in words.

    full       (default) the levels below in one layered text, no body parts:
               effort action or state, qualities, movement in space, timing
               ("Thrusting, decisive, explosive. Vigorous, pinpoint, staccato.
               Opening wide, rising. Pulsing at about 120 beats per minute.")
    body       concrete: what moves and how ("Lively movement … arms raised high")
    qualities  adjectives only, no body parts ("Quick, forceful, direct. Expansive, rising.")
    abstract   Laban effort actions and states with spatial imagery
               ("Thrusting, decisive, explosive. Radiating outward, a racing pulse.")"""
    if reading.coverage < 0.3 or (not reading.qualities and not reading.hands):
        return ""
    if level == "full":
        return _full_text(reading)
    if level == "qualities":
        return _qualities_text(reading)
    if level == "abstract":
        return _abstract_text(reading)
    lines = _body_text(reading.qualities, reading.trends) if reading.qualities else []
    lines += _hands_text(reading.hands)
    return " ".join(l for l in lines if l)


def prototype_texts(level: str = "full") -> List[str]:
    """Descriptions spanning the effort space (every Time × Weight × Space ×
    Flow pole, closed and open, low and high) at one level.  Their mean
    embedding is what all movement descriptions share; the live query
    subtracts it so what is distinctive in each description carries more."""
    texts = []
    for i, (t, w, s, f) in enumerate(itertools.product((0.1, 0.9), repeat=4)):
        for k, energy in enumerate((0.35, 0.55, 0.8)):          # one per register
            shape, lvl, size = ((0.15, 0.2, 0.2), (0.85, 0.85, 0.8))[(i + k) % 2]
            r = MotionReading(tracker="body", seconds=3.0, coverage=1.0)
            vals = dict(energy=energy, time=t, weight=w, space=s, flow=f, shape=shape, level=lvl, size=size,
                        symmetry=0.5, rhythm=0.5)
            r.qualities = {k: _q(k, v, float("nan"), "", "", "") for k, v in vals.items()}
            texts.append(to_text(r, level))
    return sorted(set(t for t in texts if t))


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    from .pose import PoseTrajectory as _T, track_video
    ap = argparse.ArgumentParser(prog="python -m delyrism.motion",
                                 description="Describe movement qualities in words, window by window.")
    ap.add_argument("input", help="video file, or a .npz trajectory from delyrism.pose")
    ap.add_argument("--tracker", default="body", choices=["body", "hands", "both"])
    ap.add_argument("--window", type=float, default=3.0, help="seconds per description")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between descriptions")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--level", default="full", choices=LEVELS,
                    help="full (default, layered, no body parts) | abstract | qualities | body (concrete)")
    ap.add_argument("--relative", type=float, default=1.0,
                    help="0 = fixed calibration, 1 (default) = each quality against the video's own range so far")
    ap.add_argument("--values", action="store_true", help="also print each quality's value")
    a = ap.parse_args(argv)
    traj = (_T.load(a.input) if a.input.endswith(".npz")
            else track_video(a.input, tracker=a.tracker, start=a.start, end=a.end, smooth=0.5))
    t, _ = traj.arrays()
    if not len(t):
        print("no landmarks"); return
    end = float(t[0]) + a.window
    norm = SessionNorm()
    while end <= float(t[-1]) + 1e-6:
        win = traj.window(a.window + 0.5, end=end)
        r = describe(win, a.window, a.level, session=norm, relative=a.relative)
        print(f"[{end - a.window:7.2f} - {end:7.2f} s] {r.text or '(nothing tracked)'}")
        if a.values:
            vals = {k: round(q.value, 2) for k, q in {**r.qualities, **r.hands}.items() if np.isfinite(q.value)}
            print("    ", vals)
        end += a.every


if __name__ == "__main__":
    main()
