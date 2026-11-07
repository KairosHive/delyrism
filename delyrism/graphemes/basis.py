"""
basis.py — an orthogonal basis of drawing motion, shared by all symbols
===========================================================================

Any symbol, from any symbol system, is drawn as a motion.  This module
breaks that motion into fragments and expresses every fragment in one
orthonormal wavelet basis, chosen over thousands of human drawings, so that

    motion  →  P(grapheme | motion)  →  P(symbol | motion)

works for symbols the basis has never seen: a new symbol only needs its
glyph (or one drawing) decomposed into the same graphemes.

Pipeline
  1. motion      a drawing becomes a 3D path sampled at FS Hz: pen-down
                 strokes lie on the drawing plane (z = 0), pen lifts arc
                 out of it (z > 0).  An IMU drawing in the air gives the
                 same signal: pointing direction for x, y, a push for z.
  2. fragments   the path is cut at pen changes and at speed minima (the
                 corners where a hand slows down).
  3. transform   each fragment's displacement per time step, normalised
                 by its path length and resampled to L steps, goes through
                 an orthonormal wavelet packet transform.  The planar
                 channels are one complex signal, so each coefficient has
                 a magnitude and a direction; z is real.
  4. basis       the wavelet family and packet tree are the Coifman-
                 Wickerhauser joint best basis: the orthonormal basis with
                 the lowest total entropy of energy over the training
                 fragments, i.e. the one that writes them with the fewest
                 strong terms.
  5. graphemes   the atoms of that basis carrying the most energy.  An
                 atom with a direction (8 bins) is a grapheme token; z
                 atoms carry the lifts.  Orthonormality makes each
                 fragment's energy split exactly across atoms (Parseval),
                 which is the distribution over graphemes.

    basis = GraphemeBasis.learn(glyphs.load_omniglot("background"))
    prof  = basis.profile(drawing_of_a_symbol)            # P(g | symbol)
    p     = basis.encode(xyz, fs=60)                      # P(g | motion)
    w     = basis.symbol_weights(p, {"venus": prof, ...}) # P(symbol | motion)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .glyphs import Drawing

FS = 60.0           # motion sample rate (Hz)
L = 32              # time steps per fragment (a power of two for the packet tree)
LEVELS = 5          # packet tree depth (log2 L)
LIFT = 0.25         # pen-up height, in symbol sizes
DIRS = 8            # direction bins for planar tokens
FAMILIES = ("haar", "db2", "db3", "db4", "db6", "sym3", "sym4", "sym6", "coif1", "coif2")


# ----------------------------------------------------------------------------
#  1 · motion
# ----------------------------------------------------------------------------

def _minjerk(u: np.ndarray) -> np.ndarray:
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def motion(d: Drawing, fs: float = FS, lift: float = LIFT) -> Tuple[np.ndarray, np.ndarray]:
    """(T, 3) path at fs Hz and the (T,) pen-down mask.  Size-normalised:
    the symbol's longer side is 1, centred on 0."""
    P = d.points
    lo, hi = P.min(0), P.max(0)
    c, s = (lo + hi) / 2, max(float(np.ptp(P, 0).max()), 1e-6)
    xyz, pen = [], []
    prev, prev_t = None, None
    for S in d.strokes:
        xy = (S[:, :2] - c) / s
        ts = S[:, 2]
        if prev is not None:
            gap = float(ts[0] - prev_t)
            dist = float(np.linalg.norm(xy[0] - prev))
            gap = gap if 0.15 <= gap <= 0.8 else float(np.clip(0.15 + dist / 1.5, 0.15, 0.8))
            n = max(int(round(gap * fs)), 4)
            u = np.linspace(0, 1, n + 2)[1:-1]
            m = _minjerk(u)[:, None]
            xyz.append(np.c_[prev + (xy[0] - prev) * m, lift * np.sin(np.pi * u)])
            pen.append(np.zeros(n, bool))
        dur = float(ts[-1] - ts[0])
        if len(S) < 2 or dur < 1.5 / fs or np.ptp(xy, 0).max() < 0.01:
            n = max(int(0.06 * fs), 3)                                  # a dot: a brief touch
            seg = np.repeat(xy[:1], n, 0)
        else:
            tt = np.maximum.accumulate(ts - ts[0])
            keep = np.r_[True, np.diff(tt) > 0]
            grid = np.arange(0, tt[keep][-1] + 1e-9, 1 / fs)
            seg = np.stack([np.interp(grid, tt[keep], xy[keep, k]) for k in range(2)], 1)
        xyz.append(np.c_[seg, np.zeros(len(seg))])
        pen.append(np.ones(len(seg), bool))
        prev, prev_t = xy[-1], ts[-1]
    X = np.concatenate(xyz)
    from scipy.ndimage import gaussian_filter1d
    X[:, :2] = gaussian_filter1d(X[:, :2], 1.2, axis=0, mode="nearest")     # tablet / sensor jitter
    return X, np.concatenate(pen)


# ----------------------------------------------------------------------------
#  2 · fragments
# ----------------------------------------------------------------------------

@dataclass
class Fragment:
    a: int                  # first sample (shared with the previous fragment's last)
    b: int                  # last sample
    down: bool              # pen down (drawing) or up (travelling between strokes)


CUT_COST = 0.01     # optimal cutting: the price of one cut, against the share of the symbol's motion the graphemes miss


def segment_optimal(P: np.ndarray, Wk: np.ndarray, lam: float = CUT_COST, max_cand: int = 110,
                    min_pts: int = 4, total: Optional[float] = None) -> List[int]:
    """Cut points of a pen-down path P (n, 2) chosen by dynamic programming:
    minimise sum over fragments of (share of its motion the graphemes Wk
    miss) x (its length / the stroke's length) + lam per fragment.  Each
    fragment is time-normalised and length-normalised before projection, so
    the cut adapts to scale: a fine wiggle splits into small bends and a
    clean arc stays whole.  Returns indices into P, first 0 and last n-1."""
    n = len(P)
    if n < 8:
        return [0, n - 1]
    m = min(n, max_cand)
    cand = np.round(np.linspace(0, n - 1, m)).astype(int)
    own = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    total = own if total is None else float(total)      # the length a cut's saving is measured against
    if own < 1e-9 or total < 1e-9:
        return [0, n - 1]
    idx, u = np.arange(n, dtype=float), np.linspace(0, 1, L + 1)
    best = np.full(m, np.inf)
    best[0] = 0.0
    frm = np.full(m, -1)
    for i in range(m - 1):
        if not np.isfinite(best[i]):
            continue
        js = np.arange(i + 1, m)
        ok = (cand[js] - cand[i] >= min_pts) | ((i == 0) & (js == m - 1))
        js = js[ok]
        if not len(js):
            continue
        pos = cand[i] + np.outer(cand[js] - cand[i], u)
        Z = np.diff(np.interp(pos, idx, P[:, 0]), axis=1) + 1j * np.diff(np.interp(pos, idx, P[:, 1]), axis=1)
        ln = np.abs(Z).sum(1)
        tot = (np.abs(Z) ** 2).sum(1)
        share = (np.abs(Z @ Wk.T) ** 2).sum(1) / np.maximum(tot, 1e-18)
        cost = best[i] + np.where(ln > 1e-12, (1 - share) * ln / total, 0.0) + lam
        upd = cost < best[js]
        best[js[upd]] = cost[upd]
        frm[js[upd]] = i
    out, j = [], m - 1
    while j > 0:
        out.append(int(cand[j]))
        j = int(frm[j])
    return [0] + out[::-1]


_W_DEFAULT: Optional[np.ndarray] = None


def _default_planar() -> Optional[np.ndarray]:
    global _W_DEFAULT
    if _W_DEFAULT is None and os.path.exists(DEFAULT_PATH):
        z = np.load(DEFAULT_PATH)
        _W_DEFAULT = z["W"][z["planar"][:6]]
    return _W_DEFAULT


def fragments(X: np.ndarray, pen: np.ndarray, fs: float = FS, corner_deg: Optional[float] = 70.0,
              method: str = "optimal", W: Optional[np.ndarray] = None, lam: float = CUT_COST,
              scale: str = "symbol") -> List[Fragment]:
    """Cut at pen changes, then each pen-down run:
    "optimal"  where the graphemes W (rows; default: the six of the saved
               basis) rebuild each fragment best, with a cost lam per cut
               (segment_optimal); scale-free.  scale "symbol" (default)
               measures what a cut saves against the whole drawing's ink,
               so cuts go where the symbol needs them; "stroke" measures
               it against each stroke on its own
    "speed"    at pronounced speed minima and sharp turns (heading change
               above corner_deg), the cut the basis was learned with."""
    if method == "optimal" and W is None:
        W = _default_planar()
        if W is None:
            method = "speed"
    out: List[Fragment] = []
    sym_len = float(np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1)[pen[1:] & pen[:-1]].sum())   # all pen-down ink
    edges = np.flatnonzero(np.diff(pen.astype(int))) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(pen)]
    gap = max(int(0.1 * fs), 3)
    for s0, e0 in zip(starts, ends):
        a, b = max(s0 - 1, 0), e0 - 1 if e0 == len(pen) else e0       # share boundary samples
        if not pen[s0]:
            out.append(Fragment(a, b, False))
            continue
        if method == "optimal":
            cuts = segment_optimal(X[a:b + 1, :2], W, lam, total=sym_len if scale == "symbol" else None)
            out += [Fragment(a + p, a + q, True) for p, q in zip(cuts[:-1], cuts[1:]) if q > p]
            continue
        v = np.linalg.norm(np.gradient(X[s0:e0, :2], axis=0), axis=1) * fs
        cand = []
        if len(v) > 2 * gap:
            med = float(np.median(v))
            for i in range(gap, len(v) - gap):
                w = v[i - gap // 2:i + gap // 2 + 1]
                if v[i] == w.min() and v[i] < 0.35 * med:
                    cand.append(i)
            if corner_deg:
                # sharp turns: the heading in and out of a point differ by more than corner_deg
                k = max(2, int(0.05 * fs))
                P = X[s0:e0, :2]
                ang = np.zeros(len(P))
                for i in range(k, len(P) - k):
                    u1, u2 = P[i] - P[i - k], P[i + k] - P[i]
                    n1, n2 = np.linalg.norm(u1), np.linalg.norm(u2)
                    if n1 > 1e-6 and n2 > 1e-6:
                        ang[i] = np.degrees(np.arccos(np.clip(u1 @ u2 / (n1 * n2), -1, 1)))
                for i in range(gap, len(P) - gap):
                    if ang[i] > corner_deg and ang[i] == ang[i - gap // 2:i + gap // 2 + 1].max():
                        cand.append(i)
        cuts = []
        for i in sorted(set(cand)):
            if not cuts or i - cuts[-1] >= gap:
                cuts.append(i)
        bounds = [a] + [s0 + i for i in cuts] + [b]
        out += [Fragment(p, q, True) for p, q in zip(bounds[:-1], bounds[1:]) if q > p]
    return out


def fragment_signal(X: np.ndarray, f: Fragment, n: int = L) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """(planar complex steps, z steps, path length, start point) of one
    fragment, time-normalised to n steps and length-normalised."""
    seg = X[f.a:f.b + 1]
    if len(seg) < 2:
        seg = np.repeat(seg, 2, 0)
    u = np.linspace(0, 1, len(seg))
    g = np.linspace(0, 1, n + 1)
    Q = np.stack([np.interp(g, u, seg[:, k]) for k in range(3)], 1)
    D = np.diff(Q, axis=0)
    length = float(np.linalg.norm(D, axis=1).sum())
    sc = max(length, 1e-4)
    return (D[:, 0] + 1j * D[:, 1]) / sc, D[:, 2] / sc, length, Q[0]


# ----------------------------------------------------------------------------
#  3, 4 · orthonormal wavelet packets and the joint best basis
# ----------------------------------------------------------------------------

def _packet_matrices(wavelet: str, n: int = L, levels: int = LEVELS) -> Dict[str, np.ndarray]:
    """For every packet node ('' = root, 'a', 'd', 'aa', ...) the matrix
    mapping a length-n signal to that node's coefficients."""
    import pywt
    import warnings
    mats: Dict[str, List[np.ndarray]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            wp = pywt.WaveletPacket(e, wavelet, mode="periodization", maxlevel=levels)
            mats.setdefault("", []).append(e)
            for lev in range(1, levels + 1):
                for node in wp.get_level(lev, order="natural"):
                    mats.setdefault(node.path, []).append(np.asarray(node.data, float))
    return {k: np.stack(v, 1) for k, v in mats.items()}


def _entropy_cost(C: np.ndarray, E: np.ndarray) -> float:
    """Additive Shannon cost of coefficients C (n_frag, k) given each
    fragment's total energy E."""
    p = np.abs(C) ** 2 / E[:, None]
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def best_basis(Zp: np.ndarray, Zz: np.ndarray, wavelet: str) -> Tuple[List[str], float, np.ndarray]:
    """Joint best basis of the fragments (planar Zp, z Zz) for one wavelet:
    leaves, total cost, and the orthonormal L × L analysis matrix."""
    mats = _packet_matrices(wavelet)
    E = (np.abs(Zp) ** 2).sum(1) + (Zz ** 2).sum(1) + 1e-12
    cost = {p: _entropy_cost(Zp @ M.T, E) + _entropy_cost(Zz @ M.T, E) for p, M in mats.items()}

    def best(p: str) -> Tuple[float, List[str]]:
        if len(p) == LEVELS:
            return cost[p], [p]
        ca, la = best(p + "a")
        cd, ld = best(p + "d")
        return (cost[p], [p]) if cost[p] <= ca + cd else (ca + cd, la + ld)

    c, leaves = best("")
    W = np.concatenate([mats[p] for p in leaves], 0)
    return leaves, c, W


# ----------------------------------------------------------------------------
#  5 · graphemes
# ----------------------------------------------------------------------------

def _dir_split(c: np.ndarray) -> np.ndarray:
    """Energy of complex coefficients c (..., k) split between the two
    nearest of DIRS direction bins → (..., k, DIRS)."""
    ang = (np.angle(c) % (2 * np.pi)) / (2 * np.pi) * DIRS
    lo = np.floor(ang).astype(int) % DIRS
    w = ang - np.floor(ang)
    out = np.zeros(c.shape + (DIRS,))
    e = np.abs(c) ** 2
    np.put_along_axis(out, lo[..., None], ((1 - w) * e)[..., None], -1)
    hi = (lo + 1) % DIRS
    cur = np.take_along_axis(out, hi[..., None], -1)
    np.put_along_axis(out, hi[..., None], cur + (w * e)[..., None], -1)
    return out


@dataclass
class GraphemeBasis:
    wavelet: str
    leaves: List[str]
    W: np.ndarray                   # (L, L) orthonormal analysis matrix (rows = atoms)
    planar: np.ndarray              # atom indices kept for planar strokes, strongest first
    lift: np.ndarray                # atom indices kept for z
    energy: np.ndarray = field(default_factory=lambda: np.zeros(0))   # mean planar energy share per atom
    family_cost: Dict[str, float] = field(default_factory=dict)
    # how a fragment's coefficients become token mass
    weighting: str = "energy"       # energy |c|² | amplitude |c| | equal (|c|² over the atom's mean share)
    axial: bool = False             # a stroke and its reverse share tokens (direction mod 180°)
    cells: int = 3                  # where in the symbol: a cells × cells grid (1 = anywhere)

    # ---- tokens: [cell ×] (pen-down atom × direction) + [cell ×] (pen-up atom × direction) + (z atom × sign)
    @property
    def n_tokens(self) -> int:
        return 2 * self.cells ** 2 * len(self.planar) * DIRS + 2 * len(self.lift)

    def token_names(self) -> List[str]:
        arrows = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"] if not self.axial else             ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW"]
        cell = [f"@{r}{c}" for r in range(self.cells) for c in range(self.cells)] if self.cells > 1 else [""]
        names = [f"{m}{k}·{arrows[d]}{c}" for m in ("draw", "move") for c in cell
                 for k in range(len(self.planar)) for d in range(DIRS)]
        return names + [f"lift{k}{s}" for k in range(len(self.lift)) for s in ("+", "-")]

    def configured(self, **kw) -> "GraphemeBasis":
        """A copy with other token settings (weighting, axial, cells)."""
        import dataclasses
        return dataclasses.replace(self, **kw)

    # ---- learning
    @classmethod
    def learn(cls, drawings: Sequence[Drawing], n_planar: int = 12, n_lift: int = 3,
              families: Sequence[str] = FAMILIES, max_fragments: int = 40000, seed: int = 0) -> "GraphemeBasis":
        Zp, Zz, _ = _fragment_matrix(drawings, max_fragments, seed)
        costs, best_ = {}, None
        for w in families:
            leaves, c, W = best_basis(Zp, Zz, w)
            costs[w] = c / len(Zp)
            if best_ is None or c < best_[1]:
                best_ = (w, c, leaves, W)
        w, _, leaves, W = best_
        Cp, Cz = Zp @ W.T, Zz @ W.T
        E = (np.abs(Cp) ** 2).sum(1) + (Cz ** 2).sum(1) + 1e-12
        ep = (np.abs(Cp) ** 2 / E[:, None]).mean(0)
        ez = (Cz ** 2 / E[:, None]).mean(0)
        return cls(w, leaves, W, np.argsort(-ep)[:n_planar], np.argsort(-ez)[:n_lift], ep, costs)

    def with_size(self, n_planar: int, n_lift: Optional[int] = None) -> "GraphemeBasis":
        lift = self.lift if n_lift is None else self.lift[:n_lift]
        import dataclasses
        return dataclasses.replace(self, planar=np.argsort(-self.energy)[:n_planar], lift=lift)

    # ---- coding
    def code(self, X: np.ndarray, pen: np.ndarray, fs: float = FS):
        """Per fragment: (down, planar coefficients on the kept atoms,
        z coefficients, path length, start)."""
        out = []
        for f in fragments(X, pen, fs, W=self.W[self.planar]):
            zp, zz, length, start = fragment_signal(X, f)
            centre = X[f.a:f.b + 1, :2].mean(0)
            out.append((f.down, (self.W @ zp)[self.planar], (self.W @ zz)[self.lift], length, start, f, centre))
        return out

    def _cell_weights(self, centre: np.ndarray) -> np.ndarray:
        """Bilinear share of a point (symbol coordinates, -0.5…0.5) in each grid cell."""
        n = self.cells
        if n == 1:
            return np.ones(1)
        gx, gy = np.clip((np.asarray(centre, float) + 0.5) * n - 0.5, 0, n - 1)
        x0, y0 = int(gx), int(gy)
        fx, fy = gx - x0, gy - y0
        w = np.zeros((n, n))                                   # [row = y (bottom first), column = x]
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                w[min(n - 1, y0 + dy), min(n - 1, x0 + dx)] += wy * wx
        return w.reshape(-1)

    def tokens(self, X: np.ndarray, pen: np.ndarray, fs: float = FS) -> np.ndarray:
        """(n_fragments, n_tokens) energy per token, each row weighted by
        the fragment's path length."""
        rows = []
        k = len(self.planar)
        block = self.cells ** 2 * k * DIRS
        for down, cp, cz, length, _, _, centre in self.code(X, pen, fs):
            r = np.zeros(self.n_tokens)
            c = np.abs(cp) * np.exp(2j * np.angle(cp)) if self.axial else cp
            e = _dir_split(c)                                            # (k, DIRS), energy
            if self.weighting == "amplitude":
                e = e / (np.abs(cp)[:, None] + 1e-12)
            elif self.weighting == "equal":
                e = e / (self.energy[self.planar][:, None] + 1e-12)
            e = (self._cell_weights(centre)[:, None, None] * e[None]).reshape(-1)
            off = 0 if down else block
            r[off:off + block] = e
            z = np.zeros(2 * len(self.lift))
            z[0::2] = np.where(cz > 0, cz ** 2, 0)
            z[1::2] = np.where(cz < 0, cz ** 2, 0)
            r[2 * block:] = z
            s = r.sum()
            rows.append(r / s * length if s > 0 else r)
        return np.array(rows) if rows else np.zeros((0, self.n_tokens))

    def encode(self, X: np.ndarray, pen: Optional[np.ndarray] = None, fs: float = FS) -> np.ndarray:
        """P(grapheme | motion) for a (T, 3) path; pen defaults to z ≈ 0."""
        if pen is None:
            pen = X[:, 2] < 0.15 * LIFT
        T = self.tokens(X, pen, fs).sum(0)
        return T / T.sum() if T.sum() > 0 else T

    def profile(self, drawings: Iterable[Drawing], alpha: float = 1e-3) -> np.ndarray:
        """P(grapheme | symbol) from one or more drawings of it (smoothed)."""
        P = np.mean([self.encode(*motion(d)) for d in drawings], 0)
        P = P + alpha
        return P / P.sum()

    def sequence(self, d: Drawing) -> np.ndarray:
        """Per-fragment token distributions, in drawing order."""
        T = self.tokens(*motion(d))
        return T / np.maximum(T.sum(1, keepdims=True), 1e-12)

    @staticmethod
    def symbol_weights(p: np.ndarray, profiles: Dict[str, np.ndarray], temperature: float = 0.05) -> Dict[str, float]:
        """P(symbol | motion) ∝ exp(Σ_g P(g|motion) log P(g|symbol) / T)."""
        names = list(profiles)
        ll = np.array([float(p @ np.log(profiles[n])) for n in names]) / temperature
        w = np.exp(ll - ll.max())
        w /= w.sum()
        return dict(zip(names, w.tolist()))

    # ---- reconstruction (how precisely the kept graphemes rebuild a drawing)
    def reconstruct(self, d: Drawing) -> Tuple[np.ndarray, np.ndarray]:
        """(original, rebuilt) pen-down points, each fragment placed at its
        true start and drawn from the kept atoms only."""
        X, pen = motion(d)
        orig, rec = [], []
        Wp = self.W[self.planar]
        for f in fragments(X, pen, W=Wp):
            if not f.down:
                continue
            zp, _, length, start = fragment_signal(X, f)
            q = np.cumsum(np.r_[0, zp]) * length + (start[0] + 1j * start[1])
            r = np.cumsum(np.r_[0, Wp.T @ (Wp @ zp)]) * length + (start[0] + 1j * start[1])
            orig.append(q)
            rec.append(r)
        o, r = np.concatenate(orig), np.concatenate(rec)
        return np.c_[o.real, o.imag], np.c_[r.real, r.imag]

    def error(self, d: Drawing) -> float:
        o, r = self.reconstruct(d)
        return float(np.linalg.norm(o - r, axis=1).mean())

    def atom_shape(self, k: int) -> np.ndarray:
        """The path drawn by planar atom k pointing east (unit length)."""
        a = self.W[self.planar[k]]
        z = np.cumsum(np.r_[0, a])
        return z / (np.abs(np.diff(z)).sum() + 1e-12)

    # ---- persistence
    def save(self, path: str) -> None:
        np.savez_compressed(path, wavelet=self.wavelet, leaves=np.array(self.leaves), W=self.W,
                            planar=self.planar, lift=self.lift, energy=self.energy,
                            weighting=self.weighting, axial=self.axial, cells=self.cells)

    @classmethod
    def load(cls, path: str) -> "GraphemeBasis":
        z = np.load(path)
        kw = {k: (str(z[k]) if k == "weighting" else bool(z[k]) if k == "axial" else int(z[k]))
              for k in ("weighting", "axial", "cells") if k in z.files}
        return cls(str(z["wavelet"]), [str(x) for x in z["leaves"]], z["W"], z["planar"], z["lift"], z["energy"], **kw)


def _fragment_matrix(drawings: Sequence[Drawing], max_fragments: int, seed: int):
    Zp, Zz, down = [], [], []
    for d in drawings:
        X, pen = motion(d)
        for f in fragments(X, pen, method="speed"):
            zp, zz, length, _ = fragment_signal(X, f)
            if length > 1e-3:
                Zp.append(zp)
                Zz.append(zz)
                down.append(f.down)
    Zp, Zz, down = np.array(Zp), np.array(Zz), np.array(down)
    if len(Zp) > max_fragments:
        idx = np.random.default_rng(seed).choice(len(Zp), max_fragments, replace=False)
        Zp, Zz, down = Zp[idx], Zz[idx], down[idx]
    return Zp, Zz, down


def dtw_distance(A: np.ndarray, B: np.ndarray) -> float:
    """DTW between two sequences of token distributions, cost 1 − Bhattacharyya."""
    S = np.sqrt(A) @ np.sqrt(B).T
    C = 1.0 - S
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


# ----------------------------------------------------------------------------
#  default basis and CLI
# ----------------------------------------------------------------------------

import os as _os

DEFAULT_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "grapheme_basis.npz")


def default_basis(n_planar: int = 6) -> GraphemeBasis:
    """The basis learned on Omniglot's 30 background alphabets, with the
    n_planar strongest atoms kept (learned and saved on first use)."""
    if not _os.path.exists(DEFAULT_PATH):
        from .glyphs import load_omniglot
        GraphemeBasis.learn(load_omniglot("background"), n_planar=32, max_fragments=60000).save(DEFAULT_PATH)
    return GraphemeBasis.load(DEFAULT_PATH).with_size(n_planar)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    from . import glyphs as _g
    ap = argparse.ArgumentParser(prog="python -m delyrism.graphemes",
                                 description="Orthogonal grapheme basis of drawing motion.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("learn", help="learn the basis on Omniglot's background alphabets and save it")
    a.add_argument("--out", default=DEFAULT_PATH)
    s = sub.add_parser("symbols", help="grapheme profile of each symbol in a font symbol set")
    s.add_argument("system", choices=list(_g.SYMBOL_SETS))
    s.add_argument("--planar", type=int, default=6)
    s.add_argument("--top", type=int, default=4)
    args = ap.parse_args(argv)
    if args.cmd == "learn":
        B = GraphemeBasis.learn(_g.load_omniglot("background"), n_planar=32, max_fragments=60000)
        B.save(args.out)
        print(f"{B.wavelet} best basis, leaves {B.leaves} → {args.out}")
        return
    B = default_basis(args.planar)
    names = B.token_names()
    for d in _g.font_drawings(args.system):
        p = B.profile([d])
        top = np.argsort(-p)[:args.top]
        print(f"{d.char}  {d.symbol.split('/')[-1][:28]:28s} " + "  ".join(f"{names[i]} {p[i]:.2f}" for i in top))


if __name__ == "__main__":
    main()
