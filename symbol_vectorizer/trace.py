"""
trace.py — ink layers to vector geometry.

Two complementary vectorizations of the same symbol:

  outlines    filled regions per colour (contours with holes), fitted with
              cubic Béziers.  Reproduces the drawing as it looks, marker
              width and all.  -> <name>.svg
  centerlines the pen's path per colour: the ink thinned to one pixel,
              traced into a graph, chains joined smoothly through junctions
              (a "+" is two strokes), ordered top-left first.  Each stroke
              keeps the pen width measured along it.  -> <name>_lines.svg
              and the stroke list in the JSON, for animation and motion.

Regions much thicker than the pen (a coloured-in triangle) are fills: they
go to the outlines only, and the stroke list records them as fills.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ----------------------------------------------------------------------------
#  Cubic Bézier fitting (Schneider, Graphics Gems 1990), with corner splits
# ----------------------------------------------------------------------------

def _bez(ctrl, t):
    t = t[:, None]
    mt = 1 - t
    return mt ** 3 * ctrl[0] + 3 * mt ** 2 * t * ctrl[1] + 3 * mt * t ** 2 * ctrl[2] + t ** 3 * ctrl[3]


def _bez_d(ctrl, t):
    t = t[:, None]
    mt = 1 - t
    return 3 * mt ** 2 * (ctrl[1] - ctrl[0]) + 6 * mt * t * (ctrl[2] - ctrl[1]) + 3 * t ** 2 * (ctrl[3] - ctrl[2])


def _bez_dd(ctrl, t):
    t = t[:, None]
    return 6 * (1 - t) * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + 6 * t * (ctrl[3] - 2 * ctrl[2] + ctrl[1])


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _chord_params(P):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(P))


def _fit_one(P, u, t1, t2):
    A1, A2 = np.outer(3 * (1 - u) ** 2 * u, t1), np.outer(3 * (1 - u) * u ** 2, t2)
    C = np.array([[np.sum(A1 * A1), np.sum(A1 * A2)], [np.sum(A1 * A2), np.sum(A2 * A2)]])
    base = _bez(np.array([P[0], P[0], P[-1], P[-1]]), u)
    tmp = P - base
    X = np.array([np.sum(A1 * tmp), np.sum(A2 * tmp)])
    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    seg = np.linalg.norm(P[-1] - P[0])
    if abs(det) > 1e-12:
        a1 = (X[0] * C[1, 1] - X[1] * C[0, 1]) / det
        a2 = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det
    else:
        a1 = a2 = seg / 3
    if a1 < 1e-6 * seg or a2 < 1e-6 * seg:
        a1 = a2 = seg / 3
    return np.array([P[0], P[0] + a1 * t1, P[-1] + a2 * t2, P[-1]])


def _reparam(ctrl, P, u):
    q = _bez(ctrl, u) - P
    d1, d2 = _bez_d(ctrl, u), _bez_dd(ctrl, u)
    num = (q * d1).sum(1)
    den = (d1 * d1).sum(1) + (q * d2).sum(1)
    ok = np.abs(den) > 1e-9
    step = np.zeros_like(u)
    step[ok] = num[ok] / den[ok]
    return np.clip(u - step, 0, 1)


def fit_beziers(P: np.ndarray, err: float, t1=None, t2=None, depth: int = 0) -> List[np.ndarray]:
    """Cubic segments within `err` px (max distance) of the polyline P."""
    if len(P) < 2:
        return []
    if t1 is None:
        t1 = _unit(P[min(3, len(P) - 1)] - P[0])
    if t2 is None:
        t2 = _unit(P[max(-4, -len(P))] - P[-1])
    if len(P) == 2 or np.linalg.norm(P[-1] - P[0]) < 1e-6 and len(P) < 4:
        d = np.linalg.norm(P[-1] - P[0]) / 3
        return [np.array([P[0], P[0] + d * t1, P[-1] + d * t2, P[-1]])]
    u = _chord_params(P)
    ctrl = _fit_one(P, u, t1, t2)
    for _ in range(6):
        e = np.linalg.norm(_bez(ctrl, u) - P, axis=1)
        if e.max() <= err:
            return [ctrl]
        u = _reparam(ctrl, P, u)
        ctrl = _fit_one(P, u, t1, t2)
    e = np.linalg.norm(_bez(ctrl, u) - P, axis=1)
    if e.max() <= err or len(P) < 6 or depth > 40:
        return [ctrl]
    k = int(np.clip(np.argmax(e), 2, len(P) - 3))
    tc = _unit(P[k - 1] - P[k + 1])
    return fit_beziers(P[:k + 1], err, t1, tc, depth + 1) + fit_beziers(P[k:], err, -tc, t2, depth + 1)


def _corners(P: np.ndarray, closed: bool, k: int = 6, angle: float = 55.0) -> List[int]:
    """Indices where the polyline turns sharper than `angle` degrees over +-k points."""
    n = len(P)
    if n < 2 * k + 3:
        return []
    idx = np.arange(n)
    a = P[(idx - k) % n] if closed else P[np.clip(idx - k, 0, n - 1)]
    b = P[(idx + k) % n] if closed else P[np.clip(idx + k, 0, n - 1)]
    v1, v2 = P - a, b - P
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    if not closed:
        turn[:k + 1] = 0
        turn[-k - 1:] = 0
    out = []
    for i in np.argsort(-turn):
        if turn[i] < angle:
            break
        if all(min(abs(i - j), n - abs(i - j)) > 2 * k for j in out):
            out.append(int(i))
    return sorted(out)


def path_beziers(P: np.ndarray, closed: bool, err: float) -> List[np.ndarray]:
    """Bézier segments for a polyline, split at corners so they stay sharp."""
    if closed:
        P = P[:-1] if np.linalg.norm(P[0] - P[-1]) < 1e-6 else P
        cs = _corners(P, True)
        if not cs:
            # a smooth loop: start anywhere, tangents continuous through the seam
            Q = np.vstack([P, P[:1]])
            t = _unit(P[1] - P[-1])
            return fit_beziers(Q, err, t, -t)
        P = np.roll(P, -cs[0], axis=0)
        cs = [(c - cs[0]) % len(P) for c in cs] + [len(P)]
        P = np.vstack([P, P[:1]])
        return [b for s, e in zip(cs[:-1], cs[1:]) for b in fit_beziers(P[s:e + 1], err)]
    cs = [0] + _corners(P, False) + [len(P) - 1]
    return [b for s, e in zip(cs[:-1], cs[1:]) for b in fit_beziers(P[s:e + 1], err)]


def svg_d(segs: List[np.ndarray], closed: bool, nd: int = 2) -> str:
    if not segs:
        return ""
    f = lambda p: f"{p[0]:.{nd}f} {p[1]:.{nd}f}"
    parts = [f"M{f(segs[0][0])}"]
    parts += [f"C{f(s[1])} {f(s[2])} {f(s[3])}" for s in segs]
    if closed:
        parts.append("Z")
    return "".join(parts)


# ----------------------------------------------------------------------------
#  Outlines
# ----------------------------------------------------------------------------

def _smooth_closed(P: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter1d(P.astype(float), sigma, axis=0, mode="wrap")


def sample_segs(segs: List[np.ndarray], per_seg: int = 12) -> np.ndarray:
    """Points along a chain of cubic segments."""
    if not segs:
        return np.zeros((0, 2))
    t = np.linspace(0, 1, per_seg, endpoint=False)
    pts = [_bez(s, t) for s in segs] + [segs[-1][3][None]]
    return np.vstack(pts)


def outline_shapes(mask: np.ndarray, err: float = 0.8, sigma: float = 1.2, min_area: float = 4.0
                   ) -> List[List[List[np.ndarray]]]:
    """Outer contours with their holes: shapes -> rings -> cubic segments (px, y down).
    Fill with the even-odd rule."""
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    for i, c in enumerate(cs):
        if hier[i][3] != -1:            # a hole: drawn with its parent
            continue
        rings = [c]
        j = hier[i][2]
        while j != -1:
            rings.append(cs[j])
            j = hier[j][0]
        ds = []
        for r in rings:
            if cv2.contourArea(r) < min_area:
                continue
            P = r[:, 0, :].astype(float) + 0.5          # pixel centres to pixel-edge frame
            if len(P) >= 6:
                P = _smooth_closed(P, sigma)
            ds.append(path_beziers(P, True, err))
        if ds:
            out.append(ds)
    return out


# ----------------------------------------------------------------------------
#  Centerlines
# ----------------------------------------------------------------------------

def _skeleton_graph(mask: np.ndarray):
    import networkx as nx
    sk = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0
    ys, xs = np.nonzero(sk)
    pix = set(zip(ys.tolist(), xs.tolist()))
    G = nx.Graph()
    G.add_nodes_from(pix)
    for (y, x) in pix:
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            q = (y + dy, x + dx)
            if q not in pix:
                continue
            if dy and dx and ((y + dy, x) in pix or (y, x + dx) in pix):
                continue                       # staircase, not a branch
            G.add_edge((y, x), q)
    return G


def _chains(G) -> List[List[Tuple[int, int]]]:
    import networkx as nx
    nodes = {n for n in G if G.degree(n) != 2}
    seen, out = set(), []
    for n in nodes:
        for nb in G.neighbors(n):
            if (n, nb) in seen:
                continue
            path, prev, cur = [n, nb], n, nb
            seen.update({(n, nb), (nb, n)})
            while cur not in nodes:
                nxt = [q for q in G.neighbors(cur) if q != prev]
                if not nxt:
                    break
                q = nxt[0]
                seen.update({(cur, q), (q, cur)})
                path.append(q)
                prev, cur = cur, q
            out.append(path)
    done = {p for c in out for p in c}
    for comp in nx.connected_components(G):          # rings with no branch point
        if comp & done or len(comp) < 8:
            continue
        start = min(comp)
        path, prev, cur = [start], None, start
        while True:
            nxt = [q for q in G.neighbors(cur) if q != prev and (q not in path or (q == start and len(path) > 2))]
            if not nxt:
                break
            q = nxt[0]
            path.append(q)
            if q == start:
                break
            prev, cur = cur, q
        out.append(path)
    return out


def _direction(chain, at_start: bool, pw: float) -> np.ndarray:
    """Leaving direction of a chain at one end, read between one and three pen
    widths away from the junction, where thinning has stopped bending it."""
    P = np.asarray(chain, dtype=float)
    if not at_start:
        P = P[::-1]
    n = len(P)
    i0, i1 = min(int(pw), n // 3), min(int(3 * pw), n - 1)
    if i1 <= i0:
        i0, i1 = 0, n - 1
    return _unit(P[i1] - P[i0])


def _prune_spurs(chains, spur: float):
    """Remove short end branches left by thinning (repeat: pruning exposes more)."""
    for _ in range(4):
        deg = defaultdict(int)
        for c in chains:
            deg[c[0]] += 1
            deg[c[-1]] += 1
        keep = [c for c in chains
                if not (len(c) < spur and (deg[c[0]] == 1) != (deg[c[-1]] == 1))]
        if len(keep) == len(chains):
            break
        chains = keep
    return chains


def _join(chains, pw: float) -> List[List[Tuple[int, int]]]:
    """Join chains that continue smoothly through a junction.

    Thinning splits a crossing of two thick lines into two branch points about
    a pen width apart, joined by a short bridge chain.  Branch points closer
    than that are one junction; the bridge is dropped and the four arms pair
    up by direction (a "+" is two strokes)."""
    import networkx as nx
    tol = max(3.0, 1.1 * pw)
    # junction clusters: chain ends linked by short bridges or lying close together
    ends = [(i, e, np.array(c[0] if e == 0 else c[-1], float)) for i, c in enumerate(chains) for e in (0, 1)
            if c[0] != c[-1] or e == 0]
    J = nx.Graph()
    J.add_nodes_from(range(len(ends)))
    pos = np.array([p for _, _, p in ends]) if ends else np.zeros((0, 2))
    for a in range(len(ends)):
        near = np.nonzero(np.abs(pos[a + 1:] - pos[a]).max(1) <= 2)[0] + a + 1
        J.add_edges_from((a, b) for b in near)                     # same branch pixel area
    deg = defaultdict(int)
    for c in chains:
        deg[c[0]] += 1
        deg[c[-1]] += 1
    bridges = set()
    idx = {(i, e): a for a, (i, e, _) in enumerate(ends)}
    for i, c in enumerate(chains):
        if c[0] != c[-1] and len(c) <= tol and deg[c[0]] >= 2 and deg[c[-1]] >= 2:
            bridges.add(i)
            J.add_edge(idx[(i, 0)], idx[(i, 1)])
    link: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for comp in nx.connected_components(J):
        g = [(ends[q][0], ends[q][1]) for q in comp if ends[q][0] not in bridges]
        if len(g) < 2:
            continue
        dirs = {ie: _direction(chains[ie[0]], ie[1] == 0, pw) for ie in g}
        pairs = sorted((float(dirs[p] @ dirs[q]), p, q) for i, p in enumerate(g) for q in g[i + 1:] if p[0] != q[0])
        free = set(g)
        for dot, p, q in pairs:
            if dot > -0.7:
                break
            if p in free and q in free:
                link[p], link[q] = q, p
                free -= {p, q}
    strokes, done = [], set(bridges)

    def walk(i, e_in):
        pts = []
        while i not in done:
            done.add(i)
            c = chains[i] if e_in == 0 else chains[i][::-1]
            if pts and np.abs(np.subtract(pts[-1], c[0])).max() > 1:
                # jumping across a collapsed junction: bridge the gap in a straight line
                a, b = np.array(pts[-1], float), np.array(c[0], float)
                m = int(np.abs(b - a).max())
                pts.extend(tuple(np.round(a + (b - a) * t).astype(int)) for t in np.linspace(0, 1, m + 1)[1:-1])
            pts.extend(c if not pts else c[1:])
            nxt = link.get((i, 1 - e_in))
            if nxt is None:
                break
            i, e_in = nxt
        return pts

    for i, e in [(i, e) for i in range(len(chains)) for e in (0, 1) if (i, e) not in link]:
        if i not in done:
            strokes.append(walk(i, e))
    for i in range(len(chains)):
        if i not in done:
            strokes.append(walk(i, 0))
    return [s for s in strokes if len(s) >= 2]


def _resample(P: np.ndarray, step: float, sigma: float) -> np.ndarray:
    closed = len(P) > 8 and np.linalg.norm(P[0] - P[-1]) < 3
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return P[:1]
    s = np.linspace(0, d[-1], max(2, int(np.ceil(d[-1] / step)) + 1))
    Q = np.stack([np.interp(s, d, P[:, k]) for k in range(2)], axis=1)
    if len(Q) > 4:
        if closed:
            Q = gaussian_filter1d(Q[:-1], sigma, axis=0, mode="wrap")
            Q = np.vstack([Q, Q[:1]])
        else:
            Q2 = gaussian_filter1d(Q, sigma, axis=0, mode="nearest")
            Q2[0], Q2[-1] = Q[0], Q[-1]
            Q = Q2
    return Q


def _order(strokes):
    """Top-to-bottom, then left-to-right; strokes start at their upper-left end,
    loops at their top point, counter-clockwise on the page."""
    out = []
    for S in strokes:
        P = S["points"]
        if S["closed"]:
            k = int(np.argmin(P[:-1, 1]))
            Q = np.roll(P[:-1], -k, axis=0)
            Q = np.vstack([Q, Q[:1]])
            area = np.sum(Q[:-1, 0] * Q[1:, 1] - Q[1:, 0] * Q[:-1, 1])   # y down
            if area > 0:
                Q = Q[::-1]
            S["points"] = Q
        elif (P[-1, 1] + 0.5 * P[-1, 0]) < (P[0, 1] + 0.5 * P[0, 0]):
            S["points"] = P[::-1]
        out.append(S)
    return sorted(out, key=lambda S: (round(S["points"][0, 1] / 40), S["points"][0, 0]))


@dataclass
class Stroke:
    points: np.ndarray          # (n, 2) px, y down
    color: int                  # palette index
    width: float                # pen width, px
    closed: bool
    kind: str = "line"          # "line" or "dot"


def centerlines(labels: np.ndarray, n_colors: int, pw: float, fill_factor: float = 2.2
                ) -> Tuple[List[Stroke], np.ndarray]:
    """Strokes per colour, plus the mask of fill regions (too thick to be a line)."""
    fills = np.zeros(labels.shape, bool)
    strokes: List[Stroke] = []
    for c in range(n_colors):
        m = labels == c
        if m.sum() == 0:
            continue
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        n, cl, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        line = np.zeros_like(m)
        for k in range(1, n):
            comp = cl == k
            w, h = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
            sk = cv2.ximgproc.thinning(comp.astype(np.uint8) * 255) > 0
            # 90th percentile: a coloured-in wedge is thick over much of its length,
            # a line only at its crossings
            thick = 2 * float(np.percentile(dt[sk], 90)) if sk.any() else 0
            if max(w, h) < 2.5 * pw and st[k, cv2.CC_STAT_AREA] > 0.4 * w * h:
                ys, xs = np.nonzero(comp)
                strokes.append(Stroke(np.array([[xs.mean() + 0.5, ys.mean() + 0.5]]), c, float(max(w, h)), False, "dot"))
            elif thick > fill_factor * pw:
                fills |= comp
            else:
                line |= comp
        if not line.any():
            continue
        G = _skeleton_graph(line)
        chains = _prune_spurs(_chains(G), spur=1.2 * pw)
        for s in _join(chains, pw):
            P = np.array([(x + 0.5, y + 0.5) for (y, x) in s], float)
            closed = len(P) > 8 and np.linalg.norm(P[0] - P[-1]) < 3
            Q = _resample(P, step=1.0, sigma=max(1.0, 0.35 * pw))
            if np.linalg.norm(np.ptp(Q, axis=0)) < 0.8 * pw:
                continue
            ix = np.clip(P.astype(int), 0, np.array(line.shape[::-1]) - 1)
            width = float(2 * np.median(dt[ix[:, 1], ix[:, 0]]))
            strokes.append(Stroke(Q, c, max(width, 1.0), closed))
    ordered = _order([dict(points=s.points, color=s.color, width=s.width, closed=s.closed, kind=s.kind) for s in strokes])
    return [Stroke(**S) for S in ordered], fills
