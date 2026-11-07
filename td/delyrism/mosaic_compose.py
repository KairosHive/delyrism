"""Procedural snowflake of Lakota symbols around the screen centre (pure numpy).

One figure with the dihedral symmetry D_n about the centre: n arms, each a
mirror line, so every figure appears 2n times (n where it sits on a mirror
line) and turns with its arm — the whole pattern is point symmetric and each
symbol's rotation follows its angle to the centre.

Growth is dendritic, like ice:
    centre   — one figure at the middle
    arms     — n main arms grow straight out along the mirror lines, and n
               shorter ones on the bisectors, starting further out
    branches — every few figures a path sprouts side branches at the branch
               angle (60° for six arms: the branches run parallel to the
               neighbouring arms, as in a snowflake); branches sprout
               sub-branches the same way
A path keeps one symbol (picked by how far each symbol is below its target
share of ink, so the ink follows the activation proportions) and grows until
it would collide with something or leave the screen.  Consecutive figures are
spaced by their real extents along the path plus a constant gap, so the gaps
along every path are uniform.

Orientation: a symbol's principal axis (long axis of its ink, from the mask's
second moments) lies along its path; symbols without a clear axis point
their 'up' along it.  On a mirror line an asymmetric figure keeps a single copy.

Coordinates: units of the screen height, centre (0, 0), y up.  A figure of
scale s is an atlas canvas of side s (the symbol fills ~88 % of it).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Sequence, Tuple

import numpy as np

PX = 200         # collision raster: pixels per screen height
MASK = 64        # symbol mask resolution
GAP = 0.05       # collision clearance per side, fraction of the canvas
ELONG = 1.2      # principal-axis anisotropy above which a symbol has a direction


# ------------------------------------------------------------------ symbols
def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    for _ in range(r):
        n = out.copy()
        n[1:] |= out[:-1]; n[:-1] |= out[1:]; n[:, 1:] |= out[:, :-1]; n[:, :-1] |= out[:, 1:]
        out = n
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def symbol_info(alpha: np.ndarray) -> Dict:
    """alpha: canvas coverage (H, W), row 0 at the top → masks, symmetries, axis, ink points."""
    h, w = alpha.shape
    f = max(1, min(h, w) // MASK)
    a = alpha[: f * MASK, : f * MASK].reshape(MASK, f, MASK, f).mean(axis=(1, 3))
    solid = a > 0.3
    v, u = np.nonzero(solid)
    x, y = (u + 0.5) / MASK - 0.5, 0.5 - (v + 0.5) / MASK          # canvas units, y up
    axis, aniso = 90.0, 1.0
    if len(u) > 2:
        wv, V = np.linalg.eigh(np.cov(np.vstack([x - x.mean(), y - y.mean()])))
        axis = math.degrees(math.atan2(V[1, 1], V[0, 1])) % 180.0  # long axis, deg from +x
        aniso = math.sqrt(max(wv[1], 1e-12) / max(wv[0], 1e-12))
    pts = np.column_stack([x, y])
    if len(pts) > 700:
        pts = pts[np.linspace(0, len(pts) - 1, 700).astype(int)]
    return dict(solid=solid, dil=_dilate(solid, max(1, int(round(GAP * MASK)))),
                fill=float(solid.mean()), axis=axis, aniso=aniso, pts=pts,
                lr=_iou(solid, solid[:, ::-1]) > 0.86,
                ud=_iou(solid, solid[::-1, :]) > 0.86,
                r90=_iou(solid, np.rot90(solid)) > 0.86)


# ------------------------------------------------------------------ geometry
def _rot(deg: float) -> np.ndarray:
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def orient(phi: float, mirror: bool = False) -> np.ndarray:
    return _rot(phi) @ (np.diag([-1.0, 1.0]) if mirror else np.eye(2))


def align(info: Dict, direction: float, mirror: bool = False, flip: bool = False) -> np.ndarray:
    """Orientation putting the symbol's long axis along `direction` (its 'up' for
    symbols without a clear axis); flip = turned 180°, mirror = reflected."""
    a = info["axis"] if info["aniso"] >= ELONG else 90.0
    if mirror:
        a = 180.0 - a                          # the reflected symbol's axis
    return orient(direction - a + (180.0 if flip else 0.0), mirror)


def dihedral(n: int) -> List[np.ndarray]:
    """D_n about the origin, mirror lines at multiples of 180/n (rotations first)."""
    F = np.diag([1.0, -1.0])
    return [_rot(360.0 * k / n) for k in range(n)] + [_rot(360.0 * k / n) @ F for k in range(n)]


def decompose(M: np.ndarray) -> Tuple[float, bool]:
    """2×2 orthogonal → (rotation deg, mirrored) with M = R(phi) · diag(±1, 1)."""
    if np.linalg.det(M) < 0:
        return math.degrees(math.atan2(-M[1, 0], -M[0, 0])), True
    return math.degrees(math.atan2(M[1, 0], M[0, 0])), False


def support(info: Dict, M: np.ndarray, s: float, d: np.ndarray) -> float:
    """How far the figure's ink reaches from its centre in direction d."""
    return s * float(np.max(info["pts"] @ M.T @ d))


class _Canvas:
    """Collision raster over the disc of radius rmax; places all symmetric copies or none."""

    def __init__(self, infos: Dict[int, Dict], group: List[np.ndarray], rmax: float):
        self.infos, self.group, self.rmax = infos, group, rmax
        self.n = int(math.ceil(2 * rmax * PX)) + 2
        self.occ = np.zeros((self.n, self.n), bool)
        self.items: List[Tuple[int, float, float, float, np.ndarray]] = []

    def _footprint(self, mask, x, y, s, M):
        rad = s * 0.7072
        o = self.rmax
        c0, c1 = int(math.floor((x - rad + o) * PX)), int(math.ceil((x + rad + o) * PX))
        r0, r1 = int(math.floor((o - y - rad) * PX)), int(math.ceil((o - y + rad) * PX))
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, self.n), min(r1, self.n)
        if c1 <= c0 or r1 <= r0:
            return None
        cols, rows = np.arange(c0, c1), np.arange(r0, r1)
        XX, YY = np.meshgrid((cols + 0.5) / PX - o, o - (rows + 0.5) / PX)
        dx, dy = XX - x, YY - y
        qx = (M[0, 0] * dx + M[1, 0] * dy) / s           # local = Mᵀ · d / s
        qy = (M[0, 1] * dx + M[1, 1] * dy) / s
        uu = np.floor((qx + 0.5) * MASK).astype(int)
        vv = np.floor((0.5 - qy) * MASK).astype(int)
        ok = (uu >= 0) & (uu < MASK) & (vv >= 0) & (vv < MASK)
        hit = np.zeros(ok.shape, bool)
        hit[ok] = mask[vv[ok], uu[ok]]
        return r0, r1, c0, c1, hit

    def place(self, k: int, x: float, y: float, s: float, M: np.ndarray) -> int:
        if math.hypot(x, y) > self.rmax - s * 0.5:
            return 0
        copies, seen = [], set()
        for G in self.group:
            p = G @ (x, y)
            key = (round(p[0], 4), round(p[1], 4))
            if key in seen:
                continue                                 # on a mirror line: one copy
            seen.add(key)
            copies.append((float(p[0]), float(p[1]), G @ M))
        scratch = self.occ.copy()
        for cx, cy, CM in copies:
            fp = self._footprint(self.infos[k]["dil"], cx, cy, s, CM)
            if fp is None:
                return 0
            r0, r1, c0, c1, hit = fp
            region = scratch[r0:r1, c0:c1]
            if (region & hit).any():
                return 0
            region |= hit
        self.occ = scratch
        for cx, cy, CM in copies:
            self.items.append((k, cx, cy, s, CM))
        return len(copies)


# ------------------------------------------------------------------ recipe
def compose(infos: Dict[int, Dict], symbols: Sequence[int], shares: Sequence[float],
            seed: int = 0, aspect: float = 16 / 9, arms: int = 6, size: float = 0.06,
            gap: float = 0.25, branching: float = 0.7) -> Dict:
    """→ {items: [(symbol, x, y, scale, rot_deg, mirrored)], achieved: {symbol: share}}

    aspect    — screen width / height (the figure grows to the screen corners)
    arms      — n of the D_n symmetry
    size      — scale of the main-arm figures (fraction of the screen height)
    gap       — space between consecutive figures on a path (fraction of the scale)
    branching — probability that a path sprouts branches at a sprouting point
    """
    rng = np.random.default_rng(seed)
    n = max(3, int(arms))
    rmax = math.hypot(aspect / 2, 0.5) + size
    cv = _Canvas(infos, dihedral(n), rmax)
    target = {k: float(w) for k, w in zip(symbols, shares)}
    ink = {k: 0.0 for k in symbols}
    syms = list(symbols)
    bangle = min(60.0, max(45.0, 360.0 / n))                 # branch angle to its parent path

    def by_deficit(cands: Sequence[int], avoid: int = -1) -> List[int]:
        tot = sum(ink.values())
        key = {k: target[k] - (ink[k] / tot if tot > 0 else 0.0) + rng.uniform(0, 0.03)
               - (0.04 if k == avoid else 0.0) for k in cands}
        return sorted(cands, key=lambda k: -key[k])

    def put(k, x, y, s, M) -> bool:
        c = cv.place(k, x, y, s, M)
        if c:
            ink[k] += c * s * s * infos[k]["fill"]
        return c > 0

    def walk(parent, direction, s, k, M, maxlen):
        """Place figures from `parent` (k, x, y, s, M) outward along `direction`."""
        d = np.array([math.cos(math.radians(direction)), math.sin(math.radians(direction))])
        pk, px, py, ps, pM = parent
        out = []
        for i in range(maxlen):
            dist = support(infos[pk], pM, ps, d) + gap * s + support(infos[k], M, s, -d) + 2.0 / PX
            ok = False
            for slide in range(8 if i == 0 else 1):        # the first figure may slide out to fit
                x, y = px + d[0] * (dist + slide * 0.2 * s), py + d[1] * (dist + slide * 0.2 * s)
                if put(k, x, y, s, M):
                    ok = True
                    break
            if not ok:
                break
            out.append((k, x, y, s, M))
            pk, px, py, ps, pM = k, x, y, s, M
        return out

    # 1 · centre
    sc = size * 1.5
    centre = None
    for k in sorted(syms, key=lambda k: (not infos[k]["r90"], infos[k]["aniso"])):
        M = align(infos[k], 90.0)
        if put(k, 0.0, 0.0, sc, M):
            centre = (k, 0.0, 0.0, sc, M)
            break
    if centre is None:
        centre = (syms[0], 0.0, 0.0, 0.0, np.eye(2))

    # 2 · paths, breadth first: (start, direction, scale, depth, on mirror line, length cap)
    queue = deque()
    queue.append((centre, 0.0, size, 0, True, 40))
    every = int(rng.integers(2, 4))                              # sprouting interval along a path
    sec_start = float(rng.uniform(0.18, 0.3))                    # bisector arms start this far out
    sec_done = False
    last_k = -1
    while queue or not sec_done:
        if not queue:
            sec_done = True
            theta = 180.0 / n                                     # the bisector, also a mirror line
            d = np.array([math.cos(math.radians(theta)), math.sin(math.radians(theta))])
            r = sec_start
            while r < rmax:
                k = by_deficit(syms, last_k)[0]
                M = align(infos[k], theta, flip=bool(rng.random() < 0.5))
                stub = (k, d[0] * (r - size), d[1] * (r - size), 0.0, M)    # a point just behind
                path = walk(stub, theta, size * 0.9, k, M, 40)
                if path:
                    queue.append(("from", path, theta, size * 0.9, 1, True))
                    break
                r += size * 0.5
            continue
        item = queue.popleft()
        if item[0] == "from":                                     # sprout branches off a placed path
            _, path, direction, s, depth, onaxis = item
            if depth > 3 or s < size * 0.35:
                continue
            offset = int(rng.integers(1, every + 1))
            for i in range(offset, len(path), every):
                if rng.random() > branching:
                    continue
                for sign in ((1.0,) if onaxis else (1.0, -1.0)):
                    queue.append((path[i], direction + sign * bangle, s * 0.8, depth, False,
                                  int(rng.integers(3, 9)) if depth else 30))
            continue
        start, direction, s, depth, onaxis, cap = item
        k = by_deficit(syms, last_k)[0]
        M = align(infos[k], direction, mirror=bool(not onaxis and rng.random() < 0.5),
                  flip=bool(rng.random() < 0.5))
        path = walk(start, direction, s, k, M, cap)
        if path:
            last_k = k
            queue.append(("from", path, direction, s, depth + 1, onaxis))

    tot = sum(ink.values()) or 1.0
    items = []
    for k, x, y, s, M in cv.items:
        phi, mir = decompose(M)
        items.append((k, x, y, s, phi, mir))
    return dict(items=items, achieved={k: ink[k] / tot for k in syms}, ink=tot)


# ------------------------------------------------------------------ preview
def render(items, atlas_cells: Dict[int, np.ndarray], height: int = 720, aspect: float = 16 / 9) -> np.ndarray:
    """Rasterise the composition on a screen (white on black) for checking."""
    from PIL import Image
    W, H = int(round(height * aspect)), height
    img = Image.new("L", (W, H), 0)
    for k, x, y, s, phi, mir in items:
        size = max(2, int(round(s * H)))
        im = Image.fromarray((atlas_cells[k] * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
        if mir:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = im.rotate(phi, resample=Image.BICUBIC, expand=True)
        cx, cy = W / 2 + x * H, H / 2 - y * H
        img.paste(255, (int(cx - im.width / 2), int(cy - im.height / 2)), im)
    return np.asarray(img)
