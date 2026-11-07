"""Generative symmetric composition of Lakota symbols (pure numpy).

Builds one square wallpaper tile with the symmetry of the square (p4m: four
rotations × mirror).  The composer works in the fundamental eighth of the tile
(0 ≤ y ≤ x ≤ ½) and every placement is expanded by the 8 symmetries; the tile
edges are mirror lines, so translated copies of the tile join seamlessly and a
figure placed by a corner becomes a 4-fold rosette across the tiles that meet
there.

Recipe (seeded, so a given mosaic always looks the same) — a procedural snowflake:
    centre figure — a single symbol if it has the full symmetry, else a rosette
    arms          — grow out along the axes and diagonals: figures on the arm,
                    side branches (mirrored by the arm) and sub-branches
    corner / edge — figures pointing into the tile from corners and edge midpoints
    fill          — small symbols on a snapped grid, angles snapped to 45°
Every step picks the symbol furthest below its target share of ink, so the
ink of each symbol follows the activation proportions.  A collision raster
with a clearance gap keeps every figure separate and inside the tile.

On a mirror line a figure must coincide with its own reflection: only
symbols with a matching mirror symmetry go there, turned so that axis lies
along the line.  Asymmetric symbols appear as mirrored pairs / pinwheels.

Coordinates: tile units, centre (0, 0), y up, tile spans [-½, ½]².  A figure
of scale s is an atlas canvas of side s (the symbol fills ~88 % of it).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

R = 200          # collision raster (pixels per tile side)
MASK = 64        # symbol mask resolution
GAP = 0.06       # clearance per side, fraction of the canvas


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
    """alpha: canvas coverage (H, W), row 0 at the top → masks + symmetries."""
    h, w = alpha.shape
    f = max(1, min(h, w) // MASK)
    a = alpha[: f * MASK, : f * MASK].reshape(MASK, f, MASK, f).mean(axis=(1, 3))
    solid = a > 0.3
    return dict(solid=solid, dil=_dilate(solid, max(1, int(round(GAP * MASK)))),
                fill=float(solid.mean()),
                lr=_iou(solid, solid[:, ::-1]) > 0.86,
                ud=_iou(solid, solid[::-1, :]) > 0.86,
                r90=_iou(solid, np.rot90(solid)) > 0.86)


# ------------------------------------------------------------------ geometry
def _rot(deg: float) -> np.ndarray:
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


_F = np.diag([1.0, -1.0])
GROUP = [_rot(90 * k) for k in range(4)] + [_rot(90 * k) @ _F for k in range(4)]


def orient(phi: float, mirror: bool = False) -> np.ndarray:
    return _rot(phi) @ (np.diag([-1.0, 1.0]) if mirror else np.eye(2))


def expand(x: float, y: float, M: np.ndarray) -> List[Tuple[float, float, np.ndarray]]:
    """The 8 symmetric copies, without duplicates (figures on a mirror line coincide)."""
    out, seen = [], set()
    for G in GROUP:
        p = G @ (x, y)
        key = (round(p[0], 5), round(p[1], 5))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(p[0]), float(p[1]), G @ M))
    return out


def decompose(M: np.ndarray) -> Tuple[float, bool]:
    """2×2 orthogonal → (rotation deg, mirrored) with M = R(phi) · diag(±1, 1)."""
    if np.linalg.det(M) < 0:
        return math.degrees(math.atan2(-M[1, 0], -M[0, 0])), True
    return math.degrees(math.atan2(M[1, 0], M[0, 0])), False


def _footprint(mask: np.ndarray, x: float, y: float, s: float, M: np.ndarray):
    rad = s * 0.7072
    c0, c1 = int(math.floor((x - rad + 0.5) * R)), int(math.ceil((x + rad + 0.5) * R))
    r0, r1 = int(math.floor((0.5 - y - rad) * R)), int(math.ceil((0.5 - y + rad) * R))
    cols, rows = np.arange(c0, c1), np.arange(r0, r1)
    XX, YY = np.meshgrid((cols + 0.5) / R - 0.5, 0.5 - (rows + 0.5) / R)
    dx, dy = XX - x, YY - y
    qx = (M[0, 0] * dx + M[1, 0] * dy) / s           # local = Mᵀ · d / s
    qy = (M[0, 1] * dx + M[1, 1] * dy) / s
    u = np.floor((qx + 0.5) * MASK).astype(int)
    v = np.floor((0.5 - qy) * MASK).astype(int)
    ok = (u >= 0) & (u < MASK) & (v >= 0) & (v < MASK)
    hit = np.zeros(ok.shape, bool)
    hit[ok] = mask[v[ok], u[ok]]
    return rows, cols, hit


class _Canvas:
    def __init__(self, infos: Dict[int, Dict]):
        self.occ = np.zeros((R, R), bool)
        self.infos = infos
        self.items: List[Tuple[int, float, float, float, np.ndarray]] = []

    def place(self, k: int, x: float, y: float, s: float, M: np.ndarray) -> int:
        """Place symbol k at a fundamental site; all copies or nothing.  → #copies."""
        copies = expand(x, y, M)
        scratch = self.occ.copy()
        for cx, cy, CM in copies:
            rows, cols, hit = _footprint(self.infos[k]["dil"], cx, cy, s, CM)
            if not hit.any():
                return 0
            inside_r = (rows >= 0) & (rows < R)
            inside_c = (cols >= 0) & (cols < R)
            if hit[~inside_r, :].any() or hit[:, ~inside_c].any():
                return 0                                    # would cross the tile edge
            sub = hit[np.ix_(inside_r, inside_c)]
            ri, ci = rows[inside_r], cols[inside_c]
            region = scratch[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1]
            if (region & sub).any():
                return 0
            region |= sub
        self.occ = scratch
        for cx, cy, CM in copies:
            self.items.append((k, cx, cy, s, CM))
        return len(copies)


# ------------------------------------------------------------------ recipe
def _ray_phis(info: Dict, theta: float) -> List[float]:
    """Orientations for which the figure is its own mirror image across the line at theta."""
    out = []
    if info["lr"]:
        out += [theta - 90, theta + 90]        # its vertical axis along the line (out / in)
    if info["ud"]:
        out += [theta, theta + 180]
    return out


def compose(infos: Dict[int, Dict], symbols: Sequence[int], shares: Sequence[float],
            seed: int = 0, density: float = 0.16, maxsize: float = 0.125) -> Dict:
    """→ {items: [(symbol, x, y, scale, rot_deg, mirrored)], achieved: {symbol: share}}

    maxsize caps every figure's scale (fraction of the tile side); the
    structure (arm spacing, rosette radii) keeps its unscaled proportions."""
    rng = np.random.default_rng(seed)
    cv = _Canvas(infos)
    target = {k: float(w) for k, w in zip(symbols, shares)}
    ink = {k: 0.0 for k in symbols}

    def by_deficit(cands: Sequence[int]) -> List[int]:
        tot = sum(ink.values())
        key = {k: target[k] - (ink[k] / tot if tot > 0 else 0.0) + rng.uniform(0, 0.02) for k in cands}
        return sorted(cands, key=lambda k: -key[k])

    def put(k: int, x: float, y: float, s: float, M: np.ndarray) -> bool:
        s = min(s, maxsize)
        n = cv.place(k, x, y, s, M)
        if n:
            ink[k] += n * s * s * infos[k]["fill"]
        return n > 0

    def ray(theta: float, r: float, s: float, cands: Sequence[int], shrink: int = 2) -> bool:
        """A figure on the mirror line at angle theta, distance r from the centre."""
        c, sn = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        for k in by_deficit([k for k in cands if _ray_phis(infos[k], theta)]):
            for phi in rng.permutation(_ray_phis(infos[k], theta)):
                ss = s
                for _ in range(shrink + 1):
                    if put(k, r * c, r * sn, ss, orient(phi)):
                        return True
                    ss *= 0.85
        return False

    syms = list(symbols)

    # 1 · centre: a fully symmetric figure, or a rosette of a mirror-symmetric one
    sc = rng.uniform(0.17, 0.25)
    placed = False
    for k in by_deficit(syms):
        inf = infos[k]
        if inf["r90"] and inf["lr"]:
            placed = put(k, 0.0, 0.0, sc, orient(rng.choice([0.0, 45.0])))
        elif inf["lr"] or inf["ud"]:
            placed = ray(rng.choice([0.0, 45.0]), sc * rng.uniform(0.45, 0.6), sc * 0.75, [k])
        if placed:
            break
    if not placed:                                     # asymmetric only: a pinwheel near the centre
        for k in by_deficit(syms):
            if put(k, sc * 0.55, sc * 0.18, sc * 0.6, orient(rng.choice([0.0, 90.0]))):
                break

    # 2 · snowflake: arms grow outward along the two mirror rays of the
    #     fundamental eighth (axis 0°, diagonal 45°; the symmetry makes 8 arms).
    #     At each step an arm places a figure on itself (oriented along the arm)
    #     or sprouts a side branch — a row of smaller figures at 45° or 90° to
    #     the arm, oriented along the branch; the arm being a mirror line, every
    #     branch gets its mirror twin.  Branches may sprout a sub-branch.
    def branch(x0: float, y0: float, direction: float, s: float, n: int, depth: int) -> None:
        dx, dy = math.cos(math.radians(direction)), math.sin(math.radians(direction))
        cands = by_deficit(syms)
        k = cands[0] if rng.random() < 0.75 else cands[min(1, len(cands) - 1)]
        mirror = bool(rng.random() < 0.5)
        phi = direction - 90.0 + (180.0 if rng.random() < 0.5 else 0.0)
        step = s * rng.uniform(1.15, 1.45)
        for i in range(n):
            d = s * 0.9 + i * step
            x, y = x0 + d * dx, y0 + d * dy
            if not (0.0 < y < x - 0.01 and x < 0.48):
                break
            if not put(k, x, y, s, orient(phi, mirror)):
                break
            if depth > 0 and i == n // 2 and rng.random() < 0.45:
                turn = float(rng.choice([45.0, -45.0, 90.0, -90.0]))
                branch(x, y, direction + turn, s * 0.7, int(rng.integers(2, 4)), depth - 1)

    for theta in rng.permutation([0.0, 45.0]):
        rmax = 0.5 if theta == 0 else 0.5 * math.sqrt(2)
        c, sn = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        s = rng.uniform(0.08, 0.13)
        grow = rng.uniform(0.82, 1.05)                         # tips shrink (or slightly swell)
        r = sc * 0.75 + s * 0.8
        steps = 0
        while r < rmax - s * 0.8 and steps < 12:
            steps += 1
            if rng.random() < 0.5:
                if ray(theta, r, s, syms, shrink=1):
                    r += s * rng.uniform(1.05, 1.35)
                else:
                    r += s * 0.5
            else:
                # side branch into the fundamental domain (below the diagonal / above the axis)
                angle = float(rng.choice([45.0, 90.0]))
                direction = theta + angle if theta == 0 else theta - angle
                branch(r * c, r * sn, direction, s * rng.uniform(0.45, 0.6),
                       int(rng.integers(2, 5)), depth=1)
                r += s * rng.uniform(0.9, 1.3)
            s = max(0.03, s * grow)

    # 4 · corners and edge midpoints: figures pointing into the tile
    for theta, rmax in ((45.0, 0.5 * math.sqrt(2)), (0.0, 0.5)):
        if rng.random() < 0.85:
            s = rng.uniform(0.07, 0.11)
            for _ in range(3):
                if ray(theta, rmax - s * rng.uniform(0.6, 0.9), s, syms, shrink=1):
                    break
                s *= 0.85

    # 5 · light fill of the leftover space: small symbols on a snapped grid,
    #     capped so the snowflake structure stays legible
    grid = 1.0 / 48
    for _ in range(60):
        if sum(ink.values()) >= density:
            break
        s = rng.uniform(0.035, 0.07)
        x = round(rng.uniform(0.05, 0.47) / grid) * grid
        y = round(rng.uniform(0.0, x) / grid) * grid
        if y < 0.02 or x - y < 0.02:
            continue                                   # stay off the mirror lines
        for k in by_deficit(syms)[:2]:
            phi = float(rng.choice([0, 45, 90, 135, 180, 225, 270, 315]))
            if put(k, x, y, s, orient(phi, bool(rng.random() < 0.3))):
                break

    tot = sum(ink.values()) or 1.0
    items = []
    for k, x, y, s, M in cv.items:
        phi, mir = decompose(M)
        items.append((k, x, y, s, phi, mir))
    return dict(items=items, achieved={k: ink[k] / tot for k in syms}, ink=tot)


# ------------------------------------------------------------------ preview
def render(items, atlas_cells: Dict[int, np.ndarray], px: int = 900, tiles: int = 1) -> np.ndarray:
    """Rasterise `tiles × tiles` copies of the tile (white on black) for checking."""
    from PIL import Image
    img = Image.new("L", (px * tiles, px * tiles), 0)
    for ty in range(tiles):
        for tx in range(tiles):
            for k, x, y, s, phi, mir in items:
                cell = atlas_cells[k]
                size = max(2, int(round(s * px)))
                im = Image.fromarray((cell * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
                if mir:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                im = im.rotate(phi, resample=Image.BICUBIC, expand=True)
                cx = (x + 0.5 + tx) * px
                cy = (0.5 - y + ty) * px
                img.paste(255, (int(cx - im.width / 2), int(cy - im.height / 2)), im)
    return np.asarray(img)
