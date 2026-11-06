"""Prepare the Lakota symbol set for the TouchDesigner background tiling.

    python td/tools/lakota_prepare.py            (from the repo root, project venv)

Input : td/assets/lakota/raw/*.png   (from AntoineBellemare/LakotaNeuralDreams,
        dream_language_symbols/no_bg_square — black symbols, transparent background)
Output: td/assets/lakota/symbols/<name>.png   clean white symbol, transparent, cropped
        td/assets/lakota/atlas.png            all symbols, one square cell each (4 × 3 grid)
        td/assets/lakota/tilings.json         optimal tiling per symbol (upright + free)
        td/assets/lakota/tilings.tsv          the same, flat, for a TouchDesigner Table DAT
        td/assets/lakota/preview_tilings.png  contact sheet of every tiling

Cleaning: the symbol's coverage is alpha × darkness; specks (components smaller
than 0.5 % of the largest) are dropped, the anti-aliased edge is kept, the result
is cropped and written as pure white with that coverage as alpha.

Tiling: every symbol sits centred in a square "canvas" (side = 1 unit).  Its
mask is dilated by half the clearance GAP, and the set of relative offsets at
which two copies would overlap is computed with an FFT correlation.  A lattice
v1 = (a, 0), v2 = (s, b) is valid if no non-zero lattice vector falls in that
set; we search a, the row shift s ∈ {0, ¼, ⅓, ½, ⅔, ¾}·a and the smallest b,
and keep the densest.  "upright" uses one copy per cell (every symbol keeps its
orientation); "free" may add a second copy rotated 180° or mirrored, placed
where it interlocks best.  Results are stored y-up in canvas units for the
shader (td/delyrism/glsl_tiles_pixel.glsl).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.signal import fftconvolve

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "lakota"))
RAW = os.path.join(ROOT, "raw")

# delyrism lakota preset symbol → image (EARTH has two drawings; both are kept)
SYMBOLS = [
    ("BAG", "bag"), ("CLOUDS", "clouds"), ("DRAGONFLY", "dragonfly"), ("EARTH", "earth_square"),
    ("FEATHER", "feather"), ("HORSE TRACK", "horse_track"), ("HOUSE", "house"),
    ("LIGHTNING", "lightning"), ("STAR", "star"), ("THUNDER", "thunder"), ("EARTH", "earth_triangle"),
]
CELL = 512            # atlas cell size (px)
ATLAS_COLS = 4
CANVAS_PAD = 0.06     # symbol occupies (1 - 2·pad) of its canvas side
GAP = 0.07            # clearance between copies, fraction of the canvas side
SEARCH = 128          # canvas size (px) for the tiling search
S_FRACS = [0.0, 0.25, 1 / 3, 0.5, 2 / 3, 0.75]
TRANSFORMS = ["rot180", "flipx", "flipy"]


# ------------------------------------------------------------------ cleaning
def clean(path: str) -> np.ndarray:
    """→ float32 coverage in [0, 1], cropped tight."""
    a = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    cov = a[..., 3] * (1.0 - a[..., :3].mean(-1))          # opaque AND dark
    solid = cov > 0.5
    lab, n = ndimage.label(solid)
    if n > 1:
        sizes = ndimage.sum(solid, lab, range(1, n + 1))
        keep = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.005 * sizes.max()))
        cov *= ndimage.binary_dilation(keep, iterations=2)  # keep the AA rim of kept parts
    ys, xs = np.nonzero(cov > 0.02)
    return cov[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def to_canvas(cov: np.ndarray, size: int) -> np.ndarray:
    """Centre the symbol in a square canvas of `size` px (symbol fills 1 − 2·pad)."""
    h, w = cov.shape
    inner = size * (1 - 2 * CANVAS_PAD)
    k = inner / max(h, w)
    nh, nw = max(1, round(h * k)), max(1, round(w * k))
    im = Image.fromarray((cov * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    out = np.zeros((size, size), np.float32)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = np.asarray(im, np.float32) / 255.0
    return out


def transform(m: np.ndarray, t: str) -> np.ndarray:
    return {"identity": m, "rot180": m[::-1, ::-1], "flipx": m[:, ::-1], "flipy": m[::-1, :]}[t]


# ------------------------------------------------------------------ geometry
def forbidden(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """F[c+dy, c+dx] true ⇔ b placed at offset (dx, dy) relative to a overlaps a.
    a, b: same-size square boolean canvases; c = size − 1."""
    corr = fftconvolve(a.astype(np.float32), b[::-1, ::-1].astype(np.float32), mode="full")
    return corr > 0.5


def lookup(F: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    c = (F.shape[0] - 1) // 2
    dx, dy = np.broadcast_arrays(np.asarray(dx), np.asarray(dy))
    ix, iy = c + dx, c + dy
    ok = (ix >= 0) & (ix < F.shape[1]) & (iy >= 0) & (iy < F.shape[0])
    out = np.zeros(dx.shape, bool)
    out[ok] = F[iy[ok], ix[ok]]
    return out


def best_lattice(F: np.ndarray, area: float, ncopies: int, size: int):
    """Densest lattice v1=(a,0), v2=(s,b) with no non-zero lattice vector in F."""
    c = (F.shape[0] - 1) // 2
    ij = [(i, j) for i in range(-3, 4) for j in range(-3, 4) if (i, j) != (0, 0)]
    I = np.array([p[0] for p in ij])[:, None]
    J = np.array([p[1] for p in ij])[:, None]
    bs = np.arange(1, F.shape[0] + 1)[None, :]
    a_min = next((a for a in range(1, F.shape[1]) if not lookup(F, np.array([a, 2 * a, 3 * a]), np.zeros(3, int)).any()), None)
    if a_min is None:
        return None
    best = None
    for a in range(a_min, int(a_min * 1.7) + 2):
        if lookup(F, np.array([a, 2 * a, 3 * a]), np.zeros(3, int)).any():
            continue
        for frac in S_FRACS:
            s = int(round(frac * a))
            bad = lookup(F, I * a + J * s, J * bs)            # (pairs, b)
            ok = ~bad.any(axis=0)
            if not ok.any():
                continue
            b = int(bs[0, np.argmax(ok)])
            dens = ncopies * area / (a * b)
            if best is None or dens > best["density"] + 1e-9:
                best = dict(a=a, s=s, b=b, frac=frac, density=dens)
    return best


def solve(canvas: np.ndarray) -> dict:
    size = canvas.shape[0]
    solid = canvas > 0.5
    r = max(1, int(round(GAP * size / 2)))
    dil = ndimage.binary_dilation(solid, iterations=r)
    area = float(solid.sum())

    # upright: one copy, translations only
    up = best_lattice(forbidden(dil, dil), area, 1, size)
    upright = dict(kind="p1", copies=[dict(offset=[0, 0], transform="identity")], **up)

    # free: + a transformed second copy, placed in contact along one of 32 directions
    free = dict(upright)
    for t in TRANSFORMS:
        tb = transform(dil, t)
        Fab = forbidden(dil, tb)
        for k in range(32):
            th = 2 * math.pi * k / 32
            for rr in range(1, 2 * size):
                dx, dy = int(round(rr * math.cos(th))), int(round(rr * math.sin(th)))
                if not lookup(Fab, np.array([dx]), np.array([dy]))[0]:
                    break
            # union of both copies on a canvas large enough to hold them
            big = 3 * size
            U = np.zeros((big, big), bool)
            o = size
            U[o:o + size, o:o + size] |= dil
            U[o + dy:o + dy + size, o + dx:o + dx + size] |= tb
            res = best_lattice(forbidden(U, U), area, 2, big)
            if res and res["density"] > free["density"] * 1.02:      # must beat upright clearly
                free = dict(kind={"rot180": "p2", "flipx": "pm", "flipy": "pg"}[t],
                            copies=[dict(offset=[0, 0], transform="identity"),
                                    dict(offset=[dx, dy], transform=t)], **res)
    return dict(upright=upright, free=free, size=size)


def to_units(sol: dict, size: int) -> dict:
    """pixels (y down) → canvas units (y up)."""
    out = {}
    for key in ("upright", "free"):
        p = sol[key]
        out[key] = dict(kind=p["kind"], density=round(p["density"], 4),
                        v1=[p["a"] / size, 0.0], v2=[p["s"] / size, -p["b"] / size],
                        copies=[dict(offset=[c["offset"][0] / size, -c["offset"][1] / size],
                                     transform=c["transform"]) for c in p["copies"]])
    return out


# ------------------------------------------------------------------ preview
def render(motif: np.ndarray, pat: dict, cells: int = 6, px: int = 64) -> np.ndarray:
    """Render the tiling (as the shader does) into an image, for checking."""
    W, H = cells * px, int(cells * px * 0.75)
    out = np.zeros((H, W), np.float32)
    m = Image.fromarray((motif * 255).astype(np.uint8)).resize((px, px), Image.LANCZOS)
    tiles = {t: np.asarray(m, np.float32)[::1] / 255.0 for t in ("identity",)}
    for t in TRANSFORMS:
        tiles[t] = transform(tiles["identity"], t)
    v1 = np.array(pat["v1"]) * px; v2 = np.array(pat["v2"]) * px
    for i in range(-cells * 3, cells * 3):
        for j in range(-cells * 3, cells * 3):
            base = i * v1 + j * v2
            for c in pat["copies"]:
                cx = base[0] + c["offset"][0] * px + W / 2
                cy = -(base[1] + c["offset"][1] * px) + H / 2          # y-up → image rows
                x0, y0 = int(round(cx - px / 2)), int(round(cy - px / 2))
                if x0 >= W or y0 >= H or x0 + px <= 0 or y0 + px <= 0:
                    continue
                tile = tiles[c["transform"]]
                xs0, ys0 = max(0, -x0), max(0, -y0)
                xs1, ys1 = min(px, W - x0), min(px, H - y0)
                region = out[y0 + ys0:y0 + ys1, x0 + xs0:x0 + xs1]
                np.maximum(region, tile[ys0:ys1, xs0:xs1], out=region)
    return out


def main() -> None:
    os.makedirs(os.path.join(ROOT, "symbols"), exist_ok=True)
    rows = (len(SYMBOLS) + ATLAS_COLS - 1) // ATLAS_COLS
    atlas = np.zeros((rows * CELL, ATLAS_COLS * CELL), np.float32)
    meta, previews = [], []
    for idx, (sym, name) in enumerate(SYMBOLS):
        cov = clean(os.path.join(RAW, f"{name}.png"))
        white = np.zeros(cov.shape + (4,), np.uint8)
        white[..., :3] = 255
        white[..., 3] = (cov * 255).astype(np.uint8)
        Image.fromarray(white).save(os.path.join(ROOT, "symbols", f"{name}.png"))

        canvas = to_canvas(cov, CELL)
        r, c = divmod(idx, ATLAS_COLS)
        atlas[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = canvas

        sol = solve(to_canvas(cov, SEARCH))
        units = to_units(sol, SEARCH)
        meta.append(dict(index=idx, symbol=sym, name=name, atlas_cell=[c, r], **units))
        up, fr = units["upright"], units["free"]
        print(f"{name:15s} upright {up['kind']} density {up['density']:.3f}   "
              f"free {fr['kind']} density {fr['density']:.3f}", flush=True)
        previews.append((name, render(canvas, up), render(canvas, fr), up, fr))

    # atlas (white, alpha = coverage); row 0 at the top of the image
    rgba = np.zeros(atlas.shape + (4,), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = (atlas * 255).astype(np.uint8)
    Image.fromarray(rgba).save(os.path.join(ROOT, "atlas.png"))
    json.dump(dict(cell=CELL, cols=ATLAS_COLS, rows=rows, canvas_pad=CANVAS_PAD, gap=GAP,
                   default_earth="earth_square", symbols=meta),
              open(os.path.join(ROOT, "tilings.json"), "w"), indent=1)

    # flat table for TouchDesigner: one row per (symbol image, mode)
    tcode = {"identity": 0, "rot180": 1, "flipx": 2, "flipy": 3}
    with open(os.path.join(ROOT, "tilings.tsv"), "w") as f:
        f.write("index\tsymbol\tname\tmode\tcellx\tcelly\tv1x\tv1y\tv2x\tv2y\tncopies\toff1x\toff1y\tt1\tdensity\n")
        for m in meta:
            for mode in ("upright", "free"):
                p = m[mode]
                c1 = p["copies"][1] if len(p["copies"]) > 1 else dict(offset=[0, 0], transform="identity")
                f.write("\t".join(str(x) for x in [
                    m["index"], m["symbol"], m["name"], mode, m["atlas_cell"][0], m["atlas_cell"][1],
                    *[round(v, 6) for v in p["v1"]], *[round(v, 6) for v in p["v2"]], len(p["copies"]),
                    *[round(v, 6) for v in c1["offset"]], tcode[c1["transform"]], p["density"]]) + "\n")

    # contact sheet
    pw, ph = previews[0][1].shape[1], previews[0][1].shape[0]
    sheet = Image.new("RGB", (2 * pw + 30, len(previews) * (ph + 22)), (20, 20, 24))
    dr = ImageDraw.Draw(sheet)
    for k, (name, up, fr, pu, pf) in enumerate(previews):
        y = k * (ph + 22)
        for j, (img, p) in enumerate(((up, pu), (fr, pf))):
            sheet.paste(Image.fromarray((img * 255).astype(np.uint8)).convert("RGB"), (j * (pw + 30), y + 18))
            dr.text((j * (pw + 30) + 2, y + 3), f"{name} · {'upright' if j == 0 else 'free'} {p['kind']} "
                    f"· density {p['density']:.2f}", fill=(220, 220, 220))
    sheet.save(os.path.join(ROOT, "preview_tilings.png"))


if __name__ == "__main__":
    sys.exit(main())
