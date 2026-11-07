"""
segment.py — from a photo or scan of a drawing to clean ink layers.

Steps
  1. load (EXIF orientation applied), resize to a working size
  2. flatten the paper: divide by a median-filtered background estimate, so
     shading, vignetting and paper tint go to white
  3. ink score per pixel = chroma (coloured markers) or darkness (black ink);
     hysteresis keeps faint stroke edges that touch confident ink and drops
     bleed-through, ruled lines and printed text, which stay grey and faint
  4. clean: drop specks, drop anything touching the photo border, close
     pin-holes
  5. crop to the ink with a margin
  6. palette: k-means in Lab on confident ink, merge near colours, label every
     ink pixel with its nearest colour, re-absorb tiny islands of one colour
     into their neighbours
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class SegmentParams:
    work_size: int = 2000          # longest side of the working image, px
    chroma_hi: float = 0.25        # confident coloured ink (Lab chroma / 100)
    chroma_lo: float = 0.12        # faint coloured ink, kept when touching confident ink
    dark_hi: float = 0.40          # confident dark ink (1 - L/100 after flattening)
    dark_lo: float = 0.22
    min_speck: float = 1.5         # drop components smaller than this x pen width^2
    margin: float = 0.06           # crop margin, fraction of the symbol's longest side
    max_colors: int = 5
    merge_delta_e: float = 22.0    # merge palette entries closer than this (CIE76)
    min_color_share: float = 0.01  # palette entries below this share of ink are merged


@dataclass
class Segmentation:
    rgb: np.ndarray                # (h, w, 3) float32, flattened paper, cropped
    ink: np.ndarray                # (h, w) bool
    alpha: np.ndarray              # (h, w) float32 soft ink coverage
    labels: np.ndarray             # (h, w) int, -1 on paper, else palette index
    palette: List[Tuple[int, int, int]]   # sRGB 0..255, ordered by ink share
    shares: List[float]
    pen_width: float               # median stroke thickness, px (working scale)
    crop_box: Tuple[int, int, int, int]   # x0, y0, x1, y1 in working-image px
    scale: float                   # working px per source px
    source_size: Tuple[int, int]   # source w, h
    notes: List[str] = field(default_factory=list)


def load_rgb(path, work_size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """`path`: an image file, a PIL image, or an RGB / RGBA uint8 array (H, W, 3|4)."""
    if isinstance(path, np.ndarray):
        img = Image.fromarray(path.astype(np.uint8) if path.dtype != np.uint8 else path)
    elif isinstance(path, Image.Image):
        img = path
    else:
        img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    w, h = img.size
    sc = min(1.0, work_size / max(w, h))
    a = np.asarray(img).astype(np.float32) / 255
    if sc < 1:
        a = cv2.resize(a, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    return a, sc, (w, h)


def flatten_paper(img: np.ndarray) -> np.ndarray:
    """Divide by the paper's own colour, estimated with a wide median filter
    on a reduced copy (the drawing covers a small share of any window)."""
    h, w = img.shape[:2]
    f = max(1, round(max(h, w) / 250))
    small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    k = 2 * (max(small.shape[:2]) // 12) + 1
    u8 = (small * 255).astype(np.uint8)
    bg = np.stack([cv2.medianBlur(u8[..., c], min(k, 255)) for c in range(3)], -1).astype(np.float32) / 255
    bg = cv2.GaussianBlur(bg, (0, 0), 2)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    # the paper's brightest plausible white, so a grey scan of white paper becomes white
    return np.clip(img / np.maximum(bg, 1e-3), 0, 1).astype(np.float32)


def _hysteresis(score_hi: np.ndarray, score_lo: np.ndarray) -> np.ndarray:
    n, lab = cv2.connectedComponents(score_lo.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[score_hi & score_lo])] = True
    keep[0] = False
    return keep[lab]


def stroke_width(mask: np.ndarray) -> float:
    """Median stroke thickness: twice the distance to paper along the skeleton."""
    if mask.sum() < 10:
        return 1.0
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    sk = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255) > 0
    v = dt[sk]
    return float(2 * np.median(v)) if v.size else 1.0


def segment(path, p: Optional[SegmentParams] = None) -> Segmentation:
    """`path`: an image file, a PIL image, or an RGB uint8 array."""
    p = p or SegmentParams()
    notes: List[str] = []
    img, scale, src_size = load_rgb(path, p.work_size)
    flat = flatten_paper(img)
    lab = cv2.cvtColor(flat, cv2.COLOR_RGB2LAB)
    dark = (100 - lab[..., 0]) / 100
    chroma = np.hypot(lab[..., 1], lab[..., 2]) / 100

    hi = (chroma > p.chroma_hi) | (dark > p.dark_hi)
    lo = (chroma > p.chroma_lo) | (dark > p.dark_lo)
    hi = cv2.morphologyEx(hi.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    ink = _hysteresis(hi, lo)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    if ink.sum() == 0:
        src = path if isinstance(path, (str, bytes)) or hasattr(path, "__fspath__") else "image"
        raise ValueError(f"{src}: no ink found (try lowering --chroma-hi / --dark-hi)")

    # components: drop border-touching ones (photo edges, table shadows) and specks
    pw = stroke_width(ink)
    n, cl, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    H, W = ink.shape
    keep = np.zeros(n, bool)
    for k in range(1, n):
        x, y, w, h, a = st[k]
        touches = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        if touches and a < 0.5 * ink.sum():
            notes.append(f"dropped a border-touching component ({a} px)")
            continue
        keep[k] = a >= p.min_speck * pw * pw
    ink = keep[cl]
    if ink.sum() == 0:
        raise ValueError(f"{path}: only specks or border shapes found")

    # fill pin-holes smaller than a pen dot (marker streaks inside a line)
    holes = (~ink).astype(np.uint8)
    n, hl, hs, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    small = np.zeros(n, bool)
    small[1:] = hs[1:, cv2.CC_STAT_AREA] < 0.6 * pw * pw
    ink |= small[hl]

    # crop
    ys, xs = np.nonzero(ink)
    side = max(np.ptp(xs), np.ptp(ys)) + 1
    m = int(round(p.margin * side))
    x0, y0 = max(0, xs.min() - m), max(0, ys.min() - m)
    x1, y1 = min(W, xs.max() + 1 + m), min(H, ys.max() + 1 + m)
    flat_c, ink_c = flat[y0:y1, x0:x1], ink[y0:y1, x0:x1]
    sc_hi, lo_c = np.maximum(chroma, dark)[y0:y1, x0:x1], lo[y0:y1, x0:x1]
    alpha = np.where(ink_c, np.clip(sc_hi / max(p.chroma_hi, 1e-3), 0, 1), 0).astype(np.float32)
    alpha = np.where(ink_c, np.maximum(alpha, 0.35), 0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)

    labels, palette, shares = _palette(flat_c, ink_c, pw, p, notes)
    return Segmentation(flat_c, ink_c, alpha, labels, palette, shares, pw,
                        (int(x0), int(y0), int(x1), int(y1)), scale, src_size, notes)


def _cdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Ink-colour distance between Lab rows of A and B, shape (len(A), len(B)).
    The same marker varies in lightness and saturation (pressure, overlaps,
    anti-aliased edges) and keeps its hue, so for chromatic pairs lightness and
    chroma count less than hue; near-grey pairs use plain ΔE."""
    A, B = np.atleast_2d(A)[:, None, :], np.atleast_2d(B)[None, :, :]
    dL = A[..., 0] - B[..., 0]
    Ca, Cb = np.hypot(A[..., 1], A[..., 2]), np.hypot(B[..., 1], B[..., 2])
    ha, hb = np.arctan2(A[..., 2], A[..., 1]), np.arctan2(B[..., 2], B[..., 1])
    dh = np.abs((ha - hb + np.pi) % (2 * np.pi) - np.pi)
    chromatic = np.hypot(0.35 * dL, np.hypot(0.5 * (Ca - Cb), dh * np.minimum(Ca, Cb)))
    plain = np.sqrt(((A - B) ** 2).sum(-1))
    w = np.clip((np.minimum(Ca, Cb) - 8) / 10, 0, 1)        # 0 = grey pair, 1 = chromatic pair
    return w * chromatic + (1 - w) * plain


def _palette(rgb: np.ndarray, ink: np.ndarray, pw: float, p: SegmentParams, notes: List[str]):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    # confident ink: away from the anti-aliased edge
    core = cv2.erode(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if core.sum() < 50:
        core = ink
    X = lab[core].astype(np.float32)
    rng = np.random.default_rng(0)
    if len(X) > 40000:
        X = X[rng.choice(len(X), 40000, replace=False)]
    k = min(p.max_colors, max(1, len(X) // 50))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    cv2.setRNGSeed(0)
    _, lbl, C = cv2.kmeans(X, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lbl.ravel(), minlength=k).astype(float)
    C, cnt = list(C), list(cnt)

    # merge closest pairs below the colour threshold, and tiny clusters into their nearest
    while len(C) > 1:
        D = _cdist(np.array(C), np.array(C))
        np.fill_diagonal(D, np.inf)
        share = np.array(cnt) / sum(cnt)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        tiny = int(np.argmin(share))
        Ca = np.array([np.hypot(c[1], c[2]) for c in C])
        paper = [q for q, c in enumerate(C) if c[0] > 90 and Ca[q] < 10]   # paper caught inside the ink
        if paper:
            i, j = paper[0], int(np.argmin(D[paper[0]]))
            w = cnt[i] + cnt[j]
            cnt[j] = w                                                     # the ink keeps its own colour
            del C[i], cnt[i]
            continue
        if D[i, j] < p.merge_delta_e:
            pass
        elif share[tiny] < p.min_color_share:
            i, j = tiny, int(np.argmin(D[tiny]))
        else:
            break
        w = cnt[i] + cnt[j]
        C[i] = (C[i] * cnt[i] + C[j] * cnt[j]) / w
        cnt[i] = w
        del C[j], cnt[j]
    C = np.array(C, np.float32)

    # label confident ink by the nearest centre, hue first; edge pixels (mixed
    # with paper, shifted in hue by shadow) take the label of the nearest confident pixel
    inner = cv2.erode(ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    if inner.sum() < 0.2 * ink.sum():
        inner = core
    lab_in = np.full(ink.shape, -1, np.int32)
    lab_in[inner] = _cdist(lab[inner], C).argmin(1)
    _, near = cv2.distanceTransformWithLabels((~inner).astype(np.uint8), cv2.DIST_L2, 5,
                                              labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(inner)
    by_idx = np.zeros(near.max() + 1, np.int32)
    by_idx[near[ys, xs]] = lab_in[ys, xs]
    labels = np.where(ink, by_idx[near], -1).astype(np.int32)

    # tiny islands of one colour (anti-aliasing, ink overlaps) join the colour around them
    for c in range(len(C)):
        m = (labels == c).astype(np.uint8)
        n, cl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for q in range(1, n):
            if st[q, cv2.CC_STAT_AREA] >= 1.2 * pw * pw:
                continue
            isl = cl == q
            ring = (cv2.dilate(isl.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~isl & (labels >= 0)
            if ring.any():
                nb = np.bincount(labels[ring], minlength=len(C))
                nb[c] = 0
                if nb.sum():
                    labels[isl] = int(nb.argmax())

    # final pass: a label holding too little ink, or whose pixels are paper-white,
    # goes to the nearest pixel of another label
    for _ in range(len(C)):
        used = np.bincount(labels[ink], minlength=len(C)).astype(float)
        bad = None
        for c in np.argsort(used):
            if used[c] == 0 or (used > 0).sum() < 2:
                continue
            Lc = lab[labels == c]
            med = np.median(Lc, 0)
            if used[c] / used.sum() < p.min_color_share or (med[0] > 90 and np.hypot(med[1], med[2]) < 10):
                bad = int(c)
                break
        if bad is None:
            break
        keep = (labels >= 0) & (labels != bad)
        _, near = cv2.distanceTransformWithLabels((~keep).astype(np.uint8), cv2.DIST_L2, 5,
                                                  labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(keep)
        by_idx = np.zeros(near.max() + 1, np.int32)
        by_idx[near[ys, xs]] = labels[ys, xs]
        labels = np.where(labels == bad, by_idx[near], labels)

    used = np.bincount(labels[ink], minlength=len(C)).astype(float)
    order = [int(i) for i in np.argsort(-used) if used[i] > 0]
    remap = np.full(len(C), -1, np.int32)
    remap[order] = np.arange(len(order))
    labels = np.where(labels >= 0, remap[np.maximum(labels, 0)], -1)
    shares = [float(used[i] / used.sum()) for i in order]
    # report the colour as the median sRGB of its own pixels (truer than the Lab centre)
    palette = []
    for i in range(len(order)):
        px = rgb[(labels == i) & core]
        if len(px) < 20:
            px = rgb[labels == i]
        palette.append(tuple(int(round(v)) for v in np.median(px, 0) * 255))
    return labels, palette, shares
