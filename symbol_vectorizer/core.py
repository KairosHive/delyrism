"""
core.py — one drawing in, one vector symbol out; or a whole folder.

    from symbol_vectorizer import vectorize_folder
    vectorize_folder("new-symbols", "symbols_out")

Per symbol, in <out>/<name>/:
  <name>.svg         filled outlines in the drawing's own colours
  <name>_lines.svg   centre-line strokes (pen paths) plus fills
  <name>.png         the cropped drawing on a transparent background
  <name>.json        geometry for the live space (normalised frame)
and for the folder:
  <out>/symbols.json   every symbol's record
  <out>/preview.png    source crop | outlines | centre lines, one row per symbol
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .segment import Segmentation, SegmentParams, segment
from .trace import Stroke, centerlines, outline_shapes, path_beziers, sample_segs, svg_d

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
SCHEMA = "delyrism.symbol/1"


@dataclass
class TraceParams:
    fit_error: float = 0.8         # max distance of a Bézier from the traced contour, px
    outline_sigma: float = 1.2     # contour smoothing before fitting, px
    fill_factor: float = 1.8       # regions thicker than this x pen width are fills (90th pct)
    json_step: float = 0.01        # stroke point spacing in the JSON, normalised units
    json_decimals: int = 4


@dataclass
class Params:
    segment: SegmentParams = field(default_factory=SegmentParams)
    trace: TraceParams = field(default_factory=TraceParams)


def _hex(c) -> str:
    return "#{:02x}{:02x}{:02x}".format(*c)


class _Frame:
    """Crop px (y down) <-> normalised frame: ink box centred on 0, longest side
    spans [-1, 1], y up."""

    def __init__(self, ink: np.ndarray):
        ys, xs = np.nonzero(ink)
        self.cx = (xs.min() + xs.max() + 1) / 2
        self.cy = (ys.min() + ys.max() + 1) / 2
        self.s = 2.0 / max(np.ptp(xs) + 1, np.ptp(ys) + 1)

    def __call__(self, P: np.ndarray) -> np.ndarray:
        P = np.atleast_2d(P)
        return np.c_[(P[:, 0] - self.cx) * self.s, -(P[:, 1] - self.cy) * self.s]


def _resample_by(P: np.ndarray, step: float) -> np.ndarray:
    if len(P) < 2:
        return P
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.linspace(0, d[-1], max(2, int(np.ceil(d[-1] / step)) + 1))
    return np.stack([np.interp(s, d, P[:, k]) for k in range(2)], axis=1)


def _layers(seg: Segmentation, tp: TraceParams):
    """Outline layers bottom to top.  The dominant colour fills the whole ink
    silhouette, so the other colours sit on top of it with no seams."""
    out = []
    for c in range(len(seg.palette)):
        mask = seg.ink if c == 0 else seg.labels == c
        shapes = outline_shapes(mask, tp.fit_error, tp.outline_sigma)
        if shapes:
            out.append((c, shapes))
    return out


def _svg(width: int, height: int, body: List[str], title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}">\n<title>{title}</title>\n' + "\n".join(body) + "\n</svg>\n")


def _outline_svg(seg, layers, name) -> str:
    h, w = seg.ink.shape
    body = []
    for c, shapes in layers:
        d = " ".join(" ".join(svg_d(r, True) for r in shape) for shape in shapes)
        body.append(f'<path fill="{_hex(seg.palette[c])}" fill-rule="evenodd" d="{d}"/>')
    return _svg(w, h, body, name)


def _lines_svg(seg, strokes: List[Stroke], fill_shapes, tp: TraceParams, name) -> str:
    h, w = seg.ink.shape
    body = []
    for c, shapes in fill_shapes:
        d = " ".join(" ".join(svg_d(r, True) for r in shape) for shape in shapes)
        body.append(f'<path fill="{_hex(seg.palette[c])}" fill-rule="evenodd" d="{d}"/>')
    body.append('<g fill="none" stroke-linecap="round" stroke-linejoin="round">')
    for s in strokes:
        col = _hex(seg.palette[s.color])
        if s.kind == "dot":
            x, y = s.points[0]
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{s.width / 2:.2f}" fill="{col}" stroke="none"/>')
            continue
        d = svg_d(path_beziers(s.points, s.closed, tp.fit_error), s.closed)
        body.append(f'<path stroke="{col}" stroke-width="{s.width:.2f}" d="{d}"/>')
    body.append("</g>")
    return _svg(w, h, body, name + " (centre lines)")


def _rgba(seg: Segmentation) -> np.ndarray:
    rgb = (np.clip(seg.rgb, 0, 1) * 255).astype(np.uint8)
    a = (np.clip(seg.alpha, 0, 1) * 255).astype(np.uint8)
    return np.dstack([rgb, a])


def _render_preview(seg, layers, strokes, fill_shapes, size: int = 360) -> np.ndarray:
    """Three panels rasterised from the vector data itself (checks the geometry)."""
    h, w = seg.ink.shape
    sc = size / max(h, w)
    W, H = max(1, int(w * sc)), max(1, int(h * sc))

    def panel():
        return np.full((size, size, 3), 255, np.uint8)

    def off(P):
        return ((P * sc) + [(size - W) / 2, (size - H) / 2])

    a = panel()
    src = cv2.resize((seg.rgb * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA)
    a[(size - H) // 2:(size - H) // 2 + H, (size - W) // 2:(size - W) // 2 + W] = src

    def fill(img, c, shapes):
        col = tuple(int(v) for v in seg.palette[c])
        for shape in shapes:
            polys = [np.round(off(sample_segs(r)) * 16).astype(np.int32) for r in shape]
            cv2.fillPoly(img, polys, col, lineType=cv2.LINE_AA, shift=4)

    b = panel()
    for c, shapes in layers:
        fill(b, c, shapes)
    d = panel()
    for c, shapes in fill_shapes:
        fill(d, c, shapes)
    for s in strokes:
        col = tuple(int(v) for v in seg.palette[s.color])
        t = max(1, int(round(s.width * sc)))
        if s.kind == "dot":
            x, y = off(s.points[0][None])[0]
            cv2.circle(d, (int(x), int(y)), max(1, t // 2), col, -1, cv2.LINE_AA)
            continue
        P = np.round(off(s.points) * 16).astype(np.int32)
        cv2.polylines(d, [P], s.closed, col, t, cv2.LINE_AA, shift=4)
        cv2.polylines(d, [P], s.closed, (40, 40, 40), 1, cv2.LINE_AA, shift=4)   # the path itself
        x, y = off(s.points[0][None])[0]
        cv2.circle(d, (int(x), int(y)), 3, (200, 0, 0), -1, cv2.LINE_AA)       # stroke start
    return np.hstack([a, np.full((size, 4, 3), 200, np.uint8), b, np.full((size, 4, 3), 200, np.uint8), d])


def vectorize_image(path: str, out_dir: str, params: Optional[Params] = None,
                    name: Optional[str] = None) -> Dict:
    """Vectorize one drawing; returns its JSON record."""
    params = params or Params()
    tp = params.trace
    name = name or os.path.splitext(os.path.basename(path))[0]
    seg = segment(path, params.segment)
    layers = _layers(seg, tp)
    strokes, fills = centerlines(seg.labels, len(seg.palette), seg.pen_width, tp.fill_factor)
    fill_shapes = []
    for c in range(len(seg.palette)):
        m = fills & (seg.labels == c)
        if m.any():
            fill_shapes.append((c, outline_shapes(m, tp.fit_error, tp.outline_sigma)))

    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    files = {"svg": f"{name}.svg", "lines_svg": f"{name}_lines.svg", "png": f"{name}.png"}
    with open(os.path.join(d, files["svg"]), "w", encoding="utf-8") as f:
        f.write(_outline_svg(seg, layers, name))
    with open(os.path.join(d, files["lines_svg"]), "w", encoding="utf-8") as f:
        f.write(_lines_svg(seg, strokes, fill_shapes, tp, name))
    cv2.imwrite(os.path.join(d, files["png"]), cv2.cvtColor(_rgba(seg), cv2.COLOR_RGBA2BGRA))

    fr = _Frame(seg.ink)
    r = tp.json_decimals
    rnd = lambda P: np.round(P, r).tolist()
    js_strokes = []
    for s in strokes:
        P = fr(s.points)
        rec = {"kind": s.kind, "color": s.color, "width": round(s.width * fr.s, r), "closed": bool(s.closed)}
        if s.kind == "line":
            P = _resample_by(P, tp.json_step)
            rec["length"] = round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), r)
        rec["points"] = rnd(P)
        js_strokes.append(rec)
    js_fills = []
    for c, shapes in fill_shapes:
        for shape in shapes:
            js_fills.append({"color": c, "rings": [rnd(_resample_by(fr(sample_segs(ring)), tp.json_step)) for ring in shape]})
    ys, xs = np.nonzero(seg.ink)
    x0, y0, x1, y1 = seg.crop_box
    record = {
        "schema": SCHEMA,
        "id": name,
        "source": os.path.abspath(path),
        "files": files,
        "frame": "ink box centred on (0, 0), longest side spans [-1, 1], y up",
        "aspect": round(float((np.ptp(xs) + 1) / (np.ptp(ys) + 1)), 4),
        "pen_width": round(seg.pen_width * fr.s, r),
        "palette": [{"hex": _hex(c), "rgb": list(c), "share": round(s, 4)} for c, s in zip(seg.palette, seg.shares)],
        "strokes": js_strokes,
        "fills": js_fills,
        "stats": {
            "n_strokes": sum(s["kind"] == "line" for s in js_strokes),
            "n_dots": sum(s["kind"] == "dot" for s in js_strokes),
            "n_fills": len(js_fills),
            "ink_length": round(sum(s.get("length", 0) for s in js_strokes), r),
            "crop_px": [x1 - x0, y1 - y0],
            "source_px": list(seg.source_size),
            "crop_box_source_px": [round(v / seg.scale) for v in seg.crop_box],
        },
        "notes": seg.notes,
    }
    with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, separators=(",", ":"))
    record["_preview"] = _render_preview(seg, layers, strokes, fill_shapes)
    return record


def vectorize_folder(src: str, out_dir: str, params: Optional[Params] = None, verbose: bool = True) -> List[Dict]:
    """Vectorize every image in `src`; writes symbols.json and preview.png in `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith(IMAGE_EXT))
    records, rows, failed = [], [], []
    for p in paths:
        try:
            rec = vectorize_image(p, out_dir, params)
        except Exception as e:                     # one bad photo must not stop the batch
            failed.append({"source": p, "error": str(e)})
            if verbose:
                print(f"  FAILED {os.path.basename(p)}: {e}")
            continue
        rows.append(rec.pop("_preview"))
        records.append(rec)
        if verbose:
            st = rec["stats"]
            pal = " ".join(c["hex"] for c in rec["palette"])
            print(f"  {rec['id']:<24} {st['n_strokes']:>3} strokes {st['n_dots']:>2} dots "
                  f"{st['n_fills']:>2} fills  palette {pal}")
    with open(os.path.join(out_dir, "symbols.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": "delyrism.symbols/1", "symbols": records, "failed": failed}, f, separators=(",", ":"))
    if rows:
        sheet = np.vstack([np.vstack([r, np.full((4, r.shape[1], 3), 200, np.uint8)]) for r in rows])
        cv2.imwrite(os.path.join(out_dir, "preview.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return records
