"""
gui.py — a local page to draw and watch the graphemes
=====================================================

Builds the grapheme page (gui.html) with the current basis and a folder of
drawing images, writes it to a single self-contained HTML file and opens it
in the browser.  On the page:

  draw one        a drawing pad; strokes are cut where the six graphemes fit
                  best and the grapheme weights update as you draw
  the graphemes   each one drawn as an animated stroke, in its 8 directions
  mix them        blend the six into a fragment
  your drawings   each image rebuilt fragment by fragment in grapheme colours
  lifts           the three pen-lift graphemes

    python -m delyrism.graphemes gui                       # the bundled sample drawings
    python -m delyrism.graphemes gui --drawings my_symbols/ --out graphemes.html
"""
from __future__ import annotations

import json
import os
import webbrowser
from typing import List, Optional, Sequence

import numpy as np

from . import basis as G
from . import bridge as Br
from . import glyphs

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _r(a, n=4):
    return [round(float(x), n) for x in a]


def _preset(Wp: np.ndarray, P: np.ndarray):
    """A shape's graphemes relative to its straight part (which points east)."""
    P = Br._arc_resample(P, 200)
    g, u = np.linspace(0, 1, G.L + 1), np.linspace(0, 1, len(P))
    Q = np.stack([np.interp(g, u, P[:, i]) for i in range(2)], 1)
    D = np.diff(Q, axis=0)
    z = D[:, 0] + 1j * D[:, 1]
    z /= np.abs(z).sum()
    c = Wp @ z
    rel = c / c[0]
    return [[round(float(x.real), 3), round(float(x.imag), 3)] for x in rel[1:]]


def build_data(B: Optional[G.GraphemeBasis] = None, drawings: Sequence[str] = ()) -> dict:
    """The page's data: the basis (six graphemes, lifts, full matrix), presets
    for the mixer, and each drawing image as 60 Hz pen-down strokes."""
    B = B or G.default_basis(6)
    W = B.W.copy()
    W[B.planar[0]] *= np.sign(W[B.planar[0]].sum())          # the straight grapheme points forward
    Wp = W[B.planar]
    data = {"wavelet": B.wavelet, "planar": [_r(a) for a in Wp], "lift": [_r(a) for a in W[B.lift]],
            "energy": _r(B.energy[B.planar], 3), "W": [_r(a) for a in W], "kept": [int(i) for i in B.planar],
            "bands": [], "drawings": [], "presets": {}}
    data["presets"] = {
        "arc": _preset(Wp, Br._arc(0, 0, 1, -np.pi / 2, 0, 80)),
        "S-curve": _preset(Wp, np.vstack([Br._arc(0, 0.5, 0.5, -np.pi / 2, np.pi / 2, 40),
                                          Br._arc(0, 1.5, 0.5, -np.pi / 2, -3 * np.pi / 2, 40)[1:]])),
        "corner": _preset(Wp, Br._poly((0, 0), (1, 0), (1, 0.8), n=40)),
        "hook": _preset(Wp, np.vstack([Br._poly((0, 0), (0.8, 0), n=40), Br._arc(0.8, 0.2, 0.2, -np.pi / 2, np.pi / 2, 30)[1:]])),
        "wave": _preset(Wp, np.c_[np.linspace(0, 1, 120), 0.12 * np.sin(2 * np.pi * 2 * np.linspace(0, 1, 120))]),
        "straight": [[0, 0]] * (len(B.planar) - 1),
    }
    for path in drawings:
        d = glyphs.image_drawing(path)
        if d is None:
            print(f"[graphemes] skipped {os.path.basename(path)}: no line drawing found")
            continue
        X, pen = G.motion(d)
        edges = np.flatnonzero(np.diff(pen.astype(int))) + 1
        runs = np.split(np.arange(len(pen)), edges)
        strokes = [[_r(X[i, :2]) for i in r] for r in runs if pen[r[0]] and len(r) > 1]
        data["drawings"].append({"name": os.path.splitext(os.path.basename(path))[0].replace("_", " "),
                                 "fragments": [], "ghost": [_r(p, 3) for p in X[pen][::2, :2]], "strokes": strokes})
    return data


def image_files(folder: str) -> List[str]:
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT))


def build_page(out: str, drawings_dir: Optional[str] = None, B: Optional[G.GraphemeBasis] = None) -> str:
    """Write the self-contained page to `out` and return its path."""
    files = image_files(drawings_dir or SAMPLES)
    with open(os.path.join(HERE, "gui.html"), encoding="utf-8") as f:
        template = f.read()
    html = template.replace("__DATA__", json.dumps(build_data(B, files), separators=(",", ":")))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return os.path.abspath(out)


def run(out: Optional[str] = None, drawings_dir: Optional[str] = None, open_browser: bool = True) -> str:
    out = out or os.path.join(os.path.expanduser("~"), ".cache", "delyrism", "graphemes.html")
    path = build_page(out, drawings_dir)
    print(f"[graphemes] page written to {path}")
    if open_browser:
        webbrowser.open("file:///" + path.replace(os.sep, "/"))
    return path
