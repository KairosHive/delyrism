"""
python -m symbol_vectorizer vectorize <drawings_dir> -o <out_dir>
python -m symbol_vectorizer space <out_dir> <descriptors.json|csv|dir> -o space.json
python -m symbol_vectorizer all <drawings_dir> <descriptors> -o <out_dir>
"""
from __future__ import annotations

import argparse
import os
import sys

from .core import Params, vectorize_folder
from .space import build_space, load_descriptors


def _params(a) -> Params:
    p = Params()
    s, t = p.segment, p.trace
    s.work_size, s.chroma_hi, s.dark_hi = a.work_size, a.chroma_hi, a.dark_hi
    s.chroma_lo, s.dark_lo = a.chroma_hi * 0.48, a.dark_hi * 0.55
    s.max_colors, s.merge_delta_e, s.margin = a.max_colors, a.merge_de, a.margin
    t.fit_error, t.fill_factor = a.fit_error, a.fill_factor
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="symbol_vectorizer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def seg_args(q):
        q.add_argument("--work-size", type=int, default=2000, help="longest side of the working image (px)")
        q.add_argument("--chroma-hi", type=float, default=0.25, help="coloured-ink threshold (Lab chroma/100)")
        q.add_argument("--dark-hi", type=float, default=0.40, help="dark-ink threshold (1 - L/100)")
        q.add_argument("--max-colors", type=int, default=5)
        q.add_argument("--merge-de", type=float, default=22.0, help="merge colours closer than this (ΔE)")
        q.add_argument("--margin", type=float, default=0.06)
        q.add_argument("--fit-error", type=float, default=0.8, help="Bézier fit tolerance (px)")
        q.add_argument("--fill-factor", type=float, default=1.8, help="thicker than this x pen width = fill")

    v = sub.add_parser("vectorize", help="drawings folder -> vector symbols")
    v.add_argument("src")
    v.add_argument("-o", "--out", default="symbols_out")
    seg_args(v)

    s = sub.add_parser("space", help="vector symbols + descriptors -> space JSON")
    s.add_argument("symbols_dir")
    s.add_argument("descriptors", help="JSON, CSV/TSV, or a folder of <symbol>.txt files")
    s.add_argument("-o", "--out", default="space.json")
    s.add_argument("--name")
    s.add_argument("--no-geometry", action="store_true", help="files and descriptors only")
    s.add_argument("--require-descriptors", action="store_true", help="leave out symbols without descriptors")

    al = sub.add_parser("all", help="vectorize, then build the space JSON")
    al.add_argument("src")
    al.add_argument("descriptors", nargs="?", help="default: <symbol>.txt sidecars in src")
    al.add_argument("-o", "--out", default="symbols_out")
    al.add_argument("--name")
    seg_args(al)

    a = ap.parse_args(argv)
    if a.cmd in ("vectorize", "all"):
        print(f"vectorizing {a.src} -> {a.out}")
        recs = vectorize_folder(a.src, a.out, _params(a))
        print(f"{len(recs)} symbols, preview: {os.path.join(a.out, 'preview.png')}")
    if a.cmd in ("space", "all"):
        sym_dir = a.symbols_dir if a.cmd == "space" else a.out
        dsrc = a.descriptors or a.src
        out = a.out if a.cmd == "space" else os.path.join(a.out, "space.json")
        sp = build_space(sym_dir, load_descriptors(dsrc), out, a.name,
                         include_geometry=not getattr(a, "no_geometry", False),
                         require_descriptors=getattr(a, "require_descriptors", False))
        print(f"space '{sp['name']}': {len(sp['symbols'])} symbols -> {out}")
        for k, v in sp["missing"].items():
            if v:
                print(f"  missing {k}: {', '.join(v)}")


if __name__ == "__main__":
    sys.exit(main())
