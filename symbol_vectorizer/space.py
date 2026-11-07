"""
space.py — vector symbols + descriptors -> the space JSON for live Delyrism.

Descriptors may come as
  JSON   {"DESTINY": ["fate", "path", ...], ...}           (the structures/*.json format)
         {"DESTINY": "fate, path, ..."}
         [{"symbol": "DESTINY", "descriptors": [...]}, ...]
  CSV    symbol,descriptor         one row per pair, or
         symbol,"fate; path; ..."  one row per symbol
  TXT    sidecar files next to the drawings: destiny.txt, one descriptor per
         line or comma-separated
Symbol names match image names case-insensitively, ignoring spaces, "-" and "_".

The output keeps `structure` exactly in the {symbol: [descriptors]} form that
SymbolSpace(symbols_to_descriptors=...) takes, and adds each symbol's geometry
and file paths (relative to the space JSON) under `symbols`.
"""
from __future__ import annotations

import csv
import json
import os
import re
from typing import Dict, List, Optional


def _key(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s).casefold()


def _split(v) -> List[str]:
    if isinstance(v, str):
        v = re.split(r"[;,\n]", v)
    return [d.strip() for d in v if d and d.strip()]


def load_descriptors(path: str) -> Dict[str, List[str]]:
    """{symbol: [descriptors]} from a JSON, CSV or a folder of .txt sidecars."""
    if os.path.isdir(path):
        out = {}
        for f in sorted(os.listdir(path)):
            if f.lower().endswith(".txt"):
                with open(os.path.join(path, f), encoding="utf-8") as fh:
                    out[os.path.splitext(f)[0]] = _split(fh.read())
        return out
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "structure" in data:
            data = data["structure"]
        if isinstance(data, list):
            return {d["symbol"]: _split(d["descriptors"]) for d in data}
        return {k: _split(v) for k, v in data.items()}
    if ext in (".csv", ".tsv"):
        out: Dict[str, List[str]] = {}
        with open(path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.reader(fh, delimiter="\t" if ext == ".tsv" else ","))
        if rows and rows[0] and rows[0][0].strip().lower() in ("symbol", "name", "id"):
            rows = rows[1:]
        for row in rows:
            if len(row) >= 2 and row[0].strip():
                out.setdefault(row[0].strip(), []).extend(_split(",".join(row[1:]) if len(row) > 2 else row[1]))
        return out
    raise ValueError(f"unknown descriptor format: {path}")


def build_space(symbols_dir: str, descriptors: Dict[str, List[str]], out_path: str,
                name: Optional[str] = None, include_geometry: bool = True,
                require_descriptors: bool = False) -> Dict:
    """Join the vectorized symbols in `symbols_dir` (its symbols.json) with
    `descriptors`; writes and returns the space JSON."""
    with open(os.path.join(symbols_dir, "symbols.json"), encoding="utf-8") as fh:
        recs = json.load(fh)["symbols"]
    by_key = {_key(k): (k, v) for k, v in descriptors.items()}
    base = os.path.dirname(os.path.abspath(out_path))
    rel = lambda rid, f: os.path.relpath(os.path.join(os.path.abspath(symbols_dir), rid, f), base).replace(os.sep, "/")

    structure: Dict[str, List[str]] = {}
    symbols, no_desc = [], []
    for r in recs:
        label, desc = by_key.get(_key(r["id"]), (r["id"].upper(), []))
        if not desc:
            no_desc.append(r["id"])
            if require_descriptors:
                continue
        structure[label] = desc
        s = {"id": label, "image_id": r["id"], "descriptors": desc,
             "files": {k: rel(r["id"], f) for k, f in r["files"].items()},
             "aspect": r["aspect"], "pen_width": r["pen_width"], "palette": r["palette"]}
        if include_geometry:
            s["strokes"], s["fills"] = r["strokes"], r["fills"]
        symbols.append(s)
    seen = {_key(r["id"]) for r in recs}
    no_image = [k for k in descriptors if _key(k) not in seen]
    space = {
        "schema": "delyrism.space/1",
        "name": name or os.path.splitext(os.path.basename(out_path))[0],
        "frame": "strokes and fills: ink box centred on (0, 0), longest side spans [-1, 1], y up",
        "structure": structure,
        "symbols": symbols,
        "missing": {"descriptors": no_desc, "images": no_image},
    }
    os.makedirs(base, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(space, fh, ensure_ascii=False, separators=(",", ":") if include_geometry else None,
                  indent=None if include_geometry else 2)
    return space
