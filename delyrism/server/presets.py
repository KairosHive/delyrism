"""The JSON presets under delyrism/structures/ for the frontend, with their drawings.

A preset is `<name>.json`: {SYMBOL: [descriptors]}.  Its drawings, made once with the website's
generator (scripts/draw_structures.py), are `art/<name>.json`: per symbol, the same record the
Generate tab keeps for a set made on the site (glyph, svg, lines_svg, photo, geometry, seed,
model, image_prompt).
"""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List

# the symbol structures ship inside the package (delyrism/structures/)
_PRESETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "structures")
ART_DIR = os.path.join(_PRESETS_DIR, "art")


def _safe(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "-_")


def list_presets() -> List[str]:
    if not os.path.isdir(_PRESETS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(_PRESETS_DIR)
        if f.endswith(".json")
    )


def load_preset(name: str) -> Dict[str, List[str]]:
    path = os.path.join(_PRESETS_DIR, f"{_safe(name)}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"preset '{name}' not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def art_path(name: str) -> str:
    return os.path.join(ART_DIR, f"{_safe(name)}.json")


def load_art(name: str) -> Dict[str, Dict[str, Any]]:
    """The preset's drawings by symbol name; {} when none were made."""
    path = art_path(name)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
