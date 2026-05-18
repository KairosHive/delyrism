"""Bundle the JSON presets living under delyrism/structures/ for the frontend."""
from __future__ import annotations
import json
import os
from typing import Dict, List

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PRESETS_DIR = os.path.join(_REPO_ROOT, "delyrism", "structures")


def list_presets() -> List[str]:
    if not os.path.isdir(_PRESETS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(_PRESETS_DIR)
        if f.endswith(".json")
    )


def load_preset(name: str) -> Dict[str, List[str]]:
    safe = "".join(c for c in name if c.isalnum() or c in "-_")
    path = os.path.join(_PRESETS_DIR, f"{safe}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"preset '{name}' not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
