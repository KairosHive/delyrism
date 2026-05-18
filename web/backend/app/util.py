"""Small conversions used across routes."""
from __future__ import annotations
from typing import Any, Dict


def to_hex(color: Any) -> str:
    """Accept hex strings, RGB tuples, or RGBA tuples and return '#rrggbb'."""
    if isinstance(color, str):
        return color
    if isinstance(color, (tuple, list)):
        r, g, b = color[:3]
        return "#{:02x}{:02x}{:02x}".format(
            int(round(255 * float(r))),
            int(round(255 * float(g))),
            int(round(255 * float(b))),
        )
    return "#888888"


def color_map_to_hex(d: Dict[str, Any]) -> Dict[str, str]:
    return {k: to_hex(v) for k, v in d.items()}
