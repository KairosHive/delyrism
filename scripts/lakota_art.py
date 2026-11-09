"""The Lakota structure's drawings are the Lakota dream-language symbols themselves, never generated
ones: delyrism/structures/art/lakota.json is written from the drawings in do-you-dream
(td/assets/lakota/raw, from AntoineBellemare/LakotaNeuralDreams, dream_language_symbols/no_bg_square,
black on transparent), vectorized like a generated drawing so the site shows them the same way.

    python scripts/lakota_art.py [path/to/do-you-dream]

EARTH has two drawings (square and triangle); the square one is used, as in do-you-dream's tiling.
scripts/draw_structures.py keeps these records (they carry a "source") even with --redo.
"""
from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from delyrism.server import presets, symbolgen  # noqa: E402

FILES = {"BAG": "bag", "CLOUDS": "clouds", "DRAGONFLY": "dragonfly", "EARTH": "earth_square",
         "FEATHER": "feather", "HORSE TRACK": "horse_track", "HOUSE": "house", "LIGHTNING": "lightning",
         "STAR": "star", "THUNDER": "thunder"}
SOURCE = "Lakota dream-language symbols (LakotaNeuralDreams)"


def on_white(path: str) -> bytes:
    """Black on transparent → black on white, as the vectorizer reads a drawing."""
    im = Image.open(path).convert("RGBA")
    page = Image.new("RGBA", im.size, (255, 255, 255, 255))
    page.alpha_composite(im)
    buf = io.BytesIO()
    page.convert("L").save(buf, "PNG")
    return buf.getvalue()


def main() -> int:
    dyd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(ROOT), "do-you-dream")
    raw = os.path.join(dyd, "td", "assets", "lakota", "raw")
    symbols = presets.load_preset("lakota")
    assert set(symbols) == set(FILES), sorted(set(symbols) ^ set(FILES))
    art = {}
    for name in symbols:
        ink = symbolgen.clean_ink(on_white(os.path.join(raw, FILES[name] + ".png")))
        v = symbolgen.vectorize(ink, FILES[name].replace("_", "-"))
        v["record"]["source"] = SOURCE
        art[name] = {"glyph": f"{name.title()}, a Lakota dream-language symbol", "source": SOURCE,
                     "photo": symbolgen._data_url(symbolgen._thumb(ink)), "svg": v["svg"],
                     "lines_svg": v["lines_svg"], "geometry": v["record"],
                     "at": datetime.now().isoformat(timespec="seconds")}
        st = v["record"]["stats"]
        print(f"  {name}: {st['n_fills']} fills, {st['n_strokes']} strokes")
    with open(presets.art_path("lakota"), "w", encoding="utf-8") as f:
        json.dump(art, f, separators=(",", ":"), ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
