"""Draw the symbols of the shipped structures (delyrism/structures/*.json) with the website's
generator, so every preset comes with drawings: delyrism/structures/art/<preset>.json holds, per
symbol name, {glyph, llm, svg, lines_svg, photo, geometry, seed, model, image_prompt, at} - the
same record the Generate tab keeps for a set made on the site.

    python scripts/draw_structures.py                 # every preset, symbols without art only
    python scripts/draw_structures.py elements planets # these presets
    python scripts/draw_structures.py --redo elements  # draw again, even where art exists
    python scripts/draw_structures.py --model @cf/black-forest-labs/flux-2-klein-4b

The descriptors stay as shipped: the language model only writes the glyph (how to draw the
symbol in a few basic shapes), read through the structure's name as the theme.  Credentials:
CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN (process environment, or the repo's .env).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from delyrism.server import presets, symbolgen  # noqa: E402

THEMES = {
    "elements": "the four classical elements",
    "planets": "the planets as celestial archetypes",
    "jungian": "Jungian archetypes",
    "lakota": "Lakota dream symbols",
    "chakras": "the chakra system",
    "chinese_zodiac": "the Chinese zodiac",
    "mayan": "the Mayan calendar day signs",
    "musical": "the musical modes",
    "architecture": "sacred architecture",
    "seasons_life": "the seasons of a life",
}


def one(preset: str, name: str, descriptors, model: str) -> tuple:
    theme = THEMES.get(preset, preset.replace("_", " "))
    others = [n for n in presets.load_preset(preset) if n != name]
    for attempt in range(3):
        try:
            entry = symbolgen.write_entry(name, theme, others)
            d = symbolgen.make_drawing(name, descriptors, entry["glyph"], model)
            return name, {"glyph": entry["glyph"], "llm": entry["llm"], **d}, None
        except symbolgen.GenError as e:
            err = str(e)
            if "flagged" not in err and "refused" not in err and attempt < 2:
                time.sleep(2)
                continue
            if "flagged" in err and attempt < 2:      # a different seed and glyph next time
                continue
            return name, None, err
    return name, None, "gave up"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("presets", nargs="*")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--model", default=symbolgen.DEFAULT_IMAGE_MODEL)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if symbolgen.missing():
        print("not set:", ", ".join(symbolgen.missing()))
        return 2
    names = a.presets or presets.list_presets()
    os.makedirs(presets.ART_DIR, exist_ok=True)
    for p in names:
        symbols = presets.load_preset(p)
        art = presets.load_art(p)
        if a.redo:      # drawings with a source (the Lakota symbols, scripts/lakota_art.py) stay
            art = {n: r for n, r in art.items() if r.get("source")}
        todo = [(n, ds) for n, ds in symbols.items() if n not in art]
        print(f"{p}: {len(symbols)} symbols, {len(todo)} to draw")
        if not todo:
            continue
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for name, rec, err in pool.map(lambda x: one(p, x[0], x[1], a.model), todo):
                if rec:
                    art[name] = rec
                    print(f"  {name}: {rec['geometry']['stats']['n_fills']} fills, "
                          f"{rec['geometry']['stats']['n_strokes']} strokes - {rec['glyph'][:70]}")
                else:
                    print(f"  {name}: FAILED - {err}")
                # keep what is done, in the structure's order
                ordered = {n: art[n] for n in symbols if n in art}
                with open(presets.art_path(p), "w", encoding="utf-8") as f:
                    json.dump(ordered, f, separators=(",", ":"), ensure_ascii=False)
        print(f"  {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
