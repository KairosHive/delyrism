"""Fetch curated free (CC/PD) ambient sounds from Wikimedia Commons for the
sound-navigation creative demo. Saves ~8s 48k mono wavs into _audio/.
Each entry: (slug, exact Commons File title) or (slug, ('search', term)).
"""
from __future__ import annotations
import json, re, urllib.parse, urllib.request
from pathlib import Path
import librosa, numpy as np, soundfile as sf

OUT = Path(__file__).resolve().parent / "_audio"; OUT.mkdir(exist_ok=True)
UA = {"User-Agent": "delyrism-research/0.1 (academic; antoine.bellemare9@gmail.com)"}
API = "https://commons.wikimedia.org/w/api.php"
PRON = re.compile(r'^[A-Za-z]{2,3}(-[a-z]{2,})?-')

CURATED = [
    ("thunder",  "File:Thunder 01.ogg"),
    ("fire",     "File:Campfire sound ambience.ogg"),
    ("ocean",    "File:Oceanwavescrushing.ogg"),
    ("birds",    "File:Birds forest.ogg"),
    ("powwow",   "File:AUDIO First Nations Pow-Wow Drums and singers stereo.ogg"),
    ("drum",     "File:Drum - Cadence A.ogg"),
    ("flute",    ("search", "flute solo instrument filetype:audio")),
    ("wind",     ("search", "wind storm ambience filetype:audio")),
    ("water",    ("search", "stream water flowing filetype:audio")),
    ("rain",     ("search", "rain thunderstorm ambience filetype:audio")),
]


def api_get(p):
    p = {**p, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(p)
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))


def url_for_title(title):
    d = api_get({"action": "query", "titles": title, "prop": "imageinfo", "iiprop": "url|mime"})
    for p in (d.get("query") or {}).get("pages", {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        return ii.get("url"), ii.get("mime")
    return None, None


def search_pick(term):
    d = api_get({"action": "query", "generator": "search", "gsrsearch": term,
                 "gsrnamespace": "6", "gsrlimit": "10", "prop": "imageinfo",
                 "iiprop": "url|mime|size"})
    cands = []
    for p in (d.get("query") or {}).get("pages", {}).values():
        ii = (p.get("imageinfo") or [{}])[0]; t = p.get("title", "")[5:]
        mime = ii.get("mime", "")
        if ("ogg" not in mime and "audio" not in mime) or t.lower().endswith(".ogv"):
            continue
        if PRON.match(t) or "alarm" in t.lower() or "midi" in mime:
            continue
        cands.append((ii.get("size", 1e12), ii.get("url"), t))
    cands.sort()
    return cands


def fetch_decode(url, suffix):
    raw = OUT / ("_tmp" + suffix)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r, open(raw, "wb") as f:
        f.write(r.read())
    y, _ = librosa.load(str(raw), sr=48000, mono=True, duration=8.0)
    raw.unlink(missing_ok=True)
    return y.astype(np.float32)


def main():
    got = []
    for slug, spec in CURATED:
        try:
            if isinstance(spec, tuple) and spec[0] == "search":
                cands = search_pick(spec[1])
            else:
                u, m = url_for_title(spec)
                cands = [(0, u, spec)] if u else []
            ok = False
            for _, url, title in cands:
                if not url:
                    continue
                try:
                    y = fetch_decode(url, Path(urllib.parse.urlparse(url).path).suffix or ".ogg")
                    if len(y) < 48000:
                        continue
                    sf.write(str(OUT / f"{slug}.wav"), y, 48000)
                    print(f"OK   {slug:8} <- {title}  ({len(y)/48000:.1f}s)")
                    got.append(slug); ok = True; break
                except Exception:
                    continue
            if not ok:
                print(f"MISS {slug:8}")
        except Exception as e:
            print(f"FAIL {slug:8} {type(e).__name__}: {str(e)[:50]}")
    print(f"\nfetched {len(got)}/{len(CURATED)}: {got}")


if __name__ == "__main__":
    main()
