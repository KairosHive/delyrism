"""
glyphs.py — human symbols as pen motion (delyrism.graphemes)
=======================================

Every symbol becomes a `Drawing`: its strokes, each an (n, 3) array of
x, y (y up) and t (seconds), in drawing order.  Two sources:

  omniglot   1,623 characters from 50 alphabets, each drawn by 20 people on
             a tablet, with the pen's timing (Lake, Salakhutdinov & Tenenbaum
             2015).  Real human motion.  ~45 MB, fetched once into
             ~/.cache/delyrism/glyphs/omniglot.
  fonts      symbol blocks rendered from the system's symbol fonts
             (alchemical, planetary, zodiac, trigrams, runes, ogham, ...),
             thinned to centre lines, traced into strokes, ordered top-left
             first, and timed with the two-thirds power law of human drawing
             (speed ∝ curvature^-1/3, Lacquaniti et al. 1983).

    from delyrism.graphemes import glyphs
    train = glyphs.load_omniglot("background")      # list[Drawing]
    astro = glyphs.font_drawings("planets")         # list[Drawing]
"""
from __future__ import annotations

import os
import unicodedata
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "delyrism", "glyphs")
OMNIGLOT_URL = "https://github.com/brendenlake/omniglot/raw/master/python/{split}.zip"


@dataclass
class Drawing:
    system: str                        # alphabet or symbol block
    symbol: str                        # symbol id, unique across systems
    drawer: str                        # omniglot drawer ("01".."20") or "font"
    strokes: List[np.ndarray] = field(default_factory=list)   # each (n, 3): x, y (up), t (s)
    char: str = ""                     # the Unicode character, for font symbols

    @property
    def points(self) -> np.ndarray:
        return np.concatenate([s[:, :2] for s in self.strokes]) if self.strokes else np.zeros((0, 2))


# ----------------------------------------------------------------------------
#  Omniglot
# ----------------------------------------------------------------------------

def _omniglot_dir(split: str) -> str:
    name = f"strokes_{split}"
    root = os.path.join(CACHE, "omniglot")
    path = os.path.join(root, name)
    if not os.path.isdir(path):
        os.makedirs(root, exist_ok=True)
        zpath = os.path.join(root, name + ".zip")
        if not os.path.exists(zpath):
            urllib.request.urlretrieve(OMNIGLOT_URL.format(split=name), zpath)
        zipfile.ZipFile(zpath).extractall(root)
    return path


def _read_omniglot(path: str) -> List[np.ndarray]:
    strokes, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line in ("START", "BREAK", "END") or not line:
                if cur:
                    strokes.append(np.array(cur, dtype=np.float64))
                cur = []
                continue
            x, y, t = (float(v) for v in line.split(","))
            cur.append((x, y, t / 1000.0))
    if cur:
        strokes.append(np.array(cur, dtype=np.float64))
    return strokes


def load_omniglot(split: str = "background", alphabets: Optional[Iterable[str]] = None) -> List[Drawing]:
    """All drawings of one Omniglot split: "background" (30 alphabets, 964
    characters) or "evaluation" (20 other alphabets, 659 characters)."""
    import pickle
    keep = set(alphabets) if alphabets else None
    cache = os.path.join(CACHE, "omniglot", f"{split}.pkl")
    if os.path.exists(cache):
        this = __name__

        class _Local(pickle.Unpickler):                        # our own cache; earlier versions lived in delyrism.glyphs
            def find_class(self, module, name):
                return super().find_class(this if module == "delyrism.glyphs" else module, name)

        with open(cache, "rb") as f:
            raw = _Local(f).load()
        out = [r if isinstance(r, Drawing) else Drawing(*r) for r in raw]
        if raw and isinstance(raw[0], Drawing):                # rewrite as plain tuples, independent of the class path
            with open(cache, "wb") as f:
                pickle.dump([(d.system, d.symbol, d.drawer, d.strokes, d.char) for d in out], f)
        return [d for d in out if not keep or d.system in keep]
    root = _omniglot_dir(split)
    out = []
    for alpha in sorted(os.listdir(root)):
        adir = os.path.join(root, alpha)
        if not os.path.isdir(adir):
            continue
        for ch in sorted(os.listdir(adir)):
            cdir = os.path.join(adir, ch)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if fn.endswith(".txt"):
                    strokes = _read_omniglot(os.path.join(cdir, fn))
                    if strokes:
                        out.append(Drawing(alpha, f"{alpha}/{ch}", fn[-6:-4], strokes))
    with open(cache, "wb") as f:
        pickle.dump([(d.system, d.symbol, d.drawer, d.strokes, d.char) for d in out], f)
    return [d for d in out if not keep or d.system in keep]


# ----------------------------------------------------------------------------
#  symbol fonts → strokes
# ----------------------------------------------------------------------------

FONT_DIRS = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
             "/usr/share/fonts", "/Library/Fonts", "/System/Library/Fonts"]
FONT_FILES = ["seguisym.ttf", "seguihis.ttf", "NotoSansSymbols-Regular.ttf", "NotoSansSymbols2-Regular.ttf",
              "NotoSansRunic-Regular.ttf", "NotoSansOgham-Regular.ttf", "DejaVuSans.ttf", "msyh.ttc",
              "NotoSansCJK-Regular.ttc"]

SYMBOL_SETS: Dict[str, List[int]] = {
    "alchemical": list(range(0x1F700, 0x1F774)),
    "planets": [0x2609, 0x263D, 0x263E, 0x263F, 0x2640, 0x2641, 0x2642, 0x2643, 0x2644, 0x2645, 0x2646, 0x2647,
                0x26B3, 0x26B4, 0x26B5, 0x26B6, 0x26B7, 0x260A, 0x260B, 0x260C, 0x260D],
    "zodiac": list(range(0x2648, 0x2654)),
    "trigrams": list(range(0x2630, 0x2638)),
    "religious": list(range(0x2625, 0x262F)) + [0x2020, 0x2021, 0x2627, 0x2628],
    "runic": list(range(0x16A0, 0x16EB)),
    "ogham": list(range(0x1681, 0x169B)),
    "phoenician": list(range(0x10900, 0x1091C)),
    "old_italic": list(range(0x10300, 0x10320)),
    "gothic": list(range(0x10330, 0x1034B)),
    "old_turkic": list(range(0x10C00, 0x10C49)),
}


def _find_font(cp: int) -> Optional[str]:
    from fontTools.ttLib import TTFont
    for name in FONT_FILES:
        for d in FONT_DIRS:
            p = os.path.join(d, name)
            if not os.path.exists(p):
                continue
            if p not in _CMAPS:
                try:
                    font = TTFont(p, lazy=True, fontNumber=0) if p.lower().endswith(".ttc") else TTFont(p, lazy=True)
                    _CMAPS[p] = set(font.getBestCmap().keys())
                except Exception:
                    _CMAPS[p] = set()
            if cp in _CMAPS[p]:
                return p
    return None


_CMAPS: Dict[str, set] = {}
SIZE = 160          # render canvas, px


def _render(ch: str, font_path: str) -> Optional[np.ndarray]:
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, int(SIZE * 0.7))
    img = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), ch, font=font)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    d.text(((SIZE - (x1 - x0)) / 2 - x0, (SIZE - (y1 - y0)) / 2 - y0), ch, font=font, fill=255)
    return np.asarray(img) > 127


def _skeleton_graph(mask: np.ndarray):
    import cv2
    import networkx as nx
    sk = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0
    ys, xs = np.nonzero(sk)
    pix = set(zip(ys.tolist(), xs.tolist()))
    G = nx.Graph()
    G.add_nodes_from(pix)
    for (y, x) in pix:
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            q = (y + dy, x + dx)
            if q not in pix:
                continue
            # a diagonal step whose corner is also skeleton is a staircase, not a branch
            if dy and dx and ((y + dy, x) in pix or (y, x + dx) in pix):
                continue
            G.add_edge((y, x), q)
    return G, sk


def _chains(G) -> List[List[Tuple[int, int]]]:
    """Pixel chains between branch points (degree != 2), plus closed loops."""
    nodes = {n for n in G if G.degree(n) != 2}
    seen = set()
    out = []
    for n in nodes:
        for nb in G.neighbors(n):
            if (n, nb) in seen:
                continue
            path, prev, cur = [n, nb], n, nb
            seen.update({(n, nb), (nb, n)})
            while cur not in nodes:
                nxt = [q for q in G.neighbors(cur) if q != prev]
                if not nxt:
                    break
                q = nxt[0]
                seen.update({(cur, q), (q, cur)})
                path.append(q)
                prev, cur = cur, q
            out.append(path)
    # rings with no branch point
    done = {p for c in out for p in c}
    for comp in __import__("networkx").connected_components(G):
        if comp & done or len(comp) < 8:
            continue
        start = min(comp)
        path, prev, cur = [start], None, start
        while True:
            nxt = [q for q in G.neighbors(cur) if q != prev and (q not in path or (q == start and len(path) > 2))]
            if not nxt:
                break
            q = nxt[0]
            if q == start:
                path.append(q)
                break
            path.append(q)
            prev, cur = cur, q
        out.append(path)
    return out


def _direction(chain, at_start: bool, k: int = 8) -> np.ndarray:
    P = np.asarray(chain, dtype=float)
    a, b = (P[0], P[min(k, len(P) - 1)]) if at_start else (P[-1], P[max(-k - 1, -len(P))])
    v = b - a
    return v / (np.linalg.norm(v) + 1e-9)


def _strokes_from_chains(chains, size: float) -> List[np.ndarray]:
    """Drop thinning spurs, then join chains that continue smoothly through a
    junction (a "+" is two strokes, not four)."""
    from collections import defaultdict
    deg = defaultdict(int)
    for c in chains:
        deg[c[0]] += 1
        deg[c[-1]] += 1
    min_len = 0.07 * size
    chains = [c for c in chains if len(c) >= min_len or (deg[c[0]] == 1 and deg[c[-1]] == 1)]
    # ends meeting at the same junction (within 3 px)
    ends = [(i, e, np.array(c[0] if e == 0 else c[-1], float)) for i, c in enumerate(chains) for e in (0, 1)
            if c[0] != c[-1] or e == 0]
    link: Dict[Tuple[int, int], Tuple[int, int]] = {}
    groups: List[List[Tuple[int, int]]] = []
    used = set()
    for a in range(len(ends)):
        if a in used:
            continue
        g = [a]
        for b in range(a + 1, len(ends)):
            if b not in used and np.abs(ends[a][2] - ends[b][2]).max() <= 3:
                g.append(b)
        used.update(g)
        if len(g) >= 2:
            groups.append([(ends[k][0], ends[k][1]) for k in g])
    for g in groups:
        dirs = {ie: _direction(chains[ie[0]], ie[1] == 0) for ie in g}
        pairs = sorted(((float(dirs[p] @ dirs[q]), p, q) for i, p in enumerate(g) for q in g[i + 1:] if p[0] != q[0]))
        free = set(g)
        for dot, p, q in pairs:
            if dot > -0.7:                    # leaving directions must be roughly opposite (< 45° bend)
                break
            if p in free and q in free:
                link[p], link[q] = q, p
                free -= {p, q}
    strokes, done = [], set()

    def walk(i, e_in):
        """Follow chain i entered at end e_in, then whatever it links to."""
        pts = []
        while i not in done:
            done.add(i)
            c = chains[i] if e_in == 0 else chains[i][::-1]
            pts.extend(c if not pts else c[1:])
            e_out = 1 - e_in
            nxt = link.get((i, e_out))
            if nxt is None:
                break
            i, e_in = nxt
        return pts

    starts = [(i, e) for i in range(len(chains)) for e in (0, 1) if (i, e) not in link]
    for i, e in starts:
        if i not in done:
            strokes.append(walk(i, e))
    for i in range(len(chains)):              # loops made only of linked chains
        if i not in done:
            strokes.append(walk(i, 0))
    return [np.array([(x, -y) for (y, x) in s], float) for s in strokes if len(s) >= 2]


def _smooth_resample(P: np.ndarray, step: float = 1.0, sigma: float = 2.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    closed = len(P) > 8 and np.linalg.norm(P[0] - P[-1]) < 3
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return P[:1]
    s = np.arange(0, d[-1] + 1e-9, step)
    Q = np.stack([np.interp(s, d, P[:, k]) for k in range(2)], axis=1)
    return gaussian_filter1d(Q, sigma, axis=0, mode="wrap" if closed else "nearest")


def _order(strokes: List[np.ndarray]) -> List[np.ndarray]:
    """Top-to-bottom, left-to-right; each stroke starts at its top-left end,
    loops at their top point going counter-clockwise."""
    out = []
    for P in strokes:
        if len(P) > 8 and np.linalg.norm(P[0] - P[-1]) < 4:          # loop
            k = int(np.argmax(P[:, 1]))
            P = np.roll(P[:-1], -k, axis=0)
            P = np.vstack([P, P[:1]])
            area = np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1])
            if area < 0:
                P = P[::-1]
        elif (P[-1, 1] - 0.5 * P[-1, 0]) > (P[0, 1] - 0.5 * P[0, 0]):
            P = P[::-1]
        out.append(P)
    return sorted(out, key=lambda P: (-round(P[0, 1] / 12), P[0, 0]))


def _time(strokes: List[np.ndarray], size: float) -> List[np.ndarray]:
    """Two-thirds power law within strokes, slow onsets and ends, and pen-up
    pauses proportional to the distance travelled."""
    out, t = [], 0.0
    prev_end = None
    for P in strokes:
        if prev_end is not None:
            t += 0.12 + np.linalg.norm(P[0] - prev_end) / (1.5 * size)
        if len(P) < 3:
            out.append(np.c_[P, [t] * len(P)])
            prev_end = P[-1]
            t += 0.05
            continue
        d1 = np.gradient(P, axis=0)
        d2 = np.gradient(d1, axis=0)
        k = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / (np.linalg.norm(d1, axis=1) ** 3 + 1e-9)
        v = (k * size + 0.3) ** (-1 / 3)                                      # 2/3 power law
        s = np.linspace(0, 1, len(P))
        v = v * (0.25 + 0.75 * np.sin(np.pi * s) ** 0.5)                      # onset and end
        ds = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
        v = v / v.mean() * (1.2 * size)                                        # ≈ 1.2 sizes per second
        tt = t + np.cumsum(ds / np.maximum(v, 1e-6))
        out.append(np.c_[P, tt])
        t, prev_end = float(tt[-1]), P[-1]
    return out


def glyph_strokes(ch: str, font_path: Optional[str] = None, max_strokes: int = 14) -> Optional[List[np.ndarray]]:
    """A character's centre-line strokes (x, y up, t), or None when the glyph
    is missing, empty, or a filled shape with no drawable centre line."""
    font_path = font_path or _find_font(ord(ch))
    if font_path is None:
        return None
    mask = _render(ch, font_path)
    if mask is None or mask.sum() < 30:
        return None
    return mask_strokes(mask, max_strokes)


def mask_strokes(mask: np.ndarray, max_strokes: int = 14, max_width: float = 0.09) -> Optional[List[np.ndarray]]:
    """Centre-line strokes (x, y up, t) of a binary ink mask: thinned,
    traced, joined through junctions, ordered top-left first and timed with
    the two-thirds power law."""
    G, sk = _skeleton_graph(mask)
    width = mask.sum() / max(sk.sum(), 1)                  # mean stroke thickness, px
    if width > max_width * max(mask.shape):
        return None                                         # a blob, not a line drawing
    ys, xs = np.nonzero(mask)
    size = float(max(np.ptp(xs), np.ptp(ys), 10))
    strokes = [_smooth_resample(P) for P in _strokes_from_chains(_chains(G), size)]
    strokes = [P for P in strokes if len(P) >= 2 and np.linalg.norm(np.ptp(P, axis=0)) > 0.04 * size]
    # small compact blobs (the dot of ☉, of i) are touches: one-point strokes
    import cv2
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        if max(w, h) < 0.22 * size and stats[k, cv2.CC_STAT_AREA] > 0.5 * w * h:
            c = np.array([cent[k][0], -cent[k][1]])
            if not any(np.min(np.linalg.norm(P - c, axis=1)) < 0.6 * max(w, h) for P in strokes):
                strokes.append(c[None, :])
    if not strokes or len(strokes) > max_strokes:
        return None
    return _time(_order(strokes), size)


def image_mask(path: str, size: int = 256) -> np.ndarray:
    """Ink mask of a drawing image (photo, scan or export; dark on light or
    light on dark), cropped to the ink and fitted in a size x size canvas."""
    import cv2
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA"):                          # transparent background: alpha is the ink
        a = np.asarray(img.getchannel("A"))
        if a.min() < 200:
            g = 255 - a
        else:
            g = np.asarray(img.convert("L"))
    else:
        g = np.asarray(img.convert("L"))
    g = cv2.GaussianBlur(g, (0, 0), max(1.0, max(g.shape) / 600))
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = m > 0
    if ink.mean() > 0.5:                                    # ink is the minority
        ink = ~ink
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > 0.002 * ink.sum()          # drop specks
    ink = keep[lab]
    ys, xs = np.nonzero(ink)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    sc = 0.8 * size / max(crop.shape)
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.uint8)
    y0, x0 = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    out[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return out > 127


def image_drawing(path: str, name: Optional[str] = None, system: str = "drawn") -> Optional[Drawing]:
    """A drawing image as a Drawing (centre-line strokes with human timing)."""
    strokes = mask_strokes(image_mask(path), max_strokes=24, max_width=0.14)
    if not strokes:
        return None
    name = name or os.path.splitext(os.path.basename(path))[0]
    return Drawing(system, f"{system}/{name}", "image", strokes)


def font_drawings(system: str, codepoints: Optional[Sequence[int]] = None) -> List[Drawing]:
    """Drawings of one symbol block from SYMBOL_SETS (or explicit code points)."""
    out = []
    for cp in (codepoints or SYMBOL_SETS[system]):
        ch = chr(cp)
        s = glyph_strokes(ch)
        if s:
            name = unicodedata.name(ch, f"U+{cp:04X}").lower()
            out.append(Drawing(system, f"{system}/{name}", "font", s, ch))
    return out


def all_font_drawings() -> List[Drawing]:
    return [d for sys_ in SYMBOL_SETS for d in font_drawings(sys_)]


# delyrism structures whose symbols have a canonical glyph; the others
# (chakras, Jungian archetypes, Mayan days, ...) enter through one drawing
STRUCTURE_GLYPHS: Dict[str, Dict[str, str]] = {
    "elements": {"EARTH": "🜃", "WATER": "🜄", "FIRE": "🜂", "AIR": "🜁"},
    "planets": {"SUN": "☉", "MOON": "☽", "MERCURY": "☿", "VENUS": "♀", "MARS": "♂",
                "JUPITER": "♃", "SATURN": "♄", "URANUS": "♅", "NEPTUNE": "♆", "PLUTO": "♇"},
    "chinese_zodiac": {"RAT": "鼠", "OX": "牛", "TIGER": "虎", "RABBIT": "兔", "DRAGON": "龙", "SNAKE": "蛇",
                       "HORSE": "马", "GOAT": "羊", "MONKEY": "猴", "ROOSTER": "鸡", "DOG": "狗", "PIG": "猪"},
}


def structure_drawings(name: str) -> Dict[str, Drawing]:
    """{symbol: Drawing} for a delyrism structure with canonical glyphs."""
    out = {}
    for sym, ch in STRUCTURE_GLYPHS[name].items():
        s = glyph_strokes(ch, max_strokes=24)
        if s:
            out[sym] = Drawing(name, f"{name}/{sym}", "font", s, ch)
    return out
