"""Symbol sets written, drawn and vectorized on Cloudflare, for the website's Symbolic Structure.

The same principle as do-you-dream's GENERATE page (dyd/symbolgen.py): a symbol set is a
`{NAME: [descriptors]}` map that every tab of the site can select, each symbol with a drawing.

  names    an optional first step: from a description of the set ("plants", "the Greek
           pantheon", "tools of a 1920s kitchen") the model writes `n` symbol names and a title
  entry    per name: ~10 descriptors (the name first) and a `glyph` - a plain description of how
           to draw the symbol in a few basic shapes (chat completions with a JSON schema,
           reasoning off)
  draw     FLUX.2 on Workers AI draws the glyph as one abstract emblem whose structure carries
           the symbol's meaning, black ink on white (the glyph alone: with the name, it drew the
           thing or its sign); the picture is cleaned to flat ink (`clean_ink`) and vectorized with
           `symbol_vectorizer` (Params() defaults, framed with PAD 0.08 like do-you-dream), so the
           outlines, centre-line strokes and geometry come out alike on both sides

The server keeps nothing: every call returns its result (images as data URLs) and the browser
holds the set.  Jobs are the browser's: one call per step per symbol, a few at a time.

The writer is GPT-6.1 Terra through Cloudflare's AI Gateway (unified billing: the Cloudflare
token pays); while the gateway refuses, DeepSeek V4.1 Flash on Workers AI writes instead (asked
again every 10 minutes).  The prompts are do-you-dream's, verbatim where they overlap, so a set
made on either side reads alike: descriptor 0 is the name lowercased, descriptors are lowercase,
whitespace-collapsed, unique, at most four words.

Credentials: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN from the process environment, which
the repo's `.env` fills (delyrism/envfile.py).  Never printed.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger("delyrism.symbolgen")

# ---------------------------------------------------------------- models
LLM_MODEL = os.environ.get("DELYRISM_GEN_LLM", "openai/gpt-6.1-terra")
FALLBACK_LLM = "@cf/deepseek-ai/deepseek-v4.1-flash-1017"
GATEWAY_RETRY_S = 600
NO_THINKING: Dict[str, Dict[str, Any]] = {
    "openai/": {"reasoning_effort": "none"},
    "@cf/deepseek-ai/": {"reasoning_effort": "none"},
    "@cf/qwen/": {"chat_template_kwargs": {"enable_thinking": False}},
    "@cf/google/": {"chat_template_kwargs": {"enable_thinking": False}},
}
N_DESCRIPTORS = 10
MAX_NAMES = 40                      # names per request (a set bigger than this is pasted in parts)
WORKERS = 4                         # entries written at once
IMAGE_SIZE = 1024                   # the vectorizer frames the ink; 1 MP keeps the strokes smooth
THUMB_SIZE = 384                    # the cleaned ink as sent to the browser (the vector files carry the detail)
PAD = 0.08                          # framing: padding on every side, as a share of the drawing's longest side
IMAGE_MODELS: List[Dict[str, str]] = [
    {"id": "@cf/black-forest-labs/flux-2-klein-9b", "label": "klein 9b",
     "hint": "the cleanest lines and the whitest paper, ~2 s a drawing (~$0.015)"},
    {"id": "@cf/black-forest-labs/flux-2-klein-4b", "label": "klein 4b",
     "hint": "the cheapest (~$0.001 a drawing), the paper a little grey"},
    {"id": "@cf/black-forest-labs/flux-2-dev", "label": "dev",
     "hint": "the full FLUX.2 model, ~10 s a drawing, priced per step"},
]
DEFAULT_IMAGE_MODEL = IMAGE_MODELS[0]["id"]

# ---------------------------------------------------------------- prompts
SYSTEM_PROMPT = """You write the symbol entries for "delyrism", a semantic graph of symbols. Each symbol is a simple hand-drawn sign with a name and a list of descriptors. Every descriptor becomes a node of a live semantic graph: a visitor's dream, their spoken words, the room's sounds and camera images are embedded in the same space as the descriptors, and the descriptors closest to them light up. Links appear between descriptors of different symbols when the dream draws them together.

For the symbol you are given, write:

descriptors: exactly {n} descriptors, lowercase, mostly one or two words, never more than three, no punctuation.
- The first descriptor is the symbol's name itself, lowercase.
- Then a mix of: concrete images and things one would see, hear or touch with it (objects, beings, places, gestures, weather, sounds); what it means (symbolic, emotional, spiritual, mythic); and how it can appear or feel in a dream.
- Every descriptor is specific and evocative, something a person might say when telling a dream. No generic words (symbol, concept, meaning, thing, idea, energy, essence), no near-synonyms of each other, no two forms of the same word, no explanations.
- When the set has a theme, read the symbol through that theme.
- Let a few descriptors be shared ground with the other symbols of the set, so the graph can link them, but keep the symbol's own core distinct.

glyph: three to five plain sentences that tell an artist exactly how to draw the symbol as one abstract emblem, black ink on white: a glyph whose structure carries the meaning. First read the symbol with its descriptors as one small ontology: what is at its centre, what gathers around it, what it reaches toward, what it holds in and what it lets out, what repeats, what opposes what. Then translate those relations into visual structure, not into pictures: enclosure and opening, a core and what radiates or orbits, layers, repetition and rhythm, density thinning out, a broken or a continuous line, balance or a deliberate tilt, a heavy solid mass against thin lines, a hollow against a filled shape, convergence or scattering. Compose it from a handful of elements that relate to each other: solid filled shapes and lines of varied weight, hollow and filled forms, open and closed curves, angles, bars, dots, parallel or nested strokes, cut-outs inside a fill. Say exactly what each element is, how large, where it sits and how it joins, crosses, contains or stands apart from the others, and which relation of the ontology each part embodies. One composition, readable at a glance at a small size: at most seven or eight elements, no fine texture, no tiny details, nothing hatched. It must not depict the thing, any part of it or anything it suggests, and must not look like an existing sign: no astrological, planetary or alchemical signs, no letters or numerals of any real alphabet (Latin, Greek, Hebrew, runic, Chinese...), no religious or political symbols, no arrows, hearts, stars, crosses, icons or emoji; nothing that reads as a figure, a face, limbs, an animal, a plant, a sun, a house or a tool. Do not mention colour, paper, shading or style. Symmetry or asymmetry is a choice made by the meaning. Vary the structures from symbol to symbol so the set reads as one family of emblems from one culture, each distinct.

Example, for the symbol "DRAGONFLY" in a set themed "Lakota cosmology":
{{"descriptors": ["dragonfly", "resilience", "ancestral", "message", "spiritual travel", "flight", "lightly", "blessing", "patience", "survival"], "glyph": "A small solid black lens, tilted, sits a little below the centre: the body of the meaning, resilience and survival, dense and closed. From its upper tip a thin line rises and splits into two long open curves that diverge like a path forking, each thinning to nothing: spiritual travel and the message carried away lightly. Three short bars cross the left curve at even intervals, the rhythm of patience; the right curve is crossed by nothing. Below the lens, apart from it, a wide shallow arc opens upward like a shallow bowl, the ancestral ground that holds without touching; a single dot rests at its centre."}}"""

USER_PROMPT = """Symbol: {name}
Theme of the set: {theme}
Other symbols in the set: {others}"""

NO_THEME = "none given; read the symbol broadly, with its everyday, mythic and dream meanings"

# The image prompt carries the glyph (the language model's design, which already holds the symbol's
# meaning as structure), never the name or the descriptors: given those, FLUX.2 draws the thing (or
# its existing symbol, e.g. a planet's astrological sign) instead of an emblem.
IMAGE_PROMPT = ("One abstract emblem drawn in black ink on plain white paper, the mark of an unknown culture: "
                "a glyph whose structure carries a meaning, not a picture of anything. {drawing}"
                "Drawn by a sure hand with brush and pen: solid matte black filled shapes next to lines of varied "
                "weight, thick and thin, hollow forms and filled forms, open and closed curves, dots and bars, every "
                "element clearly apart or clearly joined, edges slightly uneven as ink is. Bold and legible at a "
                "glance, no fine texture, no hatching, no tiny details. It is not any existing letter, numeral, "
                "symbol, sign, logo or icon, and not a figure, face, body, animal, plant, sun, house or object. "
                "Centered and filling most of the page, with a clear white margin on every side; the white paper "
                "fills the whole image edge to edge, in flat even light. Black ink on white only: one emblem, "
                "no words, no shading, no gradients, no shadow, no texture, no colour, no frame.")

NAMES_SYSTEM_PROMPT = """You name the symbols of a set for "delyrism", a semantic graph of symbols. Each symbol will become a sign with a name and a list of descriptors, and the set is explored as one semantic space: the symbols should be distinct from each other, each one a clear thing, being, figure, place or force that can be told in a dream, and together they should cover the set's ground evenly.

The description of the set is the brief, and it governs everything: every one of the names must be a member of exactly what it describes (if it says animals, every name is an animal; if it says tools, every name is a tool; if it says a time, a place, a culture, a mood or a story, every name belongs there), read it literally and follow every constraint it states (a language, a kind of name, a number of words, things to leave out, an order). Do not drift to the atmosphere around the subject, to related categories or to the obvious.

Write:

names: exactly {n} symbol names, each one to three words, in the language of the description unless it asks for another. Concrete and specific, never abstract nouns, categories or qualities; no two names for the same thing, no name that contains another; ordered from the most central to the description to the most peripheral. Symbols already in the set are listed only so that you do not repeat them (nor name the same thing in other words): they may be of another kind than the description asks for, and then the description wins, not their kind.

title: a short title for the set, two to four words, in the same language, taken from the description.

Example, for the description "the tools of an old kitchen", 6 names:
{{"names": ["Hearth", "Kettle", "Mortar", "Cleaver", "Sieve", "Ladle"], "title": "Old Kitchen"}}

Example, for the description "animals of the night", 4 names, with Owl and Bat already in the set:
{{"names": ["Moth", "Fox", "Hedgehog", "Nightjar"], "title": "Animals of the Night"}}"""

NAMES_USER_PROMPT = """Description of the set: {description}
Number of names: {n}
Symbols already in the set (do not repeat; the description decides the kind): {existing}"""

ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "descriptors": {"type": "array", "items": {"type": "string"}},
        "glyph": {"type": "string"},
    },
    "required": ["descriptors", "glyph"],
    "additionalProperties": False,
}
NAMES_SCHEMA = {
    "type": "object",
    "properties": {
        "names": {"type": "array", "items": {"type": "string"}},
        "title": {"type": "string"},
    },
    "required": ["names", "title"],
    "additionalProperties": False,
}


class GenError(RuntimeError):
    pass


# ---------------------------------------------------------------- credentials
def secret(name: str) -> Optional[str]:
    """From the process environment, which the repo's `.env` fills when the
    package is imported (re-read here in case `.env` was written since).
    Never logged."""
    v = (os.environ.get(name) or "").strip()
    if not v:
        from .. import envfile
        envfile.load()
        v = (os.environ.get(name) or "").strip()
    return v or None


def missing() -> List[str]:
    return [n for n in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN") if not secret(n)]


def _creds() -> tuple:
    acc, tok = secret("CLOUDFLARE_ACCOUNT_ID"), secret("CLOUDFLARE_API_TOKEN")
    if not (acc and tok):
        raise GenError("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN are not set")
    return acc, tok


_gateway = {"off_until": 0.0, "why": ""}     # the gateway refused: Workers AI writes meanwhile


def llm() -> str:
    """The model that writes now."""
    if LLM_MODEL.startswith("@cf/") or time.time() >= _gateway["off_until"]:
        return LLM_MODEL
    return FALLBACK_LLM


def _llm_request(model: str) -> tuple:
    """(url, headers) for a chat completion: Workers AI for @cf/ models, else the AI Gateway."""
    acc, tok = _creds()
    if model.startswith("@cf/"):
        return (f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/v1/chat/completions",
                {"Authorization": f"Bearer {tok}"})
    gateway = os.environ.get("CLOUDFLARE_GATEWAY_ID") or os.environ.get("DELYRISM_GEN_GATEWAY") or "default"
    return (f"https://gateway.ai.cloudflare.com/v1/{acc}/{gateway}/compat/chat/completions",
            {"cf-aig-authorization": f"Bearer {tok}"})


def _no_thinking(model: str) -> Dict[str, Any]:
    return next((v for k, v in NO_THINKING.items() if model.startswith(k)), {})


def _probe() -> None:
    """One-token request through the gateway, so the page names the model that will really write."""
    try:
        url, headers = _llm_request(LLM_MODEL)
        r = requests.post(url, headers=headers, timeout=20, json={
            "model": LLM_MODEL, "max_tokens": 1, "messages": [{"role": "user", "content": "ok"}],
            **_no_thinking(LLM_MODEL)})
        if r.status_code in (400, 401, 402, 403, 404):
            _gateway.update(off_until=time.time() + GATEWAY_RETRY_S,
                            why=f"AI Gateway refused ({r.status_code}): {_api_error(r)}")
            log.warning("%s through the AI Gateway: %s - %s writes for now", LLM_MODEL, _gateway["why"], FALLBACK_LLM)
    except Exception:  # noqa: BLE001  (no answer: the first real request decides)
        pass


_probed = threading.Event()


def can_vectorize() -> bool:
    try:
        import cv2  # noqa: F401
        import symbol_vectorizer  # noqa: F401
        return hasattr(cv2, "ximgproc")
    except ImportError:
        return False


def config() -> Dict[str, Any]:
    if not LLM_MODEL.startswith("@cf/") and not _probed.is_set() and not missing():
        _probed.set()
        threading.Thread(target=_probe, daemon=True, name="symbolgen-probe").start()
    note = "" if llm() == LLM_MODEL else f"{LLM_MODEL.split('/')[-1]} unavailable: {_gateway['why']}"
    return {"llm": llm(), "llm_note": note, "n_descriptors": N_DESCRIPTORS, "max_names": MAX_NAMES,
            "missing": missing(), "available": not missing(),
            "image_models": IMAGE_MODELS, "default_model": DEFAULT_IMAGE_MODEL,
            "drawing": not missing() and can_vectorize()}


# ---------------------------------------------------------------- the chat call
def _retrying(fn, what: str, tries: int = 4):
    err: Exception = GenError(what)
    for k in range(tries):
        try:
            return fn()
        except GenError as e:              # a bad answer: worth another try; a refusal is not
            err = e
            if "refused" in str(e):
                break
        except requests.RequestException as e:
            err = GenError(f"{what}: {e.__class__.__name__}")
        if k < tries - 1:
            time.sleep(2.0 * 2 ** k)       # 2, 4, 8 s: a busy Workers AI answers 500 for a few seconds
    raise err


def _api_error(r: requests.Response) -> str:
    try:
        j = r.json()
        e = j.get("error") or j.get("errors") or j
        if isinstance(e, list):
            e = e[0] if e else j
        msg = str(e.get("message") if isinstance(e, dict) else e)
    except Exception:
        msg = r.text
    msg = re.sub(r"^(?:AiError:\s*)+", "", msg.strip())
    msg = re.sub(r"\s*\([0-9a-f]{8}-[0-9a-f-]{27}\)\s*$", "", msg)
    return msg[:240]


def _complete(what: str, system: str, user: str, schema_name: str, schema: Dict[str, Any],
              max_tokens: int, parse) -> Dict[str, Any]:
    """A JSON chat completion with the current writer; on any gateway failure the Workers AI
    model takes over for the next ten minutes.  `parse(dict) -> dict` validates the answer."""
    model = llm()
    try:
        return _complete_with(model, what, system, user, schema_name, schema, max_tokens, parse)
    except GenError as e:
        if model == FALLBACK_LLM or not getattr(e, "gateway", False):
            raise
        _gateway.update(off_until=time.time() + GATEWAY_RETRY_S, why=str(e))
        log.warning("%s through the AI Gateway: %s - %s writes for now", model, e, FALLBACK_LLM)
        return _complete_with(FALLBACK_LLM, what, system, user, schema_name, schema, max_tokens, parse)


def _complete_with(model, what, system, user, schema_name, schema, max_tokens, parse) -> Dict[str, Any]:
    url, headers = _llm_request(model)
    gateway = not model.startswith("@cf/")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
        "max_tokens": max_tokens,
        **_no_thinking(model),
    }

    def once():
        r = requests.post(url, json=body, headers=headers, timeout=60)
        if r.status_code in (401, 402, 403, 404) or (gateway and r.status_code == 400):
            raise GenError(f"{'AI Gateway' if gateway else 'Cloudflare'} refused ({r.status_code}): {_api_error(r)}")
        if r.status_code != 200:
            raise GenError(f"Cloudflare {r.status_code}: {_api_error(r)}")
        try:
            msg = r.json()["choices"][0]["message"]["content"] or ""
            out = json.loads(msg[msg.find("{"):msg.rfind("}") + 1])
        except Exception:
            raise GenError("the language model gave no readable answer")
        return {**parse(out), "llm": model}

    try:
        return _retrying(once, what)
    except GenError as e:
        err = e if "refused" in str(e) else GenError(f"{what} failed: {e}")
        err.gateway = gateway          # the gateway failing in any way: Workers AI takes over
        raise err


# ---------------------------------------------------------------- cleaning
def clean_name(name: str) -> str:
    """A symbol name as typed: list markers dropped, whitespace collapsed, at most 40 characters."""
    n = re.sub(r"^\s*(?:[-*•·]+|\d+[.)]|\(\d+\))\s*", "", str(name))
    n = " ".join(n.split()).strip(" ,;").strip()
    return n[:40]


def clean_names(names) -> List[str]:
    out: List[str] = []
    for n in names:
        n = clean_name(n)
        if n and not any(o.lower() == n.lower() for o in out):
            out.append(n)
    return out


def clean_descriptors(ds) -> List[str]:
    out: List[str] = []
    for d in ds:
        d = " ".join(str(d).strip(" .;:-\"'").split()).lower()
        if d and d not in out:
            out.append(d)
    return out


def _clean_entry(name: str, out: Dict[str, Any]) -> Dict[str, Any]:
    ds = clean_descriptors(out.get("descriptors") or [])
    own = " ".join(name.split()).lower()
    ds = [own] + [d for d in ds if d != own]               # the name is the first descriptor
    ds = [d for d in ds if len(d.split()) <= 4][:N_DESCRIPTORS]
    glyph = " ".join(str(out.get("glyph") or "").split())
    if len(ds) < max(4, N_DESCRIPTORS // 2) or not glyph:
        raise GenError("the language model's entry was too thin")
    return {"descriptors": ds, "glyph": glyph}


# ---------------------------------------------------------------- 1. names
def write_names(description: str, n: int = 8, existing: Optional[List[str]] = None) -> Dict[str, Any]:
    """{names, title, llm}: `n` new symbol names for a described set, continuing `existing`."""
    description = " ".join((description or "").split())
    if not description:
        raise GenError("describe the set first")
    n = max(1, min(int(n), MAX_NAMES))
    existing = clean_names(existing or [])
    have = {e.lower() for e in existing}

    def parse(out):
        names = [m for m in clean_names(out.get("names") or []) if m.lower() not in have][:n]
        if len(names) < max(1, n // 2):
            raise GenError("the language model named too few symbols")
        return {"names": names, "title": " ".join(str(out.get("title") or "").split())[:60]}

    return _complete("writing the names", NAMES_SYSTEM_PROMPT.format(n=n),
                     NAMES_USER_PROMPT.format(description=description, n=n,
                                              existing=", ".join(existing) or "none yet"),
                     "symbol_names", NAMES_SCHEMA, 80 + 12 * n, parse)


# ---------------------------------------------------------------- 2. entries
def write_entry(name: str, theme: str = "", others: Optional[List[str]] = None) -> Dict[str, Any]:
    """{descriptors, glyph, llm} for a symbol name."""
    return _complete("writing the descriptors", SYSTEM_PROMPT.format(n=N_DESCRIPTORS),
                     USER_PROMPT.format(name=name, theme=(theme or "").strip() or NO_THEME,
                                        others=", ".join(o for o in (others or []) if o.lower() != name.lower())
                                        or "none yet"),
                     "symbol_entry", ENTRY_SCHEMA, 600, lambda out: _clean_entry(name, out))


def generate(names: List[str], theme: str = "", others: Optional[List[str]] = None) -> Dict[str, Any]:
    """Entries for every name, `WORKERS` at a time.  `others`: names already in the set (shared
    ground for the descriptors; a name among them is skipped).  Returns
    {entries: [{name, descriptors, glyph, llm}], errors: [{name, error}], skipped: [name]}."""
    names = clean_names(names)[:MAX_NAMES]
    others = clean_names(others or [])
    have = {o.lower() for o in others}
    todo = [n for n in names if n.lower() not in have]
    skipped = [n for n in names if n.lower() in have]
    everyone = others + todo

    def one(name: str) -> Dict[str, Any]:
        try:
            return {"name": name, **write_entry(name, theme, everyone)}
        except GenError as e:
            return {"name": name, "error": str(e)}

    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="symbolgen") as pool:
        results = list(pool.map(one, todo))
    return {"entries": [r for r in results if "error" not in r],
            "errors": [r for r in results if "error" in r],
            "skipped": skipped}


# ---------------------------------------------------------------- 3. draw
def image_prompt(name: str, descriptors: List[str], glyph: str = "") -> str:
    """The glyph only: the name and descriptors stay out (see IMAGE_PROMPT)."""
    glyph = glyph.strip()
    if glyph and glyph[-1] not in ".!?":
        glyph += "."
    return IMAGE_PROMPT.format(drawing=f"The emblem: {glyph} " if glyph else "")


def draw(prompt: str, model: str = DEFAULT_IMAGE_MODEL, seed: Optional[int] = None) -> bytes:
    """PNG bytes from FLUX.2 on Workers AI (multipart in, base64 out)."""
    import base64

    acc, tok = _creds()
    if model not in {m["id"] for m in IMAGE_MODELS}:
        raise GenError(f"unknown image model {model!r}")
    url = f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/{model}"
    fields = {"prompt": prompt, "width": str(IMAGE_SIZE), "height": str(IMAGE_SIZE)}
    if seed is not None:
        fields["seed"] = str(seed)

    def once():
        r = requests.post(url, headers={"Authorization": f"Bearer {tok}"},
                          files={k: (None, v) for k, v in fields.items()}, timeout=120)
        if r.status_code in (401, 403):
            raise GenError(f"Cloudflare refused ({r.status_code}): {_api_error(r)}")
        if r.status_code != 200:
            raise GenError(f"Cloudflare {r.status_code}: {_api_error(r)}")
        img = ((r.json() or {}).get("result") or {}).get("image")
        if not img:
            raise GenError("Cloudflare sent no image")
        return base64.b64decode(img)

    try:
        return _retrying(once, "drawing")
    except GenError as e:
        if "refused" in str(e):
            raise
        raise GenError(f"drawing failed: {e}")


def clean_ink(png: bytes) -> bytes:
    """The drawing as clean ink on white, before vectorization.  A generated marker drawing has
    glossy highlights inside its solid shapes and a slightly uneven paper; the vectorizer reads a
    highlight as a hole, so a filled shape turns into strokes around it.  Here: the paper is
    flattened, the ink thresholded, small holes inside the ink filled, loose specks dropped, and
    the edges softened by a pixel."""
    import io

    import numpy as np
    from PIL import Image, ImageFilter
    from scipy import ndimage

    im = Image.open(io.BytesIO(png)).convert("L")
    a = np.asarray(im, np.float32) / 255.0
    n = a.size
    small = Image.fromarray((a * 255).astype(np.uint8)).resize((128, 128), Image.BILINEAR)
    paper = ndimage.maximum_filter(np.asarray(small, np.float32) / 255.0, size=9)
    paper = np.asarray(Image.fromarray((paper * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)
                       .filter(ImageFilter.GaussianBlur(8)), np.float32) / 255.0
    flat = np.clip(a / np.maximum(paper, 0.2), 0.0, 1.0)
    ink = flat < 0.55
    holes, k = ndimage.label(ndimage.binary_fill_holes(ink) & ~ink)
    if k:
        idx = np.arange(1, k + 1)
        area = ndimage.sum(np.ones_like(flat), holes, idx)
        light = ndimage.mean(flat, holes, idx)
        ink |= np.isin(holes, idx[(area < 0.0006 * n) | (light < 0.8)])
    blobs, k = ndimage.label(ink)
    if k:
        idx = np.arange(1, k + 1)
        area = ndimage.sum(np.ones_like(flat), blobs, idx)
        ink &= ~np.isin(blobs, idx[area < 0.00015 * n])
    out = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    buf = io.BytesIO()
    out.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _ink(png: bytes) -> bytes:
    try:
        return clean_ink(png)
    except Exception:  # noqa: BLE001  (an odd image: the vectorizer still gets the drawing as it is)
        log.exception("cleaning a generated drawing failed")
        return png


# ---------------------------------------------------------------- 4. vectorize (framed like do-you-dream)
_SVG_ROOT = re.compile(r"<svg\b[^>]*>")
_SVG_D = re.compile(r'\bd="([^"]*)"')
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _bbox(svg: str):
    xs, ys = [], []
    for d in _SVG_D.findall(svg):
        v = [float(x) for x in _NUM.findall(d)]
        xs += v[0::2]
        ys += v[1::2]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def _reframe(svg: str, box) -> str:
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0, 1.0) * (1 + 2 * PAD)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    root = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{cx - side / 2:.2f} {cy - side / 2:.2f} '
            f'{side:.2f} {side:.2f}" width="{side:.0f}" height="{side:.0f}" data-framed="1">')
    return _SVG_ROOT.sub(root, svg, count=1)


def frame(svg: str, lines_svg: str):
    """Both drawings in one square frame around the ink, the same padding on every side."""
    box = _bbox(svg) or _bbox(lines_svg)
    return (svg, lines_svg) if box is None else (_reframe(svg, box), _reframe(lines_svg, box))


_vec_lock = threading.Lock()        # one vectorization at a time (CPU-bound, a few seconds)


def vectorize(png: bytes, name: str) -> Dict[str, Any]:
    """{svg, lines_svg, png (transparent crop, bytes), record} from a cleaned drawing."""
    import io

    from PIL import Image
    from symbol_vectorizer import Params, vectorize_array

    img = Image.open(io.BytesIO(png)).convert("RGB")
    with _vec_lock:
        try:
            r = vectorize_array(img, Params(), name=name)
        except Exception as e:  # noqa: BLE001  (no ink found, ...)
            raise GenError(f"vectorizing failed: {e or e.__class__.__name__}")
    svg, lines_svg = frame(r["svg"], r["lines_svg"])
    rec = dict(r["record"])
    rec.pop("_preview", None)
    rec["source"] = None
    return {"svg": svg, "lines_svg": lines_svg, "png": r["png"], "record": rec}


def _data_url(data: bytes, mime: str = "image/png") -> str:
    import base64
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _thumb(png: bytes, size: int = THUMB_SIZE) -> bytes:
    import io

    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("L")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def make_drawing(name: str, descriptors: List[str], glyph: str = "", model: str = DEFAULT_IMAGE_MODEL,
                 seed: Optional[int] = None) -> Dict[str, Any]:
    """Draw a symbol and vectorize it: {photo (data URL, the cleaned ink), svg, lines_svg, geometry,
    seed, model, image_prompt, glyph, at}."""
    import random
    from datetime import datetime

    if not can_vectorize():
        raise GenError("the server cannot vectorize (opencv-contrib / symbol_vectorizer missing)")
    seed = int(seed) if seed is not None else random.randrange(1, 2 ** 31)
    prompt = image_prompt(name, descriptors, glyph)
    ink = _ink(draw(prompt, model, seed))
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "symbol"
    v = vectorize(ink, slug)
    return {"photo": _data_url(_thumb(ink)), "svg": v["svg"], "lines_svg": v["lines_svg"],
            "geometry": v["record"], "seed": seed, "model": model,
            "image_prompt": prompt, "glyph": glyph, "at": datetime.now().isoformat(timespec="seconds")}
