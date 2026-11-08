"""Symbol sets written by a language model, for the website's Symbolic Structure.

The same principle as do-you-dream's GENERATE page (dyd/symbolgen.py), minus the drawing: here a
symbol set is a `{NAME: [descriptors]}` map that every tab of the site can select, so the output
is the text part only.  Two steps, both chat completions with a JSON schema, reasoning off:

  names    an optional first step: from a description of the set ("plants", "the Greek
           pantheon", "tools of a 1920s kitchen") the model writes `n` symbol names and a title
  entry    per name: ~10 descriptors (the name first) and a `glyph` - a plain description of how
           to draw the symbol in a few basic shapes.  The glyph is kept with the set so a drawing
           pipeline (do-you-dream's FLUX.2 step) can pick the set up later.

The writer is GPT-6.1 Terra through Cloudflare's AI Gateway (unified billing: the Cloudflare
token pays); while the gateway refuses, DeepSeek V4.1 Flash on Workers AI writes instead (asked
again every 10 minutes).  The prompts are do-you-dream's, verbatim where they overlap, so a set
made on either side reads alike: descriptor 0 is the name lowercased, descriptors are lowercase,
whitespace-collapsed, unique, at most four words.

Credentials: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN from the process environment, else the
Windows User environment.  Never printed.
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

# ---------------------------------------------------------------- prompts
SYSTEM_PROMPT = """You write the symbol entries for "delyrism", a semantic graph of symbols. Each symbol is a simple hand-drawn sign with a name and a list of descriptors. Every descriptor becomes a node of a live semantic graph: a visitor's dream, their spoken words, the room's sounds and camera images are embedded in the same space as the descriptors, and the descriptors closest to them light up. Links appear between descriptors of different symbols when the dream draws them together.

For the symbol you are given, write:

descriptors: exactly {n} descriptors, lowercase, mostly one or two words, never more than three, no punctuation.
- The first descriptor is the symbol's name itself, lowercase.
- Then a mix of: concrete images and things one would see, hear or touch with it (objects, beings, places, gestures, weather, sounds); what it means (symbolic, emotional, spiritual, mythic); and how it can appear or feel in a dream.
- Every descriptor is specific and evocative, something a person might say when telling a dream. No generic words (symbol, concept, meaning, thing, idea, energy, essence), no near-synonyms of each other, no two forms of the same word, no explanations.
- When the set has a theme, read the symbol through that theme.
- Let a few descriptors be shared ground with the other symbols of the set, so the graph can link them, but keep the symbol's own core distinct.

glyph: one or two plain sentences that tell an illustrator exactly how to draw the symbol as a bold, wordless sign with a thick black marker. Name the few basic shapes (straight or wavy lines, zigzags, arcs, circles, dots, spirals, triangles, squares, crosses; outlined or filled) and how they are arranged (above, inside, crossing, side by side, radiating, mirrored). At most five elements, flat and frontal, each part clearly apart from the others and no tiny details: something one could draw from memory in ten seconds, like a pictogram or an emblem from an old book of signs. No letters, numerals or words, no realistic detail, no shading; do not mention colour, paper or drawing style.

Example, for the symbol "DRAGONFLY" in a set themed "Lakota cosmology":
{{"descriptors": ["dragonfly", "resilience", "ancestral", "message", "spiritual travel", "flight", "lightly", "blessing", "patience", "survival"], "glyph": "A long vertical line crossed near its top by two short horizontal bars of equal length, like two pairs of wings on a slender body."}}"""

USER_PROMPT = """Symbol: {name}
Theme of the set: {theme}
Other symbols in the set: {others}"""

NO_THEME = "none given; read the symbol broadly, with its everyday, mythic and dream meanings"

NAMES_SYSTEM_PROMPT = """You name the symbols of a set for "delyrism", a semantic graph of symbols. Each symbol will become a simple hand-drawn sign with a name and a list of descriptors, and the set is explored as one semantic space: the symbols should be distinct from each other, each one a clear thing or figure that can be drawn as a sign and told in a dream, and together they should cover the set's ground evenly.

From the description you are given, write:

names: exactly {n} symbol names, each one to three words, in the language of the description. Concrete and specific (a thing, a being, a figure, a place, a force), never abstract nouns or categories, no two names for the same thing, no name that contains another, ordered from the most central to the set to the most peripheral. When some symbols are already in the set, continue it: do not repeat them and keep the same kind of name.

title: a short title for the set, two to four words, in the same language.

Example, for the description "the tools of an old kitchen", 6 names:
{{"names": ["Hearth", "Kettle", "Mortar", "Cleaver", "Sieve", "Ladle"], "title": "Old Kitchen"}}"""

NAMES_USER_PROMPT = """Description of the set: {description}
Symbols already in the set: {existing}"""

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
    """From the process environment, else the Windows User environment.  Never logged."""
    v = (os.environ.get(name) or "").strip()
    if v or os.name != "nt":
        return v or None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return str(winreg.QueryValueEx(k, name)[0]).strip() or None
    except (OSError, ImportError):
        return None


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


def config() -> Dict[str, Any]:
    if not LLM_MODEL.startswith("@cf/") and not _probed.is_set() and not missing():
        _probed.set()
        threading.Thread(target=_probe, daemon=True, name="symbolgen-probe").start()
    note = "" if llm() == LLM_MODEL else f"{LLM_MODEL.split('/')[-1]} unavailable: {_gateway['why']}"
    return {"llm": llm(), "llm_note": note, "n_descriptors": N_DESCRIPTORS, "max_names": MAX_NAMES,
            "missing": missing(), "available": not missing()}


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
                     NAMES_USER_PROMPT.format(description=description,
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
