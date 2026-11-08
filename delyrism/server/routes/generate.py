"""Symbol sets written, drawn and vectorized on Cloudflare (delyrism/server/symbolgen.py).

    GET  /generate                 the writer, the image models, whether credentials / the vectorizer are there
    POST /generate/names           {description, n?, existing?} -> {names, title, llm}
    POST /generate/entry           {name, theme?, others?} -> {name, descriptors, glyph, llm}
    POST /generate/draw            {name, descriptors, glyph?, model?, seed?}
                                   -> {photo, svg, lines_svg, geometry, seed, model, image_prompt, glyph, at}
    POST /generate                 {names, theme?, existing?} -> {symbols: {NAME: [..]}, entries, errors, skipped}

The server keeps no state: the Generate tab calls `entry` then `draw` per symbol, a few at a time,
and holds the set in the browser.  `existing` / `others` are the names already in the set (shared
ground for the descriptors; a name among them is skipped by the batch call).  Keys of `symbols`
are upper-cased like the shipped presets.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .. import symbolgen

router = APIRouter(prefix="/generate", tags=["generate"])


class NamesRequest(BaseModel):
    description: str
    n: int = Field(8, ge=1, le=symbolgen.MAX_NAMES)
    existing: List[str] = Field(default_factory=list)


class EntryRequest(BaseModel):
    name: str
    theme: str = ""
    others: List[str] = Field(default_factory=list)


class DrawRequest(BaseModel):
    name: str
    descriptors: List[str] = Field(default_factory=list)
    glyph: str = ""
    model: Optional[str] = None
    seed: Optional[int] = Field(None, ge=0, lt=2 ** 31)


class GenerateRequest(BaseModel):
    names: List[str]
    theme: str = ""
    existing: Dict[str, List[str]] = Field(default_factory=dict)


def _ready() -> None:
    if symbolgen.missing():
        raise HTTPException(503, "not set: " + ", ".join(symbolgen.missing()))


@router.get("")
def generate_info() -> dict:
    return symbolgen.config()


@router.post("/names")
async def generate_names(body: NamesRequest) -> dict:
    _ready()
    try:
        return await run_in_threadpool(symbolgen.write_names, body.description, body.n, body.existing)
    except symbolgen.GenError as e:
        raise HTTPException(502, str(e))


@router.post("/entry")
async def generate_entry(body: EntryRequest) -> dict:
    _ready()
    name = symbolgen.clean_name(body.name)
    if not name:
        raise HTTPException(400, "no symbol name given")
    try:
        entry = await run_in_threadpool(symbolgen.write_entry, name, body.theme, body.others)
    except symbolgen.GenError as e:
        raise HTTPException(502, str(e))
    return {"name": name, **entry}


@router.post("/draw")
async def generate_draw(body: DrawRequest) -> dict:
    _ready()
    if not symbolgen.can_vectorize():
        raise HTTPException(503, "the server cannot vectorize drawings (opencv-contrib / symbol_vectorizer missing)")
    name = symbolgen.clean_name(body.name)
    if not name:
        raise HTTPException(400, "no symbol name given")
    try:
        return await run_in_threadpool(symbolgen.make_drawing, name, symbolgen.clean_descriptors(body.descriptors),
                                       body.glyph, body.model or symbolgen.DEFAULT_IMAGE_MODEL, body.seed)
    except symbolgen.GenError as e:
        raise HTTPException(502, str(e))


@router.post("")
async def generate(body: GenerateRequest) -> dict:
    _ready()
    if not symbolgen.clean_names(body.names):
        raise HTTPException(400, "no symbol names given")
    out = await run_in_threadpool(symbolgen.generate, body.names, body.theme, list(body.existing.keys()))
    if not out["entries"] and out["errors"]:
        raise HTTPException(502, out["errors"][0]["error"])
    symbols: Dict[str, List[str]] = {}
    for e in out["entries"]:
        symbols[e["name"].upper()] = e["descriptors"]
    return {"symbols": symbols, **out, "llm": symbolgen.llm()}
