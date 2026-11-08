"""Symbol sets written by a language model (delyrism/server/symbolgen.py).

    GET  /generate                 the writer, whether credentials are set
    POST /generate/names           {description, n?, existing?} -> {names, title, llm}
    POST /generate                 {names, theme?, existing?} -> {symbols: {NAME: [..]}, entries, errors, skipped}

`existing` is the set the new symbols join (its names give the descriptors shared ground, and a
name already in it is skipped).  The response's `symbols` is the map in the site's own shape,
keys upper-cased like the shipped presets; `entries` keep the glyph for a drawing pipeline.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import symbolgen

router = APIRouter(prefix="/generate", tags=["generate"])


class NamesRequest(BaseModel):
    description: str
    n: int = Field(8, ge=1, le=symbolgen.MAX_NAMES)
    existing: List[str] = Field(default_factory=list)


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
def generate_names(body: NamesRequest) -> dict:
    _ready()
    try:
        return symbolgen.write_names(body.description, body.n, body.existing)
    except symbolgen.GenError as e:
        raise HTTPException(502, str(e))


@router.post("")
def generate(body: GenerateRequest) -> dict:
    _ready()
    if not symbolgen.clean_names(body.names):
        raise HTTPException(400, "no symbol names given")
    out = symbolgen.generate(body.names, body.theme, list(body.existing.keys()))
    if not out["entries"] and out["errors"]:
        raise HTTPException(502, out["errors"][0]["error"])
    symbols: Dict[str, List[str]] = {}
    for e in out["entries"]:
        symbols[e["name"].upper()] = e["descriptors"]
    return {"symbols": symbols, **out, "llm": symbolgen.llm()}
