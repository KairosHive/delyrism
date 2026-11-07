"""Archetype Builder (Egregore) proxy.

The standalone Egregore service (an upstream FastAPI app, traditionally
called `miner_server`) handles real-time archetype composition with
WebSocket progress.  We don't reimplement it here — we simply expose its
URL so the frontend can embed it via iframe.

(The legacy /miner endpoint is preserved as an alias to avoid breaking
older clients; new code should call /builder.)
"""
from __future__ import annotations
import os
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/builder", tags=["builder"])
# Legacy alias — kept so older client builds keep working through a redeploy.
legacy_router = APIRouter(prefix="/miner", tags=["builder (legacy)"])


class BuilderInfo(BaseModel):
    url: str
    available: bool


def _info() -> BuilderInfo:
    # Default to the public hosted instance so the "open ↗" link works out of
    # the box on prod (Railway etc.). Local dev / self-hosters can override
    # with EGREGORE_URL=http://localhost:8765.
    url = os.environ.get("EGREGORE_URL") or "https://egregore.kairos-hive.org"
    return BuilderInfo(url=url, available=True)


@router.get("", response_model=BuilderInfo)
def builder_info() -> BuilderInfo:
    return _info()


@legacy_router.get("", response_model=BuilderInfo)
def builder_info_legacy() -> BuilderInfo:
    return _info()
