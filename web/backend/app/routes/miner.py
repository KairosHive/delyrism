"""Archetype Builder (Egregore) proxy.

The existing repo ships a separate FastAPI miner under
delyrism/miner_server.py exposing real-time mining + WebSocket progress.  We
don't reimplement it here — instead we proxy a couple of read endpoints and
expose its URL so the frontend can embed/iframe it directly."""
from __future__ import annotations
import os
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/miner", tags=["miner"])


class MinerInfo(BaseModel):
    url: str
    available: bool


@router.get("", response_model=MinerInfo)
def miner_info() -> MinerInfo:
    url = os.environ.get("EGREGORE_URL") or "http://localhost:8765"
    return MinerInfo(url=url, available=True)
