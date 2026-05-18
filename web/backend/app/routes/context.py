"""Text/audio context encoding.

The frontend doesn't know the embedding dim ahead of time, so we expose
endpoints that:
- encode an arbitrary sentence with the same embedder a given space uses
- accept an uploaded audio file (wav/mp3) and return the encoded vector
- set/clear a context vector override on a cached space
"""
from __future__ import annotations
import io
import numpy as np
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from .. import engine_cache
from ..schemas import EncodeAudioResponse

router = APIRouter(prefix="/context", tags=["context"])


class EncodeTextRequest(BaseModel):
    space_id: str
    text: str


@router.post("/encode-text")
def encode_text(req: EncodeTextRequest) -> dict:
    space = engine_cache.get_space(req.space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")
    v = space.embedder.encode([req.text])[0]
    v = v / (np.linalg.norm(v) + 1e-9)
    return {"vector": v.astype(float).tolist(), "dim": int(v.shape[0])}


@router.post("/encode-audio", response_model=EncodeAudioResponse)
async def encode_audio(
    space_id: str = Form(...),
    file: UploadFile = File(...),
    max_seconds: int = Form(15),
) -> EncodeAudioResponse:
    space = engine_cache.get_space(space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")
    if not hasattr(space.embedder, "embed_audio_array"):
        raise HTTPException(
            status_code=400,
            detail="active embedder backend does not support audio (use clap or audioclip)",
        )
    try:
        import librosa
    except ImportError:
        raise HTTPException(status_code=500, detail="librosa not installed on server")

    blob = await file.read()
    wave, sr = librosa.load(io.BytesIO(blob), sr=None, mono=True)
    if max_seconds and len(wave) > max_seconds * sr:
        wave = wave[: max_seconds * sr]
    v = space.embedder.embed_audio_array(wave, sr)
    v = v / (np.linalg.norm(v) + 1e-9)
    return EncodeAudioResponse(vector=v.astype(float).tolist(), dim=int(v.shape[0]))


class SetOverrideRequest(BaseModel):
    space_id: str
    vector: Optional[list] = None  # None clears the override


@router.post("/set-override")
def set_context_override(req: SetOverrideRequest) -> dict:
    space = engine_cache.get_space(req.space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")
    if req.vector is None:
        space.set_context_vec(None)
    else:
        v = np.asarray(req.vector, dtype=np.float32)
        if v.shape != (space.embedder.dim,):
            raise HTTPException(
                status_code=400,
                detail=f"override dim {v.shape[0]} != embedder dim {space.embedder.dim}",
            )
        space.set_context_vec(v)
    # The result memo keys are derived from request params only — they do NOT
    # include `context_override` (it lives on the cached space, not in the
    # request).  Without this invalidation /propose etc. with sentence=null,
    # weights=null would keep returning the pre-audio cached result forever.
    engine_cache.invalidate_results(req.space_id)
    return {"ok": True}
