"""Text/audio/image context encoding.

The frontend doesn't know the embedding dim ahead of time, so we expose
endpoints that:
- encode an arbitrary sentence with the same embedder a given space uses
- accept an uploaded audio file (wav/mp3) and return the encoded vector
- accept an uploaded image, ask a Cloudflare vision LLM to *describe* it
  symbolically, then embed that description with the space's text embedder
  (this is the "vision-LLM shim" approach — works with any text embedder)
- set/clear a context vector override on a cached space
"""
from __future__ import annotations
import base64
import io
import os
import numpy as np
import requests
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from .. import engine_cache
from ..schemas import EncodeAudioResponse, EncodeImageResponse

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


# ---------------------------------------------------------------------------
# Image context — vision-LLM shim
# ---------------------------------------------------------------------------
# Cloudflare Workers AI doesn't ship a CLIP-style image embedder.  Instead of
# adding a heavy local CLIP model to the container, we route images through
# a vision LLM (Llama-3.2-Vision by default) that produces a short symbolic
# description, then embed THAT text with whichever embedder the space was
# built on.  Result: image context lands natively in the descriptors'
# embedding space, regardless of backend (BGE-M3, Qwen3, sentence-transformer,
# CLAP, OpenCLIP…), and the image works without rebuilding the space.

DEFAULT_VISION_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct"

# The prompt is tuned to delyrism's purpose — surface archetypal qualities,
# mythic resonance, emotional register, symbolic motifs.  This is what gets
# embedded; it's NOT a generic caption.
DEFAULT_VISION_PROMPT = (
    "You are a symbolic image reader. Render this image as a short, vivid "
    "paragraph (2–4 sentences) that captures its archetypal qualities, "
    "mythic resonance, emotional register, and symbolic motifs. Focus on "
    "the inner field the image opens — what kinds of myths, dreams, or "
    "elemental forces it evokes — rather than literal surface details "
    "(brand names, furniture, exact colors). Flow as image. No analysis "
    "or lists."
)


def _resize_image_bytes(blob: bytes, max_side: int) -> bytes:
    """Downsize the image (preserving aspect) so we don't ship a 50 MB jpeg
    to Cloudflare.  Vision LLMs work at 336–768 px tile sizes; ~1024 px on
    the long side gives plenty of headroom."""
    try:
        from PIL import Image
    except ImportError:
        return blob  # Pillow missing — let CF do its own scaling
    try:
        img = Image.open(io.BytesIO(blob))
        img.load()
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        out = io.BytesIO()
        img.convert("RGB").save(out, format="JPEG", quality=88, optimize=True)
        return out.getvalue()
    except Exception:
        return blob  # best-effort — fall through with the original bytes


def _call_cloudflare_vision(image_b64: str, prompt: str, model: str) -> str:
    """One-shot call to a CF vision LLM, returns the model's text response."""
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        raise HTTPException(
            status_code=500,
            detail="CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN env vars not set",
        )
    # CF's OpenAI-compatible chat completions endpoint accepts the standard
    # vision message format (content as an array of text + image_url parts).
    gateway = os.environ.get("CLOUDFLARE_GATEWAY_ID")
    if gateway:
        url = f"https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway}/workers-ai/v1/chat/completions"
    else:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    if gateway:
        gtok = os.environ.get("CLOUDFLARE_GATEWAY_TOKEN")
        if gtok:
            headers["cf-aig-authorization"] = f"Bearer {gtok}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        "max_tokens": 300,
        "temperature": 0.5,
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Cloudflare vision call failed: {e}")
    # OpenAI-compatible response shape
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected Cloudflare response shape: {str(data)[:300]}",
        )


@router.post("/encode-image", response_model=EncodeImageResponse)
async def encode_image(
    space_id: str = Form(...),
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    max_pixels: int = Form(1024),
    model: str = Form(DEFAULT_VISION_MODEL),
) -> EncodeImageResponse:
    """Encode an image as a context vector via the vision-LLM shim path.

    Flow: image bytes → Pillow resize → Cloudflare vision LLM (returns a
    short symbolic description) → space's text embedder → vector.  The
    description is returned alongside so the UI can show the user exactly
    what the engine "read" from their image.
    """
    space = engine_cache.get_space(space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="empty file")
    if len(blob) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image too large (>25MB)")

    blob = _resize_image_bytes(blob, max_pixels)
    image_b64 = base64.b64encode(blob).decode("ascii")

    description = _call_cloudflare_vision(
        image_b64=image_b64,
        prompt=prompt or DEFAULT_VISION_PROMPT,
        model=model,
    )
    if not description:
        raise HTTPException(status_code=502, detail="vision model returned empty description")

    # Embed the description through the space's text embedder — the resulting
    # vector lives natively in the descriptors' space.
    v = space.embedder.encode([description])[0]
    v = v / (np.linalg.norm(v) + 1e-9)
    return EncodeImageResponse(
        vector=v.astype(float).tolist(),
        dim=int(v.shape[0]),
        description=description,
        model=model,
    )


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
