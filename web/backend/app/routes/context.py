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


class AlchemistBlendRequest(BaseModel):
    space_id: str
    sentence_a: str
    sentence_b: str
    blend: float = 0.5  # 0 → pure A, 1 → pure B


@router.post("/set-alchemist-blend")
def set_alchemist_blend(req: AlchemistBlendRequest) -> dict:
    """Morph between two context sentences and install the result as the
    space's context_override.

    Server-side encode + lerp + normalize avoids round-tripping two vectors
    on every slider tick.  The encoder's single-flight cache means repeated
    calls with the same sentences are basically free — only the lerp
    changes on each slider movement.

    Reuses the same override slot audio/image use, so they're mutually
    exclusive (the UI mirrors this).
    """
    space = engine_cache.get_space(req.space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")
    a = (req.sentence_a or "").strip()
    b = (req.sentence_b or "").strip()
    lam = max(0.0, min(1.0, float(req.blend)))

    # If only one side is filled, behave like a plain sentence (no blend
    # needed) — keeps the slider feel honest at the endpoints when the
    # other textarea is empty.
    if not a and not b:
        space.set_context_vec(None)
        engine_cache.invalidate_results(req.space_id)
        return {"ok": True, "active": False}

    vecs = []
    if a:
        vecs.append(("a", space.embedder.encode([a])[0]))
    if b:
        vecs.append(("b", space.embedder.encode([b])[0]))

    if len(vecs) == 1:
        v = vecs[0][1]
    else:
        v = (1.0 - lam) * vecs[0][1] + lam * vecs[1][1]

    v = v.astype(np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        # degenerate (both sentences encoded to ~zero) — clear the override
        space.set_context_vec(None)
        engine_cache.invalidate_results(req.space_id)
        return {"ok": True, "active": False}
    v /= n
    space.set_context_vec(v)
    engine_cache.invalidate_results(req.space_id)
    return {"ok": True, "active": True}


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

# Plain descriptive prompt — the downstream pipeline (text embedder +
# symbol PPR) is what does the archetypal mapping.  We just want a clean,
# faithful description so that mapping has good signal to work from.
DEFAULT_VISION_PROMPT = (
    "Describe this image in 2–4 sentences. Cover the subjects and what "
    "they are doing, the setting, the lighting and color palette, the "
    "overall mood, and any notable objects, textures, or composition "
    "details. Be concrete and faithful to what is visible. No "
    "interpretation, no symbolism, no lists."
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


def _cf_vision_url(account_id: str, model: str) -> str:
    """Native CF /ai/run/ URL (with optional AI Gateway prefix)."""
    gateway = os.environ.get("CLOUDFLARE_GATEWAY_ID")
    if gateway:
        return f"https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway}/workers-ai/{model}"
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"


def _cf_headers(api_token: str) -> dict:
    h = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    gateway = os.environ.get("CLOUDFLARE_GATEWAY_ID")
    if gateway:
        gtok = os.environ.get("CLOUDFLARE_GATEWAY_TOKEN")
        if gtok:
            h["cf-aig-authorization"] = f"Bearer {gtok}"
    return h


def _cf_extract_text(data: dict) -> str:
    """CF vision responses come back as either {success, result} where result
    is a string OR {success, result:{response: "..."}}. Be defensive."""
    if not isinstance(data, dict):
        return ""
    result = data.get("result")
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("response", "description", "text", "output"):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    # OpenAI-compat fallback (in case CF starts routing through that shape)
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def _call_cloudflare_vision(image_bytes: bytes, prompt: str, model: str) -> str:
    """One-shot call to a CF vision LLM, returns the model's text response.

    Uses the NATIVE /ai/run/{model} endpoint.  Payload format follows the
    working sample from cloudflare/cloudflare-docs#19185 (the docs page's
    code samples don't actually work):
      - `image` is an ARRAY OF INTEGER BYTES, not a base64 string and not
        a data URL.  CF's own JS example uses `[...new Uint8Array(blob)]`.
      - `prompt` is a single top-level string.  Do NOT send both `prompt`
        and `messages` — the model accepts one or the other.

    On a license-agreement error (first use per account), automatically
    sends the "agree" prompt and retries the real request — so users don't
    hit a confusing 502 on day one.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        raise HTTPException(
            status_code=500,
            detail="CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN env vars not set",
        )
    url = _cf_vision_url(account_id, model)
    headers = _cf_headers(api_token)
    payload = {
        "prompt": prompt,
        "image": list(image_bytes),
        "max_tokens": 300,
        "temperature": 0.5,
    }

    def _post(p):
        try:
            r = requests.post(url, json=p, headers=headers, timeout=60)
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Cloudflare vision call failed: {e}")
        return r

    r = _post(payload)
    if r.status_code >= 400:
        body = r.text[:500]
        # First-use license dance for Llama-3.2-Vision: CF returns a 4xx with
        # the word "license"/"agreement" in the body; one "agree" call unblocks
        # all future requests for this account.
        if any(k in body.lower() for k in ("license", "agreement", "terms", "policy")):
            try:
                _post({"prompt": "agree"})
            except Exception:
                pass
            r = _post(payload)
        if r.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Cloudflare vision error {r.status_code}: {body}",
            )

    try:
        data = r.json()
    except ValueError:
        raise HTTPException(status_code=502, detail=f"Cloudflare vision: non-JSON response ({r.text[:200]})")

    if not data.get("success", True):
        raise HTTPException(status_code=502, detail=f"Cloudflare vision error: {data.get('errors')}")

    text = _cf_extract_text(data)
    if not text:
        # Surface the actual payload to make debugging tractable instead of
        # a generic 502 with a bare message.
        raise HTTPException(
            status_code=502,
            detail=f"Cloudflare vision returned no text. Raw response: {str(data)[:400]}",
        )
    return text


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

    description = _call_cloudflare_vision(
        image_bytes=blob,
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
