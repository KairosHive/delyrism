"""A deterministic stand-in embedder for tests and offline runs.

With DELYRISM_TEST_EMBEDDER=1 every space is built with `HashEmbedder`,
whatever backend the request names: no network, no model download, the same
vectors on every machine.  Texts sharing words or character trigrams get
nearby vectors, so contexts still move the Δ graph in a meaningful
direction.  The pytest suite (tests/) and the browser tests (web/frontend/e2e)
run the real server with it.

    DELYRISM_TEST_EMBEDDER=1 python -m delyrism.server --port 8000
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional

import numpy as np

from ..delyrism import TextEmbedder

ENV = "DELYRISM_TEST_EMBEDDER"
DIM = 256
AUDIO_BACKENDS = {"gemini-embedding-2", "clap", "peav"}     # spaces asking for these get an audio tower


def enabled() -> bool:
    return os.environ.get(ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _bucket(token: str) -> tuple:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h[:4], "little") % DIM, 1.0 if h[4] & 1 else -1.0


def _vector(text: str) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    words = re.findall(r"[a-z0-9']+", text.lower())
    for w in words:
        i, s = _bucket("w:" + w)
        v[i] += 2.0 * s
        p = f"#{w}#"
        for k in range(len(p) - 2):
            i, s = _bucket("t:" + p[k:k + 3])
            v[i] += s
    if not words:
        i, s = _bucket("empty")
        v[i] = s
    return v / (np.linalg.norm(v) + 1e-9)


def stt_stand_in() -> bool:
    """DELYRISM_TEST_STT=1 (browser tests): speech mode hears fixed words."""
    return enabled() and os.environ.get("DELYRISM_TEST_STT", "").strip() in ("1", "true", "yes", "on")


def fake_transcribe(wave, sr, language=None, prompt=None) -> str:
    """Two alternating phrases for any sound, nothing for silence."""
    x = np.asarray(wave, dtype=np.float32)
    if len(x) == 0 or float(np.sqrt(np.mean(x * x))) < 1e-4:
        return ""
    return "the water rises" if not prompt or prompt.endswith("answers") else "and the fire answers"


def router():
    """Test-only routes, mounted only when the stand-in embedder is on:
    POST /__test__/forget-spaces drops every built space, as a restart does."""
    from fastapi import APIRouter
    from . import engine_cache

    r = APIRouter(prefix="/__test__", tags=["test"])

    @r.post("/forget-spaces")
    def forget_spaces() -> dict:
        ids = list(engine_cache._space_cache)
        for sid in ids:
            engine_cache.invalidate_space(sid)
        return {"forgot": len(ids)}

    return r


class HashEmbedder(TextEmbedder):
    """TextEmbedder interface, hashing-trick vectors (DIM floats, unit norm)."""

    def __init__(self, default_instruction: Optional[str] = None, default_context: Optional[str] = None,
                 audio: bool = False):
        # the parent's __init__ loads a model; set its attributes by hand instead
        self.backend_type = "test-hash"
        self.model_name = "hash"
        self.dim = DIM
        self.device = "cpu"
        self.pooling = "mean"
        self.default_instruction = default_instruction
        self.default_context = default_context
        self._backend = None
        self._tokenizer = None
        self._proj = None
        self.audio_capable = bool(audio)
        self.image_capable = False
        self.video_capable = False
        self.av_capable = False
        self.head = "av"

    def encode(self, texts: List[str], instruction: Optional[str] = None,
               context: Optional[str] = None, batch_size: int = 32) -> np.ndarray:
        ctx = context if context is not None else self.default_context
        out = [_vector(f"{ctx} {t}" if ctx and ctx != "Distributed" else t) for t in texts]
        return np.stack(out).astype(np.float32) if out else np.zeros((0, DIM), np.float32)

    def embed_audio_array(self, wave: np.ndarray, sr: int) -> np.ndarray:
        """A sound lands near the words for its loudness and pitch register,
        so different sounds give different, repeatable queries."""
        if not self.audio_capable:
            raise RuntimeError("this stand-in space has no audio tower")
        x = np.asarray(wave, dtype=np.float32)
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        spec = np.abs(np.fft.rfft(x)) if len(x) else np.zeros(1)
        freq = float(np.argmax(spec)) * sr / max(1, len(x))
        loud = "silence" if rms < 0.01 else "whisper" if rms < 0.1 else "roar"
        pitch = "deep low earth" if freq < 300 else "voice air" if freq < 2000 else "bright sharp fire"
        return _vector(f"{loud} {pitch}")
