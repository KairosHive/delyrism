"""Gemini Embedding 2: text and audio in one embedding space, over REST.

The API equivalent of CLAP for the hosted site.  A space built with it
embeds its descriptors here, and the Live tab's microphone windows land in
the same space, so sound queries the words directly.

    g = GeminiEmbedder()                      # GEMINI_API_KEY (or GOOGLE_API_KEY)
    g.embed_text(["fire", "water"])           # (2, dim), unit norm
    g.embed_audio(wave, sr)                   # (dim,), unit norm; ≤ 180 s, mono

Pricing (Nov 2026): text $0.20 per million tokens, audio $0.00016 per
second, so a 2.5 s window every 600 ms costs about $2.40 per hour of
continuous listening.  Google notes that audio is optimised for speech.
Docs: https://ai.google.dev/gemini-api/docs/embeddings
"""
from __future__ import annotations

import base64
import os
import time
from typing import List, Optional, Sequence

import numpy as np

from .wav import to_wav16k     # 16 kHz mono WAV keeps the request small

MODEL = "gemini-embedding-2"
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}"
BATCH = 100               # requests per batchEmbedContents call


def api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def available() -> bool:
    return bool(api_key())


def _values(item: dict) -> List[float]:
    if "values" in item:
        return item["values"]
    if "embedding" in item:
        return _values(item["embedding"])
    raise RuntimeError(f"unexpected Gemini embedding payload: {str(item)[:200]}")


def _unit(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


class GeminiEmbedder:
    """Thin REST client; `dim` is the output dimensionality (768 recommended)."""

    def __init__(self, model: Optional[str] = None, dim: int = 768, timeout: float = 60.0):
        self.model = (model or MODEL).removeprefix("models/")
        self.dim = int(dim)
        self.timeout = timeout
        self.key = api_key()
        if not self.key:
            raise RuntimeError("the Gemini embedder needs GEMINI_API_KEY (or GOOGLE_API_KEY)")

    def _post(self, method: str, body: dict, retries: int = 4) -> dict:
        import requests
        url = URL.format(model=self.model, method=method)
        headers = {"x-goog-api-key": self.key, "Content-Type": "application/json"}
        delay = 1.0
        for attempt in range(retries):
            r = requests.post(url, json=body, headers=headers, timeout=self.timeout)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt < retries - 1:
                    time.sleep(delay); delay *= 2
                    continue
            if r.status_code >= 400:
                raise RuntimeError(f"Gemini {method} {r.status_code}: {r.text[:300]}")
            return r.json()
        raise RuntimeError(f"Gemini {method}: no answer after {retries} tries")

    def _request(self, parts: list) -> dict:
        return {"model": f"models/{self.model}", "content": {"parts": parts}, "output_dimensionality": self.dim}

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        out: List[List[float]] = []
        for i in range(0, len(texts), BATCH):
            chunk = [str(t) for t in texts[i:i + BATCH]]
            data = self._post("batchEmbedContents", {"requests": [self._request([{"text": t}]) for t in chunk]})
            embs = data.get("embeddings") or []
            if len(embs) != len(chunk):
                raise RuntimeError(f"Gemini returned {len(embs)} embeddings for {len(chunk)} texts")
            out.extend(_values(e) for e in embs)
        if not out:
            return np.zeros((0, self.dim), np.float32)
        return _unit(np.asarray(out, dtype=np.float32))

    def embed_audio(self, wave: np.ndarray, sr: int) -> np.ndarray:
        data = base64.b64encode(to_wav16k(wave, sr)).decode("ascii")
        body = self._request([{"inline_data": {"mime_type": "audio/wav", "data": data}}])
        res = self._post("embedContent", body)
        item = res["embeddings"][0] if res.get("embeddings") else res
        return _unit(np.asarray(_values(item), dtype=np.float32))
