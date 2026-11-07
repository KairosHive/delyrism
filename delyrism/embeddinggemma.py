"""EmbeddingGemma 2: text and audio in one embedding space, on the local GPU.

The open, local counterpart of Gemini Embedding 2 (delyrism/gemini.py): a space built with it
embeds its descriptors here, and the Live tab's microphone windows land in the same 768-d
space.  Weights: google/embeddinggemma-2 (Apache 2.0, about 1.5 GB, downloaded on first use);
the vision tower is not loaded.  Needs sentence-transformers >= 6.1 (the `media` extra).

    g = GemmaEmbedder()
    g.embed_text(["fire", "water"])           # (2, 768), unit norm
    g.embed_audio(wave, sr)                   # (768,), unit norm; mono, any rate (sent at 16 kHz)

On 120 GTZAN chunks read through the Musical Modes network it recovered the genre from the
12-symbol profile for 34 % of chunks against 54 % for CLAP, and two 10 s chunks of the same
piece agreed less (0.46 against 0.81).  Its relevance needs centring on the session (the Live
default): raw, one word ("song") topped 116 of 120 chunks on Four Elements.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional, Sequence

import numpy as np

from .wav import to_wav16k

MODEL = "google/embeddinggemma-2"
DIM = 768


class GemmaEmbedder:
    def __init__(self, model: Optional[str] = None, device: Optional[str] = None, batch_size: int = 16):
        import torch
        from sentence_transformers import SentenceTransformer

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # bfloat16 where supported (the model card warns against float16)
        dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = SentenceTransformer(model or MODEL, device=dev, model_kwargs={"torch_dtype": dtype},
                                         config_kwargs={"vision_config": None})
        self.dim = DIM
        self.batch_size = batch_size

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), np.float32)
        v = self.model.encode([{"text": str(t)} for t in texts], batch_size=self.batch_size,
                              normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def embed_audio(self, wave: np.ndarray, sr: int) -> np.ndarray:
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(to_wav16k(wave, sr))
            v = self.model.encode([{"audio": path}], normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False)[0]
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        return np.asarray(v, dtype=np.float32)
