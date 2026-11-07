"""EmbeddingGemma 2 (delyrism/embeddinggemma.py), the local audio + text backend: what it hands
sentence-transformers (stubbed, no download), and where the website offers it."""
from __future__ import annotations

import os
import sys
import types
import wave

import numpy as np
import pytest

from delyrism import embeddinggemma


class FakeST:
    """Stands in for sentence_transformers.SentenceTransformer."""
    calls: list = []

    def __init__(self, name, device=None, model_kwargs=None, config_kwargs=None):
        FakeST.init = dict(name=name, device=device, config_kwargs=config_kwargs)

    def encode(self, items, **kw):
        seen = []
        for it in items:
            if "audio" in it:
                with wave.open(it["audio"]) as w:          # the file exists while it is encoded
                    seen.append(("audio", w.getframerate(), w.getnchannels()))
            else:
                seen.append(("text", it["text"]))
        FakeST.calls.append((seen, kw))
        v = np.ones((len(items), 768), np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True) if kw.get("normalize_embeddings") else v


@pytest.fixture()
def fake_st(monkeypatch):
    FakeST.calls = []
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    return FakeST


def test_text_and_audio_calls(fake_st, monkeypatch):
    removed = []
    real_remove = os.remove
    monkeypatch.setattr(embeddinggemma.os, "remove", lambda p: (removed.append(p), real_remove(p)))
    g = embeddinggemma.GemmaEmbedder(device="cpu")
    assert fake_st.init["name"] == "google/embeddinggemma-2"
    assert fake_st.init["config_kwargs"] == {"vision_config": None}        # no vision tower
    t = g.embed_text(["fire", "water"])
    assert t.shape == (2, 768) and np.allclose(np.linalg.norm(t, axis=1), 1)
    assert fake_st.calls[-1][0] == [("text", "fire"), ("text", "water")]
    sr = 48000
    a = g.embed_audio(0.2 * np.sin(2 * np.pi * 330 * np.arange(sr) / sr).astype(np.float32), sr)
    assert a.shape == (768,)
    assert fake_st.calls[-1][0] == [("audio", 16000, 1)]                  # sent as 16 kHz mono
    assert len(removed) == 1 and not os.path.exists(removed[0])          # temp file cleaned up


def test_text_embedder_routes_to_gemma(fake_st):
    from delyrism import TextEmbedder
    e = TextEmbedder(backend="embeddinggemma", device="cpu")
    assert e.audio_capable and e.dim == 768
    assert e.encode(["a", "b"]).shape == (2, 768)
    assert e.embed_audio_array(np.full(1600, 0.1, np.float32), 16000).shape == (768,)


def test_offered_locally_hidden_online(client, monkeypatch):
    monkeypatch.setenv("DELYRISM_MEDIA", "1")
    assert "embeddinggemma-2" in {e["id"] for e in client.get("/backends").json()["embedders"]}
    monkeypatch.setenv("DELYRISM_MEDIA", "0")
    ids = {e["id"] for e in client.get("/backends").json()["embedders"]}
    assert "embeddinggemma-2" not in ids and "gemini-embedding-2" in ids


def test_resolves_to_the_local_engine():
    from delyrism.server.engine_cache import _resolve_backend
    assert _resolve_backend("embeddinggemma-2", None) == ("embeddinggemma", "google/embeddinggemma-2")
    assert _resolve_backend("gemini-embedding-2", None) == ("gemini", "gemini-embedding-2")
