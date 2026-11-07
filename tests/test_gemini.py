"""Gemini Embedding 2 (delyrism/gemini.py): the REST calls for text and
audio, batching, retries, and the backend as the website offers it."""
from __future__ import annotations

import base64
import io
import wave

import numpy as np
import pytest

from delyrism import gemini


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self._payload


@pytest.fixture()
def key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")


@pytest.fixture()
def calls(monkeypatch):
    """Record every POST; answer with one vector per request."""
    import requests
    log = []

    def fake_post(url, json=None, headers=None, timeout=None):
        log.append({"url": url, "json": json, "headers": headers})
        if url.endswith(":batchEmbedContents"):
            n = len(json["requests"])
            return FakeResponse({"embeddings": [{"values": [float(i + 1)] * 4} for i in range(n)]})
        return FakeResponse({"embedding": {"values": [3.0, 0.0, 4.0, 0.0]}})
    monkeypatch.setattr(requests, "post", fake_post)
    return log


def test_text_batches_and_unit_vectors(key, calls):
    g = gemini.GeminiEmbedder()
    v = g.embed_text([f"word {i}" for i in range(150)])
    assert v.shape == (150, 4)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert [len(c["json"]["requests"]) for c in calls] == [100, 50]
    first = calls[0]
    assert first["url"].endswith("/models/gemini-embedding-2:batchEmbedContents")
    assert first["headers"]["x-goog-api-key"] == "g-key"
    req = first["json"]["requests"][0]
    assert req["model"] == "models/gemini-embedding-2"
    assert req["content"]["parts"] == [{"text": "word 0"}]
    assert req["output_dimensionality"] == 768
    assert "task_type" not in req                    # not supported by gemini-embedding-2


def test_audio_is_inline_wav(key, calls):
    g = gemini.GeminiEmbedder()
    t = np.arange(48000) / 48000
    v = g.embed_audio((0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), 48000)
    assert np.allclose(v, [0.6, 0.0, 0.8, 0.0])
    (call,) = calls
    assert call["url"].endswith(":embedContent")
    part = call["json"]["content"]["parts"][0]["inline_data"]
    assert part["mime_type"] == "audio/wav"
    with wave.open(io.BytesIO(base64.b64decode(part["data"]))) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1


def test_rate_limit_is_retried(key, monkeypatch):
    import requests
    answers = iter([FakeResponse({}, 429), FakeResponse({"embedding": {"values": [1.0, 0.0]}})])
    monkeypatch.setattr(requests, "post", lambda *a, **k: next(answers))
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    assert gemini.GeminiEmbedder().embed_audio(np.ones(1600, np.float32) * 0.1, 16000).shape == (2,)


def test_error_is_raised_with_its_status(key, monkeypatch):
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({"error": "bad key"}, 403))
    with pytest.raises(RuntimeError, match="403"):
        gemini.GeminiEmbedder().embed_text(["x"])


def test_no_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert not gemini.available()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini.GeminiEmbedder()


def test_text_embedder_routes_to_gemini(key, calls):
    from delyrism import TextEmbedder
    e = TextEmbedder(backend="gemini")
    assert e.audio_capable and e.dim == 768
    assert e.encode(["a", "b"]).shape == (2, 4)
    assert e.embed_audio_array(np.ones(1600, np.float32) * 0.1, 16000).shape == (4,)


def test_hosted_site_offers_gemini(client, monkeypatch):
    monkeypatch.setenv("DELYRISM_MEDIA", "0")
    ids = {e["id"]: e for e in client.get("/backends").json()["embedders"]}
    assert ids["gemini-embedding-2"]["audio"] is True
    assert "clap" not in ids
