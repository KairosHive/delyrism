"""Speech to text (delyrism/server/transcribe.py): the request sent to
Cloudflare Whisper, and silence never reaching it."""
from __future__ import annotations

import base64
import io
import wave

import numpy as np
import pytest

from delyrism.server import transcribe


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.fixture()
def creds(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.delenv("CLOUDFLARE_GATEWAY_ID", raising=False)


def tone(seconds=1.0, sr=48000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr


def test_request_shape_and_text(creds, monkeypatch):
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, json=json, headers=headers)
        return FakeResponse({"success": True, "result": {"text": " the river speaks "}})
    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    w, sr = tone()
    assert transcribe.transcribe(w, sr, language="en", prompt="earlier words") == "the river speaks"
    assert sent["url"].endswith("/accounts/acc/ai/run/@cf/openai/whisper-large-v3-turbo")
    assert sent["headers"]["Authorization"] == "Bearer tok"
    body = sent["json"]
    assert body["language"] == "en" and body["initial_prompt"] == "earlier words"
    with wave.open(io.BytesIO(base64.b64decode(body["audio"]))) as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1
        assert abs(wf.getnframes() - 16000) < 5           # 1 s resampled from 48 kHz


def test_silence_is_not_sent(creds, monkeypatch):
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("silence reached Whisper"))
    assert transcribe.transcribe(np.zeros(16000, np.float32), 16000) == ""


def test_missing_credentials_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    assert not transcribe.available()
    with pytest.raises(RuntimeError, match="CLOUDFLARE"):
        transcribe.transcribe(*tone())


def test_upstream_error_is_raised(creds, monkeypatch):
    import requests
    bad = FakeResponse({"errors": ["nope"]}); bad.status_code = 400
    monkeypatch.setattr(requests, "post", lambda *a, **k: bad)
    with pytest.raises(RuntimeError, match="400"):
        transcribe.transcribe(*tone())
