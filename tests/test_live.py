"""The Live tab as it runs online: the WebSocket (/realtime/graph/{space_id})
with context, typed queries and spoken-text queries, and /live/setup."""
from __future__ import annotations

import struct

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from .conftest import CONTEXT


def recv_until(ws, kind: str, limit: int = 60, where=lambda m: True) -> dict:
    """The next message of `kind` matching `where` (other messages are skipped)."""
    for _ in range(limit):
        m = ws.receive_json()
        if m.get("type") == kind and where(m):
            return m
        if m.get("type") == "error":
            raise AssertionError(f"server error: {m.get('detail')}")
    raise AssertionError(f"no {kind!r} message within {limit} messages")


def test_unknown_space_closes_with_4404(client):
    with client.websocket_connect("/realtime/graph/0000000000000000") as ws:
        m = ws.receive_json()
        assert m["type"] == "error" and "unknown space_id" in m["detail"]
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 4404


def test_ready_context_graph_query(client, space):
    with client.websocket_connect(f"/realtime/graph/{space['space_id']}") as ws:
        ready = recv_until(ws, "ready")
        assert "params" in ready

        ws.send_json({"type": "context", "text": CONTEXT})
        graph = recv_until(ws, "graph")
        assert graph["edges"], "the context should open some Δ edges"
        labels = {n["id"] for n in graph["nodes"]}

        ws.send_json({"type": "query", "text": "rain over the river"})
        rel = recv_until(ws, "relevance", where=lambda m: m.get("active"))
        assert rel["kind"] == "text"
        assert set(rel["relevance"]) <= labels
        vals = list(rel["relevance"].values())
        assert vals and all(0.0 <= v <= 1.0 for v in vals)

        ws.send_json({"type": "query_clear"})
        recv_until(ws, "relevance", where=lambda m: not m.get("active"))


def test_transcript_is_a_speech_query(client, space):
    with client.websocket_connect(f"/realtime/graph/{space['space_id']}") as ws:
        recv_until(ws, "ready")
        ws.send_json({"type": "context", "text": CONTEXT})
        recv_until(ws, "graph")
        ws.send_json({"type": "transcript", "text": "the water rises and the fire answers", "tau": 0.2, "hz": 20})
        rel = recv_until(ws, "relevance", where=lambda m: m.get("kind") == "speech" and m.get("active"))
        assert "water" in rel["query"]


def test_live_setup_is_idempotent(client, symbols):
    a = client.post("/live/setup", json={"symbols": symbols})
    assert a.status_code == 200, a.text
    b = client.post("/live/setup", json={"symbols": symbols})
    assert a.json()["space_id"] == b.json()["space_id"]


def test_live_profile_roundtrip(client):
    prof = client.get("/live/profile").json()
    r = client.post("/live/profile", json=prof)
    assert r.status_code == 200, r.text


def speech_chunk(seconds: float = 1.0, sr: int = 16000, freq: float = 220.0) -> bytes:
    """Binary kind 7: uint32 LE sample rate + int16 LE PCM (a tone, so not silence)."""
    t = np.arange(int(seconds * sr)) / sr
    pcm = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    return bytes([7]) + struct.pack("<I", sr) + pcm.tobytes()


def test_ready_says_whether_speech_mode_is_available(client, space):
    with client.websocket_connect(f"/realtime/graph/{space['space_id']}") as ws:
        assert recv_until(ws, "ready")["stt_capable"] is True     # the test stand-in counts


def test_speech_chunks_become_a_rolling_spoken_query(client, space, monkeypatch):
    from delyrism.server import transcribe
    said = iter(["the water rises", "and the fire answers"])
    monkeypatch.setattr(transcribe, "transcribe", lambda wave, sr, language=None, prompt=None: next(said))
    with client.websocket_connect(f"/realtime/graph/{space['space_id']}") as ws:
        recv_until(ws, "ready")
        ws.send_json({"type": "context", "text": CONTEXT})
        recv_until(ws, "graph")
        ws.send_bytes(speech_chunk())
        first = recv_until(ws, "heard")
        assert first["text"] == "the water rises"
        ws.send_bytes(speech_chunk())
        second = recv_until(ws, "heard")
        assert second["transcript"] == "the water rises and the fire answers"
        rel = recv_until(ws, "relevance", where=lambda m: m.get("kind") == "speech" and m.get("active"))
        assert "fire" in rel["query"]


def test_a_failed_chunk_reports_and_the_session_goes_on(client, space, monkeypatch):
    from delyrism.server import transcribe

    def boom(*a, **k):
        raise RuntimeError("upstream down")
    monkeypatch.setattr(transcribe, "transcribe", boom)
    with client.websocket_connect(f"/realtime/graph/{space['space_id']}") as ws:
        recv_until(ws, "ready")
        ws.send_bytes(speech_chunk())
        m = ws.receive_json()
        while m.get("type") != "error":
            m = ws.receive_json()
        assert "speech to text" in m["detail"]
        ws.send_json({"type": "query", "text": "still here"})
        recv_until(ws, "relevance", where=lambda m: m.get("active"))


def mic_window(freq: float, seconds: float = 2.5, sr: int = 16000) -> bytes:
    """Binary kind 1: the mic window for an audio tower."""
    t = np.arange(int(seconds * sr)) / sr
    pcm = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    return bytes([1]) + struct.pack("<I", sr) + pcm.tobytes()


def test_mic_queries_an_audio_space(client, symbols):
    body = {"symbols": symbols, "embedder": {"backend": "gemini-embedding-2"}}
    sid = client.post("/spaces", json=body).json()["space_id"]
    with client.websocket_connect(f"/realtime/graph/{sid}") as ws:
        assert recv_until(ws, "ready")["audio_capable"] is True
        ws.send_json({"type": "context", "text": CONTEXT})
        recv_until(ws, "graph")
        ws.send_bytes(mic_window(150))
        rel = recv_until(ws, "relevance", where=lambda m: m.get("kind") == "audio" and m.get("active"))
        assert rel["relevance"]
