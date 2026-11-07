"""The Live tab as it runs online: the WebSocket (/realtime/graph/{space_id})
with context, typed queries and spoken-text queries, and /live/setup."""
from __future__ import annotations

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
