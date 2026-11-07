"""Live sessions: the analysis of a recording (delyrism/session.py) and the session messages and
endpoints of the Live socket."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from delyrism.session import SessionRecorder, analyse

from .conftest import CONTEXT
from .test_live import recv_until


@pytest.fixture(scope="module")
def synthetic():
    """Five minutes on symbol A, then B, then back to A, in a 4-symbol space."""
    rng = np.random.default_rng(0)
    syms = ["A", "B", "C", "D"]
    desc = [f"{s.lower()}{i}" for s in syms for i in range(10)]
    owners = {d: d[0].upper() for d in desc}
    centres = rng.normal(size=(4, 32))
    D = np.stack([centres[syms.index(owners[d])] + 0.4 * rng.normal(size=32) for d in desc])
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    XY = np.stack([np.r_[np.cos(a), np.sin(a)] * 3 + rng.normal(size=2) * 0.3
                   for a in [syms.index(owners[d]) * 1.57 for d in desc]])
    rec = SessionRecorder("x", desc, owners, syms)
    for t in range(300):
        target = 0 if t < 100 or t >= 200 else 1
        q = centres[target] + 0.5 * rng.normal(size=32)
        rec.snapshot(D @ (q / np.linalg.norm(q)), "text", "q", t=float(t))
        if t in (0, 100, 200):
            rec.event("graph", t=float(t), edges=[["a1", "b2", 0.1], [f"{'ab'[target]}3", "c1", 0.2]])
    rec.ended = 300.0
    return analyse(rec.to_raw(), D, XY), desc


def test_phases_follow_the_session(synthetic):
    res, _ = synthetic
    ph = res["phases"]["phases"]
    assert [p["symbol"] for p in ph] == ["A", "B", "A"]
    assert abs(ph[1]["t0"] - 100) <= 6 and abs(ph[2]["t0"] - 200) <= 6


def test_the_return_to_a_is_found(synthetic):
    res, _ = synthetic
    rets = res["phases"]["returns"]
    assert rets and all(r["t"] >= 200 for r in rets)               # only coming back counts, not staying
    assert any(r["to"] < 100 for r in rets)


def test_territory_and_fingerprint(synthetic):
    res, desc = synthetic
    terr = res["territory"]
    assert terr["symbol_share"]["A"] > terr["symbol_share"]["B"] > 0
    assert terr["symbol_share"]["C"] == 0 and any(d.startswith("c") for d in terr["never"])
    assert 0 < terr["reached"] < 1
    fp = res["fingerprint"]
    assert fp["phases"] == 3 and abs(sum(fp["symbol_share"]) - 1) < 1e-6


def test_trajectory_topology_archetypes(synthetic):
    res, _ = synthetic
    tr = res["trajectory"]
    assert len(tr["path"]) == 300 and tr["dwells"]
    assert all(0 <= p["x"] <= 1 and 0 <= p["y"] <= 1 for p in tr["path"])
    ge = res["topology"]["graph_events"]
    assert [g["born"] for g in ge] == [1, 1] and ge[0]["top_born"] == ["b3 — c1"]
    assert res["topology"]["windows"]
    arche = res["archetypes"]
    assert arche and all(len(v) >= 2 for v in arche.values())


def test_moments_without_a_query_are_valid_json():
    """The raw download is parsed by browsers: no NaN, a moment with no query is null."""
    from delyrism.session import to_json
    rec = SessionRecorder("x", ["a", "b"], {"a": "A", "b": "B"}, ["A", "B"])
    rec.snapshot(None, "text", "", t=0.0)
    rec.snapshot(np.array([0.2, 0.1]), "text", "q", t=1.0)
    rec.snapshot(np.array([0.1, 0.3]), "text", "q", t=2.0)
    rec.ended = 2.0
    raw = rec.to_raw()
    assert raw["rel"][0] is None
    text = to_json({"raw": raw, "result": analyse(raw, np.eye(2), np.array([[0.0, 0.0], [1.0, 1.0]]))})
    assert "NaN" not in text and json.loads(text)["raw"]["rel"][0] is None


def test_empty_session_is_safe():
    rec = SessionRecorder("x", ["a", "b"], {"a": "A", "b": "B"}, ["A", "B"])
    rec.snapshot(None, "text", "")
    rec.stop()
    res = analyse(rec.to_raw(), np.eye(2), np.zeros((2, 2)))
    assert res.get("empty") is True


def test_a_live_session_end_to_end(client, space):
    with client.websocket_connect(f"/realtime/graph/{space['space_id']}") as ws:
        recv_until(ws, "ready")
        ws.send_json({"type": "context", "text": CONTEXT})
        recv_until(ws, "graph")
        ws.send_json({"type": "session_start", "name": "test"})
        assert recv_until(ws, "session")["state"] == "recording"
        ws.send_json({"type": "query", "text": "rain over the river"})
        time.sleep(2.6)
        ws.send_json({"type": "context", "text": "a fire in the night"})
        ws.send_json({"type": "query", "text": "embers and smoke"})
        time.sleep(2.6)
        ws.send_json({"type": "session_stop"})
        done = recv_until(ws, "session_result", limit=400)
    res = done["result"]
    assert res["summary"]["snapshots"] >= 4 and res["summary"]["name"] == "test"
    assert {"trajectory", "territory", "topology", "phases", "fingerprint"} <= set(res)
    assert len(res["trajectory"]["context_path"]) == 2
    raw = client.get(f"/realtime/session/{done['id']}/raw")
    assert raw.status_code == 200 and "attachment" in raw.headers["content-disposition"]
    body = json.loads(raw.text)
    assert len(body["raw"]["rel"][-1]) == len(space["descriptors"])
    assert [e["type"] for e in body["raw"]["events"]].count("query") == 2
    story = client.post(f"/realtime/session/{done['id']}/story", json={})
    assert story.status_code == 200 and story.json()["story"]
    assert done["saved"] and done["saved"].endswith(".json")      # local installs keep a copy


def test_unknown_session_is_404(client):
    assert client.get("/realtime/session/nope").status_code == 404
