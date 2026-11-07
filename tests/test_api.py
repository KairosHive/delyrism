"""Every HTTP route the website calls, on a built space: status and shape."""
from __future__ import annotations

import importlib.util

import pytest

from .conftest import CONTEXT, build_body

HAS_RIPSER = importlib.util.find_spec("ripser") is not None


# ---------- process and catalogue ----------

def test_health(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_api_probe(client):
    assert client.get("/api").json()["name"] == "delyrism-api"


def test_backends_lists_cloudflare_embedders(client):
    ids = [e["id"] for e in client.get("/backends").json()["embedders"]]
    assert "cloudflare-bge-m3" in ids


def test_backends_hide_media_models_when_hosted(client, monkeypatch):
    monkeypatch.setenv("DELYRISM_MEDIA", "0")
    embs = client.get("/backends").json()["embedders"]
    local_media = [e for e in embs if not e.get("api") and (e.get("audio") or e.get("image") or e.get("video"))]
    assert local_media == []                      # API models (Gemini) stay: they need no local model


def test_presets(client):
    names = [p if isinstance(p, str) else p.get("name") for p in client.get("/spaces/presets").json()["presets"]]
    assert "elements" in names and "musical" in names


def test_unknown_preset_is_404(client):
    assert client.get("/spaces/presets/no-such-preset").status_code == 404


# ---------- spaces ----------

def test_build_space(space, symbols):
    assert len(space["space_id"]) == 16
    assert set(space["symbols"]) == set(symbols)
    assert space["embedding_dim"] > 0
    assert set(space["color_map"]) >= set(space["symbols"])


def test_space_id_is_stable(client, symbols, space):
    """The id is a hash of the build config: the lost-space recovery relies on it."""
    again = client.post("/spaces", json=build_body(symbols)).json()
    assert again["space_id"] == space["space_id"]


def test_get_space(client, space):
    r = client.get(f"/spaces/{space['space_id']}")
    assert r.status_code == 200 and r.json()["symbols"] == space["symbols"]


def test_empty_symbols_rejected(client):
    assert client.post("/spaces", json=build_body({})).status_code == 400


def test_cache_stats(client, space):
    assert client.get("/spaces/cache/stats").json()["spaces"] >= 1


# ---------- Explorer ----------

def test_propose(client, space):
    r = client.post("/propose", json={"space_id": space["space_id"], "sentence": CONTEXT, "topk": 4})
    assert r.status_code == 200, r.text
    rows = r.json()["rows"]
    assert len(rows) == 4 and {row["symbol"] for row in rows} <= set(space["symbols"])


def test_attention(client, space):
    sym = space["symbols"][0]
    r = client.post("/attention", json={"space_id": space["space_id"], "symbol": sym, "sentence": CONTEXT})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["descriptors"]) == len(body["weights"]) > 0


def test_ambiguity(client, space):
    r = client.post("/ambiguity", json={"space_id": space["space_id"]})
    assert r.status_code == 200, r.text
    assert len(r.json()["rows"]) == len(space["symbols"])


@pytest.mark.parametrize("method", ["pca", "umap"])
def test_reduce_2d(client, space, method):
    r = client.post("/reduce-2d", json={"space_id": space["space_id"], "method": method})
    assert r.status_code == 200, r.text
    kinds = {p["kind"] for p in r.json()["points"]}
    assert kinds == {"descriptor", "centroid"}


def test_shift(client, space):
    r = client.post("/shift", json={"space_id": space["space_id"], "sentence": CONTEXT, "reducer": "pca"})
    assert r.status_code == 200, r.text


def test_delta_graph_has_edges(client, space):
    r = client.post("/delta-graph", json={"space_id": space["space_id"], "sentence": CONTEXT})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["edges"], "a context should move some relations"
    ids = {n["id"] for n in body["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in body["edges"])


def test_subgraph(client, space):
    r = client.post("/subgraph", json={"space_id": space["space_id"], "sentence": CONTEXT})
    assert r.status_code == 200, r.text
    assert r.json()["nodes"]


def test_similarity(client, space):
    sym = space["symbols"][0]
    r = client.post("/similarity", json={"space_id": space["space_id"], "sentence": CONTEXT, "symbol": sym})
    assert r.status_code == 200, r.text
    body = r.json()
    n = len(body["descriptors"])
    assert len(body["before"]) == len(body["after"]) == n


def test_similarity_symbols(client, space):
    r = client.post("/similarity-symbols", json={"space_id": space["space_id"], "sentence": CONTEXT})
    assert r.status_code == 200, r.text
    assert len(r.json()["symbols"]) == len(space["symbols"])


def test_transformations(client, space):
    r = client.post("/transformations", json={"space_id": space["space_id"], "sentence": CONTEXT})
    assert r.status_code == 200, r.text


def test_global_mode_roundtrip(client, space):
    sid = space["space_id"]
    on = client.post("/global-mode", json={"space_id": sid, "mode": "descriptor", "pcs": 1}).json()
    assert on["mode"] == "descriptor" and on["pcs"] == 1
    assert client.get(f"/global-mode/{sid}").json()["mode"] == "descriptor"
    off = client.post("/global-mode", json={"space_id": sid, "mode": "none", "pcs": 0}).json()
    assert off["mode"] == "none"


# ---------- context ----------

def test_encode_text(client, space):
    r = client.post("/context/encode-text", json={"space_id": space["space_id"], "text": CONTEXT})
    assert r.status_code == 200, r.text


def test_morphing_blend_and_clear(client, space):
    sid = space["space_id"]
    r = client.post("/context/set-morphing-blend",
                    json={"space_id": sid, "sentence_a": "fire", "sentence_b": "water", "blend": 0.3})
    assert r.status_code == 200 and r.json()["active"] is True
    r = client.post("/context/set-override", json={"space_id": sid, "vector": None})
    assert r.status_code == 200


def test_encode_audio_needs_audio_tower(client, space):
    """The hosted text embedders have no audio tower: a readable 4xx, not a crash."""
    wav = _silent_wav()
    r = client.post("/context/encode-audio", data={"space_id": space["space_id"]},
                    files={"file": ("a.wav", wav, "audio/wav")})
    assert 400 <= r.status_code < 500


def _silent_wav(seconds: float = 0.5, sr: int = 16000) -> bytes:
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))
    return buf.getvalue()


# ---------- Topology ----------

topology = pytest.mark.skipif(not HAS_RIPSER, reason="ripser not installed (pip install delyrism[topology])")


@topology
def test_topology_summary(client, space):
    r = client.post("/topology/summary", json={"space_id": space["space_id"]})
    assert r.status_code == 200, r.text


@topology
@pytest.mark.parametrize("route", ["diagrams", "cycles", "catalysts"])
def test_topology_per_symbol(client, space, route):
    r = client.post(f"/topology/{route}", json={"space_id": space["space_id"], "symbol": space["symbols"][0]})
    assert r.status_code == 200, r.text


@topology
def test_topology_synergy(client, space):
    r = client.post("/topology/synergy", json={"space_id": space["space_id"]})
    assert r.status_code == 200, r.text


@topology
def test_topology_pair_cycles(client, space):
    a, b = space["symbols"][:2]
    r = client.post("/topology/pair-cycles", json={"space_id": space["space_id"], "a": a, "b": b})
    assert r.status_code == 200, r.text


# ---------- Story, Builder ----------

def test_story_models(client):
    assert client.get("/story/models").status_code == 200


def test_builder_info(client):
    assert client.get("/builder").status_code == 200


# ---------- a space the server lost (restart, redeploy) ----------

LOST_ROUTES = [
    ("post", "/propose", {"sentence": CONTEXT}),
    ("post", "/delta-graph", {"sentence": CONTEXT}),
    ("post", "/context/encode-text", {"text": CONTEXT}),
    ("post", "/topology/summary", {}),
    ("get", "/global-mode/{sid}", None),
    ("get", "/spaces/{sid}", None),
]


@pytest.mark.parametrize("method,path,extra", LOST_ROUTES)
def test_lost_space_is_404_with_the_detail_the_client_expects(client, symbols, forget_spaces, method, path, extra):
    """web/frontend/lib/api.ts rebuilds on exactly this status and detail."""
    sid = client.post("/spaces", json=build_body(symbols)).json()["space_id"]
    forget_spaces()
    if method == "get":
        r = client.get(path.format(sid=sid))
    else:
        r = client.post(path, json={"space_id": sid, **extra})
    assert r.status_code == 404
    assert "unknown space_id" in r.json()["detail"]
    # the recovery: the same config rebuilds the same id, and the call succeeds
    assert client.post("/spaces", json=build_body(symbols)).json()["space_id"] == sid
    r2 = client.get(path.format(sid=sid)) if method == "get" else client.post(path, json={"space_id": sid, **extra})
    assert r2.status_code == 200, r2.text


# ---------- sound in an audio-capable space (Gemini Embedding 2 online) ----------

@pytest.fixture(scope="module")
def audio_space(client, symbols):
    body = build_body(symbols)
    body["embedder"] = {"backend": "gemini-embedding-2"}
    r = client.post("/spaces", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _tone_wav(freq: float, seconds: float = 1.0, sr: int = 16000, amp: float = 0.3) -> bytes:
    import io, wave
    import numpy as np
    t = np.arange(int(seconds * sr)) / sr
    pcm = (amp * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_encode_audio_in_an_audio_space(client, audio_space):
    """The sidebar's Audio Context: a WAV becomes a context vector (no librosa needed)."""
    r = client.post("/context/encode-audio", data={"space_id": audio_space["space_id"]},
                    files={"file": ("a.wav", _tone_wav(220), "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json()["dim"] > 0


def test_unreadable_audio_is_400(client, audio_space):
    r = client.post("/context/encode-audio", data={"space_id": audio_space["space_id"]},
                    files={"file": ("a.wav", b"not audio at all", "audio/wav")})
    assert r.status_code == 400
