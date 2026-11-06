"""Real-time Δ-graph traversal — WebSocket surface over `delyrism.traversal`.

One `DeltaWalker` per socket.  The client streams *control* (pointer
position, anchor blends, a sentence, parameter tweaks, play/step/seal) and
the server streams *frames* (path, candidates, per-node heat, hot edges,
the symbol under construction).

    GET /realtime/layout/{space_id}?layout=umap|pca   initial draw data
    WS  /realtime/ws/{space_id}?layout=umap|pca       the live channel

Client → server messages (JSON, `type` discriminates):
    pointer   {u, v, bandwidth}        continuous mouse over the layout
    anchors   {weights: {SYMBOL: w}}   archetype-ring blend
    text      {text, weight}           sentence context (embedded once, cached)
    context   {pointer_weight, anchors_weight, text_weight}  mixing weights
    params    {...WalkerParams patch}
    play      {playing: bool, tempo: steps/sec}
    step      {choice?: int}
    undo | reset | seal {name?} | drop_sealed {index}

Server → client: {type:"ready", layout}, {type:"frame", ...}, {type:"event", ...},
{type:"sealed", symbol}, {type:"error", detail}.

Compute is coalesced: pointer messages only mark the context dirty; a pump
coroutine at ~25 Hz recomputes the field once per tick (in a thread, so the
event loop keeps serving other sockets) and emits one frame.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .. import engine_cache
from ..util import to_hex
from delyrism.traversal import DeltaWalker, WalkerParams

router = APIRouter(prefix="/realtime", tags=["realtime"])

PUMP_HZ = 25.0


def _positions(space, layout: str):
    """(descriptor positions, centroid positions) in the requested projection."""
    if layout == "umap":
        try:
            _model, Y0, Yc = engine_cache.get_umap_layout(space)
            return Y0, Yc
        except Exception:
            pass
    cents = np.stack([space.symbol_centroids[s] for s in space.symbols])
    return space._pca.transform(space.D), space._pca.transform(cents)


def _layout_payload(space, walker: DeltaWalker) -> Dict[str, Any]:
    cmap = {k: to_hex(v) for k, v in
            space.get_symbol_color_dict(palette=engine_cache.get_palette(space)).items()}
    lay = walker.layout()
    for n in lay["nodes"]:
        n["color"] = cmap.get(n["symbol"], "#888")
    for c in lay["centroids"]:
        c["color"] = cmap.get(c["symbol"], "#888")
    lay["color_map"] = cmap
    return lay


def _make_walker(space, layout: str) -> DeltaWalker:
    P, PC = _positions(space, layout)
    return DeltaWalker(space, WalkerParams(), positions=P, centroid_positions=PC)


@router.get("/layout/{space_id}")
def layout(space_id: str, layout: str = "umap") -> Dict[str, Any]:
    space = engine_cache.get_space(space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")
    return _layout_payload(space, _make_walker(space, layout))


class _Session:
    """Mutable per-socket state shared between the reader and the pump."""

    def __init__(self, walker: DeltaWalker):
        self.w = walker
        self.lock = asyncio.Lock()
        # context sources — mixed into one vector on every field recompute
        self.v_pointer: Optional[np.ndarray] = None
        self.v_anchors: Optional[np.ndarray] = None
        self.v_text: Optional[np.ndarray] = None
        self.point: Optional[list] = None
        self.mix = {"pointer": 1.0, "anchors": 1.0, "text": 1.0}
        self.ctx_dirty = True
        self.frame_dirty = True
        self.playing = False
        self.tempo = 2.0
        self.last_step = 0.0
        self.events: list = []

    def recompute(self) -> None:
        w = self.w
        vec = w.blend(
            [self.v_pointer, self.v_anchors, self.v_text],
            [self.mix["pointer"], self.mix["anchors"], self.mix["text"]],
        )
        w.set_context(vec, point=self.point)
        w.compute_field()
        self.ctx_dirty = False
        self.frame_dirty = True


@router.websocket("/ws/{space_id}")
async def walker_socket(ws: WebSocket, space_id: str, layout: str = "umap"):
    space = engine_cache.get_space(space_id)
    await ws.accept()
    if space is None:
        await ws.send_json({"type": "error", "detail": "unknown space_id — build a space first"})
        await ws.close(code=4404)
        return

    walker = await asyncio.to_thread(_make_walker, space, layout)
    sess = _Session(walker)
    await ws.send_json({"type": "ready", "layout": _layout_payload(space, walker),
                        "params": walker.params.to_dict()})

    async def handle(msg: Dict[str, Any]) -> None:
        t = msg.get("type")
        w = sess.w
        if t == "pointer":
            u, v = float(msg.get("u", 0.5)), float(msg.get("v", 0.5))
            bw = float(msg.get("bandwidth", 0.08))
            sess.v_pointer = w.context_from_point(u, v, bw)
            sess.point = [u, v]
            sess.ctx_dirty = True
        elif t == "pointer_off":
            sess.v_pointer = None
            sess.point = None
            sess.ctx_dirty = True
        elif t == "anchors":
            weights = msg.get("weights") or {}
            sess.v_anchors = w.context_from_anchors(weights) if weights else None
            sess.ctx_dirty = True
        elif t == "text":
            text = (msg.get("text") or "").strip()
            if "weight" in msg:
                sess.mix["text"] = float(msg["weight"])
            sess.v_text = (await asyncio.to_thread(w.context_from_text, text)) if text else None
            sess.ctx_dirty = True
        elif t == "context":
            for k in ("pointer", "anchors", "text"):
                key = f"{k}_weight"
                if key in msg:
                    sess.mix[k] = float(msg[key])
            sess.ctx_dirty = True
        elif t == "params":
            w.params.update(msg.get("params") or {})
            w._cand_cache = None
            sess.ctx_dirty = True  # shift params may have changed
        elif t == "play":
            if "playing" in msg:
                sess.playing = bool(msg["playing"])
            if "tempo" in msg:
                sess.tempo = max(0.1, float(msg["tempo"]))
            sess.last_step = 0.0
            sess.frame_dirty = True
        elif t == "step":
            if sess.ctx_dirty:
                await asyncio.to_thread(sess.recompute)
            choice = msg.get("choice")
            ev = w.step(None if choice is None else int(choice))
            sess.events.append(ev)
            sess.frame_dirty = True
        elif t == "undo":
            w.undo(); sess.frame_dirty = True
        elif t == "reset":
            w.reset(); sess.frame_dirty = True
        elif t == "seal":
            sym = w.seal(msg.get("name"))
            if sym:
                sess.events.append({"event": "step", "sealed": sym})
            sess.frame_dirty = True
        elif t == "drop_sealed":
            w.drop_sealed(int(msg.get("index", -1))); sess.frame_dirty = True
        elif t == "sealed":
            await ws.send_json({"type": "sealed_list", "sealed": w.sealed})

    async def reader() -> None:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            async with sess.lock:
                try:
                    await handle(msg)
                except Exception as e:  # keep the socket alive on bad input
                    await ws.send_json({"type": "error", "detail": str(e)})

    async def pump() -> None:
        period = 1.0 / PUMP_HZ
        while True:
            t0 = time.perf_counter()
            async with sess.lock:
                if sess.ctx_dirty:
                    await asyncio.to_thread(sess.recompute)
                if sess.playing and (t0 - sess.last_step) >= 1.0 / sess.tempo:
                    ev = sess.w.step()
                    sess.events.append(ev)
                    sess.last_step = t0
                    sess.frame_dirty = True
                if sess.frame_dirty:
                    frame = sess.w.frame()
                    frame["events"] = sess.events[-8:]
                    frame["sealed"] = sess.w.sealed[-12:]
                    sess.events.clear()
                    sess.frame_dirty = False
                    await ws.send_json(frame)
            dt = time.perf_counter() - t0
            await asyncio.sleep(max(0.0, period - dt))

    tasks = [asyncio.create_task(reader()), asyncio.create_task(pump())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for d in done:
            exc = d.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, RuntimeError)):
                print(f"[realtime] socket task failed: {exc!r}")
    finally:
        for t in tasks:
            t.cancel()


# ============================================================================
#  Live Δ-graph socket — context + query, continuously recomputed
# ============================================================================
#
#   WS /realtime/graph/{space_id}
#
#   client → server
#     context {text}                     recompute the Δ-graph for this sentence
#     query   {text}                     re-score node relevance to this sentence
#     params  {...DeltaGraphRequest-ish} top_abs_edges, sign_filter, beta, …
#   server → client
#     graph     {nodes, edges, relevance, context, query}   on context/params change
#     relevance {relevance, query}                           on query change only
#
# Relevance is cosine(D1_i, v_query) over the nodes of the current graph,
# min-max normalised to [0,1].  The client turns that into edge lengths: an
# edge whose two ends are both relevant contracts, the rest stretch — so the
# query *sculpts* the context graph instead of filtering it.

from delyrism import context_delta_graph  # noqa: E402

_GRAPH_DEFAULTS: Dict[str, Any] = dict(
    strategy="gate", beta=1.2, gate="relu", tau=0.3, within_symbol_softmax=True,
    gamma=0.5, prompt_template="{sent}, {desc}", pool_type="avg", pool_w=0.7,
    membership_alpha=0.0, top_abs_edges=30, min_abs_delta=0.01, within_symbol=False,
    connected_only=True, sign_filter="up", sym_filter=None,
)
_SHIFT_KEYS = ("strategy", "beta", "gate", "tau", "within_symbol_softmax", "gamma",
               "prompt_template", "pool_type", "pool_w", "membership_alpha")


def _live_graph(space_id: str, space, context: str, params: Dict[str, Any]) -> Dict[str, Any]:
    cmap = {k: to_hex(v) for k, v in
            space.get_symbol_color_dict(palette=engine_cache.get_palette(space)).items()}
    if not context.strip():
        return {"nodes": [], "edges": [], "idx": {}}
    G = context_delta_graph(
        space, sentence=context, weights=None,
        **{k: params[k] for k in _SHIFT_KEYS},
        top_abs_edges=int(params["top_abs_edges"]),
        min_abs_delta=float(params["min_abs_delta"]),
        within_symbol=bool(params["within_symbol"]),
        connected_only=bool(params["connected_only"]),
        sign_filter=params.get("sign_filter"),
        sym_filter=params.get("sym_filter"),
        only_symbol=None,
    )
    D1 = engine_cache.get_or_compute_shifted_matrix(space_id, space, dict(
        weights=None, sentence=context, **{k: params[k] for k in _SHIFT_KEYS}))
    vctx = space.ctx_vec(sentence=context, ignore_override=True)
    pull = D1 @ vctx
    nodes, idx = [], {}
    for n in G.nodes():
        i = space.desc_idx[n]
        idx[n] = i
        nodes.append({"id": n, "symbol": space.owner.get(n, ""),
                      "color": cmap.get(space.owner.get(n, ""), "#888"),
                      "pull": round(float(pull[i]), 4)})
    edges = [{"source": u, "target": v, "delta": round(float(a["delta"]), 5),
              "sign": a["sign"], "abs_delta": round(float(a["abs_delta"]), 5)}
             for u, v, a in G.edges(data=True)]
    return {"nodes": nodes, "edges": edges, "idx": idx, "D1": D1}


def _relevance_vec(graph: Dict[str, Any], vq: Optional[np.ndarray]) -> Dict[str, float]:
    """cosine(D1_i, v_query) over the graph's nodes, min-max normalised."""
    if vq is None or not graph["nodes"]:
        return {}
    D1 = graph.get("D1")
    if D1 is None:
        return {}
    ids = [n["id"] for n in graph["nodes"]]
    rows = np.asarray([graph["idx"][i] for i in ids])
    r = D1[rows] @ vq
    lo, hi = float(r.min()), float(r.max())
    r = (r - lo) / (hi - lo) if hi - lo > 1e-9 else np.ones_like(r)
    return {i: round(float(x), 4) for i, x in zip(ids, r)}


def _text_vec(space, text: str) -> Optional[np.ndarray]:
    if not text.strip():
        return None
    v = np.asarray(space._sent_vec(text, ignore_override=True), dtype=np.float32)
    return v if float(np.linalg.norm(v)) > 1e-9 else None


_CLAP_SR = 48000


def _audio_vec(space, pcm16: np.ndarray, sr: int) -> np.ndarray:
    """Embed a mic window through the space's audio tower (CLAP / AudioCLIP).
    Resamples to 48 kHz with linear interpolation — plenty for a query."""
    if not getattr(space.embedder, "audio_capable", False) and not hasattr(space.embedder, "_clap"):
        raise RuntimeError("the space's embedder has no audio tower — rebuild it with the CLAP backend")
    wave = pcm16.astype(np.float32) / 32768.0
    if sr != _CLAP_SR and len(wave) > 1:
        n_new = int(round(len(wave) * _CLAP_SR / sr))
        t_old = np.linspace(0.0, 1.0, len(wave), endpoint=False)
        t_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
        wave = np.interp(t_new, t_old, wave).astype(np.float32)
        sr = _CLAP_SR
    v = np.asarray(space.embedder.embed_audio_array(wave, sr), dtype=np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-9)


@router.websocket("/graph/{space_id}")
async def live_graph_socket(ws: WebSocket, space_id: str):
    space = engine_cache.get_space(space_id)
    await ws.accept()
    if space is None:
        await ws.send_json({"type": "error", "detail": "unknown space_id — build a space first"})
        await ws.close(code=4404)
        return

    state: Dict[str, Any] = {
        "context": "", "params": dict(_GRAPH_DEFAULTS),
        "query_text": "", "query_kind": "text",     # text | audio
        "vq": None,                                 # active query vector
        "audio_pending": None,                      # latest (pcm16, sr) not yet embedded
        "audio_smoothing": 0.5,                     # EMA on successive audio vectors
        "graph_dirty": False, "query_dirty": False,
    }
    graph: Dict[str, Any] = {"nodes": [], "edges": [], "idx": {}}
    wake = asyncio.Event()
    await ws.send_json({"type": "ready", "params": state["params"],
                        "audio_capable": bool(getattr(space.embedder, "audio_capable", False) or hasattr(space.embedder, "_clap"))})

    def _rel_msg(rel: Dict[str, float]) -> Dict[str, Any]:
        return {"relevance": rel, "active": state["vq"] is not None,
                "query": state["query_text"], "kind": state["query_kind"]}

    async def reader() -> None:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code", 1000))
            if msg.get("bytes"):
                # binary frame: uint32 LE sample-rate + int16 PCM mono
                buf = msg["bytes"]
                if len(buf) > 4 + 2 * 1000:
                    sr = int(np.frombuffer(buf[:4], dtype="<u4")[0])
                    pcm = np.frombuffer(buf[4:], dtype="<i2")
                    state["audio_pending"] = (pcm, sr)   # latest wins
                    state["query_kind"] = "audio"
                    state["query_dirty"] = True
                    wake.set()
                continue
            raw = msg.get("text")
            if not raw:
                continue
            try:
                m = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = m.get("type")
            if t == "context":
                state["context"] = str(m.get("text") or "")
                state["graph_dirty"] = True
            elif t == "query":
                state["query_text"] = str(m.get("text") or "")
                state["query_kind"] = "text"
                state["audio_pending"] = None
                state["vq"] = await asyncio.to_thread(_text_vec, space, state["query_text"])
                state["query_dirty"] = True
            elif t == "query_clear":
                state["vq"] = None
                state["audio_pending"] = None
                state["query_text"] = ""
                state["query_dirty"] = True
            elif t == "audio_smoothing":
                state["audio_smoothing"] = float(np.clip(float(m.get("value", 0.5)), 0.0, 0.95))
            elif t == "params":
                patch = {k: v for k, v in (m.get("params") or {}).items() if k in _GRAPH_DEFAULTS}
                state["params"].update(patch)
                state["graph_dirty"] = True
            wake.set()

    async def _embed_pending_audio() -> None:
        pend = state["audio_pending"]
        if pend is None:
            return
        state["audio_pending"] = None
        pcm, sr = pend
        v = await asyncio.to_thread(_audio_vec, space, pcm, sr)
        a = state["audio_smoothing"]
        prev = state["vq"] if state["query_kind"] == "audio" else None
        if prev is not None and a > 0:
            v = a * prev + (1.0 - a) * v
            v = v / (float(np.linalg.norm(v)) + 1e-9)
        state["vq"] = v
        state["query_text"] = "(audio)"

    async def worker() -> None:
        nonlocal graph
        while True:
            await wake.wait()
            wake.clear()
            try:
                if state["graph_dirty"]:
                    state["graph_dirty"] = False
                    ctx, params = state["context"], dict(state["params"])
                    graph = await asyncio.to_thread(_live_graph, space_id, space, ctx, params)
                    await _embed_pending_audio()
                    rel = _relevance_vec(graph, state["vq"])
                    state["query_dirty"] = False
                    await ws.send_json({"type": "graph", "nodes": graph["nodes"], "edges": graph["edges"],
                                        "context": ctx, **_rel_msg(rel)})
                elif state["query_dirty"]:
                    state["query_dirty"] = False
                    await _embed_pending_audio()
                    rel = _relevance_vec(graph, state["vq"])
                    await ws.send_json({"type": "relevance", **_rel_msg(rel)})
            except Exception as e:
                await ws.send_json({"type": "error", "detail": str(e)})
            if state["graph_dirty"] or state["query_dirty"]:
                wake.set()  # something arrived while we were computing

    tasks = [asyncio.create_task(reader()), asyncio.create_task(worker())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for d in done:
            exc = d.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, RuntimeError)):
                print(f"[realtime] graph socket failed: {exc!r}")
    finally:
        for t in tasks:
            t.cancel()
