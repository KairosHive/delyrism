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
