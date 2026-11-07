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
#     pose_tick {window, frames}         embed the dancer's recent trajectory now
#     pose_params {tracker, trace, fit, joints, smooth}   how the dance is tracked and drawn
#                                        (tracker: body | hands | both; switching starts a new trajectory)
#     pose_reset                         forget the pose trajectory
#     pose_export {seconds, format}      mp4 of the moving landmarks, or the npz trajectory
#   server → client
#     graph     {nodes, edges, relevance, context, query}   on context/params change
#     relevance {relevance, query}                           on query change only
#     pose      {jpeg, present}          the pose drawing (base64), once per tracked frame
#     pose_export {format, data, seconds}                    base64 file
#
# Relevance is cosine(D1_i, v_query) over the nodes of the current graph,
# min-max normalised to [0,1].  The client turns that into edge lengths: an
# edge whose two ends are both relevant contracts, the rest stretch — so the
# query *sculpts* the context graph instead of filtering it.

from delyrism import context_delta_graph  # noqa: E402
from delyrism.pose import (  # noqa: E402
    PoseStream, PoseStyle, encode_mp4, iter_frames, render_clip, render_frame,
)
from ..live_sim import LiveSim  # noqa: E402
from ..osc_bridge import bridge as osc, frame_messages, mixed, sim_frame  # noqa: E402
from .. import speech  # noqa: E402
from .. import lakota  # noqa: E402
from .. import profile as live_profile  # noqa: E402


# ---- spoken word (TTS → TouchDesigner) — see app/speech.py ------------------

@router.get("/speech")
def speech_status() -> Dict[str, Any]:
    return speech.cfg.status()


@router.post("/speech")
def speech_configure(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Patch the voice: {enabled, delay, url, repeat, mood, mood_threshold, mood_temperature, mood_tau}."""
    return speech.cfg.configure(**cfg)


@router.get("/speech/moods/{space_id}")
def speech_moods(space_id: str) -> Dict[str, Any]:
    """Mood balance across the space's descriptors + the most typical descriptors per mood."""
    space = engine_cache.get_space(space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")
    return speech.mood_report(space)


# ---- OSC output (TouchDesigner) — format documented in app/osc_bridge.py ----

@router.get("/osc")
def osc_status() -> Dict[str, Any]:
    return osc.status()


@router.post("/osc")
def osc_configure(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Patch the OSC target / rate: {host, port, hz, scale, enabled, bundle}."""
    return osc.configure(**cfg)

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


def _relevance_vec(graph: Dict[str, Any], vq: Optional[np.ndarray], D1: Optional[np.ndarray] = None) -> Dict[str, float]:
    """cosine(D1_i, v_query) over the graph's nodes, min-max normalised.
    `D1` defaults to the graph's own shifted matrix; PE-AV queries pass the
    shifted matrix of the head their vector lives in."""
    if vq is None or not graph["nodes"]:
        return {}
    D1 = graph.get("D1") if D1 is None else D1
    if D1 is None:
        return {}
    ids = [n["id"] for n in graph["nodes"]]
    rows = np.asarray([graph["idx"][i] for i in ids])
    r = D1[rows] @ vq
    lo, hi = float(r.min()), float(r.max())
    r = (r - lo) / (hi - lo) if hi - lo > 1e-9 else np.ones_like(r)
    return {i: round(float(x), 4) for i, x in zip(ids, r)}


# ---- PE-AV heads -----------------------------------------------------------
# A PE-AV space keeps one descriptor matrix per text head (av / audio / video).
# Media vectors must be compared inside the head they came from, so the
# context shift is recomputed in that head on demand (cached per context).

def _head_for_kind(space, kind: str) -> Optional[str]:
    heads = getattr(space, "D_heads", None) or {}
    if not heads:
        return None
    return {"audio": "audio", "video": "video", "pose": "video", "av": "av"}.get(kind, "av")


def _shifted_for_head(space_id: str, space, graph: Dict[str, Any], context: str,
                      params: Dict[str, Any], head: Optional[str]) -> Optional[np.ndarray]:
    if head is None or head == getattr(space.embedder, "head", None):
        return graph.get("D1")
    cache = graph.setdefault("D1_heads", {})
    if head not in cache:
        with space.using_head(head):
            cache[head] = engine_cache.get_or_compute_shifted_matrix(
                f"{space_id}:{head}", space,
                dict(weights=None, sentence=context, **{k: params[k] for k in _SHIFT_KEYS}))
    return cache[head]


def _text_vec(space, text: str) -> Optional[np.ndarray]:
    if not text.strip():
        return None
    v = np.asarray(space._sent_vec(text, ignore_override=True), dtype=np.float32)
    return v if float(np.linalg.norm(v)) > 1e-9 else None


_CLAP_SR = 48000


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-9)


def _parse_audio(buf: bytes):
    """uint32 LE sample-rate + int16 LE PCM mono → (float32 wave, sr)."""
    sr = int(np.frombuffer(buf[:4], dtype="<u4")[0])
    pcm = np.frombuffer(buf[4:], dtype="<i2")
    return pcm.astype(np.float32) / 32768.0, sr


def _parse_frames(buf: bytes) -> list:
    """uint8 n + n × (uint32 LE length + JPEG) → list of HxWx3 uint8 arrays."""
    import io
    from PIL import Image
    n = buf[0]
    off, frames = 1, []
    for _ in range(n):
        ln = int(np.frombuffer(buf[off:off + 4], dtype="<u4")[0])
        off += 4
        frames.append(np.asarray(Image.open(io.BytesIO(buf[off:off + ln])).convert("RGB")))
        off += ln
    return frames


def _resample_48k(wave: np.ndarray, sr: int):
    if sr != _CLAP_SR and len(wave) > 1:
        n_new = int(round(len(wave) * _CLAP_SR / sr))
        t_old = np.linspace(0.0, 1.0, len(wave), endpoint=False)
        t_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
        wave = np.interp(t_new, t_old, wave).astype(np.float32)
    return wave, _CLAP_SR


def _image_vec(space, jpeg: bytes) -> np.ndarray:
    """Embed a camera frame through the space's image tower (CLIP)."""
    if not hasattr(space.embedder, "embed_image") or not hasattr(space.embedder, "oc_model"):
        raise RuntimeError("the space's embedder has no image tower — rebuild it with the CLIP backend")
    import io
    from PIL import Image
    return _unit(space.embedder.embed_image(Image.open(io.BytesIO(jpeg))))


def _audio_vec(space, payload: bytes) -> np.ndarray:
    """Embed a mic window through the space's audio tower (CLAP / AudioCLIP / PE-AV)."""
    emb = space.embedder
    if not getattr(emb, "audio_capable", False) and not hasattr(emb, "_clap"):
        raise RuntimeError("the space's embedder has no audio tower — rebuild it with the CLAP or PE-AV backend")
    wave, sr = _resample_48k(*_parse_audio(payload))
    return _unit(emb.embed_audio_array(wave, sr))


def _video_vec(space, payload: bytes) -> np.ndarray:
    """Embed a sliding window of camera frames through the PE-AV video tower."""
    if not getattr(space.embedder, "video_capable", False):
        raise RuntimeError("the space's embedder has no video tower — rebuild it with the PE-AV backend")
    frames = _parse_frames(payload)
    if not frames:
        raise RuntimeError("empty video window")
    return _unit(space.embedder.embed_video_frames(frames))


def _av_vec(space, payload: bytes) -> np.ndarray:
    """uint32 LE audio-length + audio payload + frames payload → joint PE-AV vector."""
    if not getattr(space.embedder, "av_capable", False):
        raise RuntimeError("the space's embedder has no audio-visual tower — rebuild it with the PE-AV backend")
    alen = int(np.frombuffer(payload[:4], dtype="<u4")[0])
    wave, sr = _resample_48k(*_parse_audio(payload[4:4 + alen]))
    frames = _parse_frames(payload[4 + alen:])
    if not frames:
        raise RuntimeError("empty video window")
    return _unit(space.embedder.embed_audio_video(wave, sr, frames))


# ---- pose (delyrism/pose.py) -----------------------------------------------
# Camera frames (kind 5) are tracked into a per-socket PoseStream as they
# arrive, and each tracked frame sends back the drawing the embedder would
# see.  Embedding happens on `pose_tick`, at the client's send rate: the
# recent trajectory is drawn as moving landmarks with traces and goes through
# the video tower as a clip (PE-AV, scored in its video head) or through the
# image tower as a still (CLIP).

_POSE_DEFAULTS: Dict[str, Any] = dict(tracker="body", trace=1.0, fit="camera", joints="extremities", smooth=0.5)
_POSE_SIZE = 224        # the embedders' input side


def _pose_capable() -> bool:
    import importlib.util
    return importlib.util.find_spec("mediapipe") is not None


def _pose_style(opts: Dict[str, Any], size: int) -> PoseStyle:
    return PoseStyle(size=size, square=True, fit=opts["fit"],
                     trace_seconds=float(opts["trace"]), trace_joints=opts["joints"])


def _jpeg_b64(rgb: np.ndarray) -> Optional[str]:
    import base64
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _pose_track(stream: PoseStream, jpeg: bytes, opts: Dict[str, Any]):
    """Track one camera frame; return (person found, base64 drawing)."""
    stream.filter.smooth = float(opts["smooth"])
    found = stream.push_jpeg(jpeg)
    return found, _jpeg_b64(render_frame(stream.trajectory, style=_pose_style(opts, _POSE_SIZE)))


_LANDMARK_KIND = 6      # landmarks tracked in the browser on the GPU (see web/frontend/lib/poseTracker.ts)
_TRACKER_IDS = {0: "body", 1: "hands", 2: "face", 3: "both"}


def _parse_landmarks(buf: bytes):
    """uint8 tracker id · 3 pad · float32 aspect · float64 t (s) · uint32 rows ·
    rows × 7 float32 (x y z visibility wx wy wz) → (tracker, aspect, t, (rows, 7))."""
    tracker = _TRACKER_IDS.get(buf[0], "body")
    aspect = float(np.frombuffer(buf[4:8], dtype="<f4")[0])
    t = float(np.frombuffer(buf[8:16], dtype="<f8")[0])
    rows = int(np.frombuffer(buf[16:20], dtype="<u4")[0])
    data = np.frombuffer(buf[20:20 + rows * 28], dtype="<f4").reshape(rows, 7) if rows else np.zeros((0, 7), np.float32)
    return tracker, aspect, t, data


def _pose_ingest(stream: PoseStream, frames: list, opts: Dict[str, Any], draw: bool):
    """Append browser-tracked landmark frames in order; return (found in the
    newest, base64 drawing or None)."""
    stream.filter.smooth = float(opts["smooth"])
    found = False
    for _, aspect, t, rows in frames:
        found = stream.push_landmarks(rows, t, aspect)
    preview = _jpeg_b64(render_frame(stream.trajectory, style=_pose_style(opts, _POSE_SIZE))) if draw else None
    return found, preview


def _motion_reading(stream: PoseStream, seconds: float, level: str, session, relative: float) -> dict:
    from delyrism.motion import describe
    d = describe(stream.trajectory, seconds, level, session=session, relative=relative).to_dict()
    d["session_samples"] = session.samples
    return d


def _motion_session(state: Dict[str, Any]):
    """The SessionNorm of the current pose stream (a new tracker or stream
    starts a new session)."""
    from delyrism.motion import SessionNorm
    cur = state.get("motion_session")
    if cur is None or cur[0] is not state["pose"]:
        cur = state["motion_session"] = (state["pose"], SessionNorm())
    return cur[1]


def _motion_baseline(space, level: str) -> Optional[np.ndarray]:
    """Mean embedding of reference descriptions spanning the effort space at
    `level`: the direction every movement description shares.  Subtracting
    it from a description's vector leaves what is distinctive about it; on
    CLIP over the Four Elements this takes the rank correlation of node
    relevance between contrasting movements from 0.73 to 0.31 (contrast 0.7)."""
    from delyrism.motion import prototype_texts
    vecs = [v for v in (_text_vec(space, t) for t in prototype_texts(level)) if v is not None]
    return np.mean(vecs, axis=0) if vecs else None


def _pose_vec(space, stream: PoseStream, opts: Dict[str, Any], window: float, frames: int) -> Optional[np.ndarray]:
    """Embed the drawn trajectory, or None when nobody is in frame."""
    emb = space.embedder
    traj = stream.trajectory
    if not traj.present():
        return None
    style = _pose_style(opts, _POSE_SIZE)
    if getattr(emb, "video_capable", False):
        clip = render_clip(traj, seconds=window, fps=frames / window, style=style)
        return _unit(emb.embed_video_frames(list(clip)))
    if not hasattr(emb, "embed_image") or not hasattr(emb, "oc_model"):
        raise RuntimeError("the space's embedder has no video or image tower — rebuild it with PE-AV or CLIP")
    from PIL import Image
    return _unit(emb.embed_image(Image.fromarray(render_frame(traj, style=style))))


def _pose_export(stream: PoseStream, opts: Dict[str, Any], seconds: float, fmt: str) -> bytes:
    traj = stream.trajectory
    if fmt == "npz":
        import io
        buf = io.BytesIO()
        traj.window(seconds).save(buf)
        return buf.getvalue()
    t0, end = traj.span
    return encode_mp4(iter_frames(traj, 25.0, start=max(t0, end - seconds), end=end,
                                  style=_pose_style(opts, 512)), 25.0)


# binary frame kinds (first byte) → (query kind, embed fn, min payload bytes)
_BIN_KINDS = {
    1: ("audio", _audio_vec, 4 + 2 * 1000),
    2: ("image", _image_vec, 100),
    3: ("video", _video_vec, 100),
    4: ("av", _av_vec, 4 + 4 + 2 * 1000 + 100),
}
_POSE_KIND = 5          # JPEG for the pose tracker; embedded on pose_tick, not on arrival
_LABELS = {"audio": "(audio)", "image": "(camera)", "video": "(video)", "av": "(audio + video)", "pose": "(pose)",
           "motion": "(movement)"}


def _lakota_safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # the background keeps its last mosaic
        print(f"[lakota] {e}")
        return None


@router.websocket("/graph/{space_id}")
async def live_graph_socket(ws: WebSocket, space_id: str):
    space = engine_cache.get_space(space_id)
    await ws.accept()
    if space is None:
        await ws.send_json({"type": "error", "detail": "unknown space_id — build a space first"})
        await ws.close(code=4404)
        return

    prof = live_profile.current()
    state: Dict[str, Any] = {
        "context": "", "params": {**_GRAPH_DEFAULTS, **{k: v for k, v in prof.graph.items() if k in _GRAPH_DEFAULTS}},
        "query_text": "", "query_kind": "text",     # text | audio | image | video | av
        "vq": None,                                 # active query vector
        "pending": None,                            # latest (kind, payload) not yet embedded
        "audio_smoothing": 0.5 if prof.audio_smoothing is None else float(prof.audio_smoothing),   # EMA on successive media vectors (before scoring)
        "rel_smoothing": 0.0 if prof.rel_smoothing is None else float(prof.rel_smoothing),         # EMA on the relevance activations (after scoring)
        "rel_prev": None,                           # previous smoothed relevance (per node id)
        "graph_dirty": False, "query_dirty": False,
        "pose": None,                               # PoseStream, made on the first pose frame
        "pose_frame": None,                         # latest camera jpeg not yet tracked
        "pose_lm": [],                              # browser landmark frames not yet appended (all kept)
        "pose_drawn": 0.0,                          # last preview time (one per tracked batch, ≤ 75 per second)
        "pose_opts": {**_POSE_DEFAULTS, **{k: v for k, v in prof.pose.items() if k in _POSE_DEFAULTS}},
        "lm_t": 0.0,                                # perf_counter of the last landmark frame
        "sim_hz": 0.0,                              # JSON stream of the layout to this client (0 = off)
    }
    graph: Dict[str, Any] = {"nodes": [], "edges": [], "idx": {}}
    wake = asyncio.Event()
    pose_wake = asyncio.Event()
    emb = space.embedder
    # server-side twin of the browser's spring layout → OSC (see live_sim.py)
    sim = LiveSim()
    if prof.shape:
        sim.set_shape(dict(prof.shape))
    sim.lakota_names = list(lakota.NAMES)       # uniform background until the first graph
    osc_token = object()
    osc.register(osc_token, lambda: {
        "layout_source": "browser" if sim.client_fresh(time.perf_counter()) else "server-sim",
        "browser_fps": round(sim.client_fps, 1), "awake": sim.awake, "nodes": len(sim.ids),
        "context": state["context"][:60]})
    speak: Dict[str, Any] = {"task": None, "last": None, "top": None, "t": 0.0, "heard": 0.0,
                             "synth": False, "again": False}

    def _speakable() -> bool:
        k = state["query_kind"]
        return speech.cfg.enabled and state["vq"] is not None and (
            k == "text" or (speech.cfg.media and k in speech.cfg.media_kinds))

    async def speak_later() -> None:
        """Debounced voice: `speech.cfg.delay` s after the last text-query edit
        — or, in a movement mode, after the top activated node changed —
        speak the word with the highest query relevance (if it changed)."""
        try:
            await asyncio.sleep(speech.cfg.delay)
            for _ in range(60):           # let the query embedding + relevance settle
                if not state["query_dirty"] and not state["graph_dirty"]:
                    break
                await asyncio.sleep(0.05)
            if not _speakable():
                return
            if state["query_kind"] != "text":
                # movement: never cut the last word off — wait until it has been
                # heard (its voice + 1 s of its reverb), at least min_gap after it began
                hold = max(speech.cfg.min_gap, speak["heard"] + 1.0)
                wait = speak["t"] + hold - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)
                if not _speakable():
                    return
            rel = state.get("rel_prev") or {}
            if not rel:
                return
            word = max(rel, key=rel.get)
            if word == speak["last"] and not speech.cfg.repeat:
                return
            speak["synth"] = True
            # mood sampled from the queried sub-network (see app/speech.py)
            mood = (await asyncio.to_thread(speech.choose_mood, space, dict(rel))
                    if speech.cfg.mood else dict(mood="neutral", tag=None, probs={}, nodes=[]))
            res = await asyncio.to_thread(speech.synthesize, word, mood["tag"])
            if not res:
                return
            speak["last"] = word
            speak["t"] = time.perf_counter()                    # playback starts now
            speak["heard"] = float(res.get("voice_duration") or res.get("duration") or 0.0)
            speech.cfg.last_mood = mood
            if osc.is_owner(osc_token):
                osc.send([mixed("/delyrism/speak", word, res["path"], float(res["duration"]), mood["mood"])])
            await ws.send_json({"type": "spoken", "word": word, "duration": res["duration"],
                                "mood": mood["mood"], "mood_probs": mood["probs"]})
        except asyncio.CancelledError:
            pass
        finally:
            speak["synth"] = False
            if speak["again"]:              # the top node moved on while this word was synthesised
                speak["again"] = False
                asyncio.get_running_loop().call_soon(schedule_speak)

    def schedule_speak() -> None:
        if speak["task"] is not None:
            speak["task"].cancel()
        speak["task"] = asyncio.create_task(speak_later())

    def media_top_changed(rel: Dict[str, float]) -> None:
        """Movement modes: (re)start the voice timer when the top node changes."""
        if not rel or not speech.cfg.media or state["query_kind"] not in speech.cfg.media_kinds:
            return
        top = max(rel, key=rel.get)
        if top != speak["top"]:
            speak["top"] = top
            if speak["synth"]:              # don't drop the word being synthesised; look again after
                speak["again"] = True
                return
            if top != speak["last"] or speech.cfg.repeat:
                schedule_speak()
            elif speak["task"] is not None:
                speak["task"].cancel()      # back to the word just spoken: nothing new to say

    if speech.cfg.enabled and speech.cfg.mood:              # embed the mood anchors now, not at the first word
        asyncio.create_task(asyncio.to_thread(speech.mood_matrix, space))
    asyncio.create_task(asyncio.to_thread(_lakota_safe, lakota.affinity, space))   # Lakota prototypes, once per space

    await ws.send_json({"type": "ready", "params": state["params"], "shape": dict(sim.shape),
                        "pose_opts": dict(state["pose_opts"]), "symbols": list(space.symbols),
                        "motion": prof.motion.model_dump(),
                        "audio_capable": bool(getattr(emb, "audio_capable", False) or hasattr(emb, "_clap")),
                        "image_capable": bool(getattr(emb, "image_capable", False) or hasattr(emb, "oc_model")),
                        "video_capable": bool(getattr(emb, "video_capable", False)),
                        "av_capable": bool(getattr(emb, "av_capable", False)),
                        "pose_capable": _pose_capable()})

    def _rel_msg(rel: Dict[str, float]) -> Dict[str, Any]:
        return {"relevance": rel, "active": state["vq"] is not None,
                "query": state["query_text"], "kind": state["query_kind"]}

    def _score() -> Dict[str, float]:
        """Relevance of the active query inside the right head, then the
        post-embedding EMA on the activations (live media kinds only)."""
        head = _head_for_kind(space, state["query_kind"])
        D1 = _shifted_for_head(space_id, space, graph, state["context"], state["params"], head)
        rel = _relevance_vec(graph, state["vq"], D1)
        a = state["rel_smoothing"]
        prev = state["rel_prev"]
        if rel and state["query_kind"] != "text" and a > 0 and prev:
            rel = {k: round(a * prev.get(k, v) + (1.0 - a) * v, 4) for k, v in rel.items()}
        state["rel_prev"] = rel if rel else None
        return rel

    def _lakota_shares(nodes: bool):
        """Lakota background (app/lakota.py): shares of the graph's nodes and of a
        text / movement query.  None if the embedder can't provide them."""
        def run():
            rows = lakota.node_rows(space, [n["id"] for n in graph["nodes"]]) if nodes else {}
            q = (lakota.query_shares(space, state["vq"])
                 if state["vq"] is not None and state["query_kind"] in ("text", "motion") else None)
            return rows, q
        return _lakota_safe(run)

    async def _lakota_update(nodes: bool) -> None:
        res = await asyncio.to_thread(_lakota_shares, nodes)
        if res is not None:
            sim.set_lakota(lakota.NAMES, res[0], res[1])

    async def reader() -> None:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code", 1000))
            if msg.get("bytes"):
                osc.claim(osc_token)   # the socket being played streams the OSC
                # binary frame, first byte = kind:
                #   0x01 audio  → uint32 LE sample-rate + int16 PCM mono
                #   0x02 image  → JPEG bytes
                #   0x03 video  → uint8 n + n × (uint32 LE len + JPEG)
                #   0x04 av     → uint32 LE audio-len + audio payload + video payload
                #   0x05 pose   → JPEG, tracked into the pose trajectory on the server
                #   0x06 marks  → landmarks tracked in the browser (see _parse_landmarks)
                buf = msg["bytes"]
                if buf and buf[0] == _POSE_KIND and len(buf) > 100:
                    state["pose_frame"] = bytes(buf[1:])   # latest wins; pose_loop tracks it
                    pose_wake.set()
                    continue
                if buf and buf[0] == _LANDMARK_KIND and len(buf) >= 21:
                    state["lm_t"] = time.perf_counter()
                    state["pose_lm"].append(_parse_landmarks(bytes(buf[1:])))
                    del state["pose_lm"][:-120]            # never more than ~4 s behind
                    pose_wake.set()
                    continue
                spec = _BIN_KINDS.get(buf[0] if buf else 0)
                if spec and len(buf) - 1 >= spec[2]:
                    state["pending"] = (spec[0], bytes(buf[1:]))   # latest wins
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
            if t == "sim":
                # subscribe to the server's layout as JSON frames (the preview of
                # what TouchDesigner draws); hz 0 = off.  Not an input either.
                state["sim_hz"] = float(np.clip(float(m.get("hz") or 0.0), 0.0, 60.0))
                continue
            if t == "layout":
                # the browser's own node positions — what the user actually sees.
                # Not an input, so it does not claim the OSC output.
                sim.sync_from_client(m.get("ids") or [], m.get("xy") or [], m.get("v") or [],
                                     bool(m.get("awake", True)), float(m.get("fps") or 0.0))
                continue
            osc.claim(osc_token)   # the socket being played streams the OSC
            if t == "context":
                state["context"] = str(m.get("text") or "")
                state["graph_dirty"] = True
            elif t == "query":
                state["query_text"] = str(m.get("text") or "")
                state["query_kind"] = "text"
                state["pending"] = None
                state["rel_prev"] = None
                state["vq"] = await asyncio.to_thread(_text_vec, space, state["query_text"])
                state["query_dirty"] = True
                schedule_speak()          # restarts the 1 s timer on every edit
            elif t == "query_clear":
                if speak["task"] is not None:
                    speak["task"].cancel()
                speak["last"] = speak["top"] = None
                state["vq"] = None
                state["pending"] = None
                state["rel_prev"] = None
                state["query_text"] = ""
                state["query_dirty"] = True
            elif t == "audio_smoothing":
                state["audio_smoothing"] = float(np.clip(float(m.get("value", 0.5)), 0.0, 0.95))
            elif t == "rel_smoothing":
                state["rel_smoothing"] = float(np.clip(float(m.get("value", 0.0)), 0.0, 0.97))
            elif t == "pose_tick":
                if state["pose"] is not None:
                    window = float(np.clip(float(m.get("window") or 2.0), 0.5, 10.0))
                    frames = int(np.clip(int(m.get("frames") or 16), 1, 48))
                    state["pending"] = ("pose", (window, frames))   # latest wins
                    state["query_dirty"] = True
            elif t == "motion_tick":
                if state["pose"] is not None:
                    level = m.get("level") if m.get("level") in ("full", "abstract", "qualities", "body") else "full"
                    rel = m.get("relative")
                    state["pending"] = ("motion", (float(np.clip(float(m.get("window") or 3.0), 1.0, 10.0)), level,
                                                   float(np.clip(float(m.get("contrast") or 0.0), 0.0, 1.0)),
                                                   1.0 if rel is None else float(np.clip(float(rel), 0.0, 1.0))))
                    state["query_dirty"] = True
            elif t == "motion_reset":
                # forget the session's range (a new dancer, a new piece)
                if state.get("motion_session") is not None:
                    state["motion_session"][1].reset()
                continue
            elif t == "pose_params":
                opts = state["pose_opts"]
                for k in _POSE_DEFAULTS:
                    if m.get(k) is not None:
                        opts[k] = m[k]
                if opts["tracker"] not in ("body", "hands", "face", "both"):
                    opts["tracker"] = "body"
                old = state["pose"]
                if old is not None and old.kind != opts["tracker"]:
                    # a different landmark layout: drop the stream, the next frame builds the new one
                    state["pose"] = None
                    await asyncio.to_thread(old.close)
            elif t == "pose_reset":
                if state["pose"] is not None:
                    state["pose"].trajectory.clear()
            elif t == "pose_export":
                if state["pose"] is None or len(state["pose"].trajectory) == 0:
                    await ws.send_json({"type": "error", "detail": "no pose trajectory yet — start the pose camera first"})
                    continue
                import base64
                fmt = "npz" if m.get("format") == "npz" else "mp4"
                seconds = float(np.clip(float(m.get("seconds") or 10.0), 1.0, 30.0))
                data = await asyncio.to_thread(_pose_export, state["pose"], dict(state["pose_opts"]), seconds, fmt)
                await ws.send_json({"type": "pose_export", "format": fmt, "seconds": seconds,
                                    "data": base64.b64encode(data).decode("ascii")})
            elif t == "shape":
                # the browser's layout knobs, mirrored into the OSC sim
                sim.set_shape(m.get("shape") or {})
                continue
            elif t == "params":
                patch = {k: v for k, v in (m.get("params") or {}).items() if k in _GRAPH_DEFAULTS}
                state["params"].update(patch)
                state["graph_dirty"] = True
            wake.set()

    async def _embed_pending() -> None:
        """Embed the latest queued media window (latest wins, one at a time)
        and EMA-blend it into the active query vector."""
        pend = state["pending"]
        if pend is None:
            return
        state["pending"] = None
        kind, payload = pend
        label = None
        if kind == "pose":
            if state["pose"] is None:
                return          # tracker switched, the new stream has no frames yet
            v = await asyncio.to_thread(_pose_vec, space, state["pose"], dict(state["pose_opts"]), *payload)
            if v is None:
                return          # nobody in frame: hold the last query
        elif kind == "motion":
            # movement → words → the text tower: works with every backend
            if state["pose"] is None:
                return
            window, level, contrast, relative = payload
            reading = await asyncio.to_thread(_motion_reading, state["pose"], window, level,
                                              _motion_session(state), relative)
            await ws.send_json({"type": "motion", "level": level, **reading})
            if not reading["text"]:
                return          # nothing tracked or nothing moving enough to say: hold the last query
            v = await asyncio.to_thread(_text_vec, space, reading["text"])
            if v is None:
                return
            if contrast > 0:
                bases = state.setdefault("motion_base", {})
                if level not in bases:
                    bases[level] = await asyncio.to_thread(_motion_baseline, space, level)
                if bases[level] is not None:
                    v = _unit(v - contrast * bases[level])
            label = reading["text"]
        else:
            fn = next(f for k, f, _ in _BIN_KINDS.values() if k == kind)
            v = await asyncio.to_thread(fn, space, payload)
        a = state["audio_smoothing"]
        prev = state["vq"] if state["query_kind"] == kind else None
        # each movement description already sums up a whole window, so it
        # replaces the query outright instead of blending into the last one
        if prev is not None and a > 0 and kind != "motion":
            v = _unit(a * prev + (1.0 - a) * v)
        if state["query_kind"] != kind:
            state["rel_prev"] = None
        state["vq"] = v
        state["query_kind"] = kind
        state["query_text"] = label or _LABELS.get(kind, "")

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
                    await _embed_pending()
                    rel = await asyncio.to_thread(_score)
                    state["query_dirty"] = False
                    sim.set_graph(graph["nodes"], graph["edges"], list(space.symbols))
                    sim.set_relevance(rel, state["vq"] is not None)
                    await _lakota_update(nodes=True)
                    await ws.send_json({"type": "graph", "nodes": graph["nodes"], "edges": graph["edges"],
                                        "context": ctx, **_rel_msg(rel)})
                    media_top_changed(rel)
                elif state["query_dirty"]:
                    state["query_dirty"] = False
                    await _embed_pending()
                    rel = await asyncio.to_thread(_score)
                    sim.set_relevance(rel, state["vq"] is not None)
                    await _lakota_update(nodes=False)
                    await ws.send_json({"type": "relevance", **_rel_msg(rel)})
                    media_top_changed(rel)
            except Exception as e:
                await ws.send_json({"type": "error", "detail": str(e)})
            if state["graph_dirty"] or state["query_dirty"]:
                wake.set()  # something arrived while we were computing

    async def pose_loop() -> None:
        """Track pose frames at camera rate, apart from the worker, so the
        trajectory stays dense and the drawing stays fluid while an
        embedding runs."""
        while True:
            await pose_wake.wait()
            pose_wake.clear()
            marks, state["pose_lm"] = state["pose_lm"], []
            if marks:
                kind = marks[-1][0]
                if state["pose"] is None or state["pose"].kind != kind:
                    old = state["pose"]
                    state["pose"] = PoseStream(smooth=float(state["pose_opts"]["smooth"]), tracker=kind)
                    if old is not None:
                        await asyncio.to_thread(old.close)
                marks = [f for f in marks if f[0] == kind]
                now = time.monotonic()
                draw = now - state["pose_drawn"] >= 1.0 / 75.0   # every tracked batch, up to 75 Hz (≈ 3 ms each)
                try:
                    found, preview = await asyncio.to_thread(_pose_ingest, state["pose"], marks,
                                                             dict(state["pose_opts"]), draw)
                except Exception as e:
                    await ws.send_json({"type": "error", "detail": f"pose landmarks failed: {e}"})
                    continue
                if draw:
                    state["pose_drawn"] = now
                    await ws.send_json({"type": "pose", "jpeg": preview, "present": found})
                continue
            jpeg, state["pose_frame"] = state["pose_frame"], None
            if jpeg is None:
                continue
            if state["pose"] is None:
                state["pose"] = PoseStream(smooth=float(state["pose_opts"]["smooth"]),
                                           tracker=state["pose_opts"]["tracker"])
            try:
                found, preview = await asyncio.to_thread(_pose_track, state["pose"], jpeg, dict(state["pose_opts"]))
            except Exception as e:
                await ws.send_json({"type": "error", "detail": f"pose tracking failed: {e}"})
                continue
            await ws.send_json({"type": "pose", "jpeg": preview, "present": found})

    async def osc_pump() -> None:
        """Step the layout at the OSC rate and stream it while this socket owns
        the output.  Full rate while anything moves or fades, a 4 Hz heartbeat
        once settled (so a TouchDesigner started late still gets the state)."""
        seq = 0
        t_start = last = time.perf_counter()
        last_send = last_labels = 0.0
        seen = (-1, -1)                       # (order_version, owner_epoch) at last label send
        last_sim = last_motion = 0.0
        sim_seq = 0
        mot = prof.motion
        while True:
            now = time.perf_counter()
            # server-side movement cadence (profile.motion.auto): describe the
            # last window every `refresh` s while landmarks are streaming
            if (mot.auto and state["pose"] is not None and now - state["lm_t"] < 1.0
                    and now - last_motion >= mot.refresh):
                last_motion = now
                state["pending"] = ("motion", (float(mot.window), mot.level, float(mot.contrast), float(mot.relative)))
                state["query_dirty"] = True
                wake.set()
            if state["sim_hz"] > 0 and now - last_sim >= 1.0 / state["sim_hz"]:
                if sim.awake or sim.fading or now - last_sim >= 0.25:
                    last_sim = now
                    await ws.send_json(sim_frame(sim, sim_seq, now - t_start, osc.scale))
                    sim_seq += 1
            # while the browser streams its layout, mirror it (physics paused,
            # fades still run); once it goes quiet, continue from its last state
            sim.step(now - last, physics=not sim.client_fresh(now))
            last = now
            if osc.is_owner(osc_token):
                moving = sim.awake or sim.fading
                if moving or now - last_send >= 0.25:
                    key = (sim.order_version, osc.owner_epoch)
                    labels = key != seen or now - last_labels >= 1.0
                    meta = {"context": state["context"], "query": state["query_text"],
                            "kind": state["query_kind"] if state["vq"] is not None else ""}
                    osc.send(frame_messages(sim, seq, now - t_start, osc.scale, meta, labels))
                    seq += 1
                    last_send = now
                    if labels:
                        seen, last_labels = key, now
            await asyncio.sleep(max(0.0, 1.0 / osc.hz - (time.perf_counter() - now)))

    tasks = [asyncio.create_task(reader()), asyncio.create_task(worker()),
             asyncio.create_task(pose_loop()), asyncio.create_task(osc_pump())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for d in done:
            exc = d.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, RuntimeError)):
                print(f"[realtime] graph socket failed: {exc!r}")
    finally:
        for t in tasks:
            t.cancel()
        if speak["task"] is not None:
            speak["task"].cancel()
        osc.unregister(osc_token)
        if state["pose"] is not None:
            await asyncio.to_thread(state["pose"].close)
