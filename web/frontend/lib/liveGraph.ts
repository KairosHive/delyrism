// Live Δ-graph — WebSocket client + store for the Live tab.
//
// Two sentences drive everything: the CONTEXT rewrites the descriptor cloud
// (same Δ-graph as the Explorer, recomputed whenever the text settles for
// ~180 ms) and the QUERY scores every node of that graph by cosine to the
// query vector.  The graph component turns relevance into geometry — edge
// lengths, a radial pull, opacity — inside a continuously running damped
// spring simulation, so edits to either box animate instead of re-layout.

import * as React from "react";
import { create } from "zustand";
import { wsUrl } from "./api";
import { useSidebar } from "./store";
import type { DeltaEdge } from "./api";
import { startMicCapture, type MicCapture } from "./liveAudio";
import { startCameraCapture, type CameraCapture } from "./liveCamera";

export interface LiveNode { id: string; symbol: string; color: string; pull: number; }
export interface LiveGraph {
  nodes: LiveNode[];
  edges: DeltaEdge[];
  context: string;
}
export interface LiveParams {
  top_abs_edges: number;
  min_abs_delta: number;
  within_symbol: boolean;
  connected_only: boolean;
  sign_filter: "up" | "down" | null;
  strategy: "gate" | "pooling" | "reembed" | "hybrid";
  beta: number;
  gate: "relu" | "cos" | "softmax" | "uniform";
  tau: number;
  within_symbol_softmax: boolean;
  gamma: number;
  pool_type: "avg" | "max" | "min";
  pool_w: number;
  membership_alpha: number;
}

type Status = "idle" | "connecting" | "ready" | "closed" | "error";
export type QueryKind = "text" | "audio" | "image" | "video" | "av";
export type QueryMode = QueryKind;

interface LiveState {
  status: Status;
  error: string | null;
  params: LiveParams | null;
  graph: LiveGraph | null;
  graphSeq: number;              // bumps on every graph payload
  relevance: Record<string, number>;
  relevanceSeq: number;          // bumps on every relevance payload
  queryActive: boolean;
  queryKind: QueryKind;          // what the server's current relevance came from
  audioCapable: boolean;         // the space's embedder has an audio tower (CLAP / PE-AV)
  imageCapable: boolean;         // … an image tower (CLIP)
  videoCapable: boolean;         // … a video tower (PE-AV, sliding window of frames)
  avCapable: boolean;            // … a joint audio-visual tower (PE-AV)
  pending: number;               // in-flight recomputes (for the status pulse)

  // ----- live media query (mic / camera sliding windows) -----
  queryMode: QueryMode;
  live: boolean;                 // capture + send ticker running
  liveError: string | null;
  audioLevel: number;            // smoothed RMS 0..~0.5
  audioWindow: number;           // seconds of audio per embedding
  videoWindow: number;           // seconds of frames per embedding
  videoFps: number;              // frames per second fed to the model (window × fps = frames)
  sendInterval: number;          // ms between embeddings
  audioSmoothing: number;        // server-side EMA on successive media vectors (before scoring)
  relSmoothing: number;          // server-side EMA on the relevance activations (after scoring)
  camVideo: HTMLVideoElement | null;

  // the two live inputs (local echo; the socket gets debounced copies)
  context: string;
  query: string;

  // client-side shaping knobs (never leave the browser)
  focus: number;        // sharpening exponent on relevance
  contraction: number;  // how far overlap edges contract / others stretch
  damping: number;      // viscous drag on node velocity c (1/s) — the single damping term
  stiffness: number;    // spring stiffness k (1/s²)
  unlinked: number;     // extra repulsion between nodes that share no edge (px³/s²)
  threshold: number;    // relevance cut for the "overlap subgraph" readout

  _send: ((msg: object) => void) | null;
  _sendBinary: ((kind: 1 | 2 | 3 | 4, buf: ArrayBuffer) => void) | null;
  _mic: MicCapture | null;
  _cam: CameraCapture | null;
  _ticker: ReturnType<typeof setInterval> | null;
  setContext: (t: string) => void;
  setQuery: (t: string) => void;
  setParams: (patch: Partial<LiveParams>) => void;
  setQueryMode: (m: QueryMode) => void;
  startLive: () => Promise<void>;
  stopLive: () => void;
  setAudioSmoothing: (v: number) => void;
  setRelSmoothing: (v: number) => void;
  setSendInterval: (v: number) => void;
  set: <K extends keyof LiveState>(k: K, v: LiveState[K]) => void;
}

const DEBOUNCE_MS = 180;
let ctxTimer: ReturnType<typeof setTimeout> | null = null;
let qTimer: ReturnType<typeof setTimeout> | null = null;

const needsMic = (m: QueryMode) => m === "audio" || m === "av";
const needsCam = (m: QueryMode) => m === "image" || m === "video" || m === "av";

/** Start / stop mic + camera so exactly what the current mode needs is running. */
async function ensureCapture(get: () => LiveState, set: (p: Partial<LiveState>) => void) {
  const mode = get().queryMode;
  try {
    if (needsMic(mode) && !get()._mic) {
      const mic = await startMicCapture({ onLevel: (rms) => set({ audioLevel: rms }) });
      set({ _mic: mic });
    }
    if (!needsMic(mode) && get()._mic) { get()._mic!.stop(); set({ _mic: null, audioLevel: 0 }); }
    if (needsCam(mode) && !get()._cam) {
      // the camera is sampled at the model's frame rate, so a window of
      // `videoWindow` seconds holds exactly videoWindow × videoFps frames
      const cam = await startCameraCapture({ fpsMs: () => 1000 / get().videoFps });
      set({ _cam: cam, camVideo: cam.video });
    }
    if (!needsCam(mode) && get()._cam) { get()._cam!.stop(); set({ _cam: null, camVideo: null }); }
  } catch (e: any) {
    set({ liveError: e?.message ?? String(e) });
  }
}

/** Build the binary frame for the current mode from the ring buffers and send it. */
function sendWindow(get: () => LiveState) {
  const s = get();
  if (!s._sendBinary) return;
  if (s.queryMode === "audio") {
    const a = s._mic?.window(s.audioWindow);
    if (a) s._sendBinary(1, a);
  } else if (s.queryMode === "image") {
    const f = s._cam?.latest();
    if (f) s._sendBinary(2, f);
  } else if (s.queryMode === "video") {
    const v = s._cam?.clip(s.videoWindow, videoFrameCount(s));
    if (v) s._sendBinary(3, v);
  } else if (s.queryMode === "av") {
    const a = s._mic?.window(Math.max(s.audioWindow, s.videoWindow));
    const v = s._cam?.clip(s.videoWindow, videoFrameCount(s));
    if (!a || !v) return;
    const out = new Uint8Array(4 + a.byteLength + v.byteLength);
    new DataView(out.buffer).setUint32(0, a.byteLength, true);
    out.set(new Uint8Array(a), 4);
    out.set(new Uint8Array(v), 4 + a.byteLength);
    s._sendBinary(4, out.buffer);
  }
}

export const MAX_VIDEO_FRAMES = 48;
export const videoFrameCount = (s: { videoWindow: number; videoFps: number }) =>
  Math.max(1, Math.min(MAX_VIDEO_FRAMES, Math.round(s.videoWindow * s.videoFps)));

export const useLive = create<LiveState>((set, get) => ({
  status: "idle",
  error: null,
  params: null,
  graph: null,
  graphSeq: 0,
  relevance: {},
  relevanceSeq: 0,
  queryActive: false,
  queryKind: "text",
  audioCapable: false,
  imageCapable: false,
  videoCapable: false,
  avCapable: false,
  pending: 0,

  queryMode: "text",
  live: false,
  liveError: null,
  audioLevel: 0,
  audioWindow: 2.5,
  videoWindow: 3.0,
  videoFps: 4,
  sendInterval: 600,
  audioSmoothing: 0.5,
  relSmoothing: 0.6,
  camVideo: null,

  context: "",
  query: "",

  focus: 1.6,
  contraction: 0.7,
  damping: 8,
  stiffness: 40,
  unlinked: 6000,
  threshold: 0.55,

  _send: null,
  _sendBinary: null,
  _mic: null,
  _cam: null,
  _ticker: null,
  setContext: (context) => {
    set({ context });
    if (ctxTimer) clearTimeout(ctxTimer);
    ctxTimer = setTimeout(() => {
      get()._send?.({ type: "context", text: get().context });
      set((s) => ({ pending: s.pending + 1 }));
    }, DEBOUNCE_MS);
  },
  setQuery: (query) => {
    set({ query });
    if (qTimer) clearTimeout(qTimer);
    qTimer = setTimeout(() => {
      get()._send?.({ type: "query", text: get().query });
    }, DEBOUNCE_MS);
  },
  setQueryMode: (queryMode) => {
    const s = get();
    if (queryMode === s.queryMode) return;
    set({ queryMode });
    if (queryMode === "text") {
      // leaving the live modes stops capture; the server drops the old vector
      s.stopLive();
      s._send?.({ type: "query", text: s.query });
      return;
    }
    s._send?.({ type: "query_clear" });
    // switching between live modes on the fly keeps whatever capture the
    // new mode still needs and starts the rest
    if (s.live) void ensureCapture(get, set);
  },
  startLive: async () => {
    set({ liveError: null });
    await ensureCapture(get, set);
    if (get()._ticker) return;
    const ticker = setInterval(() => sendWindow(get), get().sendInterval);
    set({ _ticker: ticker, live: !!(get()._mic || get()._cam) });
  },
  stopLive: () => {
    const s = get();
    if (s._ticker) clearInterval(s._ticker);
    s._mic?.stop();
    s._cam?.stop();
    set({ _ticker: null, _mic: null, _cam: null, live: false, audioLevel: 0, camVideo: null });
    if (s.queryMode !== "text") s._send?.({ type: "query_clear" });
  },
  setSendInterval: (v) => {
    set({ sendInterval: v });
    const s = get();
    if (s._ticker) {
      clearInterval(s._ticker);
      set({ _ticker: setInterval(() => sendWindow(get), v) });
    }
  },
  setRelSmoothing: (v) => { set({ relSmoothing: v }); get()._send?.({ type: "rel_smoothing", value: v }); },
  setAudioSmoothing: (v) => { set({ audioSmoothing: v }); get()._send?.({ type: "audio_smoothing", value: v }); },
  setParams: (patch) => {
    const cur = get().params;
    if (cur) set({ params: { ...cur, ...patch } });
    get()._send?.({ type: "params", params: patch });
    set((s) => ({ pending: s.pending + 1 }));
  },
  set: (k, v) => set({ [k]: v } as any),
}));

/** Δ-graph params mirrored from the Explorer sidebar, so the live graph
 *  starts out identical to the one in the Explorer tab. */
export function explorerDeltaParams(): Partial<LiveParams> {
  const s = useSidebar.getState();
  return {
    top_abs_edges: s.topAbsEdges,
    min_abs_delta: s.minAbsDelta,
    within_symbol: s.withinSymbolEdges,
    connected_only: s.connectedOnly,
    sign_filter: s.deltaSign === "both" ? null : s.deltaSign,
    strategy: s.strategy,
    beta: s.beta,
    gate: s.gate,
    tau: s.shiftTau,
    within_symbol_softmax: s.withinSymbolSoftmax,
    gamma: s.gamma,
    pool_type: s.poolType,
    pool_w: s.poolW,
    membership_alpha: s.membershipAlpha,
  };
}

const asKind = (k: unknown): QueryKind =>
  k === "audio" || k === "image" || k === "video" || k === "av" ? k : "text";

/** Open / maintain the live graph socket for the current space. */
export function useLiveSocket() {
  const spaceId = useSidebar((s) => s.spaceId);

  React.useEffect(() => {
    if (!spaceId) {
      useLive.setState({ status: "idle", graph: null, relevance: {}, params: null, _send: null });
      return;
    }
    let closed = false;
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;

    const connect = () => {
      if (closed) return;
      useLive.setState({ status: "connecting", error: null });
      const sock = new WebSocket(wsUrl(`/realtime/graph/${spaceId}`));
      ws = sock;
      sock.onopen = () => {
        attempt = 0;
        const send = (msg: object) => { if (sock.readyState === WebSocket.OPEN) sock.send(JSON.stringify(msg)); };
        const sendBinary = (kind: 1 | 2 | 3 | 4, buf: ArrayBuffer) => {
          if (sock.readyState !== WebSocket.OPEN || sock.bufferedAmount > 2_000_000) return;
          const out = new Uint8Array(1 + buf.byteLength);
          out[0] = kind;
          out.set(new Uint8Array(buf), 1);
          sock.send(out.buffer);
        };
        useLive.setState({ _send: send, _sendBinary: sendBinary });
        // seed: Explorer's Δ params + whatever is already typed
        const st = useLive.getState();
        const seeded = explorerDeltaParams();
        send({ type: "params", params: seeded });
        if (st.context.trim()) send({ type: "context", text: st.context });
        if (st.queryMode === "text" && st.query.trim()) send({ type: "query", text: st.query });
        if (st.audioSmoothing !== 0.5) send({ type: "audio_smoothing", value: st.audioSmoothing });
        if (st.relSmoothing > 0) send({ type: "rel_smoothing", value: st.relSmoothing });
        useLive.setState((s) => ({ params: { ...(s.params ?? ({} as LiveParams)), ...seeded } as LiveParams }));
      };
      sock.onmessage = (ev) => {
        let msg: any;
        try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg.type === "ready") {
          useLive.setState((s) => ({ status: "ready", params: { ...msg.params, ...(s.params ?? {}) },
            audioCapable: !!msg.audio_capable, imageCapable: !!msg.image_capable,
            videoCapable: !!msg.video_capable, avCapable: !!msg.av_capable }));
        } else if (msg.type === "graph") {
          useLive.setState((s) => ({
            graph: { nodes: msg.nodes, edges: msg.edges, context: msg.context },
            graphSeq: s.graphSeq + 1,
            relevance: msg.relevance ?? {},
            relevanceSeq: s.relevanceSeq + 1,
            queryActive: !!msg.active,
            queryKind: asKind(msg.kind),
            pending: Math.max(0, s.pending - 1),
          }));
        } else if (msg.type === "relevance") {
          useLive.setState((s) => ({
            relevance: msg.relevance ?? {},
            relevanceSeq: s.relevanceSeq + 1,
            queryActive: !!msg.active,
            queryKind: asKind(msg.kind),
          }));
        } else if (msg.type === "error") {
          useLive.setState((s) => ({ error: String(msg.detail ?? "error"), pending: Math.max(0, s.pending - 1) }));
        }
      };
      sock.onerror = () => useLive.setState({ status: "error" });
      sock.onclose = () => {
        useLive.setState({ status: closed ? "closed" : "connecting", _send: null });
        if (!closed) {
          attempt += 1;
          retry = setTimeout(connect, Math.min(8000, 500 * 2 ** attempt));
        }
      };
    };
    connect();
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      ws?.close();
      useLive.getState().stopLive();
      useLive.setState({ status: "closed", _send: null, _sendBinary: null });
    };
  }, [spaceId]);
}
