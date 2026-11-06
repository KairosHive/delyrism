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

interface LiveState {
  status: Status;
  error: string | null;
  params: LiveParams | null;
  graph: LiveGraph | null;
  graphSeq: number;              // bumps on every graph payload
  relevance: Record<string, number>;
  relevanceSeq: number;          // bumps on every relevance payload
  queryActive: boolean;
  queryKind: "text" | "audio" | "image";   // what the server's current relevance came from
  audioCapable: boolean;         // the space's embedder has an audio tower (CLAP)
  imageCapable: boolean;         // … an image tower (CLIP)
  pending: number;               // in-flight recomputes (for the status pulse)

  // ----- CLAP query (continuous microphone) / CLIP query (camera) -----
  queryMode: "text" | "clap" | "clip";
  listening: boolean;
  micError: string | null;
  audioLevel: number;            // smoothed RMS 0..~0.5
  audioWindow: number;           // seconds of audio per embedding
  audioInterval: number;         // ms between embeddings
  audioSmoothing: number;        // server-side EMA on successive audio / image vectors
  watching: boolean;
  camError: string | null;
  camInterval: number;           // ms between camera frames
  camVideo: HTMLVideoElement | null;

  // the two live inputs (local echo; the socket gets debounced copies)
  context: string;
  query: string;

  // client-side shaping knobs (never leave the browser)
  focus: number;        // sharpening exponent on relevance
  contraction: number;  // how far overlap edges contract / others stretch
  damping: number;      // viscous drag on node velocity c (1/s) — the single damping term
  stiffness: number;    // spring stiffness k (1/s²)
  threshold: number;    // relevance cut for the "overlap subgraph" readout

  _send: ((msg: object) => void) | null;
  _sendBinary: ((kind: 1 | 2, buf: ArrayBuffer) => void) | null;
  _mic: MicCapture | null;
  _cam: CameraCapture | null;
  setContext: (t: string) => void;
  setQuery: (t: string) => void;
  setParams: (patch: Partial<LiveParams>) => void;
  setQueryMode: (m: "text" | "clap" | "clip") => void;
  startListening: () => Promise<void>;
  stopListening: () => void;
  startWatching: () => Promise<void>;
  stopWatching: () => void;
  setAudioSmoothing: (v: number) => void;
  set: <K extends keyof LiveState>(k: K, v: LiveState[K]) => void;
}

const DEBOUNCE_MS = 180;
let ctxTimer: ReturnType<typeof setTimeout> | null = null;
let qTimer: ReturnType<typeof setTimeout> | null = null;

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
  pending: 0,

  queryMode: "text",
  listening: false,
  micError: null,
  audioLevel: 0,
  audioWindow: 2.5,
  audioInterval: 600,
  audioSmoothing: 0.5,
  watching: false,
  camError: null,
  camInterval: 500,
  camVideo: null,

  context: "",
  query: "",

  focus: 1.6,
  contraction: 0.7,
  damping: 8,
  stiffness: 40,
  threshold: 0.55,

  _send: null,
  _sendBinary: null,
  _mic: null,
  _cam: null,
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
    // leaving a live mode stops its capture; the server is told to drop
    // the old vector so the new mode starts clean
    if (s.queryMode === "clap") s.stopListening();
    if (s.queryMode === "clip") s.stopWatching();
    set({ queryMode });
    if (queryMode === "text") s._send?.({ type: "query", text: s.query });
    else s._send?.({ type: "query_clear" });
  },
  startListening: async () => {
    const s = get();
    if (s._mic) return;
    set({ micError: null });
    try {
      const mic = await startMicCapture({
        windowSec: () => get().audioWindow,
        intervalMs: () => get().audioInterval,
        onWindow: (buf) => get()._sendBinary?.(1, buf),
        onLevel: (rms) => set({ audioLevel: rms }),
      });
      set({ _mic: mic, listening: true });
    } catch (e: any) {
      set({ micError: e?.message ?? String(e), listening: false });
    }
  },
  stopListening: () => {
    const s = get();
    s._mic?.stop();
    set({ _mic: null, listening: false, audioLevel: 0 });
    if (s.queryMode === "clap") s._send?.({ type: "query_clear" });
  },
  startWatching: async () => {
    const s = get();
    if (s._cam) return;
    set({ camError: null });
    try {
      const cam = await startCameraCapture({
        intervalMs: () => get().camInterval,
        onFrame: (buf) => get()._sendBinary?.(2, buf),
      });
      set({ _cam: cam, watching: true, camVideo: cam.video });
    } catch (e: any) {
      set({ camError: e?.message ?? String(e), watching: false });
    }
  },
  stopWatching: () => {
    const s = get();
    s._cam?.stop();
    set({ _cam: null, watching: false, camVideo: null });
    if (s.queryMode === "clip") s._send?.({ type: "query_clear" });
  },
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
        const sendBinary = (kind: 1 | 2, buf: ArrayBuffer) => {
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
        useLive.setState((s) => ({ params: { ...(s.params ?? ({} as LiveParams)), ...seeded } as LiveParams }));
      };
      sock.onmessage = (ev) => {
        let msg: any;
        try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg.type === "ready") {
          useLive.setState((s) => ({ status: "ready", params: { ...msg.params, ...(s.params ?? {}) }, audioCapable: !!msg.audio_capable, imageCapable: !!msg.image_capable }));
        } else if (msg.type === "graph") {
          useLive.setState((s) => ({
            graph: { nodes: msg.nodes, edges: msg.edges, context: msg.context },
            graphSeq: s.graphSeq + 1,
            relevance: msg.relevance ?? {},
            relevanceSeq: s.relevanceSeq + 1,
            queryActive: !!msg.active,
            queryKind: msg.kind === "audio" ? "audio" : msg.kind === "image" ? "image" : "text",
            pending: Math.max(0, s.pending - 1),
          }));
        } else if (msg.type === "relevance") {
          useLive.setState((s) => ({
            relevance: msg.relevance ?? {},
            relevanceSeq: s.relevanceSeq + 1,
            queryActive: !!msg.active,
            queryKind: msg.kind === "audio" ? "audio" : msg.kind === "image" ? "image" : "text",
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
      useLive.getState().stopListening();
      useLive.getState().stopWatching();
      useLive.setState({ status: "closed", _send: null, _sendBinary: null });
    };
  }, [spaceId]);
}
