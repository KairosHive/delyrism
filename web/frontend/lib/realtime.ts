// Real-time Δ-graph walker — WebSocket client + zustand store.
//
// One socket per built space (opened by `useWalkerSocket`, mounted in the
// Live tab).  The server owns the walker; this store mirrors its latest
// frame and exposes imperative controls.  High-rate inputs (pointer, anchor
// blends, slider drags) are coalesced client-side and flushed at ~30 Hz so a
// fast mouse never floods the socket.
//
// Canvas components should NOT select `frame` through React (that would
// re-render at frame rate) — they read `useWalker.getState().frame` inside
// their own requestAnimationFrame loop.  DOM panels select narrow slices.

import * as React from "react";
import { create } from "zustand";
import { wsUrl } from "./api";
import { useSidebar } from "./store";

export type WalkerMode = "field" | "ring" | "choose";
export type LayoutKind = "umap" | "pca";

export interface LayoutNode { id: string; symbol: string; x: number; y: number; color: string; }
export interface LayoutCentroid { symbol: string; x: number; y: number; color: string; }
export interface LayoutPayload {
  nodes: LayoutNode[];
  centroids: LayoutCentroid[];
  symbols: string[];
  color_map: Record<string, string>;
}

export interface WalkerParams {
  strategy: "gate" | "pooling";
  beta: number;
  gate: "relu" | "cos" | "softmax" | "uniform";
  tau: number;
  within_symbol_softmax: boolean;
  pool_type: "avg" | "max" | "min";
  pool_w: number;
  membership_alpha: number;
  sign: "up" | "down" | "abs";
  temperature: number;
  memory: number;
  diversity: number;
  coherence: number;
  n_candidates: number;
  max_len: number;
  auto_seal: boolean;
  self_weight: number;
  hot_edges: number;
  seed: number;
}

export interface Candidate { i: number; id: string; symbol: string; score: number; p: number; delta: number; }

export interface WalkerSymbol {
  descriptors: string[];
  owners: string[];
  gains: number[];
  composition: Record<string, number>;
  coherence: number;
  context_fit: number;
  nearest: { symbol: string; sim: number }[];
  name: string;
  pairs: number[][];
  index?: number;
  path?: number[];
}

export interface WalkerEvent { event: string; i?: number; id?: string; sealed?: WalkerSymbol; len?: number; }

export interface WalkerFrame {
  type: "frame";
  ctx_point: [number, number] | null;
  has_context: boolean;
  path: number[];
  gains: number[];
  candidates: Candidate[];
  heat: number[];
  hot_edges: [number, number, number][];
  symbol: WalkerSymbol;
  sealed_count: number;
  params: WalkerParams;
  events: WalkerEvent[];
  sealed: WalkerSymbol[];
}

type Status = "idle" | "connecting" | "ready" | "closed" | "error";

interface WalkerState {
  status: Status;
  error: string | null;
  layout: LayoutPayload | null;
  layoutKind: LayoutKind;
  params: WalkerParams | null;
  frame: WalkerFrame | null;
  frameSeq: number;          // bumps per frame — for components that want a cheap "something changed"
  lastEvent: WalkerEvent | null;

  // local UI state (not owned by the server)
  mode: WalkerMode;
  playing: boolean;
  tempo: number;
  bandwidth: number;         // pointer kernel width in layout units
  text: string;
  textWeight: number;
  pointerWeight: number;
  localPointer: [number, number] | null;   // last pointer, for the cursor halo between frames
  anchorWeights: Record<string, number>;   // ring mode blend, for drawing

  _send: ((msg: object) => void) | null;

  // actions
  send: (msg: object) => void;
  setMode: (m: WalkerMode) => void;
  setLayoutKind: (k: LayoutKind) => void;
  setPlaying: (p: boolean) => void;
  setTempo: (t: number) => void;
  setBandwidth: (b: number) => void;
  setParams: (patch: Partial<WalkerParams>) => void;
  pointer: (u: number, v: number) => void;
  pointerOff: () => void;
  anchors: (weights: Record<string, number>) => void;
  setText: (text: string, weight?: number) => void;
  setPointerWeight: (w: number) => void;
  step: (choice?: number) => void;
  undo: () => void;
  reset: () => void;
  seal: (name?: string) => void;
  dropSealed: (index: number) => void;
  clearContext: () => void;
}

// ---- coalescing flush -------------------------------------------------------
// Latest-wins buffers per channel, flushed on a 33 ms timer.
const pending: { pointer?: object; anchors?: object; params?: Record<string, unknown> } = {};
let flushTimer: ReturnType<typeof setTimeout> | null = null;
function scheduleFlush() {
  if (flushTimer) return;
  flushTimer = setTimeout(() => {
    flushTimer = null;
    const send = useWalker.getState()._send;
    if (!send) return;
    if (pending.pointer) { send(pending.pointer); pending.pointer = undefined; }
    if (pending.anchors) { send(pending.anchors); pending.anchors = undefined; }
    if (pending.params) { send({ type: "params", params: pending.params }); pending.params = undefined; }
  }, 33);
}

export const useWalker = create<WalkerState>((set, get) => ({
  status: "idle",
  error: null,
  layout: null,
  layoutKind: "umap",
  params: null,
  frame: null,
  frameSeq: 0,
  lastEvent: null,

  mode: "field",
  playing: false,
  tempo: 2,
  bandwidth: 0.08,
  text: "",
  textWeight: 0.6,
  pointerWeight: 1,
  localPointer: null,
  anchorWeights: {},

  _send: null,

  send: (msg) => { get()._send?.(msg); },
  setMode: (mode) => {
    const s = get();
    // leaving a continuous mode: drop that mode's context source so the
    // next mode starts from the remaining sources only
    if (s.mode === "field" && mode !== "field") s._send?.({ type: "pointer_off" });
    if (s.mode === "ring" && mode !== "ring") s._send?.({ type: "anchors", weights: {} });
    set({ mode, localPointer: null, anchorWeights: {} });
  },
  setLayoutKind: (layoutKind) => set({ layoutKind }),
  setPlaying: (playing) => { set({ playing }); get()._send?.({ type: "play", playing, tempo: get().tempo }); },
  setTempo: (tempo) => { set({ tempo }); get()._send?.({ type: "play", tempo }); },
  setBandwidth: (bandwidth) => set({ bandwidth }),
  setParams: (patch) => {
    const cur = get().params;
    if (cur) set({ params: { ...cur, ...patch } });
    pending.params = { ...(pending.params ?? {}), ...patch };
    scheduleFlush();
  },
  pointer: (u, v) => {
    set({ localPointer: [u, v] });
    pending.pointer = { type: "pointer", u, v, bandwidth: get().bandwidth };
    scheduleFlush();
  },
  pointerOff: () => { set({ localPointer: null }); get()._send?.({ type: "pointer_off" }); },
  anchors: (weights) => {
    set({ anchorWeights: weights });
    pending.anchors = { type: "anchors", weights };
    scheduleFlush();
  },
  setText: (text, weight) => {
    const w = weight ?? get().textWeight;
    set({ text, textWeight: w });
    get()._send?.({ type: "text", text, weight: w });
  },
  setPointerWeight: (w) => {
    set({ pointerWeight: w });
    get()._send?.({ type: "context", pointer_weight: w, anchors_weight: w });
  },
  step: (choice) => get()._send?.(choice === undefined ? { type: "step" } : { type: "step", choice }),
  undo: () => get()._send?.({ type: "undo" }),
  reset: () => get()._send?.({ type: "reset" }),
  seal: (name) => get()._send?.(name ? { type: "seal", name } : { type: "seal" }),
  dropSealed: (index) => get()._send?.({ type: "drop_sealed", index }),
  clearContext: () => {
    const s = get();
    set({ localPointer: null, anchorWeights: {}, text: "" });
    s._send?.({ type: "pointer_off" });
    s._send?.({ type: "anchors", weights: {} });
    s._send?.({ type: "text", text: "" });
  },
}));

/** Open / maintain the walker socket for the current space.  Mount once in
 *  the Live tab.  Reconnects when the space or layout kind changes. */
export function useWalkerSocket() {
  const spaceId = useSidebar((s) => s.spaceId);
  const layoutKind = useWalker((s) => s.layoutKind);

  React.useEffect(() => {
    if (!spaceId) {
      useWalker.setState({ status: "idle", layout: null, frame: null, params: null, _send: null });
      return;
    }
    let closed = false;
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;

    const connect = () => {
      if (closed) return;
      useWalker.setState({ status: "connecting", error: null });
      ws = new WebSocket(wsUrl(`/realtime/ws/${spaceId}?layout=${layoutKind}`));
      const sock = ws;

      sock.onopen = () => {
        attempt = 0;
        useWalker.setState({
          _send: (msg) => { if (sock.readyState === WebSocket.OPEN) sock.send(JSON.stringify(msg)); },
        });
        // re-apply local state so a reconnect restores the session shape
        const s = useWalker.getState();
        if (s.text.trim()) sock.send(JSON.stringify({ type: "text", text: s.text, weight: s.textWeight }));
        if (s.playing) sock.send(JSON.stringify({ type: "play", playing: true, tempo: s.tempo }));
        else sock.send(JSON.stringify({ type: "play", tempo: s.tempo }));
      };
      sock.onmessage = (ev) => {
        let msg: any;
        try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg.type === "ready") {
          useWalker.setState({ status: "ready", layout: msg.layout, params: msg.params });
        } else if (msg.type === "frame") {
          const f = msg as WalkerFrame;
          const last = f.events?.length ? f.events[f.events.length - 1] : null;
          useWalker.setState((st) => ({
            frame: f,
            frameSeq: st.frameSeq + 1,
            params: f.params ?? st.params,
            lastEvent: last ?? st.lastEvent,
          }));
        } else if (msg.type === "error") {
          useWalker.setState({ error: String(msg.detail ?? "error") });
        }
      };
      sock.onerror = () => { useWalker.setState({ status: "error" }); };
      sock.onclose = () => {
        useWalker.setState({ status: closed ? "closed" : "connecting", _send: null });
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
      useWalker.setState({ status: "closed", _send: null, playing: false });
    };
  }, [spaceId, layoutKind]);
}

/** Positions of the archetype anchors in ring mode (unit square). */
export function ringAnchors(symbols: string[]): Record<string, [number, number]> {
  const out: Record<string, [number, number]> = {};
  const n = Math.max(1, symbols.length);
  symbols.forEach((s, k) => {
    const a = -Math.PI / 2 + (2 * Math.PI * k) / n;
    out[s] = [0.5 + 0.4 * Math.cos(a), 0.5 + 0.4 * Math.sin(a)];
  });
  return out;
}

/** Gaussian blend weights of the ring anchors around a pointer. */
export function ringWeights(
  anchors: Record<string, [number, number]>, u: number, v: number, sigma: number,
): Record<string, number> {
  const w: Record<string, number> = {};
  let tot = 0;
  for (const [s, [x, y]] of Object.entries(anchors)) {
    const d2 = (x - u) ** 2 + (y - v) ** 2;
    const g = Math.exp(-d2 / (2 * sigma * sigma));
    w[s] = g; tot += g;
  }
  if (tot < 1e-9) return {};
  for (const k of Object.keys(w)) w[k] = +(w[k] / tot).toFixed(4);
  return w;
}
