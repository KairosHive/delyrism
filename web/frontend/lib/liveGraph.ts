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
import { createTracker, packLandmarks, type Tracker } from "./poseTracker";

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
export interface TrackStats {
  camFps: number;       // new frames per second from the camera
  trackFps: number;     // frames tracked per second (target: poseFps)
  detectMs: number;     // MediaPipe time per tracked frame
  previewFps: number;   // pose drawings back from the server per second
}

export type QueryKind = "text" | "audio" | "image" | "video" | "av" | "pose" | "motion";
export type QueryMode = QueryKind;
const tracks = (m: QueryMode) => m === "pose" || m === "motion";   // modes fed by the landmark tracker

/** How the dance is tracked and drawn for the embedder (server-side, see delyrism/pose.py). */
export type PoseTracker = "body" | "hands" | "face" | "both";
export interface PoseOpts {
  tracker: PoseTracker;                   // which MediaPipe model: full body · hands with fingers · face · body + hands
  trace: number;                          // seconds of temporal trace behind each traced joint
  fit: "camera" | "window" | "body";      // framing: source camera · follow the movement · centred on the body (hands)
  joints: string;                         // which joints leave trails — a set from POSE_TRAILS[tracker]
  smooth: number;                         // landmark smoothing 0 (raw) .. 1 (heavy), One Euro filter
}

/** Trail sets per tracker (names match delyrism/pose.py Skeleton.joint_sets); first is the default. */
export const POSE_TRAILS: Record<PoseTracker, { value: string; label: string }[]> = {
  body: [
    { value: "extremities", label: "head, wrists, ankles" }, { value: "wrists", label: "wrists" },
    { value: "feet", label: "ankles" }, { value: "limbs", label: "elbows to ankles" }, { value: "all", label: "every joint" },
  ],
  hands: [
    { value: "fingertips", label: "fingertips" }, { value: "index", label: "index fingertips" },
    { value: "wrists", label: "wrists" }, { value: "all", label: "every joint" },
  ],
  face: [
    { value: "features", label: "nose, chin, mouth, brows" }, { value: "mouth", label: "mouth" },
    { value: "eyes", label: "irises (gaze)" }, { value: "nose", label: "nose tip" }, { value: "all", label: "every contour point" },
  ],
  both: [
    { value: "extremities", label: "head, wrists, ankles" }, { value: "fingertips", label: "fingertips" },
    { value: "limbs", label: "elbows to ankles" }, { value: "all", label: "every joint" },
  ],
};

/** One movement quality as measured by delyrism/motion.py (value 0 = low pole, 1 = high pole). */
export interface MotionQuality { key: string; value: number | null; raw: number | null; unit: string; low: string; high: string; }
export interface MotionReading {
  tracker: string; seconds: number; coverage: number;
  qualities: MotionQuality[]; hands: MotionQuality[];
  trends: Record<string, number | null>; text: string;
}

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
  poseCapable: boolean;          // the server has MediaPipe for pose tracking
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
  poseFps: number;               // camera frames tracked per second in pose / movement mode
  poseOpts: PoseOpts;
  posePreview: string | null;    // data URL of the latest pose drawing
  posePresent: boolean;          // a dancer was found in the latest tracked frame
  trackerState: "off" | "loading" | "GPU" | "CPU" | "error";   // the browser landmark tracker
  trackStats: TrackStats | null;  // measured rates while tracking (2 Hz)
  motionWindow: number;          // seconds of movement per description
  motionContrast: number;        // 0..1, how much of the shared "movement description" direction is removed
  motionRefresh: number;         // seconds between descriptions (each one replaces the query)
  motion: MotionReading | null;  // the latest description from the server
  motionApplied: { text: string; at: number } | null;   // the description the query last took, and when

  // the two live inputs (local echo; the socket gets debounced copies)
  context: string;
  query: string;

  // layout shaping knobs — drive the browser's SpringSim and are mirrored to
  // the server (`shape` messages) for its twin sim, which feeds the OSC output
  focus: number;        // sharpening exponent on relevance
  contraction: number;  // how far overlap edges contract / others stretch
  damping: number;      // viscous drag on node velocity c (1/s) — the single damping term
  stiffness: number;    // spring stiffness k (1/s²)
  repulsion: number;    // base pair repulsion between all nodes (px³/s²)
  unlinked: number;     // extra repulsion between nodes that share no edge (px³/s²)
  threshold: number;    // relevance cut for the "overlap subgraph" readout

  _send: ((msg: object) => void) | null;
  _sendBinary: ((kind: 1 | 2 | 3 | 4 | 5 | 6, buf: ArrayBuffer) => void) | null;
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
  setMotionRefresh: (v: number) => void;
  setPoseOpts: (patch: Partial<PoseOpts>) => void;
  resetPose: () => void;
  exportPose: (format: "mp4" | "npz", seconds: number) => void;
  set: <K extends keyof LiveState>(k: K, v: LiveState[K]) => void;
}

const DEBOUNCE_MS = 180;
let ctxTimer: ReturnType<typeof setTimeout> | null = null;
let qTimer: ReturnType<typeof setTimeout> | null = null;

const needsMic = (m: QueryMode) => m === "audio" || m === "av";
const needsCam = (m: QueryMode) => m === "image" || m === "video" || m === "av" || tracks(m);

// ---- browser landmark tracker (pose / movement modes) ----------------------
// MediaPipe runs here on the GPU (see poseTracker.ts).  It is driven by the
// camera itself (requestVideoFrameCallback: once per new camera frame, so no
// frame is tracked twice and none is skipped by display-rate jitter), paced
// to `poseFps` with a deadline accumulator, so the average rate is exactly the
// setting whenever the camera and the GPU can deliver it.  Each result goes to
// the server as a kind-6 landmark frame.  Kept outside the store so tracking
// never triggers a React render; `trackStats` (2 Hz) reports the real rates.
let tracker: Tracker | null = null;
let trackerBuild: Promise<void> | null = null;
let trackHandle: { stop: () => void } | null = null;

/** Events per second over the last second. */
class Rate {
  private t: number[] = [];
  hit(now: number) { this.t.push(now); this.trim(now); }
  fps(now: number) { this.trim(now); return this.t.length > 1 ? (this.t.length - 1) * 1000 / Math.max(1, now - this.t[0]) : 0; }
  private trim(now: number) { while (this.t.length && now - this.t[0] > 1000) this.t.shift(); }
}
const rates = { cam: new Rate(), track: new Rate(), preview: new Rate() };
let detectMs = 0;
export const notePreview = () => rates.preview.hit(performance.now());

function stopTracking() {
  trackHandle?.stop();
  trackHandle = null;
}

function startTracking(get: () => LiveState, set: (p: Partial<LiveState>) => void) {
  if (trackHandle !== null) return;
  type VF = HTMLVideoElement & { requestVideoFrameCallback?: (cb: (now: number) => void) => number;
                                 cancelVideoFrameCallback?: (h: number) => void };
  let stopped = false, due = 0, lastStats = 0, raf = -1;
  let video: VF | null = null, vHandle = -1, fresh = false, lastFrame = -1e9;

  // mark each new camera frame (requestVideoFrameCallback); rAF does the work
  const watch = (v: VF | null) => {
    if (video && vHandle >= 0) video.cancelVideoFrameCallback?.(vHandle);
    video = v; vHandle = -1;
    if (!v?.requestVideoFrameCallback) return;
    const cb = (now: number) => {
      fresh = true; lastFrame = now; rates.cam.hit(now);
      if (!stopped && video === v) vHandle = v.requestVideoFrameCallback!(cb);
    };
    vHandle = v.requestVideoFrameCallback(cb);
  };

  const loop = (now: number) => {
    if (stopped) return;
    raf = requestAnimationFrame(loop);
    const s = get();
    const v = (s._cam?.video ?? null) as VF | null;
    if (v !== video) watch(v);
    if (now - lastStats >= 500) {
      lastStats = now;
      set({ trackStats: { camFps: rates.cam.fps(now), trackFps: rates.track.fps(now), detectMs, previewFps: rates.preview.fps(now) } });
    }
    if (!tracker || !v || v.readyState < 2 || !tracks(s.queryMode)) return;
    // new camera frames only, while the frame callback is alive (it can pause
    // for an off-screen video — then pace on time alone)
    if (now - lastFrame < 500) {
      if (!fresh) return;
      fresh = false;
    }
    const interval = 1000 / Math.max(1, s.poseFps);
    if (now < due - 4) return;                                  // ahead of the set rate: skip this frame
    due = now - due > interval ? now + interval : due + interval;   // far behind → resync, else keep the cadence
    const t0 = performance.now();
    let rows: Float32Array | null = null;
    try { rows = tracker.detect(v, now); } catch { return; }
    const dt = performance.now() - t0;
    detectMs = detectMs ? 0.9 * detectMs + 0.1 * dt : dt;
    rates.track.hit(now);
    s._sendBinary?.(6, packLandmarks(tracker.kind, v.videoWidth / Math.max(1, v.videoHeight), now / 1000, rows));
  };
  raf = requestAnimationFrame(loop);
  trackHandle = {
    stop: () => {
      stopped = true;
      cancelAnimationFrame(raf);
      watch(null);
      set({ trackStats: null });
    },
  };
}

/** Build, swap or retire the tracker so it matches the current mode and choice. */
async function ensureTracker(get: () => LiveState, set: (p: Partial<LiveState>) => void): Promise<void> {
  const s = get();
  const want = tracks(s.queryMode) && s._cam ? s.poseOpts.tracker : null;
  if (!want) {
    stopTracking();
    if (tracker) { tracker.close(); tracker = null; }
    if (s.trackerState !== "off") set({ trackerState: "off" });
    return;
  }
  if (tracker?.kind === want) { startTracking(get, set); return; }
  if (trackerBuild) return trackerBuild;              // building; it re-checks when done
  set({ trackerState: "loading" });
  trackerBuild = (async () => {
    try {
      const t = await createTracker(want);
      tracker?.close();
      tracker = t;
      set({ trackerState: t.delegate });
    } catch (e: any) {
      set({ trackerState: "error", liveError: `landmark tracker: ${e?.message ?? String(e)}` });
    } finally {
      trackerBuild = null;
    }
    if (get().trackerState !== "error") await ensureTracker(get, set);   // the wanted kind may have changed
  })();
  return trackerBuild;
}

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
      // `videoWindow` seconds holds exactly videoWindow × videoFps frames;
      // in pose / movement mode the tracker reads the video itself and the
      // JPEG ring idles
      const cam = await startCameraCapture({ fpsMs: () => (tracks(get().queryMode) ? 1000 : 1000 / get().videoFps) });
      set({ _cam: cam, camVideo: cam.video });
    }
    if (!needsCam(mode) && get()._cam) { get()._cam!.stop(); set({ _cam: null, camVideo: null }); }
    await ensureTracker(get, set);
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
  } else if (s.queryMode === "pose") {
    // landmarks already stream to the server; this asks for an embedding of
    // the recent trajectory (a clip of videoWindow s for a video tower, else a still)
    s._send?.({ type: "pose_tick", window: s.videoWindow, frames: videoFrameCount(s) });
  } else if (s.queryMode === "motion") {
    // … or for a description of its movement qualities, embedded as text
    s._send?.({ type: "motion_tick", window: s.motionWindow, contrast: s.motionContrast });
  }
}

/** Each mode's send rate: movement mode refreshes at its own pace. */
const tickMs = (s: LiveState) => (s.queryMode === "motion" ? s.motionRefresh * 1000 : s.sendInterval);

function restartTicker(get: () => LiveState, set: (p: Partial<LiveState>) => void) {
  const s = get();
  if (!s._ticker) return;
  clearInterval(s._ticker);
  set({ _ticker: setInterval(() => sendWindow(get), tickMs(get())) });
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
  poseCapable: false,
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
  poseFps: 15,
  poseOpts: { tracker: "body", trace: 1.0, fit: "camera", joints: "extremities", smooth: 0.5 },
  posePreview: null,
  posePresent: false,
  trackStats: null,
  trackerState: "off",
  motionWindow: 3.0,
  motionContrast: 0.7,
  motionRefresh: 1.0,
  motion: null,
  motionApplied: null,

  context: "",
  query: "",

  focus: 1.6,
  contraction: 0.7,
  damping: 8,
  stiffness: 40,
  repulsion: 60000,
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
    // movement words come from body and hands; a face has its own description (next)
    if (queryMode === "motion" && s.poseOpts.tracker === "face") get().setPoseOpts({ tracker: "body" });
    // switching between live modes on the fly keeps whatever capture the
    // new mode still needs and starts the rest
    if (s.live) { void ensureCapture(get, set); restartTicker(get, set); }
  },
  startLive: async () => {
    set({ liveError: null });
    await ensureCapture(get, set);
    if (get()._ticker) return;
    const ticker = setInterval(() => sendWindow(get), tickMs(get()));
    set({ _ticker: ticker, live: !!(get()._mic || get()._cam) });
  },
  stopLive: () => {
    const s = get();
    if (s._ticker) clearInterval(s._ticker);
    s._mic?.stop();
    s._cam?.stop();
    stopTracking();
    if (tracker) { tracker.close(); tracker = null; }
    set({ _ticker: null, _mic: null, _cam: null, live: false, audioLevel: 0, camVideo: null, posePresent: false,
          trackerState: "off" });
    if (s.queryMode !== "text") s._send?.({ type: "query_clear" });
  },
  setPoseOpts: (patch) => {
    // switching tracker resets the trails to that tracker's default set
    // unless the current one also exists there
    if (patch.tracker && !patch.joints) {
      const sets = POSE_TRAILS[patch.tracker];
      if (!sets.some((o) => o.value === get().poseOpts.joints)) patch = { ...patch, joints: sets[0].value };
    }
    set((s) => ({ poseOpts: { ...s.poseOpts, ...patch }, ...(patch.tracker ? { posePreview: null, posePresent: false } : {}) }));
    get()._send?.({ type: "pose_params", ...patch });
    if (patch.tracker && get().live) void ensureTracker(get, set);
  },
  resetPose: () => get()._send?.({ type: "pose_reset" }),
  exportPose: (format, seconds) => get()._send?.({ type: "pose_export", format, seconds }),
  setSendInterval: (v) => { set({ sendInterval: v }); restartTicker(get, set); },
  setMotionRefresh: (v) => { set({ motionRefresh: v }); restartTicker(get, set); },
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

/** The Live tab's own default for the number of strongest Δ edges (the
 *  Explorer keeps its own sidebar value). */
export const LIVE_TOP_ABS_EDGES = 15;

/** Δ-graph params mirrored from the Explorer sidebar, so the live graph
 *  starts out like the one in the Explorer tab — except the edge count, which
 *  is the Live default (or whatever was set with the Live slider, so a
 *  reconnect doesn't reset it). */
export function explorerDeltaParams(): Partial<LiveParams> {
  const s = useSidebar.getState();
  return {
    top_abs_edges: useLive.getState().params?.top_abs_edges ?? LIVE_TOP_ABS_EDGES,
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

/** The layout knobs the server's OSC sim mirrors (web/backend/app/live_sim.py). */
const SHAPE_KEYS = ["focus", "contraction", "damping", "stiffness", "repulsion", "unlinked"] as const;
const shapeOf = (s: LiveState) => Object.fromEntries(SHAPE_KEYS.map((k) => [k, s[k]]));

const asKind = (k: unknown): QueryKind =>
  k === "audio" || k === "image" || k === "video" || k === "av" || k === "pose" || k === "motion" ? k : "text";

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
        const sendBinary = (kind: 1 | 2 | 3 | 4 | 5 | 6, buf: ArrayBuffer) => {
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
        send({ type: "pose_params", ...st.poseOpts });
        send({ type: "shape", shape: shapeOf(st) });
        useLive.setState((s) => ({ params: { ...(s.params ?? ({} as LiveParams)), ...seeded } as LiveParams }));
      };
      sock.onmessage = (ev) => {
        let msg: any;
        try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg.type === "ready") {
          useLive.setState((s) => ({ status: "ready", params: { ...msg.params, ...(s.params ?? {}) },
            audioCapable: !!msg.audio_capable, imageCapable: !!msg.image_capable,
            videoCapable: !!msg.video_capable, avCapable: !!msg.av_capable, poseCapable: !!msg.pose_capable }));
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
            ...(msg.kind === "motion" && msg.active ? { motionApplied: { text: String(msg.query ?? ""), at: performance.now() } } : {}),
          }));
        } else if (msg.type === "motion") {
          if (useLive.getState().queryMode !== "motion") return;
          useLive.setState({ motion: msg as MotionReading });
        } else if (msg.type === "pose") {
          if (!tracks(useLive.getState().queryMode)) return;   // a late frame after switching away
          notePreview();
          useLive.setState({ posePreview: msg.jpeg ? `data:image/jpeg;base64,${msg.jpeg}` : null, posePresent: !!msg.present });
        } else if (msg.type === "pose_export") {
          downloadBase64(msg.data, msg.format === "npz" ? "application/octet-stream" : "video/mp4",
                         `delyrism-pose-${new Date().toISOString().replace(/[:.]/g, "-")}.${msg.format}`);
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
    // mirror layout-knob changes to the server's OSC sim
    const unsubShape = useLive.subscribe((s, prev) => {
      if (SHAPE_KEYS.some((k) => s[k] !== prev[k])) s._send?.({ type: "shape", shape: shapeOf(s) });
    });
    return () => {
      closed = true;
      unsubShape();
      if (retry) clearTimeout(retry);
      ws?.close();
      useLive.getState().stopLive();
      useLive.setState({ status: "closed", _send: null, _sendBinary: null });
    };
  }, [spaceId]);
}

function downloadBase64(b64: string, mime: string, filename: string) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
