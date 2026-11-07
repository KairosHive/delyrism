/// <reference lib="webworker" />
// Landmark tracking off the main thread.
//
// The camera's VideoFrames arrive here straight from the track
// (MediaStreamTrackProcessor, transferred as a ReadableStream), so nothing on
// the page — graph simulation, React renders, a hidden or covered window —
// can delay a frame or stall the tracker.  A pump keeps only the newest frame
// (older ones are closed unseen); the tracker takes it when the set rate is
// due, so latency is one MediaPipe run, never a queue.  Landmarks go back to
// the page as kind-6 payloads, stamped with the frame's capture time on the
// page's performance.now() clock.
//
//   in : start {frames, kind, fps, origin} · fps {fps} · kind {kind} · stop
//   out: state {state, error?} · lm {buf} · stats {camFps, trackFps, detectMs, lagMs}

import { createTracker, packLandmarks, type Tracker, type TrackerKind } from "./poseTracker";

declare const self: DedicatedWorkerGlobalScope;

let tracker: Tracker | null = null;
let wantKind: TrackerKind | null = null;
let building = false;
let fps = 20;
let origin = 0;                  // page's performance.timeOrigin
let running = false;
let session = 0;                 // each start runs its own pump + work loop
let reader: ReadableStreamDefaultReader<VideoFrame> | null = null;
let latest: VideoFrame | null = null;
let latestT = 0;                 // capture time of `latest`, page clock (ms)
let wake: (() => void) | null = null;

/** Page clock: the main thread's performance.now(), from this worker. */
const clock = () => performance.timeOrigin + performance.now() - origin;

class Rate {
  private t: number[] = [];
  hit(now: number) { this.t.push(now); this.trim(now); }
  fps(now: number) { this.trim(now); return this.t.length > 1 ? ((this.t.length - 1) * 1000) / Math.max(1, now - this.t[0]) : 0; }
  private trim(now: number) { while (this.t.length && now - this.t[0] > 1000) this.t.shift(); }
}
const cam = new Rate(), track = new Rate();
let detectMs = 0;
let lagMs = 0;                   // camera capture → landmarks ready (EMA)

// capture timestamp (µs, the camera's clock) → page clock: the smallest
// arrival − capture offset seen is the pipeline's fixed delay; a jump of more
// than a second (camera restarted) starts over
let offset: number | null = null;
function captureTime(frame: VideoFrame, arrival: number): number {
  const ts = (frame.timestamp ?? 0) / 1000;
  const o = arrival - ts;
  if (offset === null || o < offset || o - offset > 1000) offset = o;
  return ts + offset;
}

async function buildTracker() {
  if (building) return;
  building = true;
  try {
    while (wantKind && tracker?.kind !== wantKind) {
      const kind = wantKind;
      self.postMessage({ type: "state", state: "loading" });
      try {
        const t = await createTracker(kind);
        tracker?.close();
        tracker = t;
        self.postMessage({ type: "state", state: t.delegate });
      } catch (e: any) {
        self.postMessage({ type: "state", state: "error", error: `landmark tracker: ${e?.message ?? String(e)}` });
        wantKind = null;
      }
    }
  } finally {
    building = false;
  }
}

async function pump(me: number, rd: ReadableStreamDefaultReader<VideoFrame>) {
  while (running && me === session) {
    let r: ReadableStreamReadResult<VideoFrame>;
    try { r = await rd.read(); } catch { break; }
    if (r.done) break;
    if (me !== session) { r.value?.close(); break; }
    const now = clock();
    cam.hit(now);
    latest?.close();
    latest = r.value;
    latestT = captureTime(r.value, now);
    wake?.();
  }
  if (me === session) { running = false; wake?.(); }
}

/** A fresh macrotask, so frames already queued in the stream land first. */
const yieldTask = () => new Promise<void>((res) => { const c = new MessageChannel(); c.port1.onmessage = () => res(); c.port2.postMessage(0); });

async function work(me: number) {
  let due = 0, lastStats = 0;
  while (running && me === session) {
    if (!latest) await new Promise<void>((res) => { wake = res; });
    wake = null;
    await yieldTask();
    if (!running || me !== session) break;
    const now = clock();
    if (now - lastStats >= 500) {
      lastStats = now;
      self.postMessage({ type: "stats", camFps: cam.fps(now), trackFps: track.fps(now), detectMs, lagMs });
    }
    const frame = latest;
    if (!frame) continue;
    if (!tracker || building) { latest = null; frame.close(); continue; }
    const interval = 1000 / Math.max(1, fps);
    // ahead of the set rate: skip.  Half an interval of slack, so camera
    // jitter never drops a frame when the target is the camera's own rate;
    // the accumulator still holds the average to the target
    if (now < due - 0.5 * interval) { latest = null; frame.close(); continue; }
    due = now - due > interval ? now + interval : due + interval;         // far behind → resync
    latest = null;
    const t = latestT;
    const t0 = performance.now();
    let rows: Float32Array | null = null;
    try { rows = tracker.detect(frame, t); } catch { frame.close(); continue; }
    const aspect = frame.displayWidth / Math.max(1, frame.displayHeight);
    frame.close();
    const dt = performance.now() - t0;
    detectMs = detectMs ? 0.9 * detectMs + 0.1 * dt : dt;
    const done = clock();
    lagMs = lagMs ? 0.9 * lagMs + 0.1 * (done - t) : done - t;
    track.hit(done);
    const buf = packLandmarks(tracker.kind, aspect, t / 1000, rows);
    self.postMessage({ type: "lm", buf }, [buf]);
  }
}

function stop() {
  running = false;
  reader?.cancel().catch(() => {});
  reader = null;
  latest?.close();
  latest = null;
  wake?.();
}

self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  if (m.type === "start") {
    stop();
    origin = m.origin;
    fps = m.fps;
    offset = null;
    reader = (m.frames as ReadableStream<VideoFrame>).getReader();
    running = true;
    const me = ++session;
    wantKind = m.kind;
    void buildTracker();
    void pump(me, reader);
    void work(me);
  } else if (m.type === "fps") {
    fps = m.fps;
  } else if (m.type === "kind") {
    wantKind = m.kind;
    void buildTracker();
  } else if (m.type === "stop") {
    stop();
    tracker?.close();
    tracker = null;
    wantKind = null;
  }
};
