"use client";
// Continuous camera capture for the live query.
//
// Grabs a frame from the webcam every `fpsMs`, downsizes it to `size` px
// on the long side, JPEG-encodes it and keeps the last few seconds in a
// ring.  `latest()` gives one frame (CLIP image query), `clip(sec, max)`
// gives an evenly thinned window of frames (PE-AV video query) packed as:
//   uint8 n · n × (uint32 LE length · JPEG)
// The <video> element is handed back so the UI can show a preview.

export interface CameraCapture {
  stop: () => void;
  video: HTMLVideoElement;
  track: MediaStreamTrack;
  /** the camera mode in use, as measured when it was opened */
  mode: { width: number; height: number; fps: number };
  latest: () => ArrayBuffer | null;
  clip: (sec: number, maxFrames: number) => ArrayBuffer | null;
  /** A stream of the camera's VideoFrames (newest first, at most `buffer`
   *  queued), straight from the track — independent of the page rendering,
   *  for the landmark worker.  null where MediaStreamTrackProcessor is missing. */
  frames: (buffer?: number) => { readable: ReadableStream<VideoFrame>; close: () => void } | null;
  /** Cap the exposure at one frame time so the camera keeps its full rate in
   *  dim light (darker picture); off = the camera's own auto exposure. */
  fastShutter: (on: boolean) => Promise<boolean>;
}

type MSTP = new (init: { track: MediaStreamTrack; maxBufferSize?: number }) => { readable: ReadableStream<VideoFrame> };
const Processor = (): MSTP | null => (typeof window !== "undefined" && (window as any).MediaStreamTrackProcessor) || null;

/** New frames per second the track delivers, over `ms` after a short warm-up. */
async function measureFps(track: MediaStreamTrack, ms = 900): Promise<number> {
  const P = Processor();
  if (!P) return track.getSettings().frameRate ?? 30;
  const probe = track.clone();
  const rd = new P({ track: probe, maxBufferSize: 2 }).readable.getReader();
  let n = 0, t0 = 0;
  const end = performance.now() + 250 + ms;
  try {
    while (performance.now() < end) {
      const r = await Promise.race([rd.read(), new Promise<null>((res) => setTimeout(() => res(null), end - performance.now()))]);
      if (!r || r.done) break;
      r.value.close();
      const now = performance.now();
      if (now >= end - ms) { if (!t0) t0 = now; else n++; }
    }
  } finally {
    rd.cancel().catch(() => {});
    probe.stop();
  }
  return t0 ? (n * 1000) / Math.max(1, performance.now() - t0) : 0;
}

// Some webcam drivers deliver a mode far below its advertised rate (here a
// Chicony "FHD Webcam" gives 2 fps at 640×480 but 30 at 1280×720), so the
// camera is opened mode by mode and the first one that really runs is kept —
// remembered per camera, re-checked on every open.
const MODES: [number, number][] = [[1280, 720], [960, 540], [1920, 1080], [640, 480], [848, 480], [320, 240]];
const MODE_KEY = "delyrism.cameraMode";
const GOOD_FPS = 14;

async function openCamera(): Promise<{ stream: MediaStream; mode: CameraCapture["mode"] }> {
  let saved: [number, number] | null = null;
  try { saved = JSON.parse(localStorage.getItem(MODE_KEY) || "null"); } catch {}
  const order = saved ? [saved, ...MODES.filter(([w, h]) => w !== saved![0] || h !== saved![1])] : MODES;
  let best: { stream: MediaStream; mode: CameraCapture["mode"] } | null = null;
  for (const [w, h] of order) {
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { exact: w }, height: { exact: h }, frameRate: { ideal: 30 }, facingMode: "user" },
        audio: false,
      });
    } catch { continue; }                                    // the camera has no such mode
    const track = stream.getVideoTracks()[0];
    const fps = await measureFps(track);
    const st = track.getSettings();
    const mode = { width: st.width ?? w, height: st.height ?? h, fps };
    if (fps >= GOOD_FPS) {
      best?.stream.getTracks().forEach((t) => t.stop());
      try { localStorage.setItem(MODE_KEY, JSON.stringify([w, h])); } catch {}
      return { stream, mode };
    }
    if (!best || fps > best.mode.fps) { best?.stream.getTracks().forEach((t) => t.stop()); best = { stream, mode }; }
    else stream.getTracks().forEach((t) => t.stop());
  }
  if (best) return best;
  // nothing matched exactly: let the browser choose
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
  const st = stream.getVideoTracks()[0].getSettings();
  return { stream, mode: { width: st.width ?? 0, height: st.height ?? 0, fps: st.frameRate ?? 0 } };
}

export function packFrames(frames: ArrayBuffer[]): ArrayBuffer {
  const total = 1 + frames.reduce((a, f) => a + 4 + f.byteLength, 0);
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  out[0] = frames.length;
  let off = 1;
  for (const f of frames) {
    dv.setUint32(off, f.byteLength, true); off += 4;
    out.set(new Uint8Array(f), off); off += f.byteLength;
  }
  return out.buffer;
}

export async function startCameraCapture(opts: {
  fpsMs: () => number;
  size?: number;
  quality?: number;
}): Promise<CameraCapture> {
  const { stream, mode } = await openCamera();
  const track = stream.getVideoTracks()[0];
  const video = document.createElement("video");
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();

  const size = opts.size ?? 336;
  const quality = opts.quality ?? 0.8;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const ring: { t: number; buf: ArrayBuffer }[] = [];
  const KEEP_MS = 12_000;

  let timer: ReturnType<typeof setTimeout> | null = null;
  let busy = false;
  const tick = () => {
    timer = setTimeout(tick, opts.fpsMs());
    if (busy || !ctx || video.videoWidth === 0) return;
    const vw = video.videoWidth, vh = video.videoHeight;
    const k = size / Math.max(vw, vh);
    canvas.width = Math.max(32, Math.round(vw * k));
    canvas.height = Math.max(32, Math.round(vh * k));
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    busy = true;
    canvas.toBlob(
      (blob) => {
        busy = false;
        if (!blob) return;
        blob.arrayBuffer().then((buf) => {
          const now = performance.now();
          ring.push({ t: now, buf });
          while (ring.length && now - ring[0].t > KEEP_MS) ring.shift();
        }).catch(() => {});
      },
      "image/jpeg",
      quality,
    );
  };
  timer = setTimeout(tick, 150);

  const latest = () => (ring.length ? ring[ring.length - 1].buf : null);
  const clip = (sec: number, maxFrames: number) => {
    const now = performance.now();
    const win = ring.filter((f) => now - f.t <= sec * 1000).map((f) => f.buf);
    if (!win.length) return null;
    const n = Math.max(1, Math.min(maxFrames, win.length));
    const picked: ArrayBuffer[] = [];
    for (let i = 0; i < n; i++) picked.push(win[Math.round((i * (win.length - 1)) / Math.max(1, n - 1))]);
    return packFrames(picked);
  };

  const clones: MediaStreamTrack[] = [];
  const frames = (buffer = 1) => {
    const P = Processor();
    if (!P) return null;
    const t = track.clone();
    clones.push(t);
    return { readable: new P({ track: t, maxBufferSize: buffer }).readable, close: () => t.stop() };
  };

  // exposure time is in 100 µs units (Image Capture spec)
  let shutter = false;
  const fastShutter = async (on: boolean) => {
    const caps: any = track.getCapabilities?.() ?? {};
    if (!caps.exposureMode?.includes?.("manual") || !caps.exposureTime) return false;
    try {
      if (on) {
        const fps = Math.max(1, track.getSettings().frameRate ?? 30);
        const step = caps.exposureTime.step || 1;
        const t = Math.max(caps.exposureTime.min, Math.floor((0.94 * 1e4) / fps / step) * step);
        await track.applyConstraints({ advanced: [{ exposureMode: "manual", exposureTime: t } as any] });
      } else if (shutter) {
        await track.applyConstraints({ advanced: [{ exposureMode: "continuous" } as any] });
      }
      shutter = on;
      return true;
    } catch {
      return false;
    }
  };

  return {
    video,
    track,
    mode,
    latest,
    clip,
    frames,
    fastShutter,
    stop: () => {
      if (timer) clearTimeout(timer);
      // the driver keeps exposure settings after the app closes: hand the
      // camera back on auto exposure
      try { video.pause(); } catch {}
      const close = () => { clones.forEach((t) => t.stop()); stream.getTracks().forEach((t) => t.stop()); };
      if (shutter) track.applyConstraints({ advanced: [{ exposureMode: "continuous" } as any] }).catch(() => {}).finally(close);
      else close();
    },
  };
}
