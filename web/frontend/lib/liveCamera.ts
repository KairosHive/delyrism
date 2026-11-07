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
  latest: () => ArrayBuffer | null;
  clip: (sec: number, maxFrames: number) => ArrayBuffer | null;
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
  const stream = await navigator.mediaDevices.getUserMedia({
    // ask for a high frame rate: the landmark tracker can only track as many
    // frames per second as the camera delivers (many default to 30 or less)
    video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60 }, facingMode: "user" },
    audio: false,
  });
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

  return {
    video,
    latest,
    clip,
    stop: () => {
      if (timer) clearTimeout(timer);
      try { video.pause(); } catch {}
      stream.getTracks().forEach((t) => t.stop());
    },
  };
}
