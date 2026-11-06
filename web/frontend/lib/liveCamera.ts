"use client";
// Continuous camera capture for the CLIP query.
//
// Grabs a frame from the webcam every `intervalMs`, downsizes it to
// `size` px on the long side, JPEG-encodes it and hands the bytes to
// `onFrame`.  The live graph socket wraps them in a kind-tagged binary
// frame (see liveGraph.ts).  The <video> element is handed back so the UI
// can show a preview.

export interface CameraCapture {
  stop: () => void;
  video: HTMLVideoElement;
}

export async function startCameraCapture(opts: {
  intervalMs: () => number;
  size?: number;
  quality?: number;
  onFrame: (jpeg: ArrayBuffer) => void;
}): Promise<CameraCapture> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  const video = document.createElement("video");
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();

  const size = opts.size ?? 320;
  const quality = opts.quality ?? 0.72;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  let timer: ReturnType<typeof setTimeout> | null = null;
  let busy = false;
  const tick = () => {
    timer = setTimeout(tick, opts.intervalMs());
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
        blob.arrayBuffer().then(opts.onFrame).catch(() => {});
      },
      "image/jpeg",
      quality,
    );
  };
  timer = setTimeout(tick, 150);

  return {
    video,
    stop: () => {
      if (timer) clearTimeout(timer);
      try { video.pause(); } catch {}
      video.srcObject = null;
      stream.getTracks().forEach((t) => t.stop());
    },
  };
}
