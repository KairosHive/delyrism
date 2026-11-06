"use client";
// Continuous microphone capture for the CLAP query.
//
// Keeps a rolling window of mono PCM and, every `intervalMs`, hands the last
// `windowSec` seconds to `onWindow` as:
//   uint32 LE sample-rate · int16 LE PCM
// The socket layer prefixes the kind byte before sending.  Also
// reports a smoothed RMS level for the meter.

export interface MicCapture {
  stop: () => void;
  sampleRate: number;
}

export async function startMicCapture(opts: {
  windowSec: () => number;
  intervalMs: () => number;
  onWindow: (frame: ArrayBuffer) => void;
  onLevel: (rms: number) => void;
}): Promise<MicCapture> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
  });
  const AC: typeof AudioContext = (window.AudioContext || (window as any).webkitAudioContext) as any;
  const ctx = new AC();
  const src = ctx.createMediaStreamSource(stream);
  const sr = ctx.sampleRate;

  // ring buffer sized for the largest window we allow (8 s)
  const cap = Math.ceil(sr * 8);
  const ring = new Float32Array(cap);
  let head = 0;
  let filled = 0;
  let level = 0;

  // ScriptProcessor is deprecated but needs no worklet file — fine for a
  // ~2 KB capture tap.  Buffer 2048 ≈ 43 ms at 48 kHz.
  const proc = ctx.createScriptProcessor(2048, 1, 1);
  proc.onaudioprocess = (e) => {
    const ch = e.inputBuffer.getChannelData(0);
    let sum = 0;
    for (let i = 0; i < ch.length; i++) {
      ring[head] = ch[i];
      head = (head + 1) % cap;
      sum += ch[i] * ch[i];
    }
    filled = Math.min(cap, filled + ch.length);
    const rms = Math.sqrt(sum / ch.length);
    level = level + (rms - level) * 0.3;
    opts.onLevel(level);
  };
  src.connect(proc);
  // some browsers only drive ScriptProcessor when it is connected to the
  // destination; a zero-gain node keeps it silent
  const mute = ctx.createGain(); mute.gain.value = 0;
  proc.connect(mute); mute.connect(ctx.destination);

  let timer: ReturnType<typeof setTimeout> | null = null;
  const tick = () => {
    const n = Math.min(filled, Math.round(sr * opts.windowSec()));
    if (n > sr * 0.25) {
      const out = new ArrayBuffer(4 + 2 * n);
      new DataView(out).setUint32(0, sr, true);
      const pcm = new Int16Array(out, 4, n);
      let idx = (head - n + cap) % cap;
      for (let i = 0; i < n; i++) {
        let s = ring[idx];
        s = s < -1 ? -1 : s > 1 ? 1 : s;
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        idx = (idx + 1) % cap;
      }
      opts.onWindow(out);
    }
    timer = setTimeout(tick, opts.intervalMs());
  };
  timer = setTimeout(tick, opts.intervalMs());

  return {
    sampleRate: sr,
    stop: () => {
      if (timer) clearTimeout(timer);
      try { proc.disconnect(); src.disconnect(); mute.disconnect(); } catch {}
      stream.getTracks().forEach((t) => t.stop());
      ctx.close().catch(() => {});
    },
  };
}
