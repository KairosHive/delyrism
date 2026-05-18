"use client";
/**
 * Convert any browser-decodable audio (webm/opus from MediaRecorder, mp3,
 * wav, ogg, m4a, flac…) into a 16-bit PCM WAV blob.
 *
 * Why: the FastAPI backend decodes uploaded audio with
 * `librosa.load → soundfile (libsndfile)`, which natively supports wav /
 * flac / ogg but NOT webm/opus.  Without ffmpeg in PATH the server can't
 * fall back to audioread, so the upload fails with
 *   "Format not recognised."
 *
 * Doing the format normalization in the browser sidesteps that entirely —
 * Web Audio API can decode every common format, and re-encoding as WAV is
 * a tight ~50-line PCM writer.  The server then only ever sees WAV.
 *
 * Tradeoff: a 10 s recording adds ~50–200 ms of in-browser CPU before the
 * network call.  Acceptable; the network round-trip dominates anyway.
 */

export async function blobToWav(file: Blob): Promise<Blob> {
  const arrayBuffer = await file.arrayBuffer();

  // Some browsers (older Safari) need the prefixed AudioContext.
  const AC: typeof AudioContext =
    (typeof window !== "undefined" && (window.AudioContext || (window as any).webkitAudioContext)) as any;
  if (!AC) throw new Error("Web Audio API not available in this browser");
  const ctx = new AC();
  try {
    // decodeAudioData accepts ArrayBuffer for any browser-supported format
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
    return audioBufferToWav(audioBuffer);
  } finally {
    // free up the AudioContext
    try { await ctx.close(); } catch {}
  }
}

function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const numSamples = buffer.length;
  const bytesPerSample = 2; // 16-bit PCM
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * blockAlign;
  const bufferSize = 44 + dataSize;

  const out = new ArrayBuffer(bufferSize);
  const view = new DataView(out);

  // ---- RIFF / WAVE header ----
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");

  // fmt sub-chunk
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);            // fmt chunk size
  view.setUint16(20, 1, true);             // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);            // bits per sample

  // data sub-chunk
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  // ---- interleaved PCM samples ----
  // Pull each channel ONCE (getChannelData is cheap but not free).
  const channels: Float32Array[] = [];
  for (let c = 0; c < numChannels; c++) channels.push(buffer.getChannelData(c));

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    for (let c = 0; c < numChannels; c++) {
      let s = channels[c][i];
      // clip and convert to signed 16-bit
      s = s < -1 ? -1 : s > 1 ? 1 : s;
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }

  return new Blob([out], { type: "audio/wav" });
}

function writeAscii(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}
