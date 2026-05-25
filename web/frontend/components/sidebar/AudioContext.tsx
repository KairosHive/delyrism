"use client";
import * as React from "react";
import { api } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { Slider } from "../ui/Slider";
import { blobToWav } from "@/lib/audioToWav";

/**
 * Audio context override.
 *
 * Flow:
 *   1) user uploads a file or records from the browser mic
 *   2) frontend POSTs the audio bytes to /context/encode-audio → vector
 *   3) frontend POSTs the vector to /context/set-override → server attaches
 *      it to the cached SymbolSpace as `context_override`
 *   4) every subsequent ctx_vec() call on the server uses this override
 *      instead of encoding the sentence — so the engine "sees" the audio
 *
 * Requires the active embedder to support audio (CLAP or AudioCLIP), since
 * the override vector must live in the same embedding space as the
 * descriptors.  We grey out the panel and explain why otherwise.
 */
export function AudioContext() {
  const spaceId = useSidebar((s) => s.spaceId);
  const backend = useSidebar((s) => s.embedderBackend);
  const audioActive = useSidebar((s) => s.audioActive);
  const audioMaxSeconds = useSidebar((s) => s.audioMaxSeconds);
  const set = useSidebar((s) => s.set);

  // Only CLAP exposes the audio path in the UI (matches the old Streamlit
  // app's behavior).  The engine technically supports audioclip too, but it
  // was never surfaced.
  const audioCapable = backend === "clap";

  const [status, setStatus] = React.useState<"idle" | "encoding" | "ok" | "error">("idle");
  const [error, setError] = React.useState<string | null>(null);
  const [recording, setRecording] = React.useState(false);
  const recorderRef = React.useRef<MediaRecorder | null>(null);
  const stopTimerRef = React.useRef<number | null>(null);

  async function uploadFile(file: Blob) {
    if (!spaceId) return;
    setStatus("encoding");
    setError(null);
    try {
      // Re-encode whatever format the user gave us (webm from MediaRecorder,
      // mp3/wav/m4a/flac/ogg from upload) into a 16-bit PCM WAV in the
      // browser.  libsndfile on the server only natively reads wav/flac/ogg —
      // converting client-side means the server never needs ffmpeg for the
      // codec fallback path.
      const wavBlob = await blobToWav(file);
      const wavFile = new File([wavBlob], "audio.wav", { type: "audio/wav" });

      const fd = new FormData();
      fd.append("space_id", spaceId);
      fd.append("file", wavFile);
      fd.append("max_seconds", String(audioMaxSeconds));
      const enc = await api.upload<{ vector: number[]; dim: number }>(
        "/context/encode-audio",
        fd,
      );
      await api.post("/context/set-override", { space_id: spaceId, vector: enc.vector });
      // Audio, image, and the alchemist blend all share the same backend
      // override slot — flipping audio on means the others just got
      // replaced.  Sync UI state to match (also turn alchemist *mode* off
      // so its effect doesn't immediately push its own blend back).
      const st = useSidebar.getState();
      if (st.imageActive) {
        if (st.imageThumbnail) URL.revokeObjectURL(st.imageThumbnail);
        set("imageActive", false);
        set("imageDescription", "");
        set("imageThumbnail", null);
        set("imageNonce", Date.now());
      }
      if (st.alchemistMode || st.alchemistActive) {
        set("alchemistMode", false);
        set("alchemistActive", false);
        set("alchemistNonce", Date.now());
      }
      set("audioActive", true);
      set("audioNonce", Date.now());
      setStatus("ok");
    } catch (e: any) {
      setStatus("error");
      setError(e?.message ?? "encoding failed");
    }
  }

  async function clearAudio() {
    if (!spaceId) return;
    try {
      await api.post("/context/set-override", { space_id: spaceId, vector: null });
    } catch {
      // server-side failure isn't critical here — still flip the UI state
    }
    set("audioActive", false);
    set("audioNonce", Date.now());
    setStatus("idle");
    setError(null);
  }

  async function startRecording() {
    if (!spaceId || !navigator.mediaDevices) {
      setError("mic not available in this browser");
      setStatus("error");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks: Blob[] = [];
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" });
        const ext = blob.type.includes("webm") ? "webm" : "ogg";
        const file = new File([blob], `recording.${ext}`, { type: blob.type });
        await uploadFile(file);
        setRecording(false);
      };
      recorder.start();
      setRecording(true);
      setStatus("idle");
      // auto-stop at max_seconds
      stopTimerRef.current = window.setTimeout(() => {
        try { recorder.stop(); } catch {}
      }, audioMaxSeconds * 1000);
    } catch (e: any) {
      setStatus("error");
      setError(e?.message ?? "mic permission denied");
    }
  }

  function stopRecording() {
    if (stopTimerRef.current) {
      clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    try {
      recorderRef.current?.stop();
    } catch {}
  }

  React.useEffect(() => () => {
    // cleanup any active stream/timer on unmount
    if (stopTimerRef.current) clearTimeout(stopTimerRef.current);
    try { recorderRef.current?.stop(); } catch {}
  }, []);

  return (
    <div className={`space-y-2 rounded-md border border-ink-700/60 bg-ink-900/40 p-2.5 ${!audioCapable ? "opacity-60" : ""}`}>
      <div className="flex items-center justify-between">
        <div className="sub-title">Audio context</div>
        {audioActive && (
          <span className="pill !text-[10px] border-accent-500/60 bg-accent-600/20 text-accent-200">● override active</span>
        )}
      </div>

      {!audioCapable ? (
        <p className="text-[11px] text-ink-400">
          Switch the Embedding Model to <span className="text-accent-300">CLAP</span> and rebuild
          the space to enable audio context.
        </p>
      ) : !spaceId ? (
        <p className="text-[11px] text-ink-400">Build the space first.</p>
      ) : (
        <>
          <Slider
            label="Max seconds"
            value={audioMaxSeconds}
            min={2}
            max={30}
            step={1}
            onChange={(v) => set("audioMaxSeconds", Math.round(v))}
            help="Length cap for both upload and recording. Audio longer than this gets trimmed before being embedded. CLAP works well on 5–15 s clips; longer clips are usually averaged anyway."
          />

          <div className="grid grid-cols-2 gap-2">
            <label className="btn cursor-pointer">
              ↑ Upload
              <input
                type="file"
                accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm,.flac"
                hidden
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) uploadFile(f);
                  e.target.value = "";
                }}
                disabled={status === "encoding" || recording}
              />
            </label>
            {recording ? (
              <button className="btn border-danger/60 bg-danger/15 text-danger" onClick={stopRecording}>
                ■ Stop
              </button>
            ) : (
              <button className="btn" onClick={startRecording} disabled={status === "encoding"}>
                ● Record
              </button>
            )}
          </div>

          {/* status row */}
          {status === "encoding" && (
            <div className="flex items-center gap-2 text-[11px] text-ink-300">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-400" />
              encoding audio…
            </div>
          )}
          {recording && (
            <div className="flex items-center gap-2 text-[11px] text-danger">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-danger" />
              recording… ({audioMaxSeconds}s max)
            </div>
          )}
          {status === "ok" && !recording && (
            <div className="text-[11px] text-accent-300">audio applied — context now overrides the sentence</div>
          )}
          {status === "error" && error && (
            <div className="text-[11px] text-danger">{error}</div>
          )}

          {audioActive && (
            <button className="btn !text-xs" onClick={clearAudio}>
              clear audio context
            </button>
          )}
        </>
      )}
    </div>
  );
}
