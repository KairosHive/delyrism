"use client";
import * as React from "react";
import { useShallow } from "zustand/react/shallow";
import { useLive, POSE_TRAILS, type PoseTracker } from "@/lib/liveGraph";
import { Slider } from "@/components/ui/Slider";
import { Select } from "@/components/ui/Select";

const TRACKERS: { value: PoseTracker; label: string; hint: string }[] = [
  { value: "body", label: "full body", hint: "MediaPipe Pose · 33 points from head to feet; needs your head and torso in view" },
  { value: "hands", label: "hands", hint: "MediaPipe Hands · 21 points per hand with every finger joint; works with only your hands in view" },
  { value: "face", label: "face", hint: "MediaPipe Face · 478 points, drawn as the contours of eyes, brows, lips and face outline, plus the irises" },
  { value: "both", label: "body + hands", hint: "both models on every frame · 75 points" },
];

/** Which MediaPipe model tracks, and whether it runs on the GPU. Shared by pose and movement modes. */
export function TrackerPicker({ allow }: { allow?: PoseTracker[] }) {
  const { poseOpts, setPoseOpts, trackerState, live } = useLive(useShallow((s) => ({
    poseOpts: s.poseOpts, setPoseOpts: s.setPoseOpts, trackerState: s.trackerState, live: s.live,
  })));
  const items = TRACKERS.filter((t) => !allow || allow.includes(t.value));
  const badge = trackerState === "GPU" ? { text: "GPU", cls: "border-accent-400/60 text-accent-300" }
    : trackerState === "CPU" ? { text: "CPU", cls: "border-warning/60 text-warning" }
    : trackerState === "loading" ? { text: "loading model…", cls: "border-ink-600 text-ink-400 animate-pulse" }
    : trackerState === "error" ? { text: "tracker error", cls: "border-danger/60 text-danger" } : null;
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px]">
      <span className="label-sm">track</span>
      <span className="inline-flex overflow-hidden rounded-md border border-ink-700">
        {items.map((t) => (
          <button
            key={t.value}
            onClick={() => setPoseOpts({ tracker: t.value })}
            className={`whitespace-nowrap px-2.5 py-1 transition ${poseOpts.tracker === t.value ? "bg-[#c2a6fe]/25 text-ink-50" : "text-ink-400 hover:text-ink-100"}`}
            title={t.hint}
          >
            {t.label}
          </button>
        ))}
      </span>
      {live && badge && (
        <span className={`rounded border px-1.5 py-0.5 font-mono text-[9px] ${badge.cls}`}
              title="where MediaPipe runs: in this browser, on the GPU through WebGL (CPU if WebGL is unavailable)">
          {badge.text}
        </span>
      )}
      <span className="min-w-[12rem] flex-1 text-[10px] leading-tight text-ink-500">{TRACKERS.find((t) => t.value === poseOpts.tracker)?.hint}</span>
    </div>
  );
}

/** Measured tracking rates, so the "tracking" setting can be checked live:
 *  camera frames in, frames tracked, MediaPipe time per frame, drawings back. */
export function TrackRate() {
  const { st, target, mode, shutter, setShutter } = useLive(useShallow((s) => ({
    st: s.trackStats, target: s.poseFps, mode: s._cam?.mode ?? null, shutter: s.fastShutter, setShutter: s.setFastShutter,
  })));
  if (!st) return null;
  const reach = Math.min(target, st.camFps || target);
  const short = st.trackFps < 0.9 * reach;
  const camShort = st.camFps > 0 && st.camFps < 0.9 * Math.min(target, 30);
  const budget = st.detectMs > 1000 / Math.max(1, target);
  const f = (x: number) => x.toFixed(x < 10 ? 1 : 0);
  return (
    <div className={`col-span-full font-mono text-[10px] leading-tight ${short ? "text-warning" : "text-ink-500"}`}
         title="camera: new frames per second delivered (and its mode) · tracked: frames run through MediaPipe (target = the tracking setting) · ms: MediaPipe time per frame · lag: camera capture to landmarks sent · drawing: pose drawings back from the server">
      camera {f(st.camFps)} fps{mode ? ` @ ${mode.width}×${mode.height}` : ""} · tracked {f(st.trackFps)}/{target} fps ·{" "}
      {st.detectMs.toFixed(1)} ms/frame{st.offThread ? ` · lag ${st.lagMs.toFixed(0)} ms` : " · on page"} · drawing {f(st.previewFps)} fps
      <label className="ml-2 inline-flex cursor-pointer items-center gap-1 align-middle text-ink-400"
             title="Cap the camera's exposure at one frame time while tracking: full frame rate in dim light, darker picture. Off = the camera's own auto exposure.">
        <input type="checkbox" className="h-3 w-3" checked={shutter} onChange={(e) => setShutter(e.target.checked)} /> fast shutter
      </label>
      {camShort && <span className="text-warning"> · the camera delivers only {f(st.camFps)} fps{shutter ? " — more light helps" : " — try fast shutter"}</span>}
      {budget && <span className="text-warning"> · MediaPipe needs more than a frame&apos;s time — lower the rate or the GPU load</span>}
    </div>
  );
}

/** The pose drawing as the server last rendered it, with an in-frame dot. */
export function PosePreview({ size = 132 }: { size?: number }) {
  const { live, posePreview, posePresent, tracker } = useLive(useShallow((s) => ({
    live: s.live, posePreview: s.posePreview, posePresent: s.posePresent, tracker: s.poseOpts.tracker,
  })));
  const what = tracker === "hands" ? "hands" : tracker === "face" ? "face" : "dancer";
  return (
    <div className="relative shrink-0 overflow-hidden rounded-md border border-ink-700 bg-black" style={{ width: size, height: size }}>
      {live && posePreview && <img src={posePreview} alt="pose drawing" className="h-full w-full" />}
      {live && (
        <span className={`absolute left-1.5 top-1.5 h-2 w-2 rounded-full ${posePresent ? "bg-accent-400" : "bg-warning animate-pulse"}`}
              title={posePresent ? `${what} in frame` : `no ${what} in frame — the query holds`} />
      )}
      {!live && <span className="absolute inset-0 flex items-center justify-center text-[10px] text-ink-600">pose drawing</span>}
    </div>
  );
}

/**
 * Pose query — the dancer defines the query with their movement.
 *
 * MediaPipe tracks in the browser on the GPU (full body, hands with
 * fingers, face, or body + hands) and streams landmarks to the server,
 * which keeps the running trajectory and draws it back as moving landmarks
 * with temporal trails.  That drawing is what gets embedded on each send
 * tick: the last video-window seconds as a clip through a video tower
 * (PE-AV), or the current pose with its trails through an image tower
 * (CLIP).  The preview is the drawing itself.  Capture, rate and the two
 * smoothings live in the shared query controls.
 */
export function PosePanel() {
  const { live, poseOpts, setPoseOpts, poseFps, resetPose, exportPose, set } = useLive(useShallow((s) => ({
    live: s.live, poseOpts: s.poseOpts, setPoseOpts: s.setPoseOpts, poseFps: s.poseFps,
    resetPose: s.resetPose, exportPose: s.exportPose, set: s.set,
  })));

  return (
    <div className="space-y-2">
      <TrackerPicker />
      <div className="flex flex-wrap gap-3">
        <PosePreview />
        <div className="grid min-w-[150px] flex-1 grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-x-3 gap-y-2">
          <Select label="trails on" value={poseOpts.joints} onChange={(v) => setPoseOpts({ joints: v })}
                  options={POSE_TRAILS[poseOpts.tracker]} help="Which joints leave fading trails in the drawing. The tracker above decides what is tracked." />
          <Slider label="trail length" value={poseOpts.trace} min={0} max={4} step={0.1} format={(v) => (v ? `${v.toFixed(1)} s` : "off")}
                  onChange={(v) => setPoseOpts({ trace: v })} help="How far back the trails reach." />
          <Select label="framing" value={poseOpts.fit} onChange={(v) => setPoseOpts({ fit: v as typeof poseOpts.fit })} options={[
            { value: "camera", label: "camera" }, { value: "window", label: "follow" },
            { value: "body", label: poseOpts.tracker === "hands" ? "hands-centred" : poseOpts.tracker === "face" ? "face-centred" : "body-centred" },
          ]} help="camera keeps travel across the frame · follow frames the recent movement · centred removes travel and keeps gesture." />
          <Slider label="landmark smoothing" value={poseOpts.smooth} min={0} max={1} step={0.05}
                  onChange={(v) => setPoseOpts({ smooth: v })} help="One Euro filter on the landmarks: calms jitter when still, keeps fast gestures sharp. 0 = raw tracker." />
          <Slider label="tracking" value={poseFps} min={5} max={60} step={1} format={(v) => `${v} fps`}
                  onChange={(v) => set("poseFps", v)} help="Frames tracked per second (on the GPU). Denser tracking draws smoother trails." />
          <TrackRate />
          <div className="flex flex-wrap items-end gap-1.5">
            <button className="btn" onClick={resetPose} disabled={!live} title="clear the trajectory and its trails">reset</button>
            <button className="btn" onClick={() => exportPose("mp4", 10)} disabled={!live} title="last 10 s as an mp4 of moving landmarks">⤓ mp4</button>
            <button className="btn" onClick={() => exportPose("npz", 30)} disabled={!live} title="last 30 s of landmarks (t, N×7, skeleton name) as npz">⤓ npz</button>
          </div>
        </div>
      </div>
    </div>
  );
}
