"use client";
import * as React from "react";
import { useShallow } from "zustand/react/shallow";
import { useLive } from "@/lib/liveGraph";
import { Slider } from "@/components/ui/Slider";
import { Select } from "@/components/ui/Select";

/**
 * Pose query — the dancer defines the query with their movement.
 *
 * In pose mode the camera streams every frame to the server, MediaPipe
 * tracks 33 body landmarks into a running trajectory, and the trajectory is
 * drawn back as moving landmarks with temporal traces.  That drawing is
 * what gets embedded on each send tick: the last video-window seconds as a
 * clip through a video tower (PE-AV), or the current pose with its traces
 * through an image tower (CLIP).  The preview is the drawing itself,
 * refreshed on every tracked frame.  Capture, rate and the two smoothings
 * live in the shared query controls.
 */
export function PosePanel() {
  const { live, posePreview, posePresent, poseOpts, setPoseOpts, poseFps, resetPose, exportPose, set } = useLive(useShallow((s) => ({
    live: s.live, posePreview: s.posePreview, posePresent: s.posePresent, poseOpts: s.poseOpts,
    setPoseOpts: s.setPoseOpts, poseFps: s.poseFps, resetPose: s.resetPose, exportPose: s.exportPose, set: s.set,
  })));

  return (
    <div className="flex gap-3">
      <div className="relative h-[132px] w-[132px] shrink-0 overflow-hidden rounded-md border border-ink-700 bg-black">
        {live && posePreview && <img src={posePreview} alt="pose drawing sent to the embedder" className="h-full w-full" />}
        {live && (
          <span className={`absolute left-1.5 top-1.5 h-2 w-2 rounded-full ${posePresent ? "bg-accent-400" : "bg-warning animate-pulse"}`}
                title={posePresent ? "dancer in frame" : "nobody in frame — the query holds"} />
        )}
        {!live && <span className="absolute inset-0 flex items-center justify-center text-[10px] text-ink-600">pose drawing</span>}
      </div>
      <div className="grid min-w-0 flex-1 grid-cols-2 gap-x-3 gap-y-2 md:grid-cols-3">
        <Slider label="trace" value={poseOpts.trace} min={0} max={4} step={0.1} format={(v) => (v ? `${v.toFixed(1)} s` : "off")}
                onChange={(v) => setPoseOpts({ trace: v })} help="How far back the fading trails of the traced joints reach." />
        <Slider label="landmark smoothing" value={poseOpts.smooth} min={0} max={1} step={0.05}
                onChange={(v) => setPoseOpts({ smooth: v })} help="One Euro filter on the landmarks: calms jitter when still, keeps fast gestures sharp. 0 = raw tracker." />
        <Slider label="tracking" value={poseFps} min={5} max={30} step={1} format={(v) => `${v} fps`}
                onChange={(v) => set("poseFps", v)} help="Camera frames tracked per second. Denser tracking draws smoother traces; ~40 ms per frame on CPU." />
        <Select label="framing" value={poseOpts.fit} onChange={(v) => setPoseOpts({ fit: v as typeof poseOpts.fit })} options={[
          { value: "camera", label: "camera" }, { value: "window", label: "follow" }, { value: "body", label: "body-centred" },
        ]} help="camera keeps travel across the room · follow frames the recent movement · body-centred removes travel and keeps gesture." />
        <Select label="traced joints" value={poseOpts.joints} onChange={(v) => setPoseOpts({ joints: v as typeof poseOpts.joints })} options={[
          { value: "extremities", label: "head · hands · feet" }, { value: "hands", label: "hands" }, { value: "feet", label: "feet" },
          { value: "limbs", label: "elbows → ankles" }, { value: "all", label: "all" },
        ]} />
        <div className="flex flex-wrap items-end gap-1.5">
          <button className="btn" onClick={resetPose} disabled={!live} title="clear the trajectory and its traces">reset</button>
          <button className="btn" onClick={() => exportPose("mp4", 10)} disabled={!live} title="last 10 s as an mp4 of moving landmarks">⤓ mp4</button>
          <button className="btn" onClick={() => exportPose("npz", 30)} disabled={!live} title="last 30 s of landmarks (t, 33×7) as npz">⤓ npz</button>
        </div>
      </div>
    </div>
  );
}
