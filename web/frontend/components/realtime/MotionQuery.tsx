"use client";
import * as React from "react";
import { useShallow } from "zustand/react/shallow";
import { useLive, type MotionQuality, type MotionReading } from "@/lib/liveGraph";
import { Slider } from "@/components/ui/Slider";
import { TrackerPicker, PosePreview, TrackRate } from "./PoseQuery";

const NAMES: Record<string, string> = {
  energy: "energy", time: "time", weight: "weight", space: "space", flow: "flow",
  shape: "shape", level: "level", symmetry: "symmetry", rhythm: "rhythm",
  openness: "openness", articulation: "articulation", hand_speed: "speed",
  hand_space: "paths", hand_flow: "smoothness", together: "apart ↔ together",
};


function QualityBar({ q, stale }: { q: MotionQuality; stale?: boolean }) {
  const v = q.value;
  const salient = v !== null && (v < 0.4 || v > 0.6);
  const raw = q.raw === null ? "–" : Math.abs(q.raw) >= 100 ? q.raw.toFixed(0) : q.raw.toFixed(2);
  return (
    <div className={`grid grid-cols-[minmax(0,1fr)_minmax(80px,1.4fr)_minmax(0,1fr)] items-center gap-2 text-[10px] transition-opacity ${stale ? "opacity-40" : ""}`}
         title={`${NAMES[q.key] ?? q.key}: ${v === null ? "not measurable here" : v.toFixed(2)} · raw ${raw} ${q.unit}${stale ? " · last measured value (not in this window)" : ""}`}>
      <span className={`truncate text-right ${v !== null && v < 0.4 ? "text-ink-100" : "text-ink-500"}`}>{q.low}</span>
      <div className="relative h-1.5 rounded-full bg-ink-800">
        <div className="absolute inset-y-0 left-[40%] right-[40%] bg-ink-700/70" />
        {v !== null && (
          <div className={`absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ${salient ? "bg-[#c2a6fe]" : "bg-ink-500"}`}
               style={{ left: `${Math.round(v * 100)}%` }} />
        )}
      </div>
      <span className={`truncate ${v !== null && v > 0.6 ? "text-ink-100" : "text-ink-500"}`}>{q.high}</span>
    </div>
  );
}

const TRENDS: [string, (v: number) => string | null][] = [
  ["opening", (v) => (v > 0.25 ? "opening" : v < -0.25 ? "closing" : null)],
  ["rising", (v) => (v > 0.2 ? "rising" : v < -0.2 ? "sinking" : null)],
  ["accelerating", (v) => (v > 0.35 ? "speeding up" : v < -0.35 ? "settling" : null)],
  ["travel", (v) => (v > 0.25 ? "travelling" : null)],
  ["approach", (v) => (v > 0.12 ? "approaching" : v < -0.12 ? "receding" : null)],
  ["turning", (v) => (v > 45 ? "turning" : null)],
  ["tempo_bpm", (v) => `≈ ${Math.round(v / 5) * 5} bpm`],
];

/** The qualities to show, held steady: a reading without some measure (the
 *  dancer out of frame for most of the window, a hand missing) keeps that
 *  quality's last value, dimmed, instead of dropping the bar.  Cleared when
 *  the tracker changes. */
function useSteadyQualities(motion: MotionReading | null): { q: MotionQuality; stale: boolean }[] {
  const held = React.useRef<{ tracker: string; byKey: Map<string, MotionQuality> }>({ tracker: "", byKey: new Map() });
  return React.useMemo(() => {
    if (!motion) return [];
    if (held.current.tracker !== motion.tracker) held.current = { tracker: motion.tracker, byKey: new Map() };
    const fresh = new Set<string>();
    for (const q of [...motion.qualities, ...motion.hands]) {
      if (q.value === null) continue;
      held.current.byKey.set(q.key, q);
      fresh.add(q.key);
    }
    return Array.from(held.current.byKey.values()).map((q) => ({ q, stale: !fresh.has(q.key) }));
  }, [motion]);
}

/** "applied 0.4 s ago", refreshed four times a second. */
function Applied() {
  const { applied, live, refresh } = useLive(useShallow((s) => ({ applied: s.motionApplied, live: s.live, refresh: s.motionRefresh })));
  const [now, setNow] = React.useState(() => performance.now());
  React.useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 250);
    return () => clearInterval(id);
  }, []);
  if (!live) return null;
  if (!applied) return <span className="text-[10px] text-ink-500">waiting for the first description…</span>;
  const age = (now - applied.at) / 1000;
  const fresh = age < Math.max(1.5, refresh * 1.5);
  return (
    <span className="flex items-center gap-1.5 text-[10px] text-ink-400" title={`query now: “${applied.text}”`}>
      <span className={`h-1.5 w-1.5 rounded-full ${fresh ? "bg-accent-400" : "bg-warning"}`} />
      query updated {age < 10 ? `${age.toFixed(1)} s` : `${Math.round(age)} s`} ago
    </span>
  );
}

/**
 * Movement → text — the dancer's movement described in words, and the
 * words are the query.
 *
 * The same GPU landmark tracking as pose mode feeds delyrism/motion.py on
 * the server, which measures movement qualities over the last few seconds
 * (Laban's Time, Weight, Space and Flow, plus shape, level, symmetry,
 * rhythm, travel and turning; hands add openness and articulation) and
 * writes the ones that stand out as one layered description that names no
 * body part: the effort action or state, the qualities behind it, the
 * movement in space, and its timing.
 * Every refresh the new description replaces the query (no blending), with
 * the shared "movement description" direction removed by `contrast` so the
 * differences between descriptions move the graph.  The text goes through
 * the space's text tower, so this works with every backend.
 */
export function MotionPanel() {
  const { motion, motionWindow, motionContrast, motionRefresh, setMotionRefresh, poseFps, poseOpts,
          setPoseOpts, set, live, setQuery } = useLive(useShallow((s) => ({
    motion: s.motion, motionWindow: s.motionWindow, motionContrast: s.motionContrast,
    motionRefresh: s.motionRefresh, setMotionRefresh: s.setMotionRefresh, poseFps: s.poseFps, poseOpts: s.poseOpts,
    setPoseOpts: s.setPoseOpts, set: s.set, live: s.live, setQuery: s.setQuery,
  })));
  const trends = motion
    ? TRENDS.map(([k, f]) => { const v = motion.trends[k]; return v === null || v === undefined ? null : f(v); }).filter(Boolean)
    : [];
  const quals = useSteadyQualities(motion);
  const [showQualities, setShowQualities] = React.useState(false);     // collapsed by default

  // the preview's trails span the description window, so the drawing shows
  // the stretch of movement being described; pose mode's own trail length
  // comes back when this panel closes
  const send = useLive((s) => s._send);
  React.useEffect(() => {
    send?.({ type: "pose_params", trace: motionWindow });
  }, [send, motionWindow]);
  React.useEffect(() => () => {
    const s = useLive.getState();
    s._send?.({ type: "pose_params", trace: s.poseOpts.trace });
  }, []);

  return (
    <div className="space-y-2">
      <TrackerPicker allow={["body", "hands", "both"]} />
      <div className="flex flex-wrap gap-3">
        <PosePreview size={96} />
        <div className="min-w-[220px] flex-1 space-y-1">
          <div className="label-sm flex flex-wrap items-center gap-x-2">
            <span>description → query</span>
            {motion && <span className="font-mono normal-case text-ink-500">{Math.round(motion.coverage * 100)}% tracked · last {motion.seconds.toFixed(1)} s</span>}
            <span className="ml-auto normal-case tracking-normal"><Applied /></span>
          </div>
          <p className="min-h-[60px] rounded-md border border-ink-700/70 bg-ink-950/40 px-3 py-2 text-[14px] leading-snug text-ink-100">
            {!live ? <span className="text-ink-500">Start moving to hear the movement described.</span>
              : motion?.text ? motion.text
              : <span className="text-ink-500">{motion ? "Nothing tracked yet, or nothing moving enough to describe." : "Waiting for the first window…"}</span>}
          </p>
          <div className="flex flex-wrap items-center gap-1.5">
            {trends.map((t) => <span key={t} className="pill text-[10px]">{t}</span>)}
            <button className="btn ml-auto text-[11px]" disabled={!motion?.text}
                    onClick={() => motion?.text && setQuery(motion.text)}
                    title="copy this description into the typed query (switch to text to freeze it there)">
              ⤓ copy to text query
            </button>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-x-3 gap-y-2">
        <Slider label="refresh" value={motionRefresh} min={0.25} max={5} step={0.25} format={(v) => `every ${v.toFixed(2)} s`}
                onChange={setMotionRefresh} help="How often a new description is generated from the movement. Each one replaces the query." />
        <Slider label="window" value={motionWindow} min={1} max={8} step={0.5} format={(v) => `${v.toFixed(1)} s`}
                onChange={(v) => set("motionWindow", v)} help="Seconds of movement each description covers. Longer is steadier and catches rhythm; shorter reacts faster." />
        <Slider label="contrast" value={motionContrast} min={0} max={1} step={0.05}
                onChange={(v) => set("motionContrast", v)} help="Removes the direction every movement description shares, so what is distinctive in this one moves the graph. 0 = the raw description; 0.7 lets contrasting movements reorder the graph about twice as much." />
        <Slider label="tracking" value={poseFps} min={10} max={60} step={1} format={(v) => `${v} fps`}
                onChange={(v) => set("poseFps", v)} help="Frames tracked per second (on the GPU). The qualities need at least 20 for speed and smoothness." />
        <TrackRate />
        <Slider label="landmark smoothing" value={poseOpts.smooth} min={0} max={1} step={0.05}
                onChange={(v) => setPoseOpts({ smooth: v })} help="One Euro filter on the landmarks. Some smoothing keeps tracker jitter from reading as jerky movement." />
      </div>
      <div className="rounded-md border border-ink-800 bg-ink-950/30">
        <button className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left text-[10px] uppercase tracking-wider text-ink-400 hover:text-ink-100"
                onClick={() => setShowQualities((v) => !v)} aria-expanded={showQualities}>
          <span className={`inline-block transition-transform ${showQualities ? "rotate-90" : ""}`}>▸</span>
          movement qualities
          <span className="normal-case tracking-normal text-ink-600">
            {quals.length ? `${quals.length} measured` : "measured once you move"}
          </span>
        </button>
        {showQualities && quals.length > 0 && (
          <div className="grid grid-cols-1 gap-x-6 gap-y-1 px-3 pb-2 md:grid-cols-2">
            {quals.map(({ q, stale }) => (
              <div key={q.key} className="space-y-0.5">
                <div className="text-[9px] uppercase tracking-wider text-ink-500">{NAMES[q.key] ?? q.key}</div>
                <QualityBar q={q} stale={stale} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
