"use client";
import * as React from "react";
import { useShallow } from "zustand/react/shallow";
import { useSidebar } from "@/lib/store";
import { useLive, useLiveSocket, videoFrameCount, MAX_VIDEO_FRAMES, type QueryMode } from "@/lib/liveGraph";
import { Slider } from "@/components/ui/Slider";
import { Select } from "@/components/ui/Select";
import { LiveDeltaGraph } from "./LiveDeltaGraph";
import { PosePanel } from "./PoseQuery";
import { MotionPanel } from "./MotionQuery";

/**
 * Live tab — the Explorer's Δ-graph driven by two continuously edited
 * sentences.  CONTEXT rewrites the cloud (which pairs move together);
 * QUERY picks out the overlap: nodes relevant to the query pull together
 * on short springs while the rest stretch away, all inside one damped
 * simulation that keeps animating as you type.
 */
export function LiveTab() {
  const spaceId = useSidebar((s) => s.spaceId);
  useLiveSocket();
  if (!spaceId) return <EmptyState />;
  return (
    <div className="w-full space-y-4">
      <Inputs />
      <LiveDeltaGraph />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.2fr,1fr]">
        <Overlap />
        <Knobs />
      </div>
    </div>
  );
}

function Inputs() {
  const { context, query, setContext, setQuery, status, pending, error, queryActive, nEdges } = useLive(useShallow((s) => ({
    context: s.context, query: s.query, setContext: s.setContext, setQuery: s.setQuery,
    status: s.status, pending: s.pending, error: s.error, queryActive: s.queryActive,
    nEdges: s.graph?.edges.length ?? 0,
  })));
  const explorerSentence = useSidebar((s) => s.contextSentence);
  const dot = status === "ready" ? (pending ? "bg-accent-300 animate-pulse" : "bg-accent-400") : status === "connecting" ? "bg-warning animate-pulse" : "bg-danger";

  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
      <div className="console-card" data-accent style={{ borderColor: "rgba(59,189,176,0.35)" }}>
        <div className="console-label" style={{ color: "#3bbdb0" }}>
          <span>◉ context</span>
          <span className="font-normal normal-case tracking-normal text-ink-500">rewrites the cloud · Δ-graph follows</span>
          {explorerSentence.trim() && explorerSentence.trim() !== context.trim() && (
            <button className="ml-auto text-[10px] normal-case tracking-normal text-accent-300 hover:text-accent-200" onClick={() => setContext(explorerSentence)}>
              ⤓ use Explorer prompt
            </button>
          )}
        </div>
        <textarea
          className="input-base min-h-[76px] resize-y text-[15px] leading-snug"
          placeholder="a slow fire under ice…"
          value={context}
          onChange={(e) => setContext(e.target.value)}
          spellCheck={false}
        />
      </div>
      <QueryCard dot={dot} status={status} nEdges={nEdges} queryActive={queryActive} error={error} />
    </div>
  );
}

const MODE_LABEL: Record<QueryMode, string> = { text: "text", speech: "speech → text", audio: "mic", image: "camera", video: "video", av: "audio + video", pose: "pose", motion: "movement → text" };
const MODE_HINT: Record<QueryMode, string> = {
  text: "typed sentence",
  speech: "the microphone transcribed (Whisper on Cloudflare); the last seconds of words are the query and it glides as they change",
  audio: "sliding window of microphone audio through the audio tower",
  image: "latest camera frame through the CLIP image tower",
  video: "sliding window of camera frames through the PE-AV video tower",
  av: "the same window of sound and frames through the joint PE-AV audio-visual tower",
  pose: "dance: MediaPipe landmarks (GPU, in the browser) drawn with temporal traces, embedded as a clip (video tower) or a still (image tower)",
  motion: "dance: movement qualities (effort, shape, rhythm …) measured from the landmarks and described in words; the words are the query",
};

function QueryCard({ dot, status, nEdges, queryActive, error }: { dot: string; status: string; nEdges: number; queryActive: boolean; error: string | null }) {
  const { query, setQuery, queryMode, setQueryMode, live, startLive, stopLive, liveError, audioLevel, queryKind,
          audioCapable, imageCapable, videoCapable, avCapable, poseCapable, sttCapable, transcript, speechChunk, speechKeep,
          audioWindow, videoWindow, videoFps, sendInterval, setSendInterval, motionRefresh,
          audioSmoothing, setAudioSmoothing, relSmoothing, setRelSmoothing, set, camVideo,
          mediaCenter, setMediaCenter, centerMemory, setCenterMemory, resetCenter, centered, centerN } = useLive(useShallow((s) => ({
    query: s.query, setQuery: s.setQuery, queryMode: s.queryMode, setQueryMode: s.setQueryMode, live: s.live,
    startLive: s.startLive, stopLive: s.stopLive, liveError: s.liveError, audioLevel: s.audioLevel, queryKind: s.queryKind,
    audioCapable: s.audioCapable, imageCapable: s.imageCapable, videoCapable: s.videoCapable, avCapable: s.avCapable,
    poseCapable: s.poseCapable, sttCapable: s.sttCapable, transcript: s.transcript, speechChunk: s.speechChunk,
    speechKeep: s.speechKeep, audioWindow: s.audioWindow, videoWindow: s.videoWindow, videoFps: s.videoFps,
    sendInterval: s.sendInterval, setSendInterval: s.setSendInterval, motionRefresh: s.motionRefresh,
    audioSmoothing: s.audioSmoothing, setAudioSmoothing: s.setAudioSmoothing, relSmoothing: s.relSmoothing, setRelSmoothing: s.setRelSmoothing,
    set: s.set, camVideo: s.camVideo,
    mediaCenter: s.mediaCenter, setMediaCenter: s.setMediaCenter, centerMemory: s.centerMemory, setCenterMemory: s.setCenterMemory,
    resetCenter: s.resetCenter, centered: s.centered, centerN: s.centerN,
  })));
  const backend = useSidebar((s) => s.embedderBackend);
  const can: Record<QueryMode, boolean> = {
    text: true,
    speech: sttCapable,
    audio: audioCapable || backend === "clap" || backend === "peav" || backend === "gemini-embedding-2",
    image: imageCapable || backend === "clip",
    video: videoCapable || backend === "peav",
    av: avCapable || backend === "peav",
    // the dance is drawn, then embedded by whichever visual tower the space has
    pose: poseCapable && (videoCapable || imageCapable || backend === "peav" || backend === "clip"),
    // movement words go through the text tower, which every backend has
    motion: poseCapable,     // movement words need the server's pose stack (local installs, not the hosted site)
  };
  // show text + every live mode this space supports; if it supports none,
  // show the three generic ones greyed out so the hint explains what to build
  const modes: QueryMode[] = (["text", "speech", "audio", "image", "video", "av", "pose", "motion"] as QueryMode[]).filter((m) => can[m]);
  if (modes.length === 1) modes.push("audio", "video", "av");
  const prefix = backend === "peav" ? "PE-AV" : backend === "clap" ? "CLAP" : backend === "clip" ? "CLIP"
    : backend === "gemini-embedding-2" ? "Gemini" : "";
  const usesMic = queryMode === "audio" || queryMode === "av" || queryMode === "speech";
  // media queries (sound, frames, the drawn dancer) are centred on the session's mean server-side
  const centerable = queryMode === "audio" || queryMode === "image" || queryMode === "video" || queryMode === "av" || queryMode === "pose";
  const usesCam = queryMode === "image" || queryMode === "video" || queryMode === "av" || queryMode === "pose" || queryMode === "motion";
  const poseClip = queryMode === "pose" && videoCapable;
  const level = Math.min(1, audioLevel * 6);
  const nFrames = videoFrameCount({ videoWindow, videoFps });
  const previewRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const host = previewRef.current;
    if (!host) return;
    host.innerHTML = "";
    if (camVideo) {
      camVideo.className = "h-full w-full object-cover";
      host.appendChild(camVideo);
    }
  }, [camVideo, queryMode]);

  return (
    <div className="console-card" data-accent style={{ borderColor: "rgba(194,166,254,0.35)" }}>
      <div className="console-label" style={{ color: "#c2a6fe" }}>
        <span>⌕ query</span>
        <span className="font-normal normal-case tracking-normal text-ink-500">pulls its overlap together · stretches the rest</span>
        <span className="ml-auto flex items-center gap-2 normal-case tracking-normal">
          <span className="inline-flex overflow-hidden rounded-md border border-ink-700 text-[10px]">
            {modes.map((m) => (
              <button
                key={m}
                onClick={() => setQueryMode(m)}
                className={`px-2 py-0.5 transition ${queryMode === m ? "bg-[#c2a6fe]/25 text-ink-50" : can[m] ? "text-ink-400 hover:text-ink-100" : "text-ink-600"}`}
                title={MODE_HINT[m]}
              >
                {m === "text" || m === "speech" || m === "motion" || !prefix ? MODE_LABEL[m] : `${prefix} · ${MODE_LABEL[m]}`}
              </button>
            ))}
          </span>
          <span className="flex items-center gap-1.5 text-[10px] text-ink-500">
            <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
            <span className="font-mono">{status}{status === "ready" ? ` · ${nEdges} edges${queryActive ? ` · ${queryKind} query on` : ""}` : ""}</span>
          </span>
        </span>
      </div>

      {queryMode === "text" ? (
        <textarea
          className="input-base min-h-[76px] resize-y text-[15px] leading-snug"
          placeholder="what are you looking for in it? e.g. grief, shelter, a threshold…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          spellCheck={false}
        />
      ) : (
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <button
              className={`btn ${live ? "border-[#c2a6fe] text-[#e2d6ff]" : ""}`}
              onClick={() => (live ? stopLive() : startLive())}
              disabled={!can[queryMode]}
              title={can[queryMode] ? "toggle live capture" : "build the space with a backend that has this tower first"}
            >
              {live ? "■ stop" : queryMode === "pose" || queryMode === "motion" ? "● dance" : usesCam && usesMic ? "● listen + watch" : usesCam ? "● watch" : "● listen"}
            </button>
            {usesCam && <div ref={previewRef} className="h-[64px] w-[86px] shrink-0 overflow-hidden rounded-md bg-ink-950/60" />}
            {usesMic && (
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-800">
                <div className="h-full rounded-full transition-[width] duration-75" style={{ width: `${Math.round(level * 100)}%`, background: "linear-gradient(90deg,#c2a6fe,#3bbdb0)" }} />
              </div>
            )}
            <span className="ml-auto font-mono text-[10px] text-ink-500">
              {live
                ? queryMode === "speech" ? `${speechChunk.toFixed(1)} s chunks`
                  : queryMode === "audio" ? `${audioWindow.toFixed(1)} s / ${sendInterval} ms`
                  : queryMode === "image" ? `1 frame / ${sendInterval} ms`
                  : queryMode === "video" ? `${videoWindow.toFixed(1)} s · ${nFrames} f / ${sendInterval} ms`
                  : queryMode === "pose" ? `${poseClip ? `${videoWindow.toFixed(1)} s · ${nFrames} f` : "still"} / ${sendInterval} ms`
                  : queryMode === "motion" ? `words / ${motionRefresh.toFixed(2)} s`
                  : `${Math.max(audioWindow, videoWindow).toFixed(1)} s · ${nFrames} f / ${sendInterval} ms`
                : "off"}
            </span>
          </div>
          {queryMode === "speech" && (
            <div className="min-h-[44px] rounded-md bg-ink-950/50 px-3 py-2 text-[13px] leading-snug text-ink-200">
              {transcript || <span className="text-ink-500">{live ? "listening…" : "press listen and speak"}</span>}
            </div>
          )}
          {centerable && (
            <div className="flex flex-wrap items-center gap-3 text-[11px] text-ink-400">
              <label className="flex cursor-pointer items-center gap-1.5" title="Every sound shares one direction that pulls a few words to the top of every query. Subtracting the mean of this session's recent queries removes that bias, so the words follow what changes.">
                <input type="checkbox" checked={mediaCenter} onChange={(e) => setMediaCenter(e.target.checked)} />
                centre on this session
              </label>
              <span className="font-mono text-ink-500">
                {!mediaCenter ? "raw" : centered ? `centred on ${centerN} queries` : centerN >= 5 ? "nothing varies yet" : `warming up · ${centerN} / 5`}
              </span>
              {mediaCenter && <button className="text-accent-300 hover:text-accent-200" onClick={resetCenter}>↺ forget</button>}
            </div>
          )}
          {queryMode === "pose" && <PosePanel />}
          {queryMode === "motion" && <MotionPanel />}
          {queryMode === "pose" && !can.pose && (
            <div className="text-[11px] text-warning">
              {poseCapable
                ? <>Pose needs a visual tower. Choose <span className="mx-1 font-mono">Local · PE-AV small</span> (clips) or <span className="mx-1 font-mono">Local · CLIP ViT-B/32</span> (stills) and rebuild.</>
                : <>The server has no MediaPipe. Install it in the backend environment (<span className="font-mono">pip install mediapipe</span>) and restart.</>}
            </div>
          )}
          {queryMode !== "pose" && !can[queryMode] && (
            <div className="text-[11px] text-warning">
              This space was built with <span className="font-mono">{backend}</span>, which has no {MODE_LABEL[queryMode]} tower. Choose
              <span className="mx-1 font-mono">{queryMode === "image" ? "Local · CLIP ViT-B/32" : queryMode === "audio" ? "Gemini Embedding 2 (or local CLAP)" : "Local · PE-AV small"}</span>
              in the sidebar and rebuild — descriptors and media then share one space.
            </div>
          )}
          {liveError && <div className="text-[11px] text-danger">{liveError}</div>}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {queryMode === "speech" && <Slider label="chunk" value={speechChunk} min={1.5} max={8} step={0.5} format={(v) => `${v.toFixed(1)} s`} onChange={(v) => set("speechChunk", v)} help="Seconds of speech per transcription. Shorter is more reactive and cuts more words." />}
            {queryMode === "speech" && <Slider label="memory" value={speechKeep} min={5} max={120} step={5} format={(v) => `${v} s`} onChange={(v) => { set("speechKeep", v); useLive.getState()._send?.({ type: "stt", keep: v }); }} help="Seconds of words kept in the query." />}
            {usesMic && queryMode !== "speech" && <Slider label="audio window" value={audioWindow} min={0.5} max={6} step={0.1} format={(v) => `${v.toFixed(1)} s`} onChange={(v) => set("audioWindow", v)} help="Seconds of recent sound embedded each time." />}
            {(queryMode === "video" || queryMode === "av" || poseClip) && <Slider label="video window" value={videoWindow} min={0.5} max={10} step={0.1} format={(v) => `${v.toFixed(1)} s`} onChange={(v) => set("videoWindow", v)} help="Seconds of recent frames fed to the model each time." />}
            {(queryMode === "video" || queryMode === "av" || poseClip) && <Slider label="model fps" value={videoFps} min={1} max={16} step={1} format={(v) => `${v} fps · ${nFrames} f`} onChange={(v) => set("videoFps", v)} help={`Frames per second inside the window (camera is sampled at this rate). window × fps = ${nFrames} frames per embedding, capped at ${MAX_VIDEO_FRAMES}; ~55 ms per frame on the GPU.`} />}
            {queryMode === "image" && <Slider label="capture" value={videoFps} min={1} max={16} step={1} format={(v) => `${v} fps`} onChange={(v) => set("videoFps", v)} help="How often a camera frame is grabbed." />}
            {queryMode !== "motion" && <Slider label="rate" value={sendInterval} min={150} max={2000} step={50} format={(v) => `${v} ms`} onChange={setSendInterval} help="How often a window is sent for embedding (latest wins if the GPU lags)." />}
            {queryMode !== "motion" && <Slider label="vector smoothing" value={audioSmoothing} min={0} max={0.95} step={0.05} onChange={setAudioSmoothing} help="Server-side blend with the previous media vector, before scoring." />}
            {centerable && <Slider label="centre memory" value={centerMemory} min={10} max={600} step={10} format={(v) => `${v} s`} onChange={setCenterMemory} help="Seconds of past queries averaged into the session's mean sound (or frame)." />}
            <Slider label="activation smoothing" value={relSmoothing} min={0} max={0.97} step={0.01} onChange={setRelSmoothing} help="Server-side blend of the per-node relevance with its previous value, after embedding — calms the springs without blurring the embedding." />
          </div>
        </div>
      )}
      {error && <div className="mt-1 text-[11px] text-danger">{error}</div>}
    </div>
  );
}

/** The overlap subgraph as text: nodes above the relevance cut, grouped by archetype. */
function Overlap() {
  const { graph, relevance, queryActive, focus, threshold } = useLive(useShallow((s) => ({
    graph: s.graph, relevance: s.relevance, queryActive: s.queryActive, focus: s.focus, threshold: s.threshold,
  })));
  const colorMap = useSidebar((s) => s.colorMap);
  const rows = React.useMemo(() => {
    if (!graph || !queryActive) return [];
    return graph.nodes
      .map((n) => ({ ...n, rel: Math.pow(relevance[n.id] ?? 0, focus) }))
      .filter((n) => n.rel >= threshold)
      .sort((a, b) => b.rel - a.rel);
  }, [graph, relevance, queryActive, focus, threshold]);
  const inIds = new Set(rows.map((r) => r.id));
  const inEdges = graph && queryActive ? graph.edges.filter((e) => inIds.has(e.source) && inIds.has(e.target)) : [];
  const bySym = new Map<string, typeof rows>();
  for (const r of rows) bySym.set(r.symbol, [...(bySym.get(r.symbol) ?? []), r]);

  return (
    <div className="panel-tight">
      <div className="mb-2 flex items-center justify-between">
        <div className="section-title">Overlap subgraph</div>
        <span className="text-[10px] text-ink-500">
          {queryActive ? `${rows.length} nodes · ${inEdges.length} edges above ${threshold.toFixed(2)}` : "type a query to select"}
        </span>
      </div>
      {!queryActive && (
        <p className="text-[12px] leading-relaxed text-ink-400">
          The context decides which descriptor pairs are moving together. The query decides which of
          those you are looking at: every node of the Δ-graph is scored against the query, edges whose two
          ends both score high contract, everything else stretches out. What gathers in the middle is the
          overlap of the two sentences.
        </p>
      )}
      {queryActive && rows.length === 0 && <div className="text-[12px] text-ink-400">nothing above the cut — lower the threshold or soften the focus</div>}
      {rows.length > 0 && (
        <div className="space-y-2">
          {Array.from(bySym.entries()).map(([sym, items]) => (
            <div key={sym} className="flex flex-wrap items-center gap-1.5">
              <span className="mr-1 inline-flex items-center gap-1.5 text-[11px] font-semibold text-ink-200">
                <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: colorMap[sym] ?? "#888" }} />{sym}
              </span>
              {items.map((n) => (
                <span key={n.id} className="pill" style={{ borderColor: `${colorMap[sym] ?? "#888"}66`, opacity: 0.55 + 0.45 * n.rel }} title={`relevance ${n.rel.toFixed(2)}`}>
                  {n.id}<span className="font-mono text-[9px] text-ink-400">{Math.round(n.rel * 100)}</span>
                </span>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Knobs() {
  const { params, setParams, focus, contraction, damping, stiffness, repulsion, unlinked, threshold, set } = useLive(useShallow((s) => ({
    params: s.params, setParams: s.setParams, focus: s.focus, contraction: s.contraction, damping: s.damping, stiffness: s.stiffness, repulsion: s.repulsion, unlinked: s.unlinked, threshold: s.threshold, set: s.set,
  })));
  const critical = 2 * Math.sqrt(stiffness);
  return (
    <div className="panel-tight space-y-3">
      <div className="section-title">Shaping</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-3">
        <Slider label="query focus" value={focus} min={0.5} max={4} step={0.1} onChange={(v) => set("focus", v)} help="Sharpens relevance (exponent). High = only the closest nodes count as overlap." />
        <Slider label="contraction" value={contraction} min={0} max={1} step={0.01} onChange={(v) => set("contraction", v)} help="How much the query reshapes the springs: overlap edges shorten, the rest lengthen and drift outward." />
        <Slider label="stiffness k" value={stiffness} min={5} max={200} step={1} format={(v) => `${v.toFixed(0)} · T≈${(2 * Math.PI / Math.sqrt(v)).toFixed(1)}s`} onChange={(v) => set("stiffness", v)} help="Spring stiffness. Natural period T = 2π/√k — how fast a rest-length change is followed." />
        <Slider label="damping c" value={damping} min={0.5} max={40} step={0.1} format={(v) => `${v.toFixed(1)} /s${v < critical * 0.9 ? " · under" : v > critical * 1.1 ? " · over" : " · critical"}`} onChange={(v) => set("damping", v)} help="The single viscous drag on every node's velocity (a = F − c·v). Rotation, translation and spring motion all decay as e^(−c·t). Critical damping is 2√k." />
        <Slider label="repulsion" value={repulsion} min={0} max={200000} step={1000} format={(v) => `${(v / 1000).toFixed(1)}k`} onChange={(v) => set("repulsion", v)} help="Base repulsion between every pair of nodes (softened 1/r², symmetric). Keeps neighbours from overlapping; higher spreads the whole graph." />
        <Slider label="disentangle" value={unlinked} min={0} max={30000} step={500} format={(v) => `${(v / 1000).toFixed(1)}k`} onChange={(v) => set("unlinked", v)} help="Extra repulsion between nodes that share no edge (softened 1/r, symmetric). Unrelated clusters slide apart; connected pairs feel only their spring and the base repulsion." />
        <Slider label="overlap cut" value={threshold} min={0.1} max={0.95} step={0.01} onChange={(v) => set("threshold", v)} help="Relevance threshold for the overlap-subgraph readout below the graph." />
        {params && (
          <>
            <Slider label="top |Δ| edges" value={params.top_abs_edges} min={5} max={150} step={1} onChange={(v) => setParams({ top_abs_edges: v })} help="How many strongest Δ edges form the context graph." />
            <Slider label="β shift" value={params.beta} min={0} max={3} step={0.05} onChange={(v) => setParams({ beta: v })} help="Context shift strength (same β as the Explorer)." />
            <Select label="Δ sign" value={params.sign_filter ?? "both"} onChange={(v) => setParams({ sign_filter: v === "both" ? null : (v as "up" | "down") })} options={[
              { value: "up", label: "↑ strengthens" }, { value: "down", label: "↓ weakens" }, { value: "both", label: "both" },
            ]} />
            <Select label="shift strategy" value={params.strategy} onChange={(v) => setParams({ strategy: v as any })} options={[
              { value: "gate", label: "gate (fast)" }, { value: "pooling", label: "pooling" }, { value: "hybrid", label: "hybrid (slow)" }, { value: "reembed", label: "reembed (slow)" },
            ]} help="reembed / hybrid re-encode every descriptor per edit — fine locally, slow with a remote embedder." />
          </>
        )}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="mx-auto max-w-2xl rounded-2xl border border-ink-700/60 bg-ink-900/40 p-8 text-center">
      <div className="font-display text-2xl text-ink-50">Build a space first</div>
      <p className="mt-2 text-sm text-ink-300">
        The live Δ-graph follows a context and a query as you type. Pick a preset and press
        <span className="ml-1 font-mono text-accent-300">Build space</span> to start.
      </p>
    </div>
  );
}
