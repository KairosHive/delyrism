"use client";
import * as React from "react";
import { useShallow } from "zustand/react/shallow";
import { useSidebar } from "@/lib/store";
import { useLive, useLiveSocket } from "@/lib/liveGraph";
import { Slider } from "@/components/ui/Slider";
import { Select } from "@/components/ui/Select";
import { LiveDeltaGraph } from "./LiveDeltaGraph";

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

function QueryCard({ dot, status, nEdges, queryActive, error }: { dot: string; status: string; nEdges: number; queryActive: boolean; error: string | null }) {
  const { query, setQuery, queryMode, setQueryMode, listening, startListening, stopListening, audioLevel, micError, audioCapable,
          audioWindow, audioInterval, audioSmoothing, setAudioSmoothing, set, queryKind } = useLive(useShallow((s) => ({
    query: s.query, setQuery: s.setQuery, queryMode: s.queryMode, setQueryMode: s.setQueryMode, listening: s.listening,
    startListening: s.startListening, stopListening: s.stopListening, audioLevel: s.audioLevel, micError: s.micError,
    audioCapable: s.audioCapable, audioWindow: s.audioWindow, audioInterval: s.audioInterval, audioSmoothing: s.audioSmoothing,
    setAudioSmoothing: s.setAudioSmoothing, set: s.set, queryKind: s.queryKind,
  })));
  const backend = useSidebar((s) => s.embedderBackend);
  const clapOk = audioCapable || backend === "clap";
  const level = Math.min(1, audioLevel * 6);

  return (
    <div className="console-card" data-accent style={{ borderColor: "rgba(194,166,254,0.35)" }}>
      <div className="console-label" style={{ color: "#c2a6fe" }}>
        <span>⌕ query</span>
        <span className="font-normal normal-case tracking-normal text-ink-500">pulls its overlap together · stretches the rest</span>
        <span className="ml-auto flex items-center gap-2 normal-case tracking-normal">
          <span className="inline-flex overflow-hidden rounded-md border border-ink-700 text-[10px]">
            {(["text", "clap"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setQueryMode(m)}
                className={`px-2 py-0.5 transition ${queryMode === m ? "bg-[#c2a6fe]/25 text-ink-50" : "text-ink-400 hover:text-ink-100"}`}
                title={m === "clap" ? "continuous microphone through the CLAP audio tower" : "typed sentence"}
              >
                {m === "clap" ? "CLAP · mic" : "text"}
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
              className={`btn ${listening ? "border-[#c2a6fe] text-[#e2d6ff]" : ""}`}
              onClick={() => (listening ? stopListening() : startListening())}
              disabled={!clapOk}
              title={clapOk ? "toggle the microphone" : "build the space with the CLAP backend first"}
            >
              {listening ? "■ stop" : "● listen"}
            </button>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-800">
              <div className="h-full rounded-full transition-[width] duration-75" style={{ width: `${Math.round(level * 100)}%`, background: "linear-gradient(90deg,#c2a6fe,#3bbdb0)" }} />
            </div>
            <span className="font-mono text-[10px] text-ink-500">{listening ? `${audioWindow.toFixed(1)} s / ${audioInterval} ms` : "mic off"}</span>
          </div>
          {!clapOk && (
            <div className="text-[11px] text-warning">
              This space was built with <span className="font-mono">{backend}</span>, which has no audio tower. Choose
              <span className="mx-1 font-mono">Local · CLAP (audio+text)</span> in the sidebar and rebuild — descriptors and sound then share one space.
            </div>
          )}
          {micError && <div className="text-[11px] text-danger">{micError}</div>}
          <div className="grid grid-cols-3 gap-3">
            <Slider label="window" value={audioWindow} min={0.5} max={6} step={0.1} format={(v) => `${v.toFixed(1)} s`} onChange={(v) => set("audioWindow", v)} help="Seconds of recent audio embedded each time." />
            <Slider label="rate" value={audioInterval} min={150} max={2000} step={50} format={(v) => `${v} ms`} onChange={(v) => set("audioInterval", v)} help="How often a window is sent." />
            <Slider label="smoothing" value={audioSmoothing} min={0} max={0.95} step={0.05} onChange={setAudioSmoothing} help="Server-side blend with the previous audio vector — higher is calmer." />
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
  const { params, setParams, focus, contraction, damping, threshold, set } = useLive(useShallow((s) => ({
    params: s.params, setParams: s.setParams, focus: s.focus, contraction: s.contraction, damping: s.damping, threshold: s.threshold, set: s.set,
  })));
  return (
    <div className="panel-tight space-y-3">
      <div className="section-title">Shaping</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-3">
        <Slider label="query focus" value={focus} min={0.5} max={4} step={0.1} onChange={(v) => set("focus", v)} help="Sharpens relevance (exponent). High = only the closest nodes count as overlap." />
        <Slider label="contraction" value={contraction} min={0} max={1} step={0.01} onChange={(v) => set("contraction", v)} help="How much the query reshapes the springs: overlap edges shorten, the rest lengthen and drift outward." />
        <Slider label="damping" value={damping} min={0.1} max={0.8} step={0.01} onChange={(v) => set("damping", v)} help="Velocity damping of the spring simulation. Low = bouncy, high = viscous." />
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
