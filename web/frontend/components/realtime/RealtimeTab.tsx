"use client";
import * as React from "react";
import { useShallow } from "zustand/react/shallow";
import { useSidebar } from "@/lib/store";
import { useWalker, useWalkerSocket, type WalkerMode, type LayoutKind } from "@/lib/realtime";
import { TraversalCanvas } from "./TraversalCanvas";
import { SymbolSigil } from "./SymbolSigil";
import { SymbolRecipe } from "./SymbolRecipe";
import { WalkerControls } from "./WalkerControls";

/**
 * Live tab — generative symbol construction by traversing the Δ-field in
 * real time.  A WebSocket-backed `DeltaWalker` (delyrism/traversal.py) sits
 * on the server; three interfaces shape its context and three views show
 * the walk and the symbol it is assembling.
 */

const MODES: { id: WalkerMode; label: string; hint: string; icon: string; color: string }[] = [
  { id: "field",  label: "Field",  hint: "pointer over the descriptor cloud is the context", icon: "◌", color: "#3bbdb0" },
  { id: "ring",   label: "Ring",   hint: "blend archetypes by moving between them",           icon: "◎", color: "#c2a6fe" },
  { id: "choose", label: "Choose", hint: "pick each step by hand under a sentence",          icon: "⋮", color: "#e67e22" },
];

export function RealtimeTab() {
  const spaceId = useSidebar((s) => s.spaceId);
  useWalkerSocket();
  if (!spaceId) return <EmptyState />;
  return (
    <div className="w-full space-y-4">
      <ModeBar />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.55fr,1fr]">
        <TraversalCanvas />
        <SymbolSigil />
      </div>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <SymbolRecipe />
        <WalkerControls />
      </div>
      <p className="text-[10px] leading-relaxed text-ink-500">
        The walker sits on one descriptor and looks at the Δ-field the current context induces
        (cosine after shift − cosine before). It steps toward the descriptor the context is
        pulling it to — weighted by the path's recent memory, nudged across archetypes by the
        diversity pressure — and the trail it leaves is a new symbol. Seal it and add it to the
        structure to make it a real archetype in the next build.
      </p>
    </div>
  );
}

function ModeBar() {
  const { mode, setMode, status, playing, setPlaying, step, seal, reset, layoutKind, setLayoutKind, pathLen, hasContext, lastEvent, error } =
    useWalker(useShallow((s) => ({
      mode: s.mode, setMode: s.setMode, status: s.status, playing: s.playing, setPlaying: s.setPlaying,
      step: s.step, seal: s.seal, reset: s.reset, layoutKind: s.layoutKind, setLayoutKind: s.setLayoutKind,
      pathLen: s.frame?.path.length ?? 0, hasContext: s.frame?.has_context ?? false, lastEvent: s.lastEvent, error: s.error,
    })));

  // keyboard: space = play/pause, → = step, s = seal, backspace = undo
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      const st = useWalker.getState();
      if (e.code === "Space") { e.preventDefault(); st.setPlaying(!st.playing); }
      else if (e.key === "ArrowRight") { e.preventDefault(); st.step(); }
      else if (e.key === "s") st.seal();
      else if (e.key === "Backspace") { e.preventDefault(); st.undo(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const dot = status === "ready" ? "bg-accent-400" : status === "connecting" ? "bg-warning animate-pulse" : "bg-danger";
  return (
    <div className="rounded-2xl border border-ink-700/60 bg-ink-900/30 p-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <nav className="flex items-center gap-1">
          {MODES.map((m) => {
            const active = m.id === mode;
            return (
              <button
                key={m.id}
                onClick={() => setMode(m.id)}
                title={m.hint}
                className="flex items-center gap-2 rounded-xl px-3 py-2 text-left transition"
                style={{
                  background: active ? `${m.color}1f` : "transparent",
                  boxShadow: active ? `inset 0 0 0 1px ${m.color}80` : "none",
                  color: active ? m.color : "#9fadc1",
                }}
              >
                <span className="text-sm">{m.icon}</span>
                <span className="flex flex-col leading-tight">
                  <span className={`text-[13px] ${active ? "font-semibold" : "font-medium"}`}>{m.label}</span>
                  <span className="hidden text-[10px] text-ink-500 lg:block">{m.hint}</span>
                </span>
              </button>
            );
          })}
        </nav>

        <div className="mx-1 hidden h-7 w-px bg-ink-700/60 md:block" />

        <div className="flex items-center gap-1.5">
          <button className={`btn ${playing ? "border-accent-500 text-accent-200" : ""}`} onClick={() => setPlaying(!playing)} title="space">
            {playing ? "❚❚ pause" : "▶ play"}
          </button>
          <button className="btn" onClick={() => step()} title="→  one step (sampled)">step ›</button>
          <button className="btn" onClick={() => seal()} disabled={pathLen === 0} title="s  seal the current symbol">seal ◉</button>
          <button className="btn" onClick={reset} disabled={pathLen === 0} title="start the walk over">reset</button>
        </div>

        <div className="ml-auto flex items-center gap-3 text-[11px] text-ink-400">
          <label className="flex items-center gap-1.5">
            <span>layout</span>
            <select className="select-base w-auto py-0.5 text-[11px]" value={layoutKind} onChange={(e) => setLayoutKind(e.target.value as LayoutKind)}>
              <option value="umap">UMAP</option>
              <option value="pca">PCA</option>
            </select>
          </label>
          <span className="hidden items-center gap-1.5 sm:flex">
            <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
            <span className="font-mono">{status}</span>
          </span>
          {status === "ready" && (
            <span className="font-mono text-ink-500">
              {hasContext ? "context ✓" : "no context"} · {pathLen} nodes
              {lastEvent?.sealed ? ` · sealed ${lastEvent.sealed.name}` : ""}
            </span>
          )}
          {error && <span className="text-danger">{error}</span>}
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="mx-auto max-w-2xl rounded-2xl border border-ink-700/60 bg-ink-900/40 p-8 text-center">
      <div className="font-display text-2xl text-ink-50">Build a space first</div>
      <p className="mt-2 text-sm text-ink-300">
        The live walker traverses the context Δ-field of a built space and assembles new symbols
        from the descriptors it passes through. Pick a preset and press
        <span className="ml-1 font-mono text-accent-300">Build space</span> to start.
      </p>
    </div>
  );
}
