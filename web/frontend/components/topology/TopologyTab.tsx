"use client";
import * as React from "react";
import { useSidebar } from "@/lib/store";
import { useTopologyContext } from "./useTopologyContext";
import { TopologyOverview } from "./TopologyOverview";
import { TopologyDiagrams } from "./TopologyDiagrams";
import { TopologyCycles } from "./TopologyCycles";
import { TopologySynergy } from "./TopologySynergy";
import { TopologyCatalysts } from "./TopologyCatalysts";

/**
 * Persistent-homology tab.  Five sub-views, lazy-loaded per click — PH
 * is expensive, so we only compute what the user is actually looking at.
 *
 *   Overview  — TopoScore table + cohesion-vs-loopiness 2D map
 *   Diagrams  — per-symbol persistence diagrams (H0/H1/H2)
 *   Cycles    — interactive cycle browser, click a cycle to trace it on PCA
 *   Synergy   — pairwise H1/H2 synergy heatmap + mixed-cycles drill-down
 *   Catalysts — leave-one-out word criticality per symbol
 *
 * PH measures the SHAPE of the unconditioned descriptor cloud — context
 * doesn't enter here.  That's intentional: this tab answers a different
 * family of questions than the Explorer ("does this archetype have
 * internal loops? do these two archetypes share structural topology?").
 */

type SubView = "overview" | "diagrams" | "cycles" | "synergy" | "catalysts";

const SUBVIEWS: { id: SubView; label: string; hint: string; color: string; icon: string }[] = [
  { id: "overview",  label: "Overview",     hint: "TopoScore + symbol map", color: "#3bbdb0", icon: "◐" },
  { id: "diagrams",  label: "Diagrams",     hint: "birth/death scatter",    color: "#9b59b6", icon: "◇" },
  { id: "cycles",    label: "Cycles",       hint: "loops in semantic space",color: "#e67e22", icon: "⊙" },
  { id: "synergy",   label: "Synergy",      hint: "shared structure",       color: "#c2a6fe", icon: "⋈" },
  { id: "catalysts", label: "Catalysts",    hint: "load-bearing words",     color: "#bf616a", icon: "✦" },
];

export function TopologyTab() {
  const spaceId = useSidebar((s) => s.spaceId);
  const [sub, setSub] = React.useState<SubView>("overview");

  if (!spaceId) {
    return (
      <div className="mx-auto max-w-2xl rounded-2xl border border-ink-700/60 bg-ink-900/40 p-8 text-center">
        <div className="font-display text-2xl text-ink-50">Build a space first</div>
        <p className="mt-2 text-sm text-ink-300">
          Persistent homology measures the <em>shape</em> of each archetype's descriptor cloud —
          loops, voids, and the bridges between symbols. Pick a preset and press
          <span className="ml-1 font-mono text-accent-300">Build space</span> to start.
        </p>
      </div>
    );
  }

  return (
    <div className="w-full space-y-4">
      <ContextOverlayBar />

      <div className="rounded-2xl border border-ink-700/60 bg-ink-900/30 p-1">
        <nav className="flex flex-wrap items-center gap-1">
          {SUBVIEWS.map((s) => {
            const active = sub === s.id;
            return (
              <button
                key={s.id}
                onClick={() => setSub(s.id)}
                className="group relative flex flex-1 items-center gap-2 rounded-xl px-3 py-2.5 text-left transition"
                style={{
                  background: active ? `${s.color}1f` : "transparent",
                  boxShadow: active ? `inset 0 0 0 1px ${s.color}80` : "none",
                  color: active ? s.color : "#9fadc1",
                }}
              >
                <span
                  className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-sm"
                  style={{
                    background: active ? `${s.color}33` : "rgba(255,255,255,0.04)",
                    color: active ? s.color : "#6e7e95",
                  }}
                >
                  {s.icon}
                </span>
                <span className="flex min-w-0 flex-col leading-tight">
                  <span
                    className={`truncate text-[13px] ${active ? "font-semibold" : "font-medium"}`}
                  >
                    {s.label}
                  </span>
                  <span className="hidden truncate text-[10px] text-ink-500 lg:block">
                    {s.hint}
                  </span>
                </span>
              </button>
            );
          })}
        </nav>
      </div>

      {sub === "overview" && <TopologyOverview />}
      {sub === "diagrams" && <TopologyDiagrams />}
      {sub === "cycles" && <TopologyCycles />}
      {sub === "synergy" && <TopologySynergy />}
      {sub === "catalysts" && <TopologyCatalysts />}

      <p className="mt-2 text-[10px] leading-relaxed text-ink-500">
        Persistent homology probes the <em>shape</em> of each archetype's descriptor cloud.
        H0 = how tight; H1 = how loopy; H2 = how full of voids.  By default this is computed
        on the unconditioned embeddings; flip "context overlay" above to re-run on the
        context-shifted cloud and see how context bends each archetype's topology.
      </p>
    </div>
  );
}

/**
 * Toggle row above the sub-view nav.  Flips every Topology endpoint
 * between intrinsic (space.D) and context-shifted (D') computation.
 * Shows a live badge of what context is currently active so the user
 * knows what they're applying without leaving the tab.
 */
function ContextOverlayBar() {
  const set = useSidebar((s) => s.set);
  const { show, active, hasContext, summary } = useTopologyContext();
  return (
    <div
      className="rounded-2xl border bg-ink-900/30 p-3 transition"
      style={{
        borderColor: active ? "#3bbdb055" : "rgba(255,255,255,0.06)",
        boxShadow: active ? "inset 0 0 0 1px #3bbdb022" : "none",
      }}
    >
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          role="switch"
          aria-checked={show}
          onClick={() => set("topologyShowContext", !show)}
          className="relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition"
          style={{
            background: show ? "#3bbdb0" : "rgba(255,255,255,0.10)",
            boxShadow: show ? "inset 0 0 0 1px rgba(0,0,0,0.2)" : "inset 0 0 0 1px rgba(255,255,255,0.08)",
          }}
        >
          <span
            className="block h-5 w-5 rounded-full bg-ink-50 shadow transition"
            style={{ transform: show ? "translateX(22px)" : "translateX(2px)" }}
          />
        </button>
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="text-[12px] font-medium text-ink-100">
            Context overlay
            <span className="ml-2 text-[10px] text-ink-400">
              {show
                ? active
                  ? "ON — running PH on the context-shifted cloud"
                  : "ON — but no context is set; views show intrinsic shape"
                : "OFF — showing intrinsic shape"}
            </span>
          </div>
          <div className="mt-0.5 text-[11px] leading-tight text-ink-400">
            {show && hasContext ? (
              <>
                <span className="text-ink-500">active context · </span>
                <span className="text-ink-200">{summary}</span>
                <span className="text-ink-500"> — adjust the Context Prompt + Δ Graph sidebar to change.</span>
              </>
            ) : show ? (
              <span className="text-warmth">
                Set a context (sentence, audio, image, or alchemist) to compare.
              </span>
            ) : (
              <span>
                Topology runs on the original embeddings.  Flip on to re-run on the
                context-shifted cloud and compare archetype shapes under your active context.
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
