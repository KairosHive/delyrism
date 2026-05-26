"use client";
import * as React from "react";
import { useSidebar } from "@/lib/store";
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
        Persistent homology probes the <em>shape</em> of each archetype's descriptor cloud — independent
        of context. H0 = how tight; H1 = how loopy; H2 = how full of voids. The Explorer asks
        "how does context shift meaning?"; this tab asks "what shape does the meaning have to begin with?"
      </p>
    </div>
  );
}
