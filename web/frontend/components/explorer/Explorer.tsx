"use client";
import * as React from "react";
import { MeaningSpace } from "./MeaningSpace";
import { Rankings } from "./Rankings";
import { AmbiguityChart } from "./AmbiguityChart";
import { AttentionHeatmap } from "./AttentionHeatmap";
import { Subgraph } from "./Subgraph";
import { DeltaGraph } from "./DeltaGraph";
import { SimilarityHeatmap } from "./SimilarityHeatmap";
import { ContextualTransformations } from "./ContextualTransformations";
import { useSidebar } from "@/lib/store";

export function Explorer() {
  const spaceId = useSidebar((s) => s.spaceId);
  if (!spaceId) return <EmptyState />;
  return (
    <div className="w-full space-y-5">
      {/* Row 1 — Map (wide) and Rankings (narrow) */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.6fr,1fr]">
        <MeaningSpace />
        <Rankings />
      </div>

      {/* Row 2 — Ambiguity (narrow) and Attention (wide) */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1fr,1.4fr]">
        <AmbiguityChart />
        <AttentionHeatmap />
      </div>

      {/* Row 3 — Subgraph + within-symbol Δ heatmap side by side */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        <Subgraph />
        <SimilarityHeatmap />
      </div>

      {/* Row 4 — Δ graph: tall and full-width, the headline relational view */}
      <DeltaGraph />

      {/* Row 5 — Contextual transformations: narrative migrations +
          per-archetype identity cards.  Placed at the bottom as a
          summary-narrative view of what the whole panel set just
          said in math. */}
      <ContextualTransformations />
    </div>
  );
}

function EmptyState() {
  return (
    <div className="mx-auto max-w-2xl rounded-2xl border border-ink-700/60 bg-ink-900/40 p-8 text-center">
      <div className="font-display text-2xl text-ink-50">Pick an archetype system above</div>
      <p className="mt-2 text-sm text-ink-300">
        Choose a preset from <span className="text-accent-300">Symbolic Structure</span>, type a sentence
        into <span className="text-accent-300">Context Prompt</span>, then press
        <span className="ml-1 font-mono text-accent-300">Build space</span> in the sidebar.
      </p>
    </div>
  );
}
