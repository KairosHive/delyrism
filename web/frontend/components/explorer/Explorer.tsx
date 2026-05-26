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
    <div className="mx-auto max-w-3xl rounded-2xl border border-ink-700/60 bg-ink-900/40 p-8">
      <div className="text-center">
        <div className="font-display text-2xl text-ink-50">Three steps to get started</div>
        <p className="mt-1 text-sm text-ink-400">
          all the controls live in the two cards at the top of this page
        </p>
      </div>
      <ol className="mt-6 space-y-4">
        <Step
          num={1}
          title="Pick an archetype system"
          color="symbolic"
          body={
            <>
              Use the <span className="text-accent-300">Symbolic Structure</span> card to choose a
              preset — Elements, Chakras, Jungian, Tarot, etc. You can also edit the JSON
              directly in the sidebar to compose your own.
            </>
          }
        />
        <Step
          num={2}
          title="Give it a context"
          color="context"
          body={
            <>
              Type a sentence into <span className="text-accent-300">Context Prompt</span>. Or drop
              an image, record audio, or flip on <span className="text-accent-300">alchemist mode</span>{" "}
              to morph between two contexts with a slider.
            </>
          }
        />
        <Step
          num={3}
          title="Press Build space"
          color="accent"
          body={
            <>
              The button just below the two cards. It encodes every descriptor through the chosen
              embedder (one-time per preset) — afterwards, context changes update every panel live.
            </>
          }
        />
      </ol>
      <div className="mt-6 rounded-lg border border-ink-700/40 bg-ink-900/30 p-3 text-[11px] text-ink-400">
        <span className="text-ink-300">Tip</span>: every sidebar control has a{" "}
        <span className="font-mono text-accent-300">?</span> hover-tooltip explaining what it
        does — no need to read docs.
      </div>
    </div>
  );
}

function Step({
  num, title, body, color,
}: {
  num: number;
  title: string;
  body: React.ReactNode;
  color: "symbolic" | "context" | "accent";
}) {
  const borderClass = {
    symbolic: "border-accent-500/50 bg-accent-600/10 text-accent-200",
    context:  "border-warmth/50 bg-warmth/10 text-warmth",
    accent:   "border-accent-500/50 bg-accent-600/20 text-accent-100",
  }[color];
  return (
    <li className="flex gap-4">
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border font-mono text-sm font-medium ${borderClass}`}
      >
        {num}
      </div>
      <div className="flex-1 pt-0.5">
        <div className="text-sm font-semibold text-ink-50">{title}</div>
        <div className="mt-0.5 text-[12px] leading-relaxed text-ink-300">{body}</div>
      </div>
    </li>
  );
}
