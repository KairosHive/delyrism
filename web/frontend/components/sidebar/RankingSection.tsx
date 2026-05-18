"use client";
import * as React from "react";
import { Section } from "../ui/Section";
import { Slider } from "../ui/Slider";
import { Toggle } from "../ui/Toggle";
import { useSidebar } from "@/lib/store";
import { SECTION_COLORS } from "@/lib/theme";

export function RankingSection() {
  const s = useSidebar();
  return (
    <Section title="Ranking (proposal)" color={SECTION_COLORS.ranking} defaultOpen={false}>
      <Slider label="τ — softmax temperature" value={s.tau} min={0.01} max={2} step={0.01}
        onChange={(v) => s.set("tau", v)}
        help="Sharpness of the attention over descriptors when computing each symbol's coherence with the context. Lower (0.05–0.2) sharpens focus on the few descriptors most aligned with the context; higher (>1) spreads attention evenly." />
      <Slider label="α — PageRank damping" value={s.alpha} min={0.1} max={0.99} step={0.01}
        onChange={(v) => s.set("alpha", v)}
        help="Probability that a PageRank random walk follows a graph edge vs jumps back to the personalization vector. Higher α (≈0.85) = walks travel further, picking up indirect graph centrality; lower α = stays closer to the seeds." />
      <Slider label="λ — coherence vs diffusion" value={s.lambda} min={0} max={1} step={0.01}
        onChange={(v) => s.set("lambda", v)}
        help="How the composite score blends two signals. λ=1 → pure cosine similarity to the context. λ=0 → pure PageRank centrality. Mid values combine semantic match with graph structure." />
      <Slider label="Top-K" value={s.topk} min={1} max={40} step={1}
        onChange={(v) => s.set("topk", Math.round(v))}
        help="Max number of archetypes shown in the Ranked archetypes panel." />
      <Toggle label="Use personalized PageRank" value={s.usePPR} onChange={(v) => s.set("usePPR", v)}
        help="When off, ranking is coherence-only (λ effectively becomes 1). Turn off to see the pure semantic ranking without graph diffusion." />
      <Toggle label="Blind-spot mode" value={s.blindSpot} onChange={(v) => s.set("blindSpot", v)}
        help="Invert the ranking — show the archetypes LEAST aligned with the context. Useful for exploring the opposite of what you typed." />
    </Section>
  );
}
