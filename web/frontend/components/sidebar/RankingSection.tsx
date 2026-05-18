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
        onChange={(v) => s.set("tau", v)} hint="lower = sharper attention" />
      <Slider label="α — PageRank damping" value={s.alpha} min={0.1} max={0.99} step={0.01}
        onChange={(v) => s.set("alpha", v)} />
      <Slider label="λ — coherence vs diffusion" value={s.lambda} min={0} max={1} step={0.01}
        onChange={(v) => s.set("lambda", v)} hint="1 = pure cosine, 0 = pure PR" />
      <Slider label="Top-K" value={s.topk} min={1} max={40} step={1}
        onChange={(v) => s.set("topk", Math.round(v))}
        hint="cap on rows in the Ranked archetypes panel" />
      <Toggle label="Use personalized PageRank" value={s.usePPR} onChange={(v) => s.set("usePPR", v)} />
      <Toggle label="Blind-spot mode" value={s.blindSpot} onChange={(v) => s.set("blindSpot", v)}
        hint="rank LEAST aligned symbols" />
    </Section>
  );
}
