"use client";
import * as React from "react";
import { Section } from "../ui/Section";
import { Slider } from "../ui/Slider";
import { Select } from "../ui/Select";
import { useSidebar } from "@/lib/store";
import { SECTION_COLORS } from "@/lib/theme";

export function SubgraphSection() {
  const s = useSidebar();
  return (
    <Section title="Contextual Subgraph" color={SECTION_COLORS.subgraph} defaultOpen={false}>
      <Slider label="Top symbols" value={s.subTopSymbols} min={1} max={12} step={1}
        onChange={(v) => s.set("subTopSymbols", Math.round(v))}
        help="How many archetype (square) nodes to include in the subgraph view, ranked by their score under the current context." />
      <Slider label="Top descriptors / symbol" value={s.subTopDescriptors} min={1} max={12} step={1}
        onChange={(v) => s.set("subTopDescriptors", Math.round(v))}
        help="For each selected archetype, how many of its strongest-attended descriptor nodes (circles) to include." />
      <Select label="Scoring method" value={s.subMethod} onChange={(v) => s.set("subMethod", v as any)}
        options={[{ value: "ppr", label: "Personalized PageRank" }, { value: "softmax", label: "Softmax attention" }]}
        help="PPR diffuses context through the descriptor graph (captures indirect relations). Softmax just ranks by direct cosine to the context." />
      <Slider label="Context focus (τ)" value={s.subTau} min={0.01} max={1} step={0.01}
        onChange={(v) => s.set("subTau", v)}
        help="Softmax temperature for distributing context mass over descriptors. Lower = sharper focus on a few descriptors; higher = spread evenly." />
      <Slider label="Subgraph α (damping)" value={s.subAlpha} min={0.5} max={0.99} step={0.01}
        onChange={(v) => s.set("subAlpha", v)}
        help="PageRank damping inside the subgraph. Higher = more graph diffusion; lower = sticks closer to seeded descriptors." />
      <Slider label="Descriptor edge threshold" value={s.subThreshold} min={0} max={0.9} step={0.02}
        onChange={(v) => s.set("subThreshold", v)}
        help="Min cosine similarity for descriptor↔descriptor edges to appear in the network. Higher = sparser, only the strongest associations remain." />
    </Section>
  );
}
