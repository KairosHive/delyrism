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
        onChange={(v) => s.set("subTopSymbols", Math.round(v))} />
      <Slider label="Top descriptors / symbol" value={s.subTopDescriptors} min={1} max={12} step={1}
        onChange={(v) => s.set("subTopDescriptors", Math.round(v))} />
      <Select label="Scoring method" value={s.subMethod} onChange={(v) => s.set("subMethod", v as any)}
        options={[{ value: "ppr", label: "Personalized PageRank" }, { value: "softmax", label: "Softmax attention" }]} />
      <Slider label="Context focus (τ)" value={s.subTau} min={0.01} max={1} step={0.01}
        onChange={(v) => s.set("subTau", v)} />
      <Slider label="Subgraph α (damping)" value={s.subAlpha} min={0.5} max={0.99} step={0.01}
        onChange={(v) => s.set("subAlpha", v)} />
      <Slider label="Descriptor edge threshold" value={s.subThreshold} min={0} max={0.9} step={0.02}
        onChange={(v) => s.set("subThreshold", v)} />
    </Section>
  );
}
