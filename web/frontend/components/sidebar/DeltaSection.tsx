"use client";
import * as React from "react";
import { Section } from "../ui/Section";
import { Slider } from "../ui/Slider";
import { Select } from "../ui/Select";
import { Toggle } from "../ui/Toggle";
import { useSidebar } from "@/lib/store";
import { SECTION_COLORS } from "@/lib/theme";

export function DeltaSection() {
  const s = useSidebar();
  const isPooling = s.strategy === "pooling";
  const isHybrid = s.strategy === "hybrid";
  const isGate = s.strategy === "gate" || s.strategy === "hybrid";
  const isSoftmax = isGate && s.gate === "softmax";

  return (
    <Section title="Δ Graph (context shift)" color={SECTION_COLORS.delta} defaultOpen={false}>
      <Select label="Strategy" value={s.strategy} onChange={(v) => s.set("strategy", v as any)}
        options={[
          { value: "gate", label: "Gate (fast additive)" },
          { value: "reembed", label: "Reembed (rich)" },
          { value: "hybrid", label: "Hybrid (blend)" },
          { value: "pooling", label: "Pooling (interpolate)" },
        ]}
        help="How to shift each descriptor toward the context. Gate adds β·gate(D·ctx)·ctx (fast, linear). Reembed re-encodes each descriptor with the context prepended (rich, slow). Hybrid blends both. Pooling interpolates each descriptor toward the context vector." />

      {isPooling && (
        <>
          <Select label="Pool type" value={s.poolType} onChange={(v) => s.set("poolType", v as any)}
            options={[{ value: "avg", label: "Average" }, { value: "max", label: "Max" }, { value: "min", label: "Min" }]}
            help="How descriptor and context combine. Avg = weighted average (uses w below). Max/min = element-wise max/min of the two vectors." />
          {s.poolType === "avg" && (
            <Slider label="Pool weight w" value={s.poolW} min={0} max={1} step={0.05}
              onChange={(v) => s.set("poolW", v)}
              help="In average pooling, w is the context's share of each shifted vector. 0 = no shift (pure descriptor); 1 = replace descriptor with context entirely." />
          )}
        </>
      )}

      {isGate && (
        <>
          <Select label="Gate" value={s.gate} onChange={(v) => s.set("gate", v as any)}
            options={[
              { value: "relu", label: "ReLU" },
              { value: "cos", label: "Cosine" },
              { value: "softmax", label: "Softmax" },
              { value: "uniform", label: "Uniform" },
            ]}
            help="Function that decides how much each descriptor moves. ReLU = only descriptors POSITIVELY aligned with context shift. Cos = signed (negatives shift opposite). Softmax = normalized — only the top few shift. Uniform = all shift equally." />
          {isSoftmax && (
            <>
              <Slider label="Softmax τ" value={s.shiftTau} min={0.01} max={2} step={0.01}
                onChange={(v) => s.set("shiftTau", v)}
                help="Temperature for softmax gate. Lower = only the top 1-2 most aligned descriptors get shifted; higher = many descriptors share the shift." />
              <Toggle label="Softmax within symbol" value={s.withinSymbolSoftmax}
                onChange={(v) => s.set("withinSymbolSoftmax", v)}
                help="Normalize softmax per-symbol instead of across all descriptors. Each symbol gets its 'most-aligned' descriptor shifted, regardless of symbol-level alignment." />
            </>
          )}
        </>
      )}

      {isHybrid && (
        <Slider label="Hybrid blend γ" value={s.gamma} min={0} max={1} step={0.05}
          onChange={(v) => s.set("gamma", v)}
          help="Mix of gate (cheap) and reembed (rich) shift results. γ=0 → pure gate, γ=1 → pure reembed, mid values blend both." />
      )}

      <Slider label="Shift strength β" value={s.beta} min={0} max={2} step={0.05}
        onChange={(v) => s.set("beta", v)} disabled={isPooling}
        help="Overall magnitude of the context shift. Larger β = descriptors move further toward the context direction. Disabled for pooling (use pool weight w instead)." />
      <Slider label="Membership α" value={s.membershipAlpha} min={0} max={1} step={0.05}
        onChange={(v) => s.set("membershipAlpha", v)}
        help="How strongly each shifted descriptor is pulled back toward its symbol's centroid after the context shift. 0 = no anchoring; 1 = stays right on the centroid (no movement)." />

      <div className="border-t border-ink-700/60 pt-3 space-y-2">
        <div className="sub-title">Graph display</div>
        <Select label="Edge sign" value={s.deltaSign} onChange={(v) => s.set("deltaSign", v as any)}
          options={[
            { value: "up",   label: "↑ Strengthens only" },
            { value: "down", label: "↓ Weakens only" },
            { value: "both", label: "Both (mixed)" },
          ]}
          help="Which kinds of association changes to render. Strengthens = descriptor pairs whose similarity INCREASED under context. Weakens = pairs that became more distant. The top-N cap below respects this filter." />
        <Slider label="Top |Δ| edges" value={s.topAbsEdges} min={2} max={100} step={1}
          onChange={(v) => s.set("topAbsEdges", Math.round(v))}
          help="Cap on how many edges to render, ranked by absolute change |after − before|. Lower for a readable graph; higher to see the long tail." />
        <Slider label="Min |Δ| threshold" value={s.minAbsDelta} min={0.0001} max={0.1} step={0.0005}
          onChange={(v) => s.set("minAbsDelta", v)} format={(v) => v.toFixed(4)}
          help="Skip edges with |Δ| below this — filters out very weak shifts that would just be noise." />
        <Toggle label="Within-symbol pairs only" value={s.withinSymbolEdges}
          onChange={(v) => s.set("withinSymbolEdges", v)}
          help="Hide edges between descriptors that belong to DIFFERENT symbols. Useful to focus on intra-archetype reorganization." />
        <Toggle label="Hide isolated nodes" value={s.connectedOnly}
          onChange={(v) => s.set("connectedOnly", v)}
          help="Drop descriptor nodes that have no surviving edges after the filters above — keeps the graph compact." />
      </div>
    </Section>
  );
}
