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
        ]} />

      {isPooling && (
        <>
          <Select label="Pool type" value={s.poolType} onChange={(v) => s.set("poolType", v as any)}
            options={[{ value: "avg", label: "Average" }, { value: "max", label: "Max" }, { value: "min", label: "Min" }]} />
          {s.poolType === "avg" && (
            <Slider label="Pool weight w" value={s.poolW} min={0} max={1} step={0.05}
              onChange={(v) => s.set("poolW", v)} />
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
            ]} />
          {isSoftmax && (
            <>
              <Slider label="Softmax τ" value={s.shiftTau} min={0.01} max={2} step={0.01}
                onChange={(v) => s.set("shiftTau", v)} />
              <Toggle label="Softmax within symbol" value={s.withinSymbolSoftmax}
                onChange={(v) => s.set("withinSymbolSoftmax", v)} />
            </>
          )}
        </>
      )}

      {isHybrid && (
        <Slider label="Hybrid blend γ" value={s.gamma} min={0} max={1} step={0.05}
          onChange={(v) => s.set("gamma", v)} hint="0=gate, 1=reembed" />
      )}

      <Slider label="Shift strength β" value={s.beta} min={0} max={2} step={0.05}
        onChange={(v) => s.set("beta", v)} disabled={isPooling} />
      <Slider label="Membership α" value={s.membershipAlpha} min={0} max={1} step={0.05}
        onChange={(v) => s.set("membershipAlpha", v)} hint="how much shift respects symbol boundaries" />

      <div className="border-t border-ink-700/60 pt-3 space-y-2">
        <div className="sub-title">Graph display</div>
        <Select label="Edge sign" value={s.deltaSign} onChange={(v) => s.set("deltaSign", v as any)}
          options={[
            { value: "up",   label: "↑ Strengthens only" },
            { value: "down", label: "↓ Weakens only" },
            { value: "both", label: "Both (mixed)" },
          ]} />
        <Slider label="Top |Δ| edges" value={s.topAbsEdges} min={2} max={100} step={1}
          onChange={(v) => s.set("topAbsEdges", Math.round(v))} />
        <Slider label="Min |Δ| threshold" value={s.minAbsDelta} min={0.0001} max={0.1} step={0.0005}
          onChange={(v) => s.set("minAbsDelta", v)} format={(v) => v.toFixed(4)} />
        <Toggle label="Within-symbol pairs only" value={s.withinSymbolEdges}
          onChange={(v) => s.set("withinSymbolEdges", v)} />
        <Toggle label="Hide isolated nodes" value={s.connectedOnly}
          onChange={(v) => s.set("connectedOnly", v)} />
      </div>
    </Section>
  );
}
