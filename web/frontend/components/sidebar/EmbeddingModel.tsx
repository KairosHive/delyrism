"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { api, BackendsResponse } from "@/lib/api";
import { Section } from "../ui/Section";
import { Select } from "../ui/Select";
import { Slider } from "../ui/Slider";
import { useSidebar } from "@/lib/store";
import { SECTION_COLORS } from "@/lib/theme";

export function EmbeddingModel() {
  const backend = useSidebar((s) => s.embedderBackend);
  const model = useSidebar((s) => s.embedderModel);
  const pooling = useSidebar((s) => s.embedderPooling);
  const instr = useSidebar((s) => s.qwenInstruction);
  const ctxMode = useSidebar((s) => s.qwenContextMode);
  const ctxText = useSidebar((s) => s.qwenGlobalContext);
  const dthr = useSidebar((s) => s.descriptorThreshold);
  const set = useSidebar((s) => s.set);

  const backends = useQuery({
    queryKey: ["backends"],
    queryFn: () => api.get<BackendsResponse>("/backends"),
  });

  const isQwen = backend.includes("qwen") || backend.startsWith("cloudflare-qwen");
  const isLocal = !backend.startsWith("cloudflare");

  return (
    <Section title="Embedding Model" color={SECTION_COLORS.embedding} defaultOpen={false}>
      <Select
        label="Backend"
        value={backend}
        onChange={(v) => set("embedderBackend", v)}
        options={(backends.data?.embedders ?? []).map((b) => ({ value: b.id, label: b.label }))}
      />
      <div className="space-y-1">
        <div className="label-sm">Model override (optional)</div>
        <input
          className="input-base font-mono text-xs"
          placeholder="e.g. Qwen/Qwen3-Embedding-0.6B"
          value={model}
          onChange={(e) => set("embedderModel", e.target.value)}
        />
      </div>

      {isLocal && (
        <Select
          label="Pooling"
          value={pooling}
          onChange={(v) => set("embedderPooling", v as any)}
          options={[
            { value: "eos", label: "EOS" },
            { value: "mean", label: "Mean" },
            { value: "cls", label: "CLS" },
            { value: "last", label: "Last" },
          ]}
        />
      )}

      {isQwen && (
        <>
          <div className="space-y-1">
            <div className="label-sm">Qwen instruction (optional)</div>
            <textarea
              className="input-base h-16 text-xs"
              placeholder="Represent the descriptor for semantic retrieval"
              value={instr}
              onChange={(e) => set("qwenInstruction", e.target.value)}
            />
          </div>
          <Select
            label="Qwen context mode"
            value={ctxMode}
            onChange={(v) => set("qwenContextMode", v as any)}
            options={[
              { value: "none", label: "None" },
              { value: "global", label: "Global string" },
              { value: "per-descriptor", label: "Per-descriptor owner" },
            ]}
          />
          {ctxMode === "global" && (
            <input
              className="input-base text-xs"
              placeholder="Global context (e.g. archetypal cosmology)"
              value={ctxText}
              onChange={(e) => set("qwenGlobalContext", e.target.value)}
            />
          )}
        </>
      )}

      <Slider
        label="Descriptor edge threshold"
        value={dthr}
        min={0.0}
        max={0.5}
        step={0.02}
        onChange={(v) => set("descriptorThreshold", v)}
        hint="Only descriptors with cosine > τ get connected"
      />
    </Section>
  );
}
