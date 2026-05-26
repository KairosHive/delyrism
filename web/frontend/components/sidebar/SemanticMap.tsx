"use client";
import * as React from "react";
import { Section } from "../ui/Section";
import { Select } from "../ui/Select";
import { Toggle } from "../ui/Toggle";
import { useSidebar } from "@/lib/store";
import { SECTION_COLORS } from "@/lib/theme";

export function SemanticMap() {
  const reducer = useSidebar((s) => s.reducer);
  const hulls = useSidebar((s) => s.drawHulls);
  const cent = useSidebar((s) => s.includeCentroids);
  const ncent = useSidebar((s) => s.normalizeCentroids);
  const arrows = useSidebar((s) => s.showArrows);
  const pullHeat = useSidebar((s) => s.pullHeatmap);
  const set = useSidebar((s) => s.set);
  return (
    <Section title="Semantic Map" color={SECTION_COLORS.map} defaultOpen={false}>
      <Select
        label="2D Reducer"
        value={reducer}
        onChange={(v) => set("reducer", v as any)}
        options={[
          { value: "umap", label: "UMAP" },
          { value: "tsne", label: "t-SNE" },
          { value: "pca", label: "PCA" },
        ]}
        help="How to flatten the high-dim descriptor cloud into 2D. UMAP preserves local neighborhoods (best for clusters), t-SNE emphasizes separation between groups, PCA is the fastest and most linear."
      />
      <div className="space-y-1.5">
        <Toggle label="Draw convex hulls" value={hulls} onChange={(v) => set("drawHulls", v)}
          help="Wrap each symbol's descriptor cluster in a translucent colored polygon. Helpful to see how clusters overlap." />
        <Toggle label="Include centroids" value={cent} onChange={(v) => set("includeCentroids", v)}
          help="Show each symbol's average position as a star — the visual 'center' of the archetype." />
        <Toggle label="Normalize centroids" value={ncent} onChange={(v) => set("normalizeCentroids", v)}
          help="Project centroids onto the unit sphere before reducing. Mostly affects spacing when descriptors have very different magnitudes." />
        <Toggle label="Show context arrows" value={arrows} onChange={(v) => set("showArrows", v)}
          help="When a context is active, draw small arrows from each descriptor to its context-shifted position." />
        <Toggle label="Color by pull intensity" value={pullHeat} onChange={(v) => set("pullHeatmap", v)}
          help="When a context is active, shade each descriptor dot by how strongly the context moved it (length of its shift arrow in 2D). Bright = pulled hard; dim = barely moved. Overrides the per-symbol coloring while active." />
      </div>
    </Section>
  );
}
