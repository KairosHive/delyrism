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
      />
      <div className="space-y-1.5">
        <Toggle label="Draw convex hulls" value={hulls} onChange={(v) => set("drawHulls", v)} />
        <Toggle label="Include centroids" value={cent} onChange={(v) => set("includeCentroids", v)} />
        <Toggle label="Normalize centroids" value={ncent} onChange={(v) => set("normalizeCentroids", v)} />
        <Toggle label="Show context arrows" value={arrows} onChange={(v) => set("showArrows", v)} />
      </div>
    </Section>
  );
}
