"use client";
import * as React from "react";
import { Plot } from "../plots/Plot";
import { useAmbiguity } from "@/lib/hooks";
import { useSidebar } from "@/lib/store";

const METRIC_INFO: { key: "dispersion" | "leakage" | "entropy"; label: string; opacity: number; tip: string }[] = [
  { key: "dispersion", label: "dispersion",   opacity: 0.95, tip: "internal diversity of descriptors" },
  { key: "leakage",    label: "leakage",      opacity: 0.6,  tip: "fraction of nearest neighbors outside the symbol" },
  { key: "entropy",    label: "soft entropy", opacity: 0.35, tip: "ambiguity across all symbols" },
];

export function AmbiguityChart() {
  const colorMap = useSidebar((s) => s.colorMap);
  const a = useAmbiguity();
  if (a.isPending) return <div className="panel-pad text-sm text-ink-300">Computing ambiguity…</div>;
  if (!a.data) return null;
  const rows = a.data.rows;
  const x = rows.map((r) => r.symbol);
  const colors = x.map((s) => colorMap[s] ?? "#888");

  return (
    <div className="panel-tight">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="section-title">Ambiguity metrics</div>
        {/* Custom legend — three opacity-graded swatches, neutral-color so the
            user reads "depth" rather than "another symbol-colored thing".
            Plotly's auto-legend would show all three swatches in the first
            symbol's color, which is what was confusing. */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-ink-300">
          {METRIC_INFO.map((m) => (
            <span key={m.key} className="inline-flex items-center gap-1.5" title={m.tip}>
              <span
                className="inline-block h-2.5 w-4 rounded-sm bg-ink-100"
                style={{ opacity: m.opacity }}
              />
              {m.label}
            </span>
          ))}
        </div>
      </div>
      <Plot
        data={[
          { x, y: rows.map((r) => r.dispersion), type: "bar", name: "dispersion",   marker: { color: colors, opacity: 0.95 } },
          { x, y: rows.map((r) => r.leakage),    type: "bar", name: "leakage",      marker: { color: colors, opacity: 0.6 } },
          { x, y: rows.map((r) => r.entropy),    type: "bar", name: "soft entropy", marker: { color: colors, opacity: 0.35 } },
        ]}
        layout={{
          autosize: true,
          height: 280,
          margin: { l: 32, r: 12, t: 8, b: 30 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#cad4e0", family: "Inter, system-ui" },
          barmode: "group",
          showlegend: false,  // we render our own — see above
          xaxis: { tickangle: -25, gridcolor: "rgba(255,255,255,0.04)" },
          yaxis: { gridcolor: "rgba(255,255,255,0.06)", zeroline: false },
        }}
        config={{ displaylogo: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%" }}
      />
    </div>
  );
}
