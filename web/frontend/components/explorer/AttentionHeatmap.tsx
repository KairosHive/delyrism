"use client";
import * as React from "react";
import { Plot } from "../plots/Plot";
import { useAttention } from "@/lib/hooks";
import { useSidebar } from "@/lib/store";

export function AttentionHeatmap() {
  const symbols = useSidebar((s) => s.symbols);
  const selected = useSidebar((s) => s.selectedSymbol);
  const set = useSidebar((s) => s.set);
  const colorMap = useSidebar((s) => s.colorMap);

  const target = selected ?? symbols[0] ?? null;
  const att = useAttention(target);
  const accent = colorMap[target ?? ""] ?? "#88C0D0";

  return (
    <div className="panel-tight">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: accent }} />
          <div className="section-title">Descriptor attention · {target ?? "—"}</div>
        </div>
        <select
          className="select-base !py-1 text-xs"
          value={target ?? ""}
          onChange={(e) => set("selectedSymbol", e.target.value)}
        >
          {symbols.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
      </div>
      {att.isPending && <div className="text-sm text-ink-300">computing…</div>}
      {att.data && (() => {
        // sort descriptors by attention weight descending
        const pairs = att.data.descriptors
          .map((d, i) => ({ d, w: att.data!.weights[i] }))
          .sort((a, b) => b.w - a.w);
        const ys = pairs.map((p) => p.d);
        const xs = pairs.map((p) => p.w);
        const maxW = Math.max(0.0001, ...xs);
        return (
          <Plot
            data={[
              {
                type: "bar",
                orientation: "h",
                y: ys,
                x: xs,
                marker: {
                  color: xs,
                  // Multi-hue Spectral_r — matches the within-symbol Δ heatmap
                  // so the eye picks up "high attention" the same way across
                  // both panels.  Low = blue, mid = green/yellow, high = red.
                  colorscale: [
                    [0.0,  "#3a86ff"],
                    [0.25, "#06d6a0"],
                    [0.5,  "#ffd166"],
                    [0.75, "#f77f00"],
                    [1.0,  "#ef476f"],
                  ],
                  cmin: 0,
                  cmax: maxW,
                  line: { color: "rgba(0,0,0,0.35)", width: 0.5 },
                  colorbar: {
                    title: { text: "w", side: "right" },
                    thickness: 10,
                    len: 0.85,
                    tickfont: { size: 9 },
                  },
                },
                hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
              },
            ]}
            layout={{
              autosize: true,
              height: Math.max(220, 22 * ys.length + 40),
              margin: { l: 130, r: 60, t: 6, b: 24 },
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              font: { color: "#cad4e0", family: "Inter, system-ui" },
              xaxis: { gridcolor: "rgba(255,255,255,0.05)", zeroline: false },
              yaxis: { autorange: "reversed", tickfont: { size: 11 } },
            }}
            useResizeHandler
            style={{ width: "100%" }}
            config={{ displaylogo: false, responsive: true }}
          />
        );
      })()}
    </div>
  );
}
