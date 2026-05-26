"use client";
import * as React from "react";
import { Plot } from "../plots/Plot";
import { useReduce2D, useShift } from "@/lib/hooks";
import { useSidebar } from "@/lib/store";

/**
 * The 2D meaning-space plot.
 *
 * Two data-source modes:
 *   • no context active        → /reduce-2d  (descriptors + centroids)
 *   • sentence or weights set  → /shift     (descriptors w/ before+after
 *                                            coords, centroids) so the arrows
 *                                            anchor exactly to the dots.
 *
 * /shift fits the chosen reducer on [D, D_shifted, centroids] *jointly*, so
 * every point in the plot lives in the same frame and arrows line up with
 * descriptor dots.
 */
export function MeaningSpace() {
  const colorMap = useSidebar((s) => s.colorMap);
  const drawHulls = useSidebar((s) => s.drawHulls);
  const showArrows = useSidebar((s) => s.showArrows);
  const includeCentroids = useSidebar((s) => s.includeCentroids);
  const reducer = useSidebar((s) => s.reducer);
  const pullHeat = useSidebar((s) => s.pullHeatmap);

  const r = useReduce2D();
  const shift = useShift();
  // If the shift query has data (i.e. a context is set), it drives the layout.
  const hasShift = !!shift.data;

  if (!hasShift && r.isPending && !r.data) return <Skeleton text="Projecting descriptors…" />;
  if (r.error && !hasShift) return <ErrorPanel text={(r.error as Error).message} />;

  // build descriptor points + centroids from whichever source is active
  type Pt = { x: number; y: number; label: string; symbol: string; kind: "descriptor" | "centroid"; x1?: number; y1?: number };
  let pts: Pt[] = [];
  if (hasShift && shift.data) {
    for (const a of shift.data.arrows) {
      pts.push({ x: a.x0, y: a.y0, x1: a.x1, y1: a.y1, label: a.descriptor, symbol: a.symbol, kind: "descriptor" });
    }
    if (includeCentroids) {
      for (const c of shift.data.centroids) {
        pts.push({ x: c.x, y: c.y, label: c.symbol, symbol: c.symbol, kind: "centroid" });
      }
    }
  } else if (r.data) {
    for (const p of r.data.points) {
      if (!includeCentroids && p.kind === "centroid") continue;
      pts.push({ x: p.x, y: p.y, label: p.label, symbol: p.symbol, kind: p.kind });
    }
  }

  // group by symbol
  const groups = new Map<string, Pt[]>();
  for (const p of pts) {
    const arr = groups.get(p.symbol) ?? [];
    arr.push(p);
    groups.set(p.symbol, arr);
  }

  // When pull-intensity mode is on, we colour descriptors by the length
  // of their 2D shift arrow rather than by archetype.  Pre-compute the
  // global magnitudes so every trace shares a single colour scale.
  const pullActive = pullHeat && hasShift;
  const pullByLabel = new Map<string, number>();
  let pullMax = 0;
  if (pullActive && shift.data) {
    for (const a of shift.data.arrows) {
      const m = Math.hypot(a.x1 - a.x0, a.y1 - a.y0);
      pullByLabel.set(a.descriptor, m);
      if (m > pullMax) pullMax = m;
    }
  }

  const traces: any[] = [];
  for (const [sym, items] of groups.entries()) {
    const color = colorMap[sym] ?? "#888";
    const descs = items.filter((i) => i.kind === "descriptor");
    const cents = items.filter((i) => i.kind === "centroid");
    const descMarker: any = pullActive
      ? {
          // Plotly's heatmap palette per-marker.  Normalised to global max so
          // colours are comparable across the whole cloud — not per-symbol.
          color: descs.map((p) => (pullByLabel.get(p.label) ?? 0) / Math.max(pullMax, 1e-9)),
          colorscale: [
            [0.0, "#1b212e"],   // dim — unmoved
            [0.3, "#3bbdb0"],   // mid — pulled some
            [0.7, "#d08770"],   // warm — pulled hard
            [1.0, "#ef476f"],   // hottest
          ],
          cmin: 0,
          cmax: 1,
          size: 9,
          opacity: 0.92,
          line: { color: "rgba(0,0,0,0.4)", width: 0.5 },
          showscale: false,
        }
      : { color, size: 9, opacity: 0.88, line: { color: "rgba(0,0,0,0.4)", width: 0.5 } };
    traces.push({
      x: descs.map((p) => p.x),
      y: descs.map((p) => p.y),
      text: descs.map((p) => p.label),
      type: "scatter",
      mode: "markers",
      name: sym,
      marker: descMarker,
      hovertemplate: pullActive
        ? `<b>${sym}</b> · %{text}<br>pull = %{customdata:.3f}<extra></extra>`
        : `<b>${sym}</b> · %{text}<extra></extra>`,
      customdata: pullActive ? descs.map((p) => pullByLabel.get(p.label) ?? 0) : undefined,
    });
    if (drawHulls && descs.length >= 3) {
      const hull = convexHull(descs.map((d) => [d.x, d.y] as [number, number]));
      hull.push(hull[0]);
      traces.push({
        x: hull.map((h) => h[0]),
        y: hull.map((h) => h[1]),
        type: "scatter",
        mode: "lines",
        line: { color, width: 1.2 },
        fill: "toself",
        fillcolor: hexToRgba(color, 0.07),
        hoverinfo: "skip",
        showlegend: false,
      });
    }
    if (cents.length) {
      traces.push({
        x: cents.map((p) => p.x),
        y: cents.map((p) => p.y),
        text: cents.map((p) => p.label),
        type: "scatter",
        mode: "markers+text",
        marker: { color, size: 18, symbol: "star", line: { color: "white", width: 1.5 } },
        textposition: "top center",
        textfont: { color: "#dbe2ee", size: 11 },
        hoverinfo: "skip",
        showlegend: false,
      });
    }
  }

  // Build arrow annotations only when shift mode active and toggle is on.
  // Filter out near-zero arrows to avoid clutter.
  let annotations: any[] = [];
  if (showArrows && hasShift && shift.data) {
    for (const a of shift.data.arrows) {
      const d = Math.hypot(a.x1 - a.x0, a.y1 - a.y0);
      if (d < 1e-3) continue;
      annotations.push({
        x: a.x1, y: a.y1,
        ax: a.x0, ay: a.y0,
        xref: "x", yref: "y", axref: "x", ayref: "y",
        showarrow: true,
        arrowhead: 2,
        arrowsize: 0.9,
        arrowwidth: 1,
        arrowcolor: colorMap[a.symbol] ?? "#cccccc",
        opacity: 0.55,
        standoff: 3,
      });
    }
  }

  return (
    <div className="panel-tight">
      <div className="mb-2 flex items-center justify-between">
        <div className="section-title">2D Meaning Space</div>
        <div className="flex items-center gap-2 text-[10px] text-ink-400">
          {pullActive && (
            <span
              className="pill border-warmth/60 bg-warmth/15 text-warmth"
              title="colouring by length of 2D shift arrow (pull intensity)"
            >
              pull heatmap
            </span>
          )}
          <span className="pill">{pts.length} points</span>
          {hasShift && (
            <span className="pill text-accent-300">{shift.data!.arrows.length} arrows</span>
          )}
          <span className="pill uppercase tracking-wider">{reducer}</span>
        </div>
      </div>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 540,
          margin: { l: 24, r: 12, t: 8, b: 24 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#cad4e0", family: "Inter, system-ui" },
          xaxis: { showgrid: true, gridcolor: "rgba(255,255,255,0.04)", zeroline: false, showticklabels: false },
          yaxis: { showgrid: true, gridcolor: "rgba(255,255,255,0.04)", zeroline: false, showticklabels: false },
          legend: { orientation: "h", y: -0.05, font: { size: 11 } },
          hoverlabel: { bgcolor: "#10131c", bordercolor: "#3a4458", font: { color: "#e8edf3" } },
          annotations,
        }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        config={{ displaylogo: false, responsive: true, scrollZoom: true }}
      />
    </div>
  );
}

function Skeleton({ text }: { text: string }) {
  return (
    <div className="panel-tight flex h-[580px] items-center justify-center text-sm text-ink-300">
      <div className="flex items-center gap-3">
        <span className="h-2 w-2 animate-pulse rounded-full bg-accent-400" />
        {text}
      </div>
    </div>
  );
}

function ErrorPanel({ text }: { text: string }) {
  return <div className="panel-pad text-sm text-danger">{text}</div>;
}

function convexHull(points: [number, number][]): [number, number][] {
  const pts = points.slice().sort((a, b) => (a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]));
  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: [number, number][] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: [number, number][] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function hexToRgba(hex: string, a: number) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}
