"use client";
import * as React from "react";
import dynamic from "next/dynamic";
import { useDeltaGraph } from "@/lib/hooks";
import { SymbolLegend } from "./SymbolLegend";
import { useMeasure } from "@/lib/useMeasure";
import { tuneForces, recenterAndFit } from "@/lib/graphForces";

// IMPORTANT: load the forwardRef wrapper, NOT react-force-graph-2d directly.
// `next/dynamic` can't forward refs through to a class component; without
// this indirection every `fgRef.current` is null and tuneForces / zoomToFit
// silently no-op.  See components/plots/ForceGraph2DInner.tsx.
const ForceGraph2D = dynamic(() => import("../plots/ForceGraph2DInner"), { ssr: false }) as any;

export function DeltaGraph() {
  const q = useDeltaGraph();
  const [wrapRef, w] = useMeasure();
  const fgRef = React.useRef<any>(null);

  React.useEffect(() => {
    if (!q.data || !fgRef.current) return;
    tuneForces(fgRef.current);
    const handles = [1200, 3000].map((d) =>
      setTimeout(() => recenterAndFit(fgRef.current, 20), d),
    );
    return () => handles.forEach(clearTimeout);
  }, [q.data]);

  // Compute per-node weight = sum of |Δ| on incident edges, then min-max
  // normalize so the visual range is independent of context strength.
  // Matches the engine's `plot_delta_graph` sizing logic.
  const { data, edgeLo, edgeRange } = React.useMemo(() => {
    if (!q.data) return { data: { nodes: [], links: [] }, edgeLo: 0, edgeRange: 1 };

    // Node weight = Σ|Δ| on incident edges, min-max normalized.
    const weight = new Map<string, number>();
    for (const e of q.data.edges) {
      weight.set(e.source, (weight.get(e.source) ?? 0) + e.abs_delta);
      weight.set(e.target, (weight.get(e.target) ?? 0) + e.abs_delta);
    }
    const wvals = Array.from(weight.values());
    const wLo = wvals.length ? Math.min(...wvals) : 0;
    const wHi = wvals.length ? Math.max(...wvals) : 1;
    const wRange = wHi - wLo > 1e-12 ? wHi - wLo : 1;
    const norm = new Map<string, number>();
    for (const [k, v] of weight.entries()) norm.set(k, (v - wLo) / wRange);

    // Edge widths: min-max normalize |Δ| so the thinnest edge maps to 0.6 px
    // and the thickest to 6.0 px — matches the engine's `plot_delta_graph`.
    const evals = q.data.edges.map((e) => e.abs_delta);
    const eLo = evals.length ? Math.min(...evals) : 0;
    const eHi = evals.length ? Math.max(...evals) : 1;
    const eRange = eHi - eLo > 1e-12 ? eHi - eLo : 1;

    return {
      data: {
        nodes: q.data.nodes.map((n) => ({
          id: n.id, label: n.id, color: n.color,
          weight: norm.get(n.id) ?? 0,  // 0..1
        })),
        links: q.data.edges.map((e) => ({
          source: e.source, target: e.target,
          value: e.abs_delta, sign: e.sign, delta: e.delta,
        })),
      },
      edgeLo: eLo,
      edgeRange: eRange,
    };
  }, [q.data]);

  const height = Math.max(560, Math.min(820, Math.round((w || 1000) * 0.45)));

  return (
    <div ref={wrapRef} className="panel-tight overflow-hidden">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
        <div className="section-title">Context Δ graph</div>
        <SymbolLegend />
        <div className="flex items-center gap-2 text-[10px]">
          <span className="pill"><span className="mr-1 inline-block h-1.5 w-3 rounded-full bg-accent-400"/> ↑ strengthens</span>
          <span className="pill"><span className="mr-1 inline-block h-1.5 w-3 rounded-full bg-danger"/> ↓ weakens</span>
          {q.data && <span className="pill">{q.data.edges.length} edges</span>}
          <span className="pill text-ink-400">size = Σ|Δ| · width = |Δ|</span>
        </div>
      </div>

      {!q.data && (
        <div className="flex items-center justify-center text-sm text-ink-300" style={{ height }}>
          Provide a context (sentence or weights) to compute Δ-edges.
        </div>
      )}
      {q.data && w > 0 && (
        <ForceGraph2D
          fwdRef={fgRef}
          graphData={data}
          width={w - 24}
          height={height}
          backgroundColor="rgba(0,0,0,0)"
          cooldownTicks={250}
          d3AlphaDecay={0.02}
          onEngineStop={() => recenterAndFit(fgRef.current, 20)}
          // Edge widths: |Δ| min-max normalized to a 0.6–6.0 range, matching
          // `plot_delta_graph`'s edge_width_min / edge_width_max.
          linkColor={(l: any) => (l.sign === "up" ? "rgba(38,161,149,0.78)" : "rgba(191,97,106,0.72)")}
          linkWidth={(l: any) => 0.6 + ((l.value - edgeLo) / edgeRange) * 5.4}
          // Node sizes scaled by normalized Σ|Δ|.  Radius range ~4 → ~14 px.
          nodeRelSize={1}
          nodeVal={(node: any) => 4 + node.weight * 24}  // val = area
          nodeCanvasObject={(node: any, ctx: any, scale: number) => {
            // Use the same weight to drive radius (matches the area implied
            // by nodeVal — react-force-graph uses sqrt(val) for radius).
            const r = (3 + node.weight * 10) + 1.5 / scale;
            ctx.beginPath();
            ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
            ctx.fillStyle = node.color;
            ctx.fill();
            ctx.lineWidth = 0.6;
            ctx.strokeStyle = "rgba(0,0,0,0.5)";
            ctx.stroke();
            if (scale > 0.55) {
              const fs = (12 + node.weight * 4) / scale;
              ctx.font = `600 ${fs}px Inter, system-ui`;
              ctx.textAlign = "center";
              ctx.textBaseline = "top";
              ctx.fillStyle = "rgba(0,0,0,0.7)";
              ctx.fillText(node.label, node.x + 0.6 / scale, node.y + r + 2.6 / scale);
              ctx.fillStyle = "#f2f6fb";
              ctx.fillText(node.label, node.x, node.y + r + 2 / scale);
            }
          }}
          // Make the link's contact area scale with line width so hover
          // detection still works on thick lines.
          linkHoverPrecision={6}
        />
      )}
    </div>
  );
}
