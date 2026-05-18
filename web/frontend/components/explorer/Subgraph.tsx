"use client";
import * as React from "react";
import dynamic from "next/dynamic";
import { useSubgraph } from "@/lib/hooks";
import { SymbolLegend } from "./SymbolLegend";
import { useMeasure } from "@/lib/useMeasure";
import { tuneForces, recenterAndFit } from "@/lib/graphForces";

// See components/plots/ForceGraph2DInner.tsx — refs only forward via this
// wrapper, otherwise tuneForces / recenterAndFit run on a null ref.
const ForceGraph2D = dynamic(() => import("../plots/ForceGraph2DInner"), { ssr: false }) as any;

export function Subgraph() {
  const q = useSubgraph();
  const [wrapRef, width] = useMeasure();
  const fgRef = React.useRef<any>(null);

  React.useEffect(() => {
    if (!q.data || !fgRef.current) return;
    tuneForces(fgRef.current, { compact: true, debug: true });
    const handles = [1000, 2500].map((d) =>
      setTimeout(() => recenterAndFit(fgRef.current, 5), d),
    );
    return () => handles.forEach(clearTimeout);
  }, [q.data]);

  const data = React.useMemo(() => {
    if (!q.data) return { nodes: [], links: [] };
    // Normalize descriptor scores so the size encoding is independent of the
    // absolute PR/softmax magnitude (which depends on graph size + tau).
    const descScores = q.data.nodes.filter((n) => n.kind === "descriptor").map((n) => n.score);
    const lo = descScores.length ? Math.min(...descScores) : 0;
    const hi = descScores.length ? Math.max(...descScores) : 1;
    const range = hi - lo > 1e-9 ? hi - lo : 1;
    return {
      nodes: q.data.nodes.map((n) => {
        const norm = n.kind === "descriptor" ? (n.score - lo) / range : 1;
        return {
          id: n.id,
          label: n.id.replace(/^[SD]:/, ""),
          kind: n.kind,
          color: n.color,
          // Symbol nodes get a fixed large size; descriptor radius grows with
          // their score so the most context-relevant ones read as bigger.
          val: n.kind === "symbol" ? 16 : 3 + norm * 6,
        };
      }),
      links: q.data.edges.map((e) => ({ source: e.source, target: e.target, value: e.weight })),
    };
  }, [q.data]);

  const height = Math.max(440, Math.min(640, Math.round((width || 700) * 0.6)));

  return (
    <div ref={wrapRef} className="panel-tight overflow-hidden">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
        <div className="section-title">Contextual subgraph</div>
        <SymbolLegend />
      </div>

      {!q.data && (
        <div className="flex items-center justify-center text-sm text-ink-300" style={{ height }}>
          Add a context (sentence or audio) to compute the contextual subgraph.
        </div>
      )}
      {q.data && width > 0 && (
        <ForceGraph2D
          fwdRef={fgRef}
          graphData={data}
          width={width - 24}
          height={height}
          backgroundColor="rgba(0,0,0,0)"
          nodeRelSize={4}
          cooldownTicks={200}
          d3AlphaDecay={0.02}
          onEngineStop={() => recenterAndFit(fgRef.current, 5)}
          nodeLabel={(n: any) => n.label}
          linkColor={() => "rgba(255,255,255,0.18)"}
          linkWidth={(l: any) => Math.max(0.5, l.value * 4)}
          nodeCanvasObject={(node: any, ctx: any, scale: number) => {
            // Symbol nodes → rounded squares, full color, thick white border
            // (matches the engine's `node_shape="s"` plus our dark UI accent).
            // Descriptor nodes → circles with a *lightened* version of the
            // parent symbol's color, matching `lighten_color(..., 0.52)` from
            // plot_contextual_subgraph_colored.
            const isSymbol = node.kind === "symbol";
            if (isSymbol) {
              const s = Math.sqrt(node.val ?? 16) * 2.2 + 1 / scale;
              const x = node.x - s, y = node.y - s, side = s * 2;
              const radius = side * 0.18;
              roundRect(ctx, x, y, side, side, radius);
              ctx.fillStyle = node.color;
              ctx.fill();
              ctx.lineWidth = 1.4 / scale;
              ctx.strokeStyle = "rgba(255,255,255,0.85)";
              ctx.stroke();
              ctx.lineWidth = 0.7;
              ctx.strokeStyle = "rgba(0,0,0,0.7)";
              ctx.stroke();
            } else {
              const r = Math.sqrt(node.val ?? 3) * 2 + 1 / scale;
              ctx.beginPath();
              ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
              ctx.fillStyle = lighten(node.color, 0.45);
              ctx.fill();
              ctx.strokeStyle = "rgba(0,0,0,0.45)";
              ctx.lineWidth = 0.6;
              ctx.stroke();
            }
            // Labels — symbols always shown, descriptors only at zoom > 0.7
            if (isSymbol || scale > 0.7) {
              const r = isSymbol
                ? Math.sqrt(node.val ?? 16) * 2.2 + 1 / scale
                : Math.sqrt(node.val ?? 3) * 2 + 1 / scale;
              const fs = (isSymbol ? 16 : 13) / scale;
              ctx.font = `${isSymbol ? "700" : "500"} ${fs}px Inter, system-ui`;
              ctx.textAlign = "center";
              ctx.textBaseline = "top";
              ctx.fillStyle = "rgba(0,0,0,0.75)";
              ctx.fillText(node.label, node.x + 0.6 / scale, node.y + r + 2.6 / scale);
              ctx.fillStyle = "#f2f6fb";
              ctx.fillText(node.label, node.x, node.y + r + 2 / scale);
            }
          }}
        />
      )}
    </div>
  );
}

// ---- canvas helpers ---------------------------------------------------------

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

/** Lighten a hex color by `amount` (0..1) toward white.
 *  Mirrors `lighten_color(hex, amount)` in delyrism/delyrism.py. */
function lighten(hex: string, amount: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  const lr = Math.round(r + (255 - r) * amount);
  const lg = Math.round(g + (255 - g) * amount);
  const lb = Math.round(b + (255 - b) * amount);
  return `rgb(${lr},${lg},${lb})`;
}
