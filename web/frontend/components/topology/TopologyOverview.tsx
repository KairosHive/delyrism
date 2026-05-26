"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Plot } from "../plots/Plot";
import { api, TopologySummaryResponse, TopologySummaryEntry } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { Skeleton } from "../ui/Skeleton";

/**
 * Topology overview — the headline single-screen read.
 *   • TopoScore-ranked table on the left
 *   • Cohesion ↑↓ Loopiness scatter on the right; dot size = voidiness
 *
 * Every archetype's "topological signature" in one frame.  Hover a row
 * in the table → its dot lights up on the map; hover a dot → the row
 * highlights.
 */
export function TopologyOverview() {
  const sid = useSidebar((s) => s.spaceId);
  const colorMap = useSidebar((s) => s.colorMap);
  const [hover, setHover] = React.useState<string | null>(null);

  const q = useQuery({
    enabled: !!sid,
    queryKey: ["topo-summary", sid],
    queryFn: () => api.post<TopologySummaryResponse>("/topology/summary", { space_id: sid }),
  });

  if (q.isPending) {
    return (
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr,1.4fr]">
        <Skeleton height={420} />
        <Skeleton height={420} />
      </div>
    );
  }
  if (q.error) {
    return <div className="panel-pad text-sm text-danger">{(q.error as Error).message}</div>;
  }
  const data = q.data!;
  if (!data.ripser_available) {
    return (
      <div className="panel-pad text-sm text-ink-300">
        <div className="mb-1 font-medium text-ink-100">ripser not installed</div>
        Persistent-homology endpoints require the <code className="text-accent-300">ripser</code> package.
        The H1/H2 metrics will be 0 until it's available on the backend host.
      </div>
    );
  }

  const sorted = [...data.entries].sort((a, b) => b.topo_score - a.topo_score);
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr,1.4fr]">
      {/* ── ranked table ── */}
      <TopoScoreTable
        entries={sorted}
        colorMap={colorMap}
        hover={hover}
        onHover={setHover}
      />

      {/* ── cohesion-vs-loopiness map ── */}
      <TopologyMap
        entries={data.entries}
        colorMap={colorMap}
        hover={hover}
        onHover={setHover}
      />
    </div>
  );
}

function TopoScoreTable({
  entries, colorMap, hover, onHover,
}: {
  entries: TopologySummaryEntry[];
  colorMap: Record<string, string>;
  hover: string | null;
  onHover: (s: string | null) => void;
}) {
  // Find max values for the inline bars
  const maxH1 = Math.max(...entries.map((e) => e.h1_sum), 1e-6);
  const maxH2 = Math.max(...entries.map((e) => e.h2_sum), 1e-6);
  const minCoh = Math.min(...entries.map((e) => e.h0_cohesion));
  const maxCoh = Math.max(...entries.map((e) => e.h0_cohesion), minCoh + 1e-6);

  return (
    <div className="panel-tight">
      <div className="mb-2">
        <div className="section-title">TopoScore</div>
        <div className="text-[11px] text-ink-400">
          z-scored composite — high = many loops/voids, tight cluster. ranked.
        </div>
      </div>
      <div className="space-y-0.5">
        <div className="grid grid-cols-[1fr,3rem,1.4fr,1.4fr,3.2rem] gap-2 px-1 pb-1 text-[10px] uppercase tracking-wider text-ink-500">
          <div>archetype</div>
          <div className="text-right">coh.</div>
          <div>H1 loops</div>
          <div>H2 voids</div>
          <div className="text-right">score</div>
        </div>
        {entries.map((e) => {
          const c = colorMap[e.symbol] ?? "#88c0d0";
          const cohN = (e.h0_cohesion - minCoh) / (maxCoh - minCoh + 1e-9);
          const isHover = hover === e.symbol;
          return (
            <div
              key={e.symbol}
              className="grid grid-cols-[1fr,3rem,1.4fr,1.4fr,3.2rem] items-center gap-2 rounded-md px-1.5 py-1 text-[11px] transition"
              style={{
                background: isHover ? `${c}18` : "transparent",
                boxShadow: isHover ? `inset 0 0 0 1px ${c}55` : "none",
              }}
              onMouseEnter={() => onHover(e.symbol)}
              onMouseLeave={() => onHover(null)}
            >
              <div className="flex items-center gap-1.5 truncate">
                <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: c }} />
                <span style={{ color: c }} className="truncate font-medium">{e.symbol}</span>
              </div>
              <div
                className="text-right font-mono text-[10px] text-ink-300"
                title={`H0 cohesion = ${e.h0_cohesion.toFixed(3)} (lower = tighter)`}
              >
                {e.h0_cohesion.toFixed(2)}
              </div>
              <InlineBar value={e.h1_sum} max={maxH1} color={c} label={`${e.h1_count}×`} />
              <InlineBar value={e.h2_sum} max={maxH2} color={c} label={`${e.h2_count}×`} />
              <div
                className="text-right font-mono text-[11px]"
                style={{ color: e.topo_score >= 0 ? "#5fcfc4" : "#9fadc1" }}
              >
                {e.topo_score > 0 ? "+" : ""}{e.topo_score.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-3 border-t border-ink-700/40 pt-2 text-[10px] leading-relaxed text-ink-400">
        Lower <span className="text-ink-200">cohesion</span> = tighter cluster.
        Higher <span className="text-ink-200">H1</span> = more meaning-loops.
        Higher <span className="text-ink-200">H2</span> = more voids (sphere-like cavities).
      </div>
    </div>
  );
}

function InlineBar({ value, max, color, label }: { value: number; max: number; color: string; label?: string }) {
  const pct = Math.max(2, Math.min(100, (value / max) * 100));
  return (
    <div className="flex items-center gap-1.5">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-800">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color, opacity: 0.85 }} />
      </div>
      {label && <span className="w-7 shrink-0 text-right font-mono text-[9px] text-ink-500">{label}</span>}
    </div>
  );
}

function TopologyMap({
  entries, colorMap, hover, onHover,
}: {
  entries: TopologySummaryEntry[];
  colorMap: Record<string, string>;
  hover: string | null;
  onHover: (s: string | null) => void;
}) {
  // Plot one trace per archetype so each gets its own colour + name in legend.
  const maxH2 = Math.max(...entries.map((e) => e.h2_sum), 1e-6);
  const traces = entries.map((e) => {
    const c = colorMap[e.symbol] ?? "#88c0d0";
    const size = 12 + 32 * (e.h2_sum / maxH2);
    const dim = hover != null && hover !== e.symbol;
    return {
      x: [e.h0_cohesion],
      y: [e.h1_sum],
      text: [e.symbol],
      type: "scatter" as const,
      mode: "markers+text" as const,
      name: e.symbol,
      marker: {
        size,
        color: c,
        opacity: dim ? 0.18 : 0.9,
        line: { color: hover === e.symbol ? "#ffffff" : "rgba(0,0,0,0.4)", width: hover === e.symbol ? 2 : 0.6 },
      },
      textposition: "top center" as const,
      textfont: { color: dim ? "#3a4458" : c, size: 11, family: "Inter, system-ui" },
      hovertemplate:
        `<b>${e.symbol}</b><br>` +
        `cohesion (H0): ${e.h0_cohesion.toFixed(3)}<br>` +
        `loopiness (H1): ${e.h1_sum.toFixed(3)} · ${e.h1_count} loops<br>` +
        `voidiness (H2): ${e.h2_sum.toFixed(3)} · ${e.h2_count} voids<br>` +
        `TopoScore: ${e.topo_score >= 0 ? "+" : ""}${e.topo_score.toFixed(2)}<extra></extra>`,
    };
  });

  return (
    <div
      className="panel-tight"
      onMouseLeave={() => onHover(null)}
    >
      <div className="mb-2">
        <div className="section-title">Cohesion × Loopiness</div>
        <div className="text-[11px] text-ink-400">
          left = tighter cluster · up = more loops · bigger dot = more voids
        </div>
      </div>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 440,
          showlegend: false,
          margin: { l: 56, r: 24, t: 8, b: 56 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#cad4e0", family: "Inter, system-ui" },
          xaxis: {
            title: { text: "H0 cohesion (lower = tighter)", standoff: 8 },
            showgrid: true, gridcolor: "rgba(255,255,255,0.05)", zeroline: false,
          },
          yaxis: {
            title: { text: "H1 loopiness", standoff: 8 },
            showgrid: true, gridcolor: "rgba(255,255,255,0.05)", zeroline: false,
          },
          hoverlabel: { bgcolor: "#10131c", bordercolor: "#3a4458", font: { color: "#e8edf3" } },
        }}
        onHover={(ev: any) => {
          const sym = ev?.points?.[0]?.data?.name as string | undefined;
          if (sym) onHover(sym);
        }}
        onUnhover={() => onHover(null)}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        config={{ displaylogo: false, responsive: true }}
      />
      <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] text-ink-400">
        <div><span className="text-ink-200">↘ lower-left</span> — diffuse, no loops (boring)</div>
        <div><span className="text-ink-200">↗ upper-right</span> — loose but loopy (rich but uncentered)</div>
        <div><span className="text-ink-200">↖ upper-left</span> — tight AND loopy (the mythic core)</div>
        <div><span className="text-ink-200">↙ lower-right</span> — diffuse with hidden structure</div>
      </div>
    </div>
  );
}
