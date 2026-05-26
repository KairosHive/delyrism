"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Plot } from "../plots/Plot";
import { api, PersistenceDiagramResponse } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { Skeleton } from "../ui/Skeleton";

/**
 * Persistence diagrams — birth/death scatter plot.
 *
 * Each dot is a topological feature.  Distance to the diagonal =
 * persistence (how long the feature lived through the filtration).
 * Far-from-diagonal dots are the "real" structure; near-diagonal dots
 * are noise.  The colour distinguishes H0 (connected components),
 * H1 (loops), H2 (voids).
 *
 * Picker: choose which symbol to look at.  H0/H1/H2 toggles live as
 * a small legend.  Hover a point to see exact birth/death.
 */
export function TopologyDiagrams() {
  const sid = useSidebar((s) => s.spaceId);
  const symbols = useSidebar((s) => s.symbols);
  const colorMap = useSidebar((s) => s.colorMap);
  const [symbol, setSymbol] = React.useState<string>("");

  React.useEffect(() => {
    if (!symbol && symbols.length) setSymbol(symbols[0]);
  }, [symbols, symbol]);

  const q = useQuery({
    enabled: !!sid && !!symbol,
    queryKey: ["topo-diagram", sid, symbol],
    queryFn: () =>
      api.post<PersistenceDiagramResponse>("/topology/diagrams", { space_id: sid, symbol }),
  });

  return (
    <div className="panel-tight">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <div className="section-title">Persistence diagram · {symbol}</div>
          <div className="text-[11px] text-ink-400">
            far from diagonal = persistent feature. close to diagonal = noise.
          </div>
        </div>
        <select
          className="select-base !w-auto !min-w-[160px]"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
        >
          {symbols.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      {q.isPending && <Skeleton height={520} />}
      {q.error && (
        <div className="text-sm text-danger">{(q.error as Error).message}</div>
      )}
      {q.data && <Diagram data={q.data} accent={colorMap[symbol] ?? "#88c0d0"} />}
    </div>
  );
}

function Diagram({ data, accent }: { data: PersistenceDiagramResponse; accent: string }) {
  // Notebook-canonical palette: H0 blue · H1 orange · H2 green.  Matches
  // the matplotlib defaults the user is comparing this view against.
  const HD_COLORS: Record<number, string> = {
    0: "#5fa8d3", // H0 — components, blue
    1: "#e67e22", // H1 — loops, orange
    2: "#2ecc71", // H2 — voids, green
  };
  const HD_LABEL: Record<number, string> = {
    0: "H0 · components",
    1: "H1 · loops",
    2: "H2 · voids",
  };

  const dims: (0 | 1 | 2)[] = [0, 1, 2];
  const lim = Math.max(0.05, data.max_finite_death * 1.15);

  const traces: any[] = [];
  // diagonal reference line
  traces.push({
    x: [0, lim], y: [0, lim],
    type: "scatter", mode: "lines",
    line: { color: "rgba(255,255,255,0.15)", width: 1, dash: "dash" },
    hoverinfo: "skip", showlegend: false,
  });
  // infinity reference line (for H0's infinite component)
  const infY = data.max_finite_death * 1.10;
  traces.push({
    x: [0, lim], y: [infY, infY],
    type: "scatter", mode: "lines",
    line: { color: "rgba(255,255,255,0.08)", width: 1, dash: "dot" },
    hoverinfo: "skip", showlegend: false,
  });
  // per-dimension scatter
  for (const d of dims) {
    const pts = data.points.filter((p) => p.dim === d);
    if (!pts.length) continue;
    traces.push({
      x: pts.map((p) => p.birth),
      y: pts.map((p) => (p.is_infinite ? infY : p.death)),
      text: pts.map((p) => (p.is_infinite ? "∞" : "")),
      type: "scatter",
      mode: "markers+text",
      name: HD_LABEL[d],
      marker: {
        size: d === 0 ? 8 : d === 1 ? 11 : 14,
        color: HD_COLORS[d],
        line: { color: "rgba(0,0,0,0.4)", width: 0.5 },
        opacity: 0.88,
      },
      textfont: { color: "#cad4e0", size: 11 },
      textposition: "middle right",
      hovertemplate:
        `<b>${HD_LABEL[d]}</b><br>birth: %{x:.3f}<br>death: %{y:.3f}<extra></extra>`,
    });
  }

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        height: 520,
        margin: { l: 56, r: 32, t: 16, b: 56 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#cad4e0", family: "Inter, system-ui" },
        legend: { orientation: "h", y: -0.12, font: { size: 11 } },
        xaxis: {
          title: { text: "birth scale ε", standoff: 8 },
          range: [0, lim],
          showgrid: true, gridcolor: "rgba(255,255,255,0.05)", zeroline: false,
        },
        yaxis: {
          title: { text: "death scale ε   (∞ pinned at top)", standoff: 8 },
          range: [0, lim],
          showgrid: true, gridcolor: "rgba(255,255,255,0.05)", zeroline: false,
        },
        hoverlabel: { bgcolor: "#10131c", bordercolor: accent, font: { color: "#e8edf3" } },
      }}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
      config={{ displaylogo: false, responsive: true }}
    />
  );
}
