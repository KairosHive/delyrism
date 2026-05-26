"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Plot } from "../plots/Plot";
import {
  api,
  TopologySynergyResponse,
  PairCyclesResponse,
  PairCycle,
} from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { Skeleton } from "../ui/Skeleton";

/**
 * Synergy view — the "shared topology" map.
 *
 *   ▸ S×S heatmap of synergy_H1 (or H2): how much loop / void mass
 *     exists ONLY when the two symbols' clouds are connected.
 *     Higher = the two archetypes share structural topology that
 *     neither has alone.
 *   ▸ Click a cell → pair drill-down panel below shows the actual
 *     persistent cycles in A∪B, tagged pure_a / pure_b / mixed.
 *     Mixed cycles are drawn on a joint PCA scatter as paths that
 *     literally cross between the two clouds — the visible "bridge"
 *     between archetypes.
 */
export function TopologySynergy() {
  const sid = useSidebar((s) => s.spaceId);
  const colorMap = useSidebar((s) => s.colorMap);
  const [dim, setDim] = React.useState<"h1" | "h2">("h1");
  const [pair, setPair] = React.useState<{ a: string; b: string } | null>(null);

  const q = useQuery({
    enabled: !!sid,
    queryKey: ["topo-synergy", sid],
    queryFn: () => api.post<TopologySynergyResponse>("/topology/synergy", { space_id: sid }),
  });

  return (
    <div className="space-y-4">
      <div className="panel-tight">
        <div className="mb-2 flex items-center justify-between gap-3">
          <div>
            <div className="section-title">Pair synergy</div>
            <div className="text-[11px] text-ink-400">
              symbols that share structural topology — click a cell to see the cycles
            </div>
          </div>
          <DimToggle dim={dim} onChange={setDim} />
        </div>
        {q.isPending && <Skeleton height={460} />}
        {q.error && <div className="text-sm text-danger">{(q.error as Error).message}</div>}
        {q.data && (
          <SynergyHeatmap
            data={q.data}
            dim={dim}
            onPickPair={(a, b) => setPair({ a, b })}
            active={pair}
          />
        )}
      </div>

      {pair && (
        <PairDrillDown
          a={pair.a}
          b={pair.b}
          colorMap={colorMap}
          onClose={() => setPair(null)}
        />
      )}
    </div>
  );
}

function DimToggle({ dim, onChange }: { dim: "h1" | "h2"; onChange: (d: "h1" | "h2") => void }) {
  return (
    <div className="inline-flex overflow-hidden rounded-md border border-ink-700">
      <button
        className={`px-2.5 py-1 text-[11px] transition ${
          dim === "h1" ? "bg-accent-600/30 text-ink-50" : "text-ink-300 hover:bg-ink-800"
        }`}
        onClick={() => onChange("h1")}
      >
        H1 loops
      </button>
      <button
        className={`px-2.5 py-1 text-[11px] transition ${
          dim === "h2" ? "bg-accent-600/30 text-ink-50" : "text-ink-300 hover:bg-ink-800"
        }`}
        onClick={() => onChange("h2")}
      >
        H2 voids
      </button>
    </div>
  );
}

function SynergyHeatmap({
  data, dim, onPickPair, active,
}: {
  data: TopologySynergyResponse;
  dim: "h1" | "h2";
  onPickPair: (a: string, b: string) => void;
  active: { a: string; b: string } | null;
}) {
  const syms = data.symbols;
  const n = syms.length;

  // S×S matrix, symmetric, NaN on diagonal
  const M: (number | null)[][] = Array.from({ length: n }, () => Array(n).fill(null));
  for (const e of data.entries) {
    const i = syms.indexOf(e.a);
    const j = syms.indexOf(e.b);
    if (i < 0 || j < 0) continue;
    const v = dim === "h1" ? e.synergy_h1 : e.synergy_h2;
    M[i][j] = v;
    M[j][i] = v;
  }

  // Find global max for colour normalisation
  const flat: number[] = [];
  for (const e of data.entries) flat.push(dim === "h1" ? e.synergy_h1 : e.synergy_h2);
  const vmax = Math.max(0.001, ...flat.map((v) => Math.abs(v)));

  return (
    <>
      <Plot
        data={[
          {
            z: M,
            x: syms,
            y: syms,
            type: "heatmap",
            colorscale: [
              [0.0, "#1b212e"],
              [0.4, "#3bbdb0"],
              [0.7, "#d08770"],
              [1.0, "#ef476f"],
            ],
            zmin: 0,
            zmax: vmax,
            colorbar: { title: { text: dim === "h1" ? "syn H1" : "syn H2", side: "right" }, thickness: 12, len: 0.8 },
            hovertemplate: "%{y} ⋈ %{x}<br>synergy = %{z:.3f}<extra></extra>",
          },
        ]}
        layout={{
          autosize: true,
          height: Math.max(360, 32 * n + 80),
          margin: { l: 110, r: 20, t: 8, b: 110 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#cad4e0", family: "Inter, system-ui" },
          xaxis: { tickangle: -55, tickfont: { size: 10 }, showgrid: false, ticks: "",
                   scaleanchor: "y", constrain: "domain" },
          yaxis: { autorange: "reversed", tickfont: { size: 10 }, showgrid: false, ticks: "",
                   constrain: "domain" },
          hoverlabel: { bgcolor: "#10131c", bordercolor: "#3a4458", font: { color: "#e8edf3" } },
        }}
        onClick={(ev: any) => {
          const pt = ev?.points?.[0];
          if (!pt) return;
          const a = pt.y as string; const b = pt.x as string;
          if (a !== b) onPickPair(a, b);
        }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        config={{ displaylogo: false, responsive: true }}
      />
      {active && (
        <div className="mt-2 text-[10px] text-ink-400">
          showing pair <span className="text-ink-100">{active.a} ⋈ {active.b}</span> below — click another cell to switch.
        </div>
      )}
    </>
  );
}

function PairDrillDown({
  a, b, colorMap, onClose,
}: {
  a: string;
  b: string;
  colorMap: Record<string, string>;
  onClose: () => void;
}) {
  const sid = useSidebar((s) => s.spaceId);
  const [activeIdx, setActiveIdx] = React.useState<number>(0);

  // Sort the two symbols so cache keys collapse for either order
  const [pa, pb] = a < b ? [a, b] : [b, a];

  const q = useQuery({
    enabled: !!sid,
    queryKey: ["topo-pair", sid, pa, pb],
    queryFn: () =>
      api.post<PairCyclesResponse>("/topology/pair-cycles", {
        space_id: sid, a: pa, b: pb, top_h1: 8, top_h2: 4,
      }),
  });
  React.useEffect(() => { setActiveIdx(0); }, [pa, pb]);

  const ca = colorMap[pa] ?? "#88c0d0";
  const cb = colorMap[pb] ?? "#d08770";

  return (
    <div className="panel-tight">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div>
          <div className="section-title flex items-center gap-2">
            Pair · <span style={{ color: ca }}>{pa}</span>
            <span className="text-ink-500">⋈</span>
            <span style={{ color: cb }}>{pb}</span>
          </div>
          <div className="text-[11px] text-ink-400">
            cycles in the union — pure or bridging
          </div>
        </div>
        <button onClick={onClose} className="text-[11px] text-ink-400 hover:text-ink-200">
          close ✕
        </button>
      </div>

      {q.isPending && (
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr,1.6fr]">
          <Skeleton lines={6} />
          <Skeleton height={520} />
        </div>
      )}
      {q.error && <div className="text-sm text-danger">{(q.error as Error).message}</div>}
      {q.data && (
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr,1.6fr]">
          <PairCycleList
            cycles={q.data.cycles}
            activeIdx={activeIdx}
            onActivate={setActiveIdx}
            ca={ca}
            cb={cb}
            a={pa}
            b={pb}
          />
          <PairCyclePlot
            data={q.data}
            active={q.data.cycles[activeIdx] ?? null}
            ca={ca}
            cb={cb}
          />
        </div>
      )}
    </div>
  );
}

const MIX_STYLE: Record<string, { label: string; colour: string }> = {
  pure_a: { label: "pure A", colour: "#9fadc1" },
  pure_b: { label: "pure B", colour: "#9fadc1" },
  mixed:  { label: "mixed (bridge)", colour: "#5fcfc4" },
};

function PairCycleList({
  cycles, activeIdx, onActivate, ca, cb, a, b,
}: {
  cycles: PairCycle[];
  activeIdx: number;
  onActivate: (i: number) => void;
  ca: string;
  cb: string;
  a: string;
  b: string;
}) {
  if (!cycles.length) {
    return (
      <div className="rounded-md border border-ink-700/40 bg-ink-900/30 p-3 text-[11px] text-ink-400">
        No persistent cycles in the union of these two clouds.
      </div>
    );
  }
  return (
    <div className="space-y-1.5">
      {cycles.map((cyc, i) => {
        const styleKey = cyc.mix === "pure_a" || cyc.mix === "pure_b" ? cyc.mix : "mixed";
        const label =
          cyc.mix === "pure_a" ? `pure ${a}` :
          cyc.mix === "pure_b" ? `pure ${b}` : "mixed (bridge)";
        const labelColour = cyc.mix === "pure_a" ? ca : cyc.mix === "pure_b" ? cb : MIX_STYLE.mixed.colour;
        const active = i === activeIdx;
        return (
          <button
            key={i}
            onClick={() => onActivate(i)}
            className="block w-full rounded-md border p-2 text-left text-[11px] transition"
            style={{
              borderColor: active ? labelColour : "rgba(255,255,255,0.08)",
              background: active ? `${labelColour}10` : "rgba(255,255,255,0.02)",
              boxShadow: active ? `inset 0 0 0 1px ${labelColour}55` : "none",
            }}
          >
            <div className="mb-1 flex items-center gap-1.5 text-[10px]">
              <span
                className="rounded px-1.5 py-0.5 font-mono"
                style={{ background: cyc.dim === 1 ? "#3bbdb022" : "#d0877022",
                         border: `1px solid ${cyc.dim === 1 ? "#3bbdb088" : "#d0877088"}`,
                         color:  cyc.dim === 1 ? "#5fcfc4" : "#d08770" }}
              >
                H{cyc.dim}
              </span>
              <span
                className="rounded px-1.5 py-0.5"
                style={{ background: `${labelColour}1a`, border: `1px solid ${labelColour}66`, color: labelColour }}
              >
                {label}
              </span>
              <span className="text-ink-400">pers</span>
              <span className="font-mono text-ink-200">{cyc.persistence.toFixed(3)}</span>
              {cyc.cross_fraction > 0 && (
                <span className="text-ink-500">· cross {(cyc.cross_fraction * 100).toFixed(0)}%</span>
              )}
            </div>
            <div className="text-ink-100">
              {cyc.dim === 1 ? "↻ " : "◆ "}
              {cyc.vertices.map((v, vi) => (
                <React.Fragment key={vi}>
                  {vi > 0 && <span className="text-ink-500">{cyc.dim === 1 ? " → " : " · "}</span>}
                  <span style={{ color: v.home_symbol === a ? ca : cb }}>{v.word}</span>
                </React.Fragment>
              ))}
              {cyc.dim === 1 && cyc.vertices.length > 0 && (
                <span className="text-ink-500"> → {cyc.vertices[0].word}</span>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function PairCyclePlot({
  data, active, ca, cb,
}: {
  data: PairCyclesResponse;
  active: PairCycle | null;
  ca: string;
  cb: string;
}) {
  const activeSet = new Set((active?.vertices ?? []).map((v) => v.index));
  const a = data.a; const b = data.b;

  const traces: any[] = [];
  // background: all descriptors, coloured by home symbol, dimmed
  // split into two traces for proper legend
  const pointsA = data.descriptors.filter((d) => d.home_symbol === a);
  const pointsB = data.descriptors.filter((d) => d.home_symbol === b);
  for (const [grp, name, col] of [[pointsA, a, ca], [pointsB, b, cb]] as const) {
    traces.push({
      x: grp.map((d) => d.x),
      y: grp.map((d) => d.y),
      text: grp.map((d) => d.word),
      type: "scatter",
      mode: "markers",
      name,
      marker: {
        size: grp.map((d) => (activeSet.has(d.index) ? 13 : 8)),
        color: grp.map((d) => (activeSet.has(d.index) ? col : `${col}55`)),
        line: {
          color: grp.map((d) => (activeSet.has(d.index) ? "white" : "rgba(0,0,0,0.3)")),
          width: grp.map((d) => (activeSet.has(d.index) ? 1.5 : 0.4)),
        },
      },
      hovertemplate: `<b>${name}</b> · %{text}<extra></extra>`,
    });
  }

  if (active && active.vertices.length >= 2) {
    if (active.dim === 1) {
      const xs = active.vertices.map((v) => v.x).concat([active.vertices[0].x]);
      const ys = active.vertices.map((v) => v.y).concat([active.vertices[0].y]);
      const edgeColour =
        active.mix === "pure_a" ? ca :
        active.mix === "pure_b" ? cb : "#5fcfc4";
      traces.push({
        x: xs, y: ys,
        type: "scatter", mode: "lines",
        line: { color: edgeColour, width: 2.5, shape: "spline" },
        hoverinfo: "skip", showlegend: false,
      });
    } else {
      const pts = active.vertices.map((v) => [v.x, v.y] as [number, number]);
      const hull = convexHull(pts);
      if (hull.length >= 3) {
        hull.push(hull[0]);
        const edgeColour =
          active.mix === "pure_a" ? ca :
          active.mix === "pure_b" ? cb : "#5fcfc4";
        traces.push({
          x: hull.map((h) => h[0]),
          y: hull.map((h) => h[1]),
          type: "scatter", mode: "lines",
          fill: "toself",
          fillcolor: `${edgeColour}25`,
          line: { color: edgeColour, width: 1.5 },
          hoverinfo: "skip", showlegend: false,
        });
      }
    }
    traces.push({
      x: active.vertices.map((v) => v.x),
      y: active.vertices.map((v) => v.y),
      text: active.vertices.map((v, i) =>
        active.dim === 1 ? `${i + 1}· ${v.word}` : v.word
      ),
      type: "scatter", mode: "text",
      textposition: "top center",
      textfont: { color: "#dbe2ee", size: 11, family: "Inter, system-ui" },
      hoverinfo: "skip", showlegend: false,
    });
  }

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        height: 540,
        margin: { l: 20, r: 12, t: 8, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#cad4e0", family: "Inter, system-ui" },
        legend: { orientation: "h", y: -0.05, font: { size: 11 } },
        xaxis: { showgrid: true, gridcolor: "rgba(255,255,255,0.04)", zeroline: false, showticklabels: false },
        yaxis: { showgrid: true, gridcolor: "rgba(255,255,255,0.04)", zeroline: false, showticklabels: false },
        hoverlabel: { bgcolor: "#10131c", bordercolor: "#3a4458", font: { color: "#e8edf3" } },
      }}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
      config={{ displaylogo: false, responsive: true, scrollZoom: true }}
    />
  );
}

function convexHull(points: [number, number][]): [number, number][] {
  if (points.length < 3) return points.slice();
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
  lower.pop(); upper.pop();
  return lower.concat(upper);
}
