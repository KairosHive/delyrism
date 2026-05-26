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
import { useTopologyContext } from "./useTopologyContext";
import { ContextPill } from "./TopologyOverview";
import { DimFilter } from "./TopologyCycles";

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

  const ctx = useTopologyContext();
  const q = useQuery({
    enabled: !!sid,
    queryKey: ["topo-synergy", sid, ...ctx.keyTail],
    queryFn: () =>
      api.post<TopologySynergyResponse>("/topology/synergy", { space_id: sid, ...ctx.payload }),
  });

  return (
    <div className="space-y-4">
      {ctx.active && <ContextPill summary={ctx.summary} />}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.4fr,1fr]">
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
        <div className="panel-tight">
          <div className="mb-2">
            <div className="section-title">Ranked pairs</div>
            <div className="text-[11px] text-ink-400">click a row to drill down</div>
          </div>
          {q.isPending && <Skeleton lines={8} />}
          {q.data && (
            <SynergyTable
              data={q.data}
              dim={dim}
              colorMap={colorMap}
              onPickPair={(a, b) => setPair({ a, b })}
              active={pair}
            />
          )}
        </div>
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

function SynergyTable({
  data, dim, colorMap, onPickPair, active,
}: {
  data: TopologySynergyResponse;
  dim: "h1" | "h2";
  colorMap: Record<string, string>;
  onPickPair: (a: string, b: string) => void;
  active: { a: string; b: string } | null;
}) {
  const synKey = dim === "h1" ? "synergy_h1" : "synergy_h2";
  const sumKey = dim === "h1" ? "sum_h1_union" : "sum_h2_union";
  const rows = [...data.entries].sort((a, b) => b[synKey] - a[synKey]);
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r[synKey])), 1e-6);

  return (
    <>
      <div className="grid grid-cols-[1fr,2.4fr,3rem,3rem] gap-2 px-1.5 pb-1 text-[10px] uppercase tracking-wider text-ink-500">
        <div>pair</div>
        <div>synergy</div>
        <div className="text-right">∪ sum</div>
        <div className="text-right">{dim.toUpperCase()}</div>
      </div>
      <div className="max-h-[420px] space-y-0.5 overflow-y-auto pr-1">
        {rows.map((r, i) => {
          const ca = colorMap[r.a] ?? "#88c0d0";
          const cb = colorMap[r.b] ?? "#88c0d0";
          const isActive =
            active != null &&
            ((active.a === r.a && active.b === r.b) ||
              (active.a === r.b && active.b === r.a));
          const val = r[synKey];
          const norm = val / maxAbs;
          const positive = norm >= 0;
          return (
            <button
              key={i}
              onClick={() => onPickPair(r.a, r.b)}
              className="grid w-full grid-cols-[1fr,2.4fr,3rem,3rem] items-center gap-2 rounded-md px-1.5 py-1 text-left text-[11px] transition hover:bg-ink-800/40"
              style={{
                background: isActive ? "rgba(95,207,196,0.10)" : undefined,
                boxShadow: isActive ? "inset 0 0 0 1px rgba(95,207,196,0.45)" : "none",
              }}
            >
              <div className="flex items-center gap-1 truncate text-ink-100">
                <span style={{ color: ca }} className="truncate">{r.a}</span>
                <span className="text-ink-500">⋈</span>
                <span style={{ color: cb }} className="truncate">{r.b}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-ink-800">
                  <div className="absolute left-1/2 top-0 h-full w-px bg-ink-600/60" />
                  <div
                    className="absolute top-0 h-full rounded"
                    style={{
                      background: positive ? "#5fcfc4" : "#d08770",
                      opacity: 0.85,
                      left: positive ? "50%" : `${50 - Math.min(50, Math.abs(norm) * 50)}%`,
                      width: `${Math.min(50, Math.abs(norm) * 50)}%`,
                    }}
                  />
                </div>
                <span
                  className="w-12 shrink-0 text-right font-mono text-[10px] tabular-nums"
                  style={{ color: positive ? "#5fcfc4" : "#d08770" }}
                >
                  {val >= 0 ? "+" : ""}{val.toFixed(3)}
                </span>
              </div>
              <div className="text-right font-mono text-[10px] text-ink-400">
                {r[sumKey].toFixed(3)}
              </div>
              <div className="text-right font-mono text-[10px] text-ink-500">
                {dim === "h1" ? "H1" : "H2"}
              </div>
            </button>
          );
        })}
      </div>
      <div className="mt-2 border-t border-ink-700/40 pt-2 text-[10px] leading-relaxed text-ink-400">
        Positive synergy = some loops/voids require both archetypes to exist. Negative =
        the union has *less* topology than the pieces alone (their clouds compete).
      </div>
    </>
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
      <div style={{ cursor: "pointer" }}>
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
            hovertemplate: "%{y} ⋈ %{x}<br>synergy = %{z:.3f}<br><i>click to drill into bridge cycles</i><extra></extra>",
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
      </div>
      <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-ink-400">
        <span className="inline-flex h-4 items-center rounded border border-accent-500/40 bg-accent-600/10 px-1.5 font-medium text-accent-300">
          tip
        </span>
        {active ? (
          <>showing <span className="text-ink-100">{active.a} ⋈ {active.b}</span> below — click another cell to switch.</>
        ) : (
          <>click any non-diagonal cell to see the actual <em>bridge cycles</em> that link the two archetypes.</>
        )}
      </div>
    </>
  );
}

// Per-cycle palette — same hues as the Cycles tab so the visual
// language is consistent across views.
const CYCLE_PALETTE = [
  "#3bbdb0", "#e67e22", "#c2a6fe", "#bf616a", "#5fa8d3",
  "#e6c068", "#2ecc71", "#d08770", "#9b59b6", "#88c0d0",
];
const cycleColor = (i: number) => CYCLE_PALETTE[i % CYCLE_PALETTE.length];

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
  const [mode, setMode] = React.useState<"single" | "all">("single");
  const [showAllLabels, setShowAllLabels] = React.useState<boolean>(false);
  const [showH1, setShowH1] = React.useState<boolean>(true);
  const [showH2, setShowH2] = React.useState<boolean>(true);

  // Sort the two symbols so cache keys collapse for either order
  const [pa, pb] = a < b ? [a, b] : [b, a];

  const ctx = useTopologyContext();
  const q = useQuery({
    enabled: !!sid,
    queryKey: ["topo-pair", sid, pa, pb, ...ctx.keyTail],
    queryFn: () =>
      api.post<PairCyclesResponse>("/topology/pair-cycles", {
        space_id: sid, a: pa, b: pb, top_h1: 8, top_h2: 4, ...ctx.payload,
      }),
  });
  React.useEffect(() => { setActiveIdx(0); }, [pa, pb]);

  const ca = colorMap[pa] ?? "#88c0d0";
  const cb = colorMap[pb] ?? "#d08770";

  const allCycles = q.data?.cycles ?? [];
  const visibleIdxToOrig: number[] = [];
  allCycles.forEach((c, i) => {
    if (c.dim === 1 && showH1) visibleIdxToOrig.push(i);
    else if (c.dim === 2 && showH2) visibleIdxToOrig.push(i);
  });
  const clampedVisible = Math.min(activeIdx, Math.max(0, visibleIdxToOrig.length - 1));
  const activeOrigIdx = visibleIdxToOrig[clampedVisible] ?? -1;

  return (
    <div className="panel-tight">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="section-title flex items-center gap-2">
            Bridges · <span style={{ color: ca }}>{pa}</span>
            <span className="text-ink-500">⋈</span>
            <span style={{ color: cb }}>{pb}</span>
          </div>
          <div className="text-[11px] text-ink-400">
            {mode === "all" && visibleIdxToOrig.length
              ? `all ${visibleIdxToOrig.length} bridge cycles overlaid`
              : "mixed cycles that require both archetypes to close"}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <DimFilter
            showH1={showH1} showH2={showH2}
            onChange={(h1, h2) => { setShowH1(h1); setShowH2(h2); setActiveIdx(0); }}
            countH1={allCycles.filter((c) => c.dim === 1).length}
            countH2={allCycles.filter((c) => c.dim === 2).length}
          />
          {mode === "all" && visibleIdxToOrig.length > 1 && (
            <button
              className={`pill !text-[10px] ${showAllLabels ? "border-accent-500/60 bg-accent-600/20 text-accent-200" : "hover:border-ink-500"}`}
              onClick={() => setShowAllLabels((v) => !v)}
              title="Show vertex labels on every visible cycle"
            >
              {showAllLabels ? "labels on" : "labels off"}
            </button>
          )}
          <div className="inline-flex shrink-0 overflow-hidden rounded-md border border-ink-700">
            <button
              className={`px-2.5 py-1 text-[11px] transition ${
                mode === "single" ? "bg-accent-600/30 text-ink-50" : "text-ink-300 hover:bg-ink-800"
              }`}
              onClick={() => setMode("single")}
            >
              single
            </button>
            <button
              className={`px-2.5 py-1 text-[11px] transition ${
                mode === "all" ? "bg-accent-600/30 text-ink-50" : "text-ink-300 hover:bg-ink-800"
              }`}
              onClick={() => setMode("all")}
              title={`Overlay all ${visibleIdxToOrig.length} cycles`}
            >
              all
            </button>
          </div>
          <button onClick={onClose} className="text-[11px] text-ink-400 hover:text-ink-200">
            close ✕
          </button>
        </div>
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
            allCycles={allCycles}
            visibleIdxToOrig={visibleIdxToOrig}
            clampedVisible={clampedVisible}
            onActivate={setActiveIdx}
            ca={ca}
            cb={cb}
            a={pa}
            b={pb}
            showSwatch={mode === "all"}
          />
          <PairCyclePlot
            data={q.data}
            visibleOrigIdx={visibleIdxToOrig}
            activeOrigIdx={activeOrigIdx}
            ca={ca}
            cb={cb}
            mode={mode}
            showAllLabels={showAllLabels}
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
  allCycles, visibleIdxToOrig, clampedVisible, onActivate, ca, cb, a, b, showSwatch = false,
}: {
  allCycles: PairCycle[];
  visibleIdxToOrig: number[];
  clampedVisible: number;
  onActivate: (i: number) => void;
  ca: string;
  cb: string;
  a: string;
  b: string;
  showSwatch?: boolean;
}) {
  if (!allCycles.length) {
    return (
      <div className="rounded-md border border-ink-700/40 bg-ink-900/30 p-3 text-[11px] text-ink-400">
        No bridge cycles between these two clouds — they don't share any
        topological structure that needs both archetypes to close.
      </div>
    );
  }
  if (!visibleIdxToOrig.length) {
    return (
      <div className="rounded-md border border-ink-700/40 bg-ink-900/30 p-3 text-[11px] text-ink-400">
        No cycles match the current H1/H2 filter.
      </div>
    );
  }
  return (
    <div className="space-y-1.5">
      {visibleIdxToOrig.map((i, visI) => {
        const cyc = allCycles[i];
        const active = visI === clampedVisible;
        // Pair-cycles is bridges-only by default — the "bridge" pill is
        // redundant.  Only show a pill if this is a pure cycle (which can
        // appear when include_pure=true is opted in).
        const isPure = cyc.mix === "pure_a" || cyc.mix === "pure_b";
        const pureLabel = cyc.mix === "pure_a" ? `pure ${a}` : `pure ${b}`;
        const pureColour = cyc.mix === "pure_a" ? ca : cb;
        const activeColour = isPure ? pureColour : MIX_STYLE.mixed.colour;
        return (
          <button
            key={i}
            onClick={() => onActivate(visI)}
            className="block w-full rounded-md border p-2 text-left text-[11px] transition"
            style={{
              borderColor: active ? activeColour : "rgba(255,255,255,0.08)",
              background: active ? `${activeColour}10` : "rgba(255,255,255,0.02)",
              boxShadow: active ? `inset 0 0 0 1px ${activeColour}55` : "none",
            }}
          >
            <div className="mb-1 flex items-center gap-1.5 text-[10px]">
              {showSwatch && (
                <span
                  className="inline-block h-2 w-2 shrink-0 rounded-sm"
                  style={{ background: cycleColor(i) }}
                  title="colour of this cycle on the plot →"
                />
              )}
              <span
                className="rounded px-1.5 py-0.5 font-mono"
                style={{ background: cyc.dim === 1 ? "#3bbdb022" : "#d0877022",
                         border: `1px solid ${cyc.dim === 1 ? "#3bbdb088" : "#d0877088"}`,
                         color:  cyc.dim === 1 ? "#5fcfc4" : "#d08770" }}
              >
                H{cyc.dim}
              </span>
              {isPure && (
                <span
                  className="rounded px-1.5 py-0.5"
                  style={{ background: `${pureColour}1a`, border: `1px solid ${pureColour}66`, color: pureColour }}
                >
                  {pureLabel}
                </span>
              )}
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
  data, visibleOrigIdx, activeOrigIdx, ca, cb, mode, showAllLabels,
}: {
  data: PairCyclesResponse;
  visibleOrigIdx: number[];
  activeOrigIdx: number;
  ca: string;
  cb: string;
  mode: "single" | "all";
  showAllLabels: boolean;
}) {
  const a = data.a; const b = data.b;
  const visibleCycles = visibleOrigIdx.map((i) => ({ origIdx: i, cyc: data.cycles[i] }));
  const active = activeOrigIdx >= 0 ? data.cycles[activeOrigIdx] : null;

  // Which descriptor indices to highlight as "in a cycle"
  let highlightedIdx: Set<number>;
  if (mode === "all") {
    highlightedIdx = new Set<number>();
    for (const { cyc } of visibleCycles)
      for (const v of cyc.vertices) highlightedIdx.add(v.index);
  } else {
    highlightedIdx = new Set((active?.vertices ?? []).map((v) => v.index));
  }

  const traces: any[] = [];
  // background: all descriptors, coloured by home symbol, dimmed
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
        size: grp.map((d) => (highlightedIdx.has(d.index) ? 13 : 8)),
        color: grp.map((d) => (highlightedIdx.has(d.index) ? col : `${col}55`)),
        line: {
          color: grp.map((d) => (highlightedIdx.has(d.index) ? "white" : "rgba(0,0,0,0.3)")),
          width: grp.map((d) => (highlightedIdx.has(d.index) ? 1.5 : 0.4)),
        },
      },
      hovertemplate: `<b>${name}</b> · %{text}<extra></extra>`,
    });
  }

  if (mode === "all") {
    // Overlay every visible cycle in its palette colour; persistence →
    // opacity; selected cycle drawn LAST with full opacity + labels.
    // Other cycles get labels when showAllLabels is on.
    const maxPers = Math.max(...visibleCycles.map(({ cyc }) => cyc.persistence), 1e-6);
    visibleCycles.forEach(({ origIdx, cyc }) => {
      if (origIdx === activeOrigIdx) return;
      const persRel = cyc.persistence / maxPers;
      drawPairCycleTraces(traces, cyc, cycleColor(origIdx),
                          0.28 + 0.32 * persRel, /* withLabels */ showAllLabels,
                          a, ca, cb);
    });
    if (active && active.vertices.length >= 2) {
      drawPairCycleTraces(traces, active, cycleColor(activeOrigIdx), 1.0,
                          /* withLabels */ true, a, ca, cb);
    }
  } else if (active && active.vertices.length >= 2) {
    // Single mode: bridge cycles get a teal edge colour; pure cycles
    // (if include_pure was ever requested) get their pure-side colour.
    const edgeColour =
      active.mix === "pure_a" ? ca :
      active.mix === "pure_b" ? cb : "#5fcfc4";
    drawPairCycleTraces(traces, active, edgeColour, 1.0, /* withLabels */ true, a, ca, cb);
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

/** Append plotly traces for one pair-cycle onto the given list.  Labels are
 *  *coloured by home symbol* (ca / cb) so it's unambiguous which words in
 *  the cycle come from which archetype — bridges become readable. */
function drawPairCycleTraces(
  traces: any[],
  cyc: PairCycle,
  color: string,
  opacity: number,
  withLabels: boolean,
  a: string,
  ca: string,
  cb: string,
) {
  if (cyc.vertices.length < 2) return;
  if (cyc.dim === 1) {
    const xs = cyc.vertices.map((v) => v.x).concat([cyc.vertices[0].x]);
    const ys = cyc.vertices.map((v) => v.y).concat([cyc.vertices[0].y]);
    traces.push({
      x: xs, y: ys,
      type: "scatter", mode: "lines",
      line: { color, width: 2.5, shape: "spline" },
      opacity,
      hoverinfo: "skip", showlegend: false,
    });
  } else {
    const pts = cyc.vertices.map((v) => [v.x, v.y] as [number, number]);
    const hull = convexHull(pts);
    if (hull.length >= 3) {
      hull.push(hull[0]);
      traces.push({
        x: hull.map((h) => h[0]),
        y: hull.map((h) => h[1]),
        type: "scatter", mode: "lines",
        fill: "toself",
        fillcolor: color + Math.round(opacity * 40).toString(16).padStart(2, "0"),
        line: { color, width: 1.5 },
        opacity,
        hoverinfo: "skip", showlegend: false,
      });
    }
  }
  if (withLabels) {
    // Split labels into two traces (one per home symbol) so each can carry
    // its own colour.  Plotly accepts a per-point textfont colour array
    // but separate traces are cleaner.
    for (const home of [a, /* the other */ null] as const) {
      const idxs = cyc.vertices
        .map((v, i) => ({ v, i }))
        .filter(({ v }) => home === null ? v.home_symbol !== a : v.home_symbol === a);
      if (idxs.length === 0) continue;
      const colour = home === a ? ca : cb;
      traces.push({
        x: idxs.map(({ v }) => v.x),
        y: idxs.map(({ v }) => v.y),
        text: idxs.map(({ v, i }) =>
          cyc.dim === 1 ? `${i + 1}· ${v.word}` : v.word,
        ),
        type: "scatter", mode: "text",
        textposition: "top center",
        textfont: { color: colour, size: 11, family: "Inter, system-ui" },
        hoverinfo: "skip", showlegend: false,
      });
    }
  }
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
