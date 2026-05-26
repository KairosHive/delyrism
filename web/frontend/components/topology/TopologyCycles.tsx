"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Plot } from "../plots/Plot";
import { api, TopologyCyclesResponse, PersistentCycle, CycleVertex } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { Skeleton } from "../ui/Skeleton";
import { useTopologyContext } from "./useTopologyContext";
import { ContextPill } from "./TopologyOverview";

/**
 * Interactive persistent-cycle browser — the killer view of the
 * topology tab.
 *
 * Layout:
 *   left column   — picker + ranked list of cycles for the chosen symbol.
 *                   each row shows dim/persistence/word-trail.  click to
 *                   activate.
 *   right column  — PCA-2D scatter of all this symbol's descriptors.  The
 *                   active cycle's vertices light up and are connected by
 *                   coloured edges, forming a visible loop in the cloud.
 *                   words labelled on each cycle vertex.
 *
 * Lets the user spin through "what semantic loops exist inside FIRE",
 * with each loop drawn as an actual closed path through the 2D layout
 * of the descriptor cloud.
 */
// Per-cycle palette — distinct hues so multiple cycles overlay cleanly.
// Cycled through if there are more cycles than colours.
const CYCLE_PALETTE = [
  "#3bbdb0", "#e67e22", "#c2a6fe", "#bf616a", "#5fa8d3",
  "#e6c068", "#2ecc71", "#d08770", "#9b59b6", "#88c0d0",
];
const cycleColor = (i: number) => CYCLE_PALETTE[i % CYCLE_PALETTE.length];

export function TopologyCycles() {
  const sid = useSidebar((s) => s.spaceId);
  const symbols = useSidebar((s) => s.symbols);
  const colorMap = useSidebar((s) => s.colorMap);
  const [symbol, setSymbol] = React.useState<string>("");
  const [activeIdx, setActiveIdx] = React.useState<number>(0);
  const [mode, setMode] = React.useState<"single" | "all">("single");
  const [showAllLabels, setShowAllLabels] = React.useState<boolean>(false);
  const [showH1, setShowH1] = React.useState<boolean>(true);
  const [showH2, setShowH2] = React.useState<boolean>(true);

  React.useEffect(() => {
    if (!symbol && symbols.length) setSymbol(symbols[0]);
  }, [symbols, symbol]);
  // Reset selection when switching symbol
  React.useEffect(() => { setActiveIdx(0); }, [symbol]);

  const ctx = useTopologyContext();
  const q = useQuery({
    enabled: !!sid && !!symbol,
    queryKey: ["topo-cycles", sid, symbol, ...ctx.keyTail],
    queryFn: () =>
      api.post<TopologyCyclesResponse>("/topology/cycles", {
        space_id: sid, symbol, top_h1: 8, top_h2: 4, ...ctx.payload,
      }),
  });

  const accent = colorMap[symbol] ?? "#88c0d0";

  // Build the filtered cycle list (respecting H1/H2 chips) but keep the
  // ORIGINAL palette index so each cycle's colour is stable as the user
  // toggles filters.  visibleIdxToOrig = absolute index of the cycle the
  // user picks in the filtered list.
  const allCycles = q.data?.cycles ?? [];
  const visibleIdxToOrig: number[] = [];
  allCycles.forEach((c, i) => {
    if (c.dim === 1 && showH1) visibleIdxToOrig.push(i);
    else if (c.dim === 2 && showH2) visibleIdxToOrig.push(i);
  });
  // Clamp activeIdx to a valid visible cycle
  const clampedVisible = Math.min(activeIdx, Math.max(0, visibleIdxToOrig.length - 1));
  const activeOrigIdx = visibleIdxToOrig[clampedVisible] ?? -1;

  return (
    <div className="space-y-3">
      {ctx.active && <ContextPill summary={ctx.summary} />}
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr,1.6fr]">
      {/* ── left: list of cycles ── */}
      <div className="panel-tight">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="section-title">Persistent cycles</div>
            <div className="text-[11px] text-ink-400">click to trace on the map →</div>
          </div>
          <select
            className="select-base !w-auto !min-w-[140px]"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
          >
            {symbols.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>

        <DimFilter
          showH1={showH1} showH2={showH2}
          onChange={(h1, h2) => { setShowH1(h1); setShowH2(h2); setActiveIdx(0); }}
          countH1={allCycles.filter((c) => c.dim === 1).length}
          countH2={allCycles.filter((c) => c.dim === 2).length}
        />

        {q.isPending && <Skeleton lines={6} />}
        {q.error && <div className="text-sm text-danger">{(q.error as Error).message}</div>}
        {q.data && (
          visibleIdxToOrig.length === 0 ? (
            <div className="mt-2 rounded-md border border-ink-700/40 bg-ink-900/30 p-3 text-[11px] text-ink-400">
              {allCycles.length === 0
                ? <>No persistent cycles found in this symbol — the descriptor cloud is essentially{" "}
                    {q.data.descriptors.length < 6 ? "too small" : "topologically trivial (no loops above noise)"}.</>
                : <>No cycles match the current H1/H2 filter.</>}
            </div>
          ) : (
            <div className="mt-2 space-y-1.5">
              {visibleIdxToOrig.map((origIdx, visI) => {
                const cyc = allCycles[origIdx];
                return (
                  <CycleRow
                    key={origIdx}
                    cycle={cyc}
                    idx={origIdx}
                    active={visI === clampedVisible}
                    accent={accent}
                    swatch={cycleColor(origIdx)}
                    showSwatch={mode === "all"}
                    onActivate={() => setActiveIdx(visI)}
                  />
                );
              })}
            </div>
          )
        )}

        <div className="mt-3 border-t border-ink-700/40 pt-2 text-[10px] leading-relaxed text-ink-400">
          <span className="text-ink-200">H1</span> = 1-dim loop. The vertex sequence traces a
          closed path through semantic neighbours, then back to start.
          <br />
          <span className="text-ink-200">H2</span> = 2-dim void. The vertices bound a cavity
          (sphere-like hole) in the cloud — no canonical ordering.
        </div>
      </div>

      {/* ── right: PCA scatter with active cycle traced (or all overlaid) ── */}
      <div className="panel-tight">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="section-title">
              Cloud · {mode === "all" ? `all ${visibleIdxToOrig.length} cycles` : "active cycle"}
            </div>
            <div className="text-[11px] text-ink-400">
              PCA-2D of <span style={{ color: accent }}>{symbol}</span>'s descriptors
            </div>
          </div>
          <div className="flex items-center gap-2">
            {mode === "single" && allCycles[activeOrigIdx] && (
              <CycleBadge cycle={allCycles[activeOrigIdx]} accent={accent} />
            )}
            {mode === "all" && visibleIdxToOrig.length > 1 && (
              <button
                className={`pill !text-[10px] ${showAllLabels ? "border-accent-500/60 bg-accent-600/20 text-accent-200" : "hover:border-ink-500"}`}
                onClick={() => setShowAllLabels((v) => !v)}
                title="Show vertex labels on every visible cycle"
              >
                {showAllLabels ? "labels on" : "labels off"}
              </button>
            )}
            <ModeToggle mode={mode} onChange={setMode} cycleCount={visibleIdxToOrig.length} />
          </div>
        </div>

        {q.isPending && <Skeleton height={520} />}
        {q.data && (
          <CyclePlot
            data={q.data}
            visibleOrigIdx={visibleIdxToOrig}
            activeOrigIdx={activeOrigIdx}
            accent={accent}
            mode={mode}
            showAllLabels={showAllLabels}
          />
        )}
      </div>
    </div>
    </div>
  );
}

export function DimFilter({
  showH1, showH2, onChange, countH1, countH2,
}: {
  showH1: boolean;
  showH2: boolean;
  onChange: (h1: boolean, h2: boolean) => void;
  countH1: number;
  countH2: number;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <button
        onClick={() => onChange(!showH1, showH2)}
        className="rounded-md border px-2 py-0.5 text-[10px] transition"
        style={{
          background: showH1 ? "#3bbdb022" : "transparent",
          borderColor: showH1 ? "#3bbdb088" : "rgba(255,255,255,0.08)",
          color: showH1 ? "#5fcfc4" : "#6e7e95",
          opacity: showH1 ? 1 : 0.6,
        }}
        title="H1 = 1-dimensional loops"
      >
        H1 · {countH1}
      </button>
      <button
        onClick={() => onChange(showH1, !showH2)}
        className="rounded-md border px-2 py-0.5 text-[10px] transition"
        style={{
          background: showH2 ? "#d0877022" : "transparent",
          borderColor: showH2 ? "#d0877088" : "rgba(255,255,255,0.08)",
          color: showH2 ? "#d08770" : "#6e7e95",
          opacity: showH2 ? 1 : 0.6,
        }}
        title="H2 = 2-dimensional voids (sphere-like cavities)"
      >
        H2 · {countH2}
      </button>
    </div>
  );
}

function ModeToggle({
  mode, onChange, cycleCount,
}: {
  mode: "single" | "all";
  onChange: (m: "single" | "all") => void;
  cycleCount: number;
}) {
  return (
    <div className="inline-flex shrink-0 overflow-hidden rounded-md border border-ink-700">
      <button
        className={`px-2.5 py-1 text-[11px] transition ${
          mode === "single" ? "bg-accent-600/30 text-ink-50" : "text-ink-300 hover:bg-ink-800"
        }`}
        onClick={() => onChange("single")}
      >
        single
      </button>
      <button
        className={`px-2.5 py-1 text-[11px] transition ${
          mode === "all" ? "bg-accent-600/30 text-ink-50" : "text-ink-300 hover:bg-ink-800"
        }`}
        onClick={() => onChange("all")}
        title={`Overlay all ${cycleCount} cycles at once`}
      >
        all
      </button>
    </div>
  );
}

function CycleRow({
  cycle, idx, active, accent, swatch, showSwatch, onActivate,
}: {
  cycle: PersistentCycle;
  idx: number;
  active: boolean;
  accent: string;
  swatch: string;
  showSwatch: boolean;
  onActivate: () => void;
}) {
  const isH1 = cycle.dim === 1;
  const dimChip = isH1
    ? { bg: "#3bbdb022", border: "#3bbdb088", color: "#5fcfc4", label: "H1" }
    : { bg: "#d0877022", border: "#d0877088", color: "#d08770", label: "H2" };
  const trail = cycle.vertices.map((v) => v.word).join(isH1 ? " → " : " · ");
  return (
    <button
      onClick={onActivate}
      className="block w-full rounded-md border p-2 text-left text-[11px] transition"
      style={{
        borderColor: active ? accent : "rgba(255,255,255,0.08)",
        background: active ? `${accent}10` : "rgba(255,255,255,0.02)",
        boxShadow: active ? `inset 0 0 0 1px ${accent}55` : "none",
      }}
    >
      <div className="mb-1 flex items-center gap-1.5 text-[10px]">
        {showSwatch && (
          <span
            className="inline-block h-2 w-2 shrink-0 rounded-sm"
            style={{ background: swatch }}
            title="colour of this cycle on the cloud →"
          />
        )}
        <span
          className="rounded px-1.5 py-0.5 font-mono"
          style={{ background: dimChip.bg, border: `1px solid ${dimChip.border}`, color: dimChip.color }}
        >
          {dimChip.label}
        </span>
        <span className="text-ink-400">persistence</span>
        <span className="font-mono text-ink-200">{cycle.persistence.toFixed(3)}</span>
        <span className="ml-auto text-ink-500">{cycle.vertices.length} verts</span>
      </div>
      <div className="text-ink-100">
        {isH1 ? "↻ " : "◆ "}{trail}
        {isH1 && cycle.vertices.length > 0 && (
          <span className="text-ink-500"> → {cycle.vertices[0].word}</span>
        )}
      </div>
    </button>
  );
}

function CycleBadge({ cycle, accent }: { cycle: PersistentCycle; accent: string }) {
  return (
    <div
      className="rounded-md border px-2 py-1 text-[10px]"
      style={{ borderColor: `${accent}55`, background: `${accent}10`, color: accent }}
    >
      H{cycle.dim} · pers {cycle.persistence.toFixed(3)} · {cycle.vertices.length} vertices
    </div>
  );
}

function CyclePlot({
  data, visibleOrigIdx, activeOrigIdx, accent, mode, showAllLabels,
}: {
  data: TopologyCyclesResponse;
  visibleOrigIdx: number[];
  activeOrigIdx: number;
  accent: string;
  mode: "single" | "all";
  showAllLabels: boolean;
}) {
  const traces: any[] = [];
  const visibleCycles = visibleOrigIdx.map((i) => ({ origIdx: i, cyc: data.cycles[i] }));
  const active = activeOrigIdx >= 0 ? data.cycles[activeOrigIdx] : null;

  // ── compute which descriptors are highlighted in either mode ──
  let highlightedIdx: Set<number>;
  if (mode === "all") {
    highlightedIdx = new Set<number>();
    for (const { cyc } of visibleCycles)
      for (const v of cyc.vertices) highlightedIdx.add(v.index);
  } else {
    highlightedIdx = new Set((active?.vertices ?? []).map((v) => v.index));
  }

  // dimmed background — descriptors with no cycle membership
  traces.push({
    x: data.descriptors.map((d) => d.x),
    y: data.descriptors.map((d) => d.y),
    text: data.descriptors.map((d) => d.word),
    type: "scatter",
    mode: "markers",
    name: "descriptors",
    showlegend: false,
    marker: {
      size: data.descriptors.map((d) => (highlightedIdx.has(d.index) ? 11 : 7)),
      color: data.descriptors.map((d) =>
        highlightedIdx.has(d.index)
          ? (mode === "all" ? "rgba(255,255,255,0.65)" : accent)
          : "rgba(255,255,255,0.16)"
      ),
      line: {
        color: data.descriptors.map((d) =>
          highlightedIdx.has(d.index) ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.3)"
        ),
        width: data.descriptors.map((d) => (highlightedIdx.has(d.index) ? 1 : 0.4)),
      },
    },
    hovertemplate: "%{text}<extra></extra>",
  });

  if (mode === "all") {
    // Draw every VISIBLE cycle in its palette colour.  Selected cycle
    // drawn last (on top) with full opacity + labels.  Other cycles dim
    // by persistence; labels optional via showAllLabels.
    const maxPers = Math.max(...visibleCycles.map(({ cyc }) => cyc.persistence), 1e-6);
    visibleCycles.forEach(({ origIdx, cyc }) => {
      if (origIdx === activeOrigIdx) return; // selected drawn last
      const col = cycleColor(origIdx);
      const persRel = cyc.persistence / maxPers;
      const opacity = 0.28 + 0.32 * persRel;
      drawCycleTraces(traces, cyc, col, opacity, /* withLabels */ showAllLabels);
    });
    if (active) {
      drawCycleTraces(traces, active, cycleColor(activeOrigIdx), 1.0, /* withLabels */ true);
    }
  } else if (active && active.vertices.length >= 2) {
    // Single mode: draw the one active cycle + its vertex labels
    drawCycleTraces(traces, active, accent, 1.0, /* withLabels */ true);
  }

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        height: 540,
        margin: { l: 20, r: 12, t: 8, b: 20 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#cad4e0", family: "Inter, system-ui" },
        xaxis: { showgrid: true, gridcolor: "rgba(255,255,255,0.04)", zeroline: false, showticklabels: false },
        yaxis: { showgrid: true, gridcolor: "rgba(255,255,255,0.04)", zeroline: false, showticklabels: false },
        hoverlabel: { bgcolor: "#10131c", bordercolor: accent, font: { color: "#e8edf3" } },
      }}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
      config={{ displaylogo: false, responsive: true, scrollZoom: true }}
    />
  );
}

/** Append the plotly traces for one cycle (loop or void) onto an existing
 *  traces array.  H1 → spline edges around the loop.  H2 → translucent
 *  filled convex-hull polygon.  Labels optional (suppressed in "all" mode
 *  to avoid clutter). */
function drawCycleTraces(
  traces: any[],
  cyc: PersistentCycle,
  color: string,
  opacity: number,
  withLabels: boolean,
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
    traces.push({
      x: cyc.vertices.map((v) => v.x),
      y: cyc.vertices.map((v) => v.y),
      text: cyc.vertices.map((v, i) =>
        cyc.dim === 1 ? `${i + 1}· ${v.word}` : v.word,
      ),
      type: "scatter", mode: "text",
      textposition: "top center",
      textfont: { color: "#dbe2ee", size: 11, family: "Inter, system-ui" },
      hoverinfo: "skip", showlegend: false,
    });
  }
}

// Andrew's monotone chain convex hull
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
