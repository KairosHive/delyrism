"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Plot } from "../plots/Plot";
import { api, TopologyCyclesResponse, PersistentCycle, CycleVertex } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { Skeleton } from "../ui/Skeleton";

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
export function TopologyCycles() {
  const sid = useSidebar((s) => s.spaceId);
  const symbols = useSidebar((s) => s.symbols);
  const colorMap = useSidebar((s) => s.colorMap);
  const [symbol, setSymbol] = React.useState<string>("");
  const [activeIdx, setActiveIdx] = React.useState<number>(0);

  React.useEffect(() => {
    if (!symbol && symbols.length) setSymbol(symbols[0]);
  }, [symbols, symbol]);
  // Reset selection when switching symbol
  React.useEffect(() => { setActiveIdx(0); }, [symbol]);

  const q = useQuery({
    enabled: !!sid && !!symbol,
    queryKey: ["topo-cycles", sid, symbol],
    queryFn: () =>
      api.post<TopologyCyclesResponse>("/topology/cycles", {
        space_id: sid, symbol, top_h1: 8, top_h2: 4,
      }),
  });

  const accent = colorMap[symbol] ?? "#88c0d0";

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr,1.6fr]">
      {/* ── left: list of cycles ── */}
      <div className="panel-tight">
        <div className="mb-2 flex items-center justify-between gap-2">
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

        {q.isPending && <Skeleton lines={6} />}
        {q.error && <div className="text-sm text-danger">{(q.error as Error).message}</div>}
        {q.data && (
          q.data.cycles.length === 0 ? (
            <div className="rounded-md border border-ink-700/40 bg-ink-900/30 p-3 text-[11px] text-ink-400">
              No persistent cycles found in this symbol — the descriptor cloud is essentially{" "}
              {q.data.descriptors.length < 6 ? "too small" : "topologically trivial (no loops above noise)"}.
            </div>
          ) : (
            <div className="space-y-1.5">
              {q.data.cycles.map((cyc, i) => (
                <CycleRow
                  key={i}
                  cycle={cyc}
                  idx={i}
                  active={i === activeIdx}
                  accent={accent}
                  onActivate={() => setActiveIdx(i)}
                />
              ))}
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

      {/* ── right: PCA scatter with active cycle traced ── */}
      <div className="panel-tight">
        <div className="mb-2 flex items-center justify-between">
          <div>
            <div className="section-title">Cloud · active cycle</div>
            <div className="text-[11px] text-ink-400">
              PCA-2D of <span style={{ color: accent }}>{symbol}</span>'s descriptors
            </div>
          </div>
          {q.data?.cycles[activeIdx] && (
            <CycleBadge cycle={q.data.cycles[activeIdx]} accent={accent} />
          )}
        </div>

        {q.isPending && <Skeleton height={520} />}
        {q.data && <CyclePlot data={q.data} active={q.data.cycles[activeIdx] ?? null} accent={accent} />}
      </div>
    </div>
  );
}

function CycleRow({
  cycle, idx, active, accent, onActivate,
}: {
  cycle: PersistentCycle;
  idx: number;
  active: boolean;
  accent: string;
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
  data, active, accent,
}: {
  data: TopologyCyclesResponse;
  active: PersistentCycle | null;
  accent: string;
}) {
  // Background scatter — all descriptors of this symbol
  const activeSet = new Set((active?.vertices ?? []).map((v) => v.index));

  const traces: any[] = [];
  // dimmed background
  traces.push({
    x: data.descriptors.map((d) => d.x),
    y: data.descriptors.map((d) => d.y),
    text: data.descriptors.map((d) => d.word),
    type: "scatter",
    mode: "markers",
    name: "descriptors",
    showlegend: false,
    marker: {
      size: data.descriptors.map((d) => (activeSet.has(d.index) ? 12 : 8)),
      color: data.descriptors.map((d) => (activeSet.has(d.index) ? accent : "rgba(255,255,255,0.18)")),
      line: {
        color: data.descriptors.map((d) => (activeSet.has(d.index) ? "white" : "rgba(0,0,0,0.3)")),
        width: data.descriptors.map((d) => (activeSet.has(d.index) ? 1.5 : 0.4)),
      },
    },
    hovertemplate: "%{text}<extra></extra>",
  });

  if (active && active.vertices.length >= 2) {
    if (active.dim === 1) {
      // close the loop visually
      const xs = active.vertices.map((v) => v.x).concat([active.vertices[0].x]);
      const ys = active.vertices.map((v) => v.y).concat([active.vertices[0].y]);
      traces.push({
        x: xs, y: ys,
        type: "scatter", mode: "lines",
        line: { color: accent, width: 2.5, shape: "spline" },
        hoverinfo: "skip", showlegend: false,
      });
    } else {
      // H2 — draw the convex hull of the vertices as a translucent polygon
      const pts = active.vertices.map((v) => [v.x, v.y] as [number, number]);
      const hull = convexHull(pts);
      if (hull.length >= 3) {
        hull.push(hull[0]);
        traces.push({
          x: hull.map((h) => h[0]),
          y: hull.map((h) => h[1]),
          type: "scatter", mode: "lines",
          fill: "toself",
          fillcolor: `${accent}25`,
          line: { color: accent, width: 1.5 },
          hoverinfo: "skip", showlegend: false,
        });
      }
    }
    // labels on the active cycle vertices
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
