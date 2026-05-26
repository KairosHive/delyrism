"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Plot } from "../plots/Plot";
import {
  api,
  AllDiagramsResponse,
  AllDiagramsEntry,
  PersistencePoint,
} from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { Skeleton } from "../ui/Skeleton";
import { useTopologyContext } from "./useTopologyContext";
import { ContextPill } from "./TopologyOverview";

/**
 * Persistence diagrams — small-multiples grid + expandable detail.
 *
 * Default view: every archetype's persistence diagram in a small card,
 * all sharing the same axes so loop/void counts pop out comparatively.
 * Each card has a one-line auto-narration ("3 loops, 1 void · richly
 * structured") so the user knows what to look for without reading the
 * dots themselves.
 *
 * Click any card → expand to a full-size view with annotated top
 * features.  A persistent reading-guide at the bottom explains what
 * dots and distance-to-diagonal mean.
 */

// Canonical palette — H0 blue / H1 orange / H2 green, matches the
// notebook colours the user is comparing against.
const HD_COLORS: Record<number, string> = {
  0: "#5fa8d3",
  1: "#e67e22",
  2: "#2ecc71",
};
const HD_LABEL: Record<number, string> = {
  0: "H0 · components",
  1: "H1 · loops",
  2: "H2 · voids",
};
const PERS_THR = 0.02;

export function TopologyDiagrams() {
  const sid = useSidebar((s) => s.spaceId);
  const colorMap = useSidebar((s) => s.colorMap);
  const [expanded, setExpanded] = React.useState<string | null>(null);

  const ctx = useTopologyContext();
  const q = useQuery({
    enabled: !!sid,
    queryKey: ["topo-diagrams-all", sid, ...ctx.keyTail],
    queryFn: () =>
      api.post<AllDiagramsResponse>("/topology/diagrams-all", { space_id: sid, ...ctx.payload }),
  });

  if (q.isPending) {
    return (
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} height={260} />)}
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
        Persistent-homology endpoints require <code className="text-accent-300">ripser</code>.
      </div>
    );
  }

  if (expanded) {
    const entry = data.entries.find((e) => e.symbol === expanded);
    if (!entry) return null;
    return (
      <ExpandedDiagram
        entry={entry}
        accent={colorMap[expanded] ?? "#88c0d0"}
        axisMax={data.max_finite_death}
        onBack={() => setExpanded(null)}
      />
    );
  }

  // Headline statistics across the whole preset
  const ranked = [...data.entries].sort((a, b) => b.h1_persistent - a.h1_persistent);
  const mostLoopy = ranked[0]?.h1_persistent > 0 ? ranked[0] : null;
  const mostVoid = [...data.entries].sort((a, b) => b.h2_persistent - a.h2_persistent)[0];

  return (
    <div className="space-y-4">
      {ctx.active && <ContextPill summary={ctx.summary} />}
      {/* ── overall summary banner ── */}
      <div className="panel-tight">
        <div className="mb-1 text-[10px] uppercase tracking-widest text-ink-400">
          Topological landscape
        </div>
        <div className="text-sm leading-relaxed text-ink-100">
          {mostLoopy ? (
            <>
              Most loopy archetype:{" "}
              <strong style={{ color: colorMap[mostLoopy.symbol] ?? "#cbd" }}>
                {mostLoopy.symbol}
              </strong>{" "}
              ({mostLoopy.h1_persistent} persistent H1 loop{mostLoopy.h1_persistent === 1 ? "" : "s"}).
            </>
          ) : (
            <span className="text-ink-400">No archetype has persistent H1 loops above noise. </span>
          )}
          {mostVoid && mostVoid.h2_persistent > 0 && (
            <>
              {" "}Most void-rich:{" "}
              <strong style={{ color: colorMap[mostVoid.symbol] ?? "#cbd" }}>
                {mostVoid.symbol}
              </strong>{" "}
              ({mostVoid.h2_persistent} H2 cavit{mostVoid.h2_persistent === 1 ? "y" : "ies"}).
            </>
          )}
        </div>
      </div>

      {/* ── small-multiples grid ── */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {data.entries.map((e) => (
          <MiniDiagram
            key={e.symbol}
            entry={e}
            accent={colorMap[e.symbol] ?? "#88c0d0"}
            axisMax={data.max_finite_death}
            onClick={() => setExpanded(e.symbol)}
          />
        ))}
      </div>

      {/* ── reading guide ── */}
      <ReadingGuide />
    </div>
  );
}

// ───────── narration ─────────

function describeEntry(e: AllDiagramsEntry): string {
  if (e.h1_persistent === 0 && e.h2_persistent === 0) {
    return "topologically flat — no persistent loops or voids";
  }
  const parts: string[] = [];
  if (e.h1_persistent > 0) {
    parts.push(`${e.h1_persistent} loop${e.h1_persistent === 1 ? "" : "s"}`);
  }
  if (e.h2_persistent > 0) {
    parts.push(`${e.h2_persistent} void${e.h2_persistent === 1 ? "" : "s"}`);
  }
  let qualifier = "";
  const score = e.h1_persistent + 2 * e.h2_persistent;
  if (score >= 5) qualifier = "richly structured";
  else if (score >= 3) qualifier = "moderately structured";
  else qualifier = "lightly structured";
  return `${parts.join(", ")} · ${qualifier}`;
}

// ───────── mini diagram ─────────

function MiniDiagram({
  entry, accent, axisMax, onClick,
}: {
  entry: AllDiagramsEntry;
  accent: string;
  axisMax: number;
  onClick: () => void;
}) {
  const lim = Math.max(0.05, axisMax * 1.15);
  const traces = buildTraces(entry.points, lim, { small: true });

  return (
    <button
      onClick={onClick}
      className="group block rounded-lg border border-ink-700/60 bg-ink-900/40 p-2.5 text-left transition hover:bg-ink-900/70 hover:shadow-lg"
      style={{
        boxShadow: "0 0 0 0 transparent",
      }}
    >
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <div className="truncate text-[13px] font-semibold" style={{ color: accent }}>
          {entry.symbol}
        </div>
        <div className="shrink-0 text-[10px] text-ink-400">
          {entry.h1_persistent > 0 && (
            <span className="mr-1.5">
              <span style={{ color: HD_COLORS[1] }}>●</span> {entry.h1_persistent}
            </span>
          )}
          {entry.h2_persistent > 0 && (
            <span>
              <span style={{ color: HD_COLORS[2] }}>●</span> {entry.h2_persistent}
            </span>
          )}
          {entry.h1_persistent === 0 && entry.h2_persistent === 0 && (
            <span className="text-ink-500">—</span>
          )}
        </div>
      </div>
      <div className="mb-1.5 text-[10px] leading-tight text-ink-400">
        {describeEntry(entry)}
      </div>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 200,
          showlegend: false,
          margin: { l: 28, r: 8, t: 4, b: 24 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#6e7e95", family: "Inter, system-ui", size: 9 },
          xaxis: {
            range: [0, lim],
            showgrid: true, gridcolor: "rgba(255,255,255,0.03)", zeroline: false,
            tickfont: { size: 8 },
          },
          yaxis: {
            range: [0, lim],
            showgrid: true, gridcolor: "rgba(255,255,255,0.03)", zeroline: false,
            tickfont: { size: 8 },
          },
          hoverlabel: { bgcolor: "#10131c", bordercolor: accent, font: { color: "#e8edf3" } },
        }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        config={{ displaylogo: false, responsive: true, staticPlot: false }}
      />
      <div className="mt-1 text-right text-[9px] text-ink-500 opacity-0 transition group-hover:opacity-100">
        click to expand →
      </div>
    </button>
  );
}

// ───────── expanded diagram ─────────

function ExpandedDiagram({
  entry, accent, axisMax, onBack,
}: {
  entry: AllDiagramsEntry;
  accent: string;
  axisMax: number;
  onBack: () => void;
}) {
  const lim = Math.max(0.05, axisMax * 1.15);
  const traces = buildTraces(entry.points, lim, { small: false });

  // Top-3 most-persistent features per dimension
  const topFeatures: Record<number, { birth: number; death: number; pers: number }[]> = {
    1: [], 2: [],
  };
  for (const p of entry.points) {
    if (p.dim === 0 || p.is_infinite) continue;
    const pers = p.death - p.birth;
    if (pers <= PERS_THR) continue;
    topFeatures[p.dim].push({ birth: p.birth, death: p.death, pers });
  }
  topFeatures[1].sort((a, b) => b.pers - a.pers);
  topFeatures[2].sort((a, b) => b.pers - a.pers);

  return (
    <div className="space-y-3">
      <button
        onClick={onBack}
        className="text-[11px] text-ink-400 hover:text-ink-200"
      >
        ← back to all symbols
      </button>

      <div className="panel-tight">
        <div className="mb-2 flex items-center justify-between gap-3">
          <div>
            <div className="section-title">Persistence diagram · <span style={{ color: accent }}>{entry.symbol}</span></div>
            <div className="text-[11px] text-ink-400">{describeEntry(entry)}</div>
          </div>
        </div>
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
      </div>

      {/* Top features call-out */}
      {(topFeatures[1].length > 0 || topFeatures[2].length > 0) && (
        <div className="panel-tight">
          <div className="mb-2 text-[10px] uppercase tracking-widest text-ink-400">
            Most persistent features
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {([1, 2] as const).map((dim) => (
              <FeatureList
                key={dim}
                dim={dim}
                features={topFeatures[dim].slice(0, 5)}
              />
            ))}
          </div>
        </div>
      )}

      <ReadingGuide />
    </div>
  );
}

function FeatureList({
  dim, features,
}: {
  dim: 1 | 2;
  features: { birth: number; death: number; pers: number }[];
}) {
  const colour = HD_COLORS[dim];
  if (features.length === 0) {
    return (
      <div>
        <div className="mb-1 text-[11px]" style={{ color: colour }}>
          {HD_LABEL[dim]}
        </div>
        <div className="text-[11px] text-ink-500">none above noise</div>
      </div>
    );
  }
  const maxPers = Math.max(...features.map((f) => f.pers));
  return (
    <div>
      <div className="mb-1.5 text-[11px]" style={{ color: colour }}>
        {HD_LABEL[dim]}
      </div>
      <div className="space-y-1">
        {features.map((f, i) => (
          <div key={i} className="grid grid-cols-[auto,1fr,auto] items-center gap-2 text-[11px]">
            <span className="font-mono text-[10px] text-ink-400">#{i + 1}</span>
            <div className="h-1.5 overflow-hidden rounded-full bg-ink-800">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${(f.pers / maxPers) * 100}%`,
                  background: colour, opacity: 0.85,
                }}
              />
            </div>
            <span className="font-mono text-[10px] text-ink-300">
              {f.birth.toFixed(2)} → {f.death.toFixed(2)} <span className="text-ink-500">pers {f.pers.toFixed(3)}</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ───────── reading guide ─────────

function ReadingGuide() {
  return (
    <div className="panel-tight">
      <div className="mb-2 text-[10px] uppercase tracking-widest text-ink-400">
        How to read these
      </div>
      <div className="grid grid-cols-1 gap-3 text-[11px] leading-relaxed text-ink-300 md:grid-cols-3">
        <div>
          <div className="mb-0.5 text-[11px] font-semibold" style={{ color: HD_COLORS[0] }}>
            H0 · components
          </div>
          Connected pieces.  All dots start at birth = 0 and die when they merge into the
          main cluster.  One dot lives forever (the cloud itself) — pinned at the top of
          the diagram.
        </div>
        <div>
          <div className="mb-0.5 text-[11px] font-semibold" style={{ color: HD_COLORS[1] }}>
            H1 · loops
          </div>
          1-dimensional cycles.  A loop is a closed semantic circuit:
          word A → word B → word C → … → A.  The further from the diagonal, the more
          robust the loop.
        </div>
        <div>
          <div className="mb-0.5 text-[11px] font-semibold" style={{ color: HD_COLORS[2] }}>
            H2 · voids
          </div>
          2-dimensional cavities.  A void is a sphere-shaped hole — the cloud encloses an
          empty region.  Rare in small clouds; meaningful when present.
        </div>
      </div>
      <div className="mt-3 border-t border-ink-700/40 pt-2 text-[10px] leading-relaxed text-ink-400">
        <span className="text-ink-200">Distance to the diagonal = persistence.</span>{" "}
        Dots hugging the diagonal lived only briefly through the filtration — usually noise.
        Dots far above are features that survived a wide range of scales: the real structure.
      </div>
    </div>
  );
}

// ───────── plotly traces (shared) ─────────

function buildTraces(points: PersistencePoint[], lim: number, opts: { small: boolean }): any[] {
  const traces: any[] = [];
  // diagonal
  traces.push({
    x: [0, lim], y: [0, lim],
    type: "scatter", mode: "lines",
    line: { color: "rgba(255,255,255,0.12)", width: 1, dash: "dash" },
    hoverinfo: "skip", showlegend: false,
  });

  // infinite-death reference (pin all is_infinite dots here)
  const infY = lim * 0.96;
  traces.push({
    x: [0, lim], y: [infY, infY],
    type: "scatter", mode: "lines",
    line: { color: "rgba(255,255,255,0.06)", width: 1, dash: "dot" },
    hoverinfo: "skip", showlegend: false,
  });

  for (const d of [0, 1, 2] as const) {
    const pts = points.filter((p) => p.dim === d);
    if (!pts.length) continue;
    // Split by persistent vs noise.  Persistent dots = full size & opacity;
    // noise dots = tiny & translucent.  Visual hierarchy makes the real
    // features unmissable.
    const persistent = pts.filter((p) => !p.is_infinite && p.death - p.birth > PERS_THR);
    const noise = pts.filter((p) => p.is_infinite || p.death - p.birth <= PERS_THR);

    if (noise.length) {
      traces.push({
        x: noise.map((p) => p.birth),
        y: noise.map((p) => (p.is_infinite ? infY : p.death)),
        type: "scatter", mode: "markers",
        name: HD_LABEL[d],
        legendgroup: `dim-${d}`,
        showlegend: !opts.small,
        marker: {
          size: opts.small ? 4 : 6,
          color: HD_COLORS[d],
          opacity: 0.32,
          line: { color: "rgba(0,0,0,0.4)", width: 0.3 },
        },
        hovertemplate:
          `<b>${HD_LABEL[d]}</b><br>birth: %{x:.3f}<br>death: %{y:.3f}<extra></extra>`,
      });
    }
    if (persistent.length) {
      traces.push({
        x: persistent.map((p) => p.birth),
        y: persistent.map((p) => p.death),
        type: "scatter", mode: "markers",
        name: HD_LABEL[d],
        legendgroup: `dim-${d}`,
        showlegend: !opts.small && !noise.length, // avoid duplicate legend entries
        marker: {
          size: opts.small ? (d === 0 ? 6 : d === 1 ? 8 : 10) : (d === 0 ? 9 : d === 1 ? 12 : 14),
          color: HD_COLORS[d],
          opacity: 0.95,
          line: { color: opts.small ? "rgba(0,0,0,0.5)" : "rgba(255,255,255,0.25)", width: opts.small ? 0.5 : 1 },
        },
        hovertemplate:
          `<b>${HD_LABEL[d]}</b><br>birth: %{x:.3f}<br>death: %{y:.3f}<br>persistence: %{customdata:.3f}<extra></extra>`,
        customdata: persistent.map((p) => p.death - p.birth),
      });
    }
  }

  return traces;
}
