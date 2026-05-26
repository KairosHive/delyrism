"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Plot } from "../plots/Plot";
import { api, TopologySummaryResponse, TopologySummaryEntry, SetQualityMetrics } from "@/lib/api";
import { useSidebar } from "@/lib/store";
import { Skeleton } from "../ui/Skeleton";
import { useTopologyContext } from "./useTopologyContext";

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

  const ctx = useTopologyContext();
  const q = useQuery({
    enabled: !!sid,
    queryKey: ["topo-summary", sid, ...ctx.keyTail],
    queryFn: () =>
      api.post<TopologySummaryResponse>("/topology/summary", { space_id: sid, ...ctx.payload }),
  });
  // When context overlay is on, also fetch the intrinsic baseline so the
  // Set Quality strip can show the *delta* — "applying this context
  // increased richness by +0.4" etc.  Same endpoint, no shift params.
  const qBaseline = useQuery({
    enabled: !!sid && ctx.active,
    queryKey: ["topo-summary", sid, "intrinsic"],
    queryFn: () =>
      api.post<TopologySummaryResponse>("/topology/summary", { space_id: sid }),
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
  const baselineQ = ctx.active ? qBaseline.data?.set_quality ?? null : null;
  return (
    <div className="space-y-4">
      {ctx.active && <ContextPill summary={ctx.summary} />}
      {data.set_quality && (
        <SetQualityStrip
          current={data.set_quality}
          baseline={baselineQ}
          underContext={ctx.active}
        />
      )}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr,1.4fr]">
        {/* ── ranked table ── */}
        <TopoScoreTable
          entries={sorted}
          colorMap={colorMap}
          hover={hover}
          onHover={setHover}
        />

        {/* ── cohesion × loopiness ── */}
        <TopologyMap
          entries={data.entries}
          colorMap={colorMap}
          hover={hover}
          onHover={setHover}
          xKey="h0_cohesion"
          yKey="h1_sum"
          sizeKey="h2_sum"
          xLabel="H0 cohesion (lower = tighter)"
          yLabel="H1 loopiness"
          title="Cohesion × Loopiness"
          subtitle="left = tighter cluster · up = more loops · bigger dot = more voids"
          quadrantLabels={[
            "↘ lower-left — diffuse, no loops",
            "↗ upper-right — loose but loopy",
            "↖ upper-left — tight AND loopy (mythic core)",
            "↙ lower-right — diffuse with hidden structure",
          ]}
        />
      </div>

      {/* ── H1 × H2 — loopiness vs voidiness ── */}
      <TopologyMap
        entries={data.entries}
        colorMap={colorMap}
        hover={hover}
        onHover={setHover}
        xKey="h1_sum"
        yKey="h2_sum"
        sizeKey="topo_score"   // dot size = absolute topo score
        xLabel="H1 loopiness"
        yLabel="H2 voidiness"
        title="Loopiness × Voidiness"
        subtitle="bigger dot = bigger overall TopoScore"
        quadrantLabels={[
          "↘ lower-left — topologically flat",
          "↗ upper-right — richly structured (loops AND cavities)",
          "↖ upper-left — flat sheets of voids, few loops",
          "↙ lower-right — many loops with no enclosed cavities",
        ]}
      />
    </div>
  );
}

// ──────────────────────────── Set Quality strip ────────────────────────────

type MetricSpec = {
  key: keyof SetQualityMetrics;
  label: string;
  hint: string;
  unit?: string;       // optional formatting (e.g. "×" for counts)
  digits?: number;     // decimal places
  goodDir: "up" | "down";  // does up mean "better"?  affects Δ colouring
};

const METRICS: MetricSpec[] = [
  { key: "richness_mean",       label: "richness",  hint: "Mean per-archetype H1 + H2 feature count.  Higher = archetypes are multi-faceted, not flat synonym clusters.",
    digits: 2, unit: "loops/voids", goodDir: "up" },
  { key: "coverage_h1",         label: "coverage H1", hint: "H1 loop mass on the union of every descriptor.  Higher = your archetypes collectively explore the semantic manifold rather than clumping in one place.",
    digits: 3, goodDir: "up" },
  { key: "coverage_h2",         label: "coverage H2", hint: "H2 void mass on the union.  Higher = the joint cloud has cavity structure — usually a sign of rich coverage.",
    digits: 3, goodDir: "up" },
  { key: "cohesion_balance",    label: "balance",   hint: "1 − std/mean of H0 cohesion across archetypes.  Higher = archetypes are similarly tight; lower = some are tight, others diffuse.",
    digits: 2, goodDir: "up" },
  { key: "separation_tightness",label: "separation",hint: "Mean pairwise centroid cosine distance.  Higher = archetypes occupy distinct regions of the embedding space.",
    digits: 2, goodDir: "up" },
  { key: "count_balance",       label: "count even",hint: "Shannon entropy of descriptor counts per archetype, normalised.  1 = perfectly balanced sizes; 0 = one archetype hoards everything.",
    digits: 2, goodDir: "up" },
];

function SetQualityStrip({
  current, baseline, underContext,
}: {
  current: SetQualityMetrics;
  baseline: SetQualityMetrics | null;
  underContext: boolean;
}) {
  return (
    <div className="panel-tight">
      <div className="mb-2 flex items-baseline justify-between">
        <div>
          <div className="section-title">Set quality</div>
          <div className="text-[11px] text-ink-400">
            {underContext
              ? "scalars under the active context · Δ shows change vs intrinsic baseline"
              : "set-level scalars on the unconditioned archetype set"}
          </div>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {METRICS.map((m) => (
          <MetricCard
            key={m.key}
            spec={m}
            value={current[m.key]}
            baseline={baseline ? baseline[m.key] : null}
          />
        ))}
      </div>
    </div>
  );
}

function MetricCard({
  spec, value, baseline,
}: {
  spec: MetricSpec;
  value: number;
  baseline: number | null;
}) {
  const delta = baseline != null ? value - baseline : null;
  const goodIsUp = spec.goodDir === "up";
  // Significance threshold so we don't flash colour for noise.
  const sig = delta != null && Math.abs(delta) > Math.max(0.005, Math.abs(value) * 0.02);
  const deltaColour =
    !sig || delta == null
      ? "#6e7e95"
      : (delta > 0) === goodIsUp
        ? "#5fcfc4" // improvement
        : "#d08770"; // regression
  const arrow =
    delta == null ? "" : !sig ? "·" : delta > 0 ? "▲" : "▼";
  return (
    <div
      className="rounded-lg border border-ink-700/60 bg-ink-900/40 p-2.5"
      title={spec.hint}
    >
      <div className="mb-0.5 text-[9px] uppercase tracking-widest text-ink-500">
        {spec.label}
      </div>
      <div className="flex items-baseline gap-1.5">
        <div className="font-mono text-[15px] text-ink-100">
          {value.toFixed(spec.digits ?? 2)}
        </div>
        {delta != null && (
          <div
            className="font-mono text-[10px] tabular-nums"
            style={{ color: deltaColour }}
          >
            {arrow}{" "}
            {delta > 0 ? "+" : ""}{delta.toFixed(spec.digits ?? 2)}
          </div>
        )}
      </div>
    </div>
  );
}

export function ContextPill({ summary }: { summary: string }) {
  return (
    <div
      className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px]"
      style={{
        borderColor: "#3bbdb060",
        background: "#3bbdb014",
        color: "#5fcfc4",
      }}
      title="Context overlay is ON — these values are computed on the shifted cloud, not the original."
    >
      <span className="h-1.5 w-1.5 rounded-full bg-[#3bbdb0]" />
      under context · {summary}
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

type SumKey = "h0_cohesion" | "h0_outlier" | "h1_sum" | "h2_sum" | "topo_score";

function TopologyMap({
  entries, colorMap, hover, onHover,
  xKey, yKey, sizeKey,
  xLabel, yLabel, title, subtitle,
  quadrantLabels,
}: {
  entries: TopologySummaryEntry[];
  colorMap: Record<string, string>;
  hover: string | null;
  onHover: (s: string | null) => void;
  xKey: SumKey;
  yKey: SumKey;
  sizeKey: SumKey;
  xLabel: string;
  yLabel: string;
  title: string;
  subtitle: string;
  /** [lower-left, upper-right, upper-left, lower-right] */
  quadrantLabels: [string, string, string, string];
}) {
  const sizeVals = entries.map((e) => Math.abs(e[sizeKey]));
  const maxSize = Math.max(...sizeVals, 1e-6);
  const traces = entries.map((e) => {
    const c = colorMap[e.symbol] ?? "#88c0d0";
    const size = 12 + 32 * (Math.abs(e[sizeKey]) / maxSize);
    const dim = hover != null && hover !== e.symbol;
    return {
      x: [e[xKey]],
      y: [e[yKey]],
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
        <div className="section-title">{title}</div>
        <div className="text-[11px] text-ink-400">{subtitle}</div>
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
            title: { text: xLabel, standoff: 8 },
            showgrid: true, gridcolor: "rgba(255,255,255,0.05)", zeroline: false,
          },
          yaxis: {
            title: { text: yLabel, standoff: 8 },
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
        {quadrantLabels.map((q, i) => (
          <div key={i}><span className="text-ink-200">{q.split(" — ")[0]}</span> — {q.split(" — ")[1]}</div>
        ))}
      </div>
    </div>
  );
}
