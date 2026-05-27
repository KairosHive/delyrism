"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Plot } from "../plots/Plot";
import {
  api,
  TopologySummaryResponse, TopologySummaryEntry, SetQualityMetrics,
  TopologySynergyResponse,
} from "@/lib/api";
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

  // Synergy is heavier — fetch separately so the fast Set Quality scalars
  // show first, synergy-derived "bridges" cards fill in when the request
  // lands.  Same delta pattern (intrinsic baseline when overlay is on).
  const qSyn = useQuery({
    enabled: !!sid,
    queryKey: ["topo-synergy", sid, ...ctx.keyTail],
    queryFn: () =>
      api.post<TopologySynergyResponse>("/topology/synergy", { space_id: sid, ...ctx.payload }),
  });
  const qSynBaseline = useQuery({
    enabled: !!sid && ctx.active,
    queryKey: ["topo-synergy", sid, "intrinsic"],
    queryFn: () =>
      api.post<TopologySynergyResponse>("/topology/synergy", { space_id: sid }),
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
          currentEntries={data.entries}
          baselineEntries={qBaseline.data?.entries ?? null}
          syn={qSyn.data ?? null}
          synBaseline={ctx.active ? qSynBaseline.data ?? null : null}
          synPending={qSyn.isPending}
          colorMap={colorMap}
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
  { key: "richness_mean",       label: "richness",  hint: "Mean per-archetype H1 + H2 feature count.  Higher = archetypes are multi-faceted, not flat synonym clusters.  Tends DOWN under typical focusing context — see 'focus' (which rises).",
    digits: 2, unit: "loops/voids", goodDir: "up" },
  { key: "coverage_h1",         label: "coverage H1", hint: "H1 loop mass on the union of every descriptor.  Higher = your archetypes collectively explore the semantic manifold.  Tends DOWN under context (the cloud concentrates).",
    digits: 3, goodDir: "up" },
  { key: "coverage_h2",         label: "coverage H2", hint: "H2 void mass on the union.  Higher = the joint cloud has cavity structure (rare and meaningful when present).",
    digits: 3, goodDir: "up" },
  { key: "cohesion_balance",    label: "balance",   hint: "1 − std/mean of H0 cohesion across archetypes.  Higher = archetypes are similarly tight; lower = some are tight, others diffuse.",
    digits: 2, goodDir: "up" },
  { key: "separation_tightness",label: "separation",hint: "Mean pairwise centroid cosine distance.  Higher = archetypes occupy distinct regions of the embedding space.  Tends DOWN under context (centroids pull toward v_ctx).",
    digits: 2, goodDir: "up" },
  { key: "count_balance",       label: "count even",hint: "Shannon entropy of descriptor counts per archetype, normalised.  1 = perfectly balanced sizes; 0 = one archetype hoards everything.  Independent of context.",
    digits: 2, goodDir: "up" },
  { key: "focus",               label: "focus",     hint: "1 / (1 + mean H0 cohesion).  Higher = the per-symbol clouds are tighter overall.  This is the counter-signal to richness/coverage/separation — it RISES under productive focusing context while the others fall.",
    digits: 2, goodDir: "up" },
];

function meanSynergy(data: TopologySynergyResponse | null, kind: "h1" | "h2"): number | null {
  if (!data || data.entries.length === 0) return null;
  const k = kind === "h1" ? "synergy_h1" : "synergy_h2";
  let s = 0;
  for (const e of data.entries) s += e[k];
  return s / data.entries.length;
}

function SetQualityStrip({
  current, baseline, underContext,
  currentEntries, baselineEntries,
  syn, synBaseline, synPending,
  colorMap,
}: {
  current: SetQualityMetrics;
  baseline: SetQualityMetrics | null;
  underContext: boolean;
  currentEntries: TopologySummaryEntry[];
  baselineEntries: TopologySummaryEntry[] | null;
  syn: TopologySynergyResponse | null;
  synBaseline: TopologySynergyResponse | null;
  synPending: boolean;
  colorMap: Record<string, string>;
}) {
  const synH1 = meanSynergy(syn, "h1");
  const synH2 = meanSynergy(syn, "h2");
  const synH1Base = meanSynergy(synBaseline, "h1");
  const synH2Base = meanSynergy(synBaseline, "h2");

  return (
    <div className="panel-tight space-y-4">
      <div>
        <div className="mb-2 flex items-baseline justify-between">
          <div>
            <div className="section-title">Set quality</div>
            <div className="text-[11px] text-ink-400">
              {underContext
                ? "state under the active context · context effect below"
                : "set-level state on the unconditioned archetype set"}
            </div>
          </div>
        </div>

        {/* ── State section ── */}
        <div className="mb-1.5 text-[10px] uppercase tracking-widest text-ink-500">State</div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-9">
          {METRICS.map((m) => (
            <MetricCard
              key={m.key}
              spec={m}
              value={current[m.key]}
              baseline={baseline ? baseline[m.key] : null}
            />
          ))}
          <SynergyMetricCard
            label="bridges H1"
            hint="Mean H1 synergy across all archetype pairs.  Synergy = loop mass that requires the pair's clouds to be joined."
            value={synH1}
            baseline={synH1Base}
            pending={synPending && synH1 == null}
          />
          <SynergyMetricCard
            label="bridges H2"
            hint="Mean H2 synergy across all archetype pairs."
            value={synH2}
            baseline={synH2Base}
            pending={synPending && synH2 == null}
          />
        </div>
      </div>

      {/* ── Context effect section (only when overlay is active) ── */}
      {underContext && baseline && baselineEntries && (
        <ContextEffectStrip
          current={current}
          baseline={baseline}
          currentEntries={currentEntries}
          baselineEntries={baselineEntries}
          synH1={synH1}
          synH1Base={synH1Base}
          colorMap={colorMap}
        />
      )}
    </div>
  );
}

// ─────── Context Effect ─────────────────────────────────────────────────

function ContextEffectStrip({
  current, baseline, currentEntries, baselineEntries, synH1, synH1Base, colorMap,
}: {
  current: SetQualityMetrics;
  baseline: SetQualityMetrics;
  currentEntries: TopologySummaryEntry[];
  baselineEntries: TopologySummaryEntry[];
  synH1: number | null;
  synH1Base: number | null;
  colorMap: Record<string, string>;
}) {
  // Align baseline entries by symbol name to be defensive.
  const baselineBySym = new Map(baselineEntries.map((e) => [e.symbol, e]));

  // 1. Focus gain
  const focusDelta = current.focus - baseline.focus;
  const focusSig = Math.abs(focusDelta) > 0.005;

  // 2. New loops — per-archetype Δ (richness count)
  const loopGains: { symbol: string; delta: number }[] = [];
  for (const cur of currentEntries) {
    const base = baselineBySym.get(cur.symbol);
    if (!base) continue;
    const delta = (cur.h1_count + cur.h2_count) - (base.h1_count + base.h2_count);
    if (delta > 0) loopGains.push({ symbol: cur.symbol, delta });
  }
  loopGains.sort((a, b) => b.delta - a.delta);

  // 3. Sharpened on — archetype with largest negative Δ in H0 cohesion
  let sharpened: { symbol: string; delta: number } | null = null;
  for (const cur of currentEntries) {
    const base = baselineBySym.get(cur.symbol);
    if (!base) continue;
    const delta = cur.h0_cohesion - base.h0_cohesion;
    if (sharpened == null || delta < sharpened.delta) {
      sharpened = { symbol: cur.symbol, delta };
    }
  }
  const sharpSig = sharpened != null && sharpened.delta < -0.01;

  // 4. Bridges built
  const bridgeDelta = synH1 != null && synH1Base != null ? synH1 - synH1Base : null;
  const bridgeSig = bridgeDelta != null && Math.abs(bridgeDelta) > 0.003;

  return (
    <div>
      <div className="mb-1.5 flex items-center gap-2 text-[10px] uppercase tracking-widest text-ink-500">
        <span className="rounded-full bg-accent-500/20 px-1.5 py-0.5 text-accent-200" style={{ fontSize: "9px" }}>
          ● live
        </span>
        Context effect — did this context do something useful?
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <EffectCard
          label="Field focus"
          headline={
            !focusSig ? "unchanged"
            : focusDelta > 0 ? "tightened"
            : "diffused"
          }
          delta={focusDelta}
          digits={3}
          subtitle={
            !focusSig ? "no notable focus shift"
            : focusDelta > 0 ? "per-symbol clouds tightened"
            : "per-symbol clouds spread out"
          }
          mood={!focusSig ? "neutral" : focusDelta > 0 ? "positive" : "negative"}
        />

        <EffectCard
          label="New loops"
          headline={
            loopGains.length === 0
              ? "none created"
              : `+${loopGains.reduce((s, e) => s + e.delta, 0)} loop${
                  loopGains.reduce((s, e) => s + e.delta, 0) === 1 ? "" : "s"
                }`
          }
          delta={null}
          digits={0}
          subtitle={
            loopGains.length === 0
              ? "no archetype gained features under context"
              : `${loopGains.length} archetype${loopGains.length === 1 ? "" : "s"} gained: ${
                  loopGains.slice(0, 3).map((g) => g.symbol).join(", ")
                }${loopGains.length > 3 ? "…" : ""}`
          }
          mood={loopGains.length > 0 ? "positive" : "neutral"}
          accentSymbols={loopGains.slice(0, 3).map((g) => ({ name: g.symbol, c: colorMap[g.symbol] ?? "#cad4e0" }))}
        />

        <EffectCard
          label="Sharpened on"
          headline={
            sharpSig && sharpened
              ? sharpened.symbol
              : "no single archetype"
          }
          delta={sharpSig && sharpened ? sharpened.delta : null}
          digits={3}
          subtitle={
            sharpSig && sharpened
              ? "this archetype tightened most under context"
              : "context affected archetypes evenly"
          }
          mood={sharpSig ? "positive" : "neutral"}
          accentSymbols={sharpSig && sharpened ? [{ name: sharpened.symbol, c: colorMap[sharpened.symbol] ?? "#cad4e0" }] : []}
        />

        <EffectCard
          label="Bridges built"
          headline={
            !bridgeSig || bridgeDelta == null
              ? "unchanged"
              : bridgeDelta > 0 ? "more bridges" : "fewer bridges"
          }
          delta={bridgeDelta}
          digits={3}
          subtitle={
            bridgeDelta == null
              ? "synergy still loading…"
              : !bridgeSig
                ? "no notable change in shared structure"
                : bridgeDelta > 0
                  ? "context wove new shared loops between archetypes"
                  : "context pulled archetypes apart"
          }
          mood={
            bridgeDelta == null ? "neutral"
              : !bridgeSig ? "neutral"
              : bridgeDelta > 0 ? "positive" : "negative"
          }
        />
      </div>

      <p className="mt-2 text-[10px] leading-snug text-ink-500">
        Most state metrics naturally drop under context — that's context <em>focusing</em> the cloud.
        These four signals tell you whether the focus was <em>productive</em>: did it tighten things,
        create new structure, sharpen one archetype, or build relations between them?
      </p>
    </div>
  );
}

const MOOD_STYLE: Record<"positive" | "negative" | "neutral", { border: string; bg: string; text: string; icon: string; iconColor: string }> = {
  positive: { border: "#3bbdb066", bg: "#3bbdb014", text: "#5fcfc4", icon: "✓", iconColor: "#5fcfc4" },
  negative: { border: "#d0877066", bg: "#d0877014", text: "#d08770", icon: "↓", iconColor: "#d08770" },
  neutral:  { border: "rgba(255,255,255,0.08)", bg: "rgba(255,255,255,0.02)", text: "#9fadc1", icon: "·", iconColor: "#6e7e95" },
};

function EffectCard({
  label, headline, delta, digits, subtitle, mood, accentSymbols,
}: {
  label: string;
  headline: string;
  delta: number | null;
  digits: number;
  subtitle: string;
  mood: "positive" | "negative" | "neutral";
  accentSymbols?: { name: string; c: string }[];
}) {
  const s = MOOD_STYLE[mood];
  return (
    <div
      className="rounded-lg border p-3"
      style={{ borderColor: s.border, background: s.bg }}
    >
      <div className="mb-1 flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-widest text-ink-500">{label}</span>
        <span className="font-mono text-base leading-none" style={{ color: s.iconColor }}>
          {s.icon}
        </span>
      </div>
      <div className="flex items-baseline gap-2">
        <div className="text-[15px] font-medium" style={{ color: s.text }}>
          {accentSymbols && accentSymbols.length > 0 && accentSymbols[0].name === headline
            ? <span style={{ color: accentSymbols[0].c }}>{headline}</span>
            : headline}
        </div>
        {delta != null && (
          <div className="font-mono text-[10px] tabular-nums text-ink-400">
            {delta > 0 ? "+" : ""}{delta.toFixed(digits)}
          </div>
        )}
      </div>
      <div className="mt-1 text-[10px] leading-snug text-ink-400">{subtitle}</div>
    </div>
  );
}

function SynergyMetricCard({
  label, hint, value, baseline, pending,
}: {
  label: string;
  hint: string;
  value: number | null;
  baseline: number | null;
  pending: boolean;
}) {
  if (pending && value == null) {
    return (
      <div
        className="rounded-lg border border-ink-700/60 bg-ink-900/40 p-2.5"
        title={hint}
      >
        <div className="mb-0.5 text-[9px] uppercase tracking-widest text-ink-500">{label}</div>
        <div className="flex items-baseline gap-1.5">
          <div className="font-mono text-[15px] text-ink-500">
            <span className="inline-block h-3 w-10 animate-pulse rounded bg-ink-800/60" />
          </div>
        </div>
      </div>
    );
  }
  if (value == null) {
    return (
      <div
        className="rounded-lg border border-ink-700/60 bg-ink-900/40 p-2.5"
        title={hint}
      >
        <div className="mb-0.5 text-[9px] uppercase tracking-widest text-ink-500">{label}</div>
        <div className="font-mono text-[15px] text-ink-500">—</div>
      </div>
    );
  }
  const delta = baseline != null ? value - baseline : null;
  const sig = delta != null && Math.abs(delta) > Math.max(0.003, Math.abs(value) * 0.02);
  // For bridges, more = more connected; whether that's "good" depends on
  // intent.  Colour Δ purely by sign (teal = positive change, warm = negative)
  // without claiming a polarity.
  const deltaColour =
    !sig || delta == null
      ? "#6e7e95"
      : delta > 0
        ? "#5fcfc4"
        : "#d08770";
  const arrow = delta == null ? "" : !sig ? "·" : delta > 0 ? "▲" : "▼";
  return (
    <div
      className="rounded-lg border border-ink-700/60 bg-ink-900/40 p-2.5"
      title={hint}
    >
      <div className="mb-0.5 text-[9px] uppercase tracking-widest text-ink-500">{label}</div>
      <div className="flex items-baseline gap-1.5">
        <div
          className="font-mono text-[15px]"
          style={{ color: value >= 0 ? "#cad4e0" : "#d08770" }}
        >
          {value >= 0 ? "" : ""}{value.toFixed(3)}
        </div>
        {delta != null && (
          <div
            className="font-mono text-[10px] tabular-nums"
            style={{ color: deltaColour }}
          >
            {arrow} {delta > 0 ? "+" : ""}{delta.toFixed(3)}
          </div>
        )}
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
