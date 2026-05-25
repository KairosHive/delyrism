"use client";
import * as React from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api, ShiftSpectrumResponse, SpectrumAxis, SpectrumMoverEntry, SpectrumProfileEntry } from "@/lib/api";
import { useSidebar, buildContextWeights } from "@/lib/store";

/**
 * Shift spectrum — top-K principal axes of Δ = D' − D, computed via SVD.
 *
 * Reads as a one-line narrative ("This context is …, pulling … toward …")
 * with a small bar chart of σ values + per-axis mover chips as supporting
 * evidence.  The bare math (signed alignment percentages, raw sigma values)
 * is shown only on hover or expansion — the headline is plain prose.
 */
export function ShiftSpectrum() {
  const sid = useSidebar((s) => s.spaceId);
  const colorMap = useSidebar((s) => s.colorMap);
  const symbols = useSidebar((s) => s.symbols);
  const sentence = useSidebar((s) => s.contextSentence);
  const weights = useSidebar(buildContextWeights);
  const strategy = useSidebar((s) => s.strategy);
  const beta = useSidebar((s) => s.beta);
  const gate = useSidebar((s) => s.gate);
  const tau = useSidebar((s) => s.shiftTau);
  const wss = useSidebar((s) => s.withinSymbolSoftmax);
  const gamma = useSidebar((s) => s.gamma);
  const poolType = useSidebar((s) => s.poolType);
  const poolW = useSidebar((s) => s.poolW);
  const mAlpha = useSidebar((s) => s.membershipAlpha);

  const audioActive = useSidebar((s) => s.audioActive);
  const imageActive = useSidebar((s) => s.imageActive);
  const alchemistActive = useSidebar((s) => s.alchemistActive);
  const audioNonce = useSidebar((s) => s.audioNonce);
  const imageNonce = useSidebar((s) => s.imageNonce);
  const alchemistNonce = useSidebar((s) => s.alchemistNonce);

  const hasCtx = !!sentence.trim() || !!weights || audioActive || imageActive || alchemistActive;

  const q = useQuery({
    enabled: !!sid && hasCtx,
    placeholderData: keepPreviousData,
    queryKey: [
      "shift-spectrum", sid, sentence, weights, strategy, beta, gate, tau, wss, gamma,
      poolType, poolW, mAlpha, audioNonce, imageNonce, alchemistNonce,
    ],
    queryFn: () =>
      api.post<ShiftSpectrumResponse>("/shift-spectrum", {
        space_id: sid,
        sentence: sentence.trim() || null,
        weights,
        strategy, beta, gate, tau,
        within_symbol_softmax: wss,
        gamma,
        pool_type: poolType,
        pool_w: poolW,
        membership_alpha: mAlpha,
        topk: 3,
      }),
  });

  return (
    <div className="panel-tight">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <div>
          <div className="section-title flex items-center gap-2">
            Shift spectrum
            {/* Subtle pulse while a refetch is in flight — keeps the user
                oriented when the textarea is debounced and the panel value
                comes in ~half a second after the last keystroke. */}
            {q.isFetching && q.data && (
              <span className="inline-flex items-center gap-1 text-[10px] text-ink-400">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-400" />
                refreshing
              </span>
            )}
          </div>
          <div className="text-[11px] text-ink-400">
            primary pull + how context affects different descriptors differently
          </div>
        </div>
      </div>

      {!hasCtx && (
        <div className="p-6 text-sm text-ink-300">
          Add a context (sentence, weights, audio, image, or alchemist blend) to see the spectrum.
        </div>
      )}
      {hasCtx && q.isPending && !q.data && (
        <div className="p-6 text-sm text-ink-300">computing…</div>
      )}
      {q.data && <SpectrumContent data={q.data} colorMap={colorMap} symbols={symbols} />}
    </div>
  );
}

// Distinct from any palette archetype colour so polygons read as "axes" not
// "archetypes".  Order matters: axis 1 gets index 0.
const AXIS_COLORS = ["#3bbdb0", "#d08770", "#c2a6fe", "#e6c068"] as const;

function SpectrumContent({
  data, colorMap, symbols,
}: {
  data: ShiftSpectrumResponse;
  colorMap: Record<string, string>;
  symbols: string[];
}) {
  const topSigma = data.axes[0]?.sigma ?? 1;
  const meanDominant = topMeanArchetypes(data.mean_shift);
  const subKind = classifySub(data);

  return (
    <div className="space-y-3">
      {/* ── primary pull (mean shift) ── */}
      <div className="rounded-lg border border-ink-700/60 bg-ink-900/40 p-3 leading-relaxed">
        <div className="mb-0.5 text-[10px] uppercase tracking-widest text-ink-400">
          Primary pull · what the rankings panel sees
        </div>
        <div className="text-sm text-ink-100">
          {meanDominant.length === 0 ? (
            <span className="text-ink-400">no net pull — context-orthogonal to every archetype</span>
          ) : (
            <>
              context pulls everything toward{" "}
              {meanDominant.map((p, i) => (
                <React.Fragment key={p.symbol}>
                  {i > 0 && <span className="text-ink-400"> + </span>}
                  <strong style={{ color: colorMap[p.symbol] ?? "#cbd" }}>{p.symbol}</strong>
                </React.Fragment>
              ))}
            </>
          )}
        </div>
      </div>

      {/* ── differential / residual structure ── */}
      <div>
        <div className="mb-1.5 text-[10px] uppercase tracking-widest text-ink-400">
          Beyond that · {subKind}
        </div>
        {data.axes.length === 0 || symbols.length === 0 ? (
          <div className="text-[11px] text-ink-500">
            no measurable secondary structure — context affects every descriptor in the same direction
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-[260px,1fr]">
            <div className="flex justify-center">
              <ArchetypeRosette
                axes={data.axes}
                symbols={symbols}
                colorMap={colorMap}
                axisColors={[...AXIS_COLORS]}
                size={260}
              />
            </div>
            <div className="space-y-1.5">
              {data.axes.map((ax, i) => (
                <AxisRow
                  key={i}
                  idx={i}
                  axis={ax}
                  topSigma={topSigma}
                  colorMap={colorMap}
                  axisColor={AXIS_COLORS[i] ?? "#888"}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── small footer for the actually-curious ── */}
      <div className="flex items-center gap-3 text-[10px] text-ink-500">
        <span title="σ₁/σ₂ of the residual — high = one sub-pattern dominates; ~1 = several comparable sub-patterns.">
          σ₁/σ₂ {data.dominance_ratio == null ? "—" : data.dominance_ratio.toFixed(2)}
        </span>
        <span title="Participation ratio — fractional number of sub-patterns actually doing work.">
          effective sub-axes {data.effective_rank.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

function topMeanArchetypes(mean: SpectrumProfileEntry[]): SpectrumProfileEntry[] {
  // The mean axis is a unit vector in archetype space; surface entries with
  // |coef| ≥ 0.25, up to 2 of them.  Sign is meaningful here (negative =
  // context pushes things AWAY from that archetype on average) but in the
  // common case the top-symbol all have positive sign.
  return [...mean]
    .filter((p) => Math.abs(p.alignment) >= 0.25)
    .sort((a, b) => Math.abs(b.alignment) - Math.abs(a.alignment))
    .slice(0, 2);
}

function classifySub(data: ShiftSpectrumResponse): string {
  if (!data.axes.length) return "no sub-structure";
  const er = data.effective_rank;
  const dom = data.dominance_ratio;
  if (er <= 1.2 || (dom != null && dom >= 3.5)) return "one differential sub-pattern";
  if (er <= 1.9 || (dom != null && dom >= 1.6)) return "primary sub-pattern + secondary";
  return "multiple comparable sub-patterns";
}

function AxisRow({
  idx, axis, topSigma, colorMap, axisColor,
}: {
  idx: number;
  axis: SpectrumAxis;
  topSigma: number;
  colorMap: Record<string, string>;
  axisColor: string;
}) {
  const pct = Math.max(2, Math.min(100, (axis.sigma / Math.max(topSigma, 1e-9)) * 100));

  // Determine whether to render bipolar (+ / −) or monopolar movers based on
  // whether the axis has a meaningful negative pole.  Profile drives this.
  const positives = axis.archetype_profile.filter((p) => p.alignment > 0.2);
  const negatives = axis.archetype_profile.filter((p) => p.alignment < -0.2);
  const isContrast = positives.length > 0 && negatives.length > 0;

  return (
    <div className="flex items-start gap-2 py-1 text-[11px]">
      <div className="flex w-14 shrink-0 items-center gap-1.5 pt-0.5">
        <span
          className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm"
          style={{ background: axisColor }}
        />
        <span className="font-mono text-ink-400">{idx + 1}</span>
      </div>
      <div className="mt-1 h-2 w-24 shrink-0 overflow-hidden rounded-full bg-ink-800">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background: axisColor,
            opacity: 0.85,
          }}
        />
      </div>
      <div className="pt-0.5 font-mono text-[10px] tabular-nums text-ink-400">
        {axis.sigma.toFixed(2)}
      </div>

      <div className="ml-2 flex min-w-0 flex-1 flex-col gap-0.5">
        <Movers
          isContrast={isContrast}
          axis={axis}
          colorMap={colorMap}
        />
      </div>
    </div>
  );
}

/* ─── Archetype rosette — overlaid polygons per residual axis ──────────
 * One polygon per axis k.  Vertex radius = baseline + (σ_k / σ_max) ×
 * v_k[s] × maxOffset.  Positive coefficients push outward, negative push
 * inward — so an axis that contrasts two archetypes appears as a polygon
 * that's bulged out on one side and dented in on the other.
 *
 * Multiple axes overlay with low opacity so their orthogonal structure
 * reads at a glance: single-axis context = one bulge; polarising context
 * = two roughly equal polygons pointing different directions; etc. */
function ArchetypeRosette({
  axes, symbols, colorMap, axisColors, size,
}: {
  axes: SpectrumAxis[];
  symbols: string[];
  colorMap: Record<string, string>;
  axisColors: string[];
  size: number;
}) {
  const cx = size / 2;
  const cy = size / 2;
  const baseR = size * 0.30;
  const maxOffset = size * 0.18;

  // Archetypes evenly placed on the circle, starting at the top (-π/2).
  const angles = symbols.map((_, i) =>
    (2 * Math.PI * i) / symbols.length - Math.PI / 2
  );
  const archDots = angles.map((a) => ({
    x: cx + baseR * Math.cos(a),
    y: cy + baseR * Math.sin(a),
  }));
  // Label position pushes a bit further out than the maximum polygon
  // extent so labels don't overlap polygon vertices.
  const labelR = baseR + maxOffset + 14;
  const labels = angles.map((a, i) => {
    const x = cx + labelR * Math.cos(a);
    const y = cy + labelR * Math.sin(a);
    const c = Math.cos(a);
    // anchor: left if archetype is on the right side, right if on left
    const anchor: "start" | "middle" | "end" =
      Math.abs(c) < 0.25 ? "middle" : c > 0 ? "start" : "end";
    return { x, y, anchor, sym: symbols[i] };
  });

  const sigmaMax = Math.max(...axes.map((a) => a.sigma), 1e-9);

  // Build polygon points for each axis.
  const polygons = axes.map((axis, idx) => {
    const lookup = new Map(axis.archetype_profile.map((p) => [p.symbol, p.alignment]));
    const sigmaScale = axis.sigma / sigmaMax;
    const points = symbols.map((sym, i) => {
      const v = lookup.get(sym) ?? 0;
      const r = baseR + sigmaScale * v * maxOffset;
      const a = angles[i];
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    });
    return {
      points: points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" "),
      color: axisColors[idx] ?? "#888",
    };
  });

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width={size}
      height={size}
      role="img"
      aria-label="archetype rosette of residual axes"
    >
      {/* baseline circle — visual anchor for "no shift" */}
      <circle
        cx={cx}
        cy={cy}
        r={baseR}
        fill="none"
        stroke="#2a3142"
        strokeWidth={1}
        strokeDasharray="3 3"
      />

      {/* polygons, drawn smallest-σ first so the largest axis stays on top */}
      {[...polygons]
        .map((p, i) => ({ ...p, idx: i }))
        .sort((a, b) => (axes[a.idx].sigma - axes[b.idx].sigma))
        .map((p) => (
          <polygon
            key={p.idx}
            points={p.points}
            fill={p.color}
            fillOpacity={0.16}
            stroke={p.color}
            strokeWidth={1.6}
            strokeOpacity={0.9}
            strokeLinejoin="round"
          />
        ))}

      {/* archetype dots */}
      {archDots.map((d, i) => (
        <circle
          key={"d" + i}
          cx={d.x}
          cy={d.y}
          r={3}
          fill={colorMap[symbols[i]] ?? "#cbd"}
          opacity={0.85}
        />
      ))}

      {/* archetype labels */}
      {labels.map((l, i) => (
        <text
          key={"l" + i}
          x={l.x}
          y={l.y}
          textAnchor={l.anchor}
          dominantBaseline="middle"
          fontSize="10"
          fontWeight={500}
          fill={colorMap[l.sym] ?? "#cad4e0"}
          style={{ fontFamily: "Inter, system-ui, sans-serif" }}
        >
          {l.sym}
        </text>
      ))}
    </svg>
  );
}

function Movers({
  isContrast, axis, colorMap,
}: {
  isContrast: boolean;
  axis: SpectrumAxis;
  colorMap: Record<string, string>;
}) {
  if (isContrast) {
    // Bipolar axis: show + movers on the "positive" side and − movers on the
    // "negative" side, with a thin separator.  Tells the user how the axis
    // actually splits the descriptor cloud.
    return (
      <div className="grid grid-cols-2 gap-2 text-[10px] text-ink-300">
        <MoverInline movers={axis.positive_movers} colorMap={colorMap} side="+" />
        <MoverInline movers={axis.negative_movers} colorMap={colorMap} side="−" />
      </div>
    );
  }
  // Monopolar axis: prefer positive movers if there are any, else negative.
  const list = axis.positive_movers.length >= axis.negative_movers.length
    ? axis.positive_movers
    : axis.negative_movers;
  return <MoverInline movers={list} colorMap={colorMap} side="" />;
}

function MoverInline({
  movers, colorMap, side,
}: {
  movers: SpectrumMoverEntry[];
  colorMap: Record<string, string>;
  side: string;
}) {
  if (movers.length === 0) {
    return <span className="text-ink-600">—</span>;
  }
  return (
    <span className="truncate text-[10px] text-ink-300">
      {side && <span className="text-ink-500">{side}&nbsp;</span>}
      {movers.slice(0, 4).map((m, i) => (
        <React.Fragment key={m.descriptor}>
          {i > 0 && <span className="text-ink-600">, </span>}
          <span style={{ color: colorMap[m.symbol] ?? "#cbd" }}>{m.descriptor}</span>
        </React.Fragment>
      ))}
    </span>
  );
}

