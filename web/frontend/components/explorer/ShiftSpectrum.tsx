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
      {q.data && <SpectrumContent data={q.data} colorMap={colorMap} />}
    </div>
  );
}

function SpectrumContent({
  data, colorMap,
}: {
  data: ShiftSpectrumResponse;
  colorMap: Record<string, string>;
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
        {data.axes.length === 0 ? (
          <div className="text-[11px] text-ink-500">
            no measurable secondary structure — context affects every descriptor in the same direction
          </div>
        ) : (
          <div className="space-y-1.5">
            {data.axes.map((ax, i) => (
              <AxisRow
                key={i}
                idx={i}
                axis={ax}
                topSigma={topSigma}
                colorMap={colorMap}
              />
            ))}
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
  idx, axis, topSigma, colorMap,
}: {
  idx: number;
  axis: SpectrumAxis;
  topSigma: number;
  colorMap: Record<string, string>;
}) {
  const pct = Math.max(2, Math.min(100, (axis.sigma / Math.max(topSigma, 1e-9)) * 100));

  // In archetype-space SVD, profile entries are signed coefficients of the
  // axis itself.  Positive = the archetype is on the + side of the axis
  // (descriptors with positive U[i,k] move toward it).  Negative = on the
  // − side.  An axis with both signs IS a contrast — show both poles.
  const sorted = [...axis.archetype_profile].sort(
    (a, b) => Math.abs(b.alignment) - Math.abs(a.alignment),
  );
  const positives = sorted.filter((p) => p.alignment > 0.2).slice(0, 2);
  const negatives = sorted.filter((p) => p.alignment < -0.2).slice(0, 2);
  const isContrast = positives.length > 0 && negatives.length > 0;

  return (
    <div className="flex items-start gap-2 py-1 text-[11px]">
      <div className="w-12 shrink-0 pt-0.5 font-mono text-ink-400">axis {idx + 1}</div>
      <div className="mt-1 h-2 w-24 shrink-0 overflow-hidden rounded-full bg-ink-800">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background: "linear-gradient(90deg, #3bbdb0 0%, #5fcfc4 100%)",
          }}
        />
      </div>
      <div className="pt-0.5 font-mono text-[10px] tabular-nums text-ink-400">
        {axis.sigma.toFixed(2)}
      </div>

      <div className="ml-2 flex min-w-0 flex-1 flex-col gap-0.5">
        {/* archetype contrast line */}
        <div className="flex flex-wrap items-center gap-1.5">
          {positives.length === 0 && negatives.length === 0 && (
            <span className="text-ink-500">no clear contrast</span>
          )}
          {positives.map((p, i) => (
            <React.Fragment key={"p" + p.symbol}>
              {i > 0 && <span className="text-ink-500">+</span>}
              <ArchChip p={p} colorMap={colorMap} />
            </React.Fragment>
          ))}
          {isContrast && (
            <span className="px-0.5 text-ink-500">vs</span>
          )}
          {negatives.map((p, i) => (
            <React.Fragment key={"n" + p.symbol}>
              {i > 0 && <span className="text-ink-500">+</span>}
              <ArchChip p={p} colorMap={colorMap} variant="against" />
            </React.Fragment>
          ))}
        </div>
        {/* movers */}
        <Movers
          positives={positives}
          negatives={negatives}
          isContrast={isContrast}
          axis={axis}
          colorMap={colorMap}
        />
      </div>
    </div>
  );
}

function ArchChip({
  p, colorMap, variant = "toward",
}: {
  p: { symbol: string; alignment: number };
  colorMap: Record<string, string>;
  variant?: "toward" | "against";
}) {
  const c = colorMap[p.symbol] ?? "#888";
  const dashed = variant === "against";
  return (
    <span
      className={`rounded-md border px-1.5 py-0.5 text-[10px] font-medium ${dashed ? "border-dashed" : ""}`}
      style={{
        color: c,
        borderColor: c + (dashed ? "88" : "55"),
        background: c + "12",
      }}
      title={`signed coefficient ${(p.alignment * 100).toFixed(0)}%`}
    >
      {p.symbol}
    </span>
  );
}

function Movers({
  positives, negatives, isContrast, axis, colorMap,
}: {
  positives: { symbol: string; alignment: number }[];
  negatives: { symbol: string; alignment: number }[];
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
  // Monopolar axis: pick the side whose archetypes are dominant.
  const list = positives.length >= negatives.length ? axis.positive_movers : axis.negative_movers;
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

