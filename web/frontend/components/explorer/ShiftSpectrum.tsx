"use client";
import * as React from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api, ShiftSpectrumResponse, SpectrumAxis, SpectrumMoverEntry } from "@/lib/api";
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
            how many independent directions of pull does this context have?
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
  const headline = buildHeadline(data, colorMap);
  const topSigma = data.axes[0]?.sigma ?? 1;

  return (
    <div className="space-y-3">
      {/* ── one-line narrative ── */}
      <div className="rounded-lg border border-ink-700/60 bg-ink-900/40 p-3 leading-relaxed">
        <div className="mb-0.5 text-[10px] uppercase tracking-widest text-ink-400">
          {headline.kind}
        </div>
        <div className="text-sm text-ink-100">{headline.sentence}</div>
      </div>

      {/* ── stacked σ-bars with inline archetype + mover labels ── */}
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

      {/* ── small footer for the actually-curious ── */}
      <div className="flex items-center gap-3 text-[10px] text-ink-500">
        <span title="σ₁/σ₂ — ratio of the two largest singular values. High = one direction dominates; ~1 = several independent axes.">
          σ₁/σ₂ {data.dominance_ratio == null ? "—" : data.dominance_ratio.toFixed(2)}
        </span>
        <span title="Participation ratio of σ² values. ~1 = a single axis carries the shift; ~k = k axes share the load.">
          effective axes {data.effective_rank.toFixed(2)}
        </span>
      </div>
    </div>
  );
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

  // Dominant archetypes for this axis — those whose |alignment| crosses a
  // light threshold.  Up to 2 strong ones surfaced inline; everything else
  // is in the tooltip.
  const dom = axis.archetype_profile
    .filter((p) => Math.abs(p.alignment) >= 0.2)
    .slice(0, 2);
  // Direction read off the top alignment's sign — positive means the axis
  // points toward that archetype, negative means descriptors of that owner
  // are moving the other way.
  const sign = dom[0] ? Math.sign(dom[0].alignment) : 0;
  const movers: SpectrumMoverEntry[] = sign >= 0 ? axis.positive_movers : axis.negative_movers;

  return (
    <div className="flex items-center gap-2 text-[11px]">
      <div className="w-12 shrink-0 font-mono text-ink-400">axis {idx + 1}</div>
      <div className="h-2 w-24 shrink-0 overflow-hidden rounded-full bg-ink-800">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background: "linear-gradient(90deg, #3bbdb0 0%, #5fcfc4 100%)",
          }}
        />
      </div>
      <div className="font-mono text-[10px] tabular-nums text-ink-400">{axis.sigma.toFixed(2)}</div>

      <div className="ml-2 flex min-w-0 flex-1 flex-wrap items-center gap-1.5">
        <span className="text-ink-400">→</span>
        {dom.length === 0 ? (
          <span className="text-ink-500">no clear archetype</span>
        ) : (
          dom.map((p, i) => (
            <React.Fragment key={p.symbol}>
              {i > 0 && <span className="text-ink-500">+</span>}
              <span
                className="rounded-md border px-1.5 py-0.5 text-[10px] font-medium"
                style={{
                  color: colorMap[p.symbol] ?? "#cbd",
                  borderColor: (colorMap[p.symbol] ?? "#888") + "55",
                  background: (colorMap[p.symbol] ?? "#888") + "12",
                }}
                title={`alignment ${(p.alignment * 100).toFixed(0)}%`}
              >
                {p.symbol}
              </span>
            </React.Fragment>
          ))
        )}
        {movers.length > 0 && (
          <span className="ml-1 truncate text-ink-300">
            <span className="text-ink-500">·&nbsp;</span>
            {movers.slice(0, 4).map((m, i) => (
              <React.Fragment key={m.descriptor}>
                {i > 0 && <span className="text-ink-600">, </span>}
                <span style={{ color: colorMap[m.symbol] ?? "#cbd" }}>{m.descriptor}</span>
              </React.Fragment>
            ))}
          </span>
        )}
      </div>
    </div>
  );
}

// ───────────────────────── headline narration ─────────────────────────

function buildHeadline(
  data: ShiftSpectrumResponse,
  _colorMap: Record<string, string>,
): { kind: string; sentence: React.ReactNode } {
  const axes = data.axes;
  if (!axes.length) {
    return { kind: "no shift", sentence: "Context produces no measurable displacement." };
  }

  const a1 = axes[0];
  const a2 = axes[1];

  // Classification thresholds — tuned for typical cosine-similarity shifts.
  const dom = data.dominance_ratio;
  const er = data.effective_rank;

  // How many axes are "real" — count axes whose σ is above 30% of σ₁.
  const realAxes = axes.filter((a) => a.sigma >= 0.3 * a1.sigma).length;

  const a1Phrase = describeAxis(a1);
  const a2Phrase = a2 ? describeAxis(a2) : null;

  // Three regimes, picked by a combination of dominance and participation.
  if ((dom != null && dom >= 3.5) || er <= 1.25 || realAxes <= 1) {
    return {
      kind: "narrow · single-axis context",
      sentence: (
        <>
          One direction of pull — <Phrase {...a1Phrase} />.
        </>
      ),
    };
  }
  if ((dom != null && dom >= 1.6) || er <= 1.9) {
    return {
      kind: "primary axis + secondary",
      sentence: (
        <>
          Primary pull <Phrase {...a1Phrase} />
          {a2Phrase && (
            <>
              , with a secondary axis <Phrase {...a2Phrase} />
            </>
          )}
          .
        </>
      ),
    };
  }
  // Roughly equal top axes — polarizing.
  return {
    kind: "polarising · multi-axis context",
    sentence: (
      <>
        Two roughly equal directions of pull — <Phrase {...a1Phrase} /> and{" "}
        {a2Phrase && <Phrase {...a2Phrase} />}.
      </>
    ),
  };
}

function describeAxis(a: SpectrumAxis): {
  archetypes: { symbol: string; sign: number }[];
  movers: string[];
} {
  const dom = a.archetype_profile.filter((p) => Math.abs(p.alignment) >= 0.2).slice(0, 2);
  const sign = dom[0] ? Math.sign(dom[0].alignment) : 0;
  const movers = (sign >= 0 ? a.positive_movers : a.negative_movers)
    .slice(0, 3)
    .map((m) => m.descriptor);
  return {
    archetypes: dom.map((p) => ({ symbol: p.symbol, sign: Math.sign(p.alignment) })),
    movers,
  };
}

function Phrase({
  archetypes, movers,
}: {
  archetypes: { symbol: string; sign: number }[];
  movers: string[];
}) {
  const colorMap = useSidebar((s) => s.colorMap);
  return (
    <>
      toward{" "}
      {archetypes.length === 0 ? (
        <span className="text-ink-300">an unlabeled direction</span>
      ) : (
        archetypes.map((a, i) => (
          <React.Fragment key={a.symbol}>
            {i > 0 && <span className="text-ink-400"> + </span>}
            <strong style={{ color: colorMap[a.symbol] ?? "#cbd" }}>{a.symbol}</strong>
          </React.Fragment>
        ))
      )}
      {movers.length > 0 && (
        <>
          {" "}
          <span className="text-ink-400">({movers.join(", ")})</span>
        </>
      )}
    </>
  );
}
