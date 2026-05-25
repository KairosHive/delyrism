"use client";
import * as React from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api, ShiftSpectrumResponse, SpectrumAxis, SpectrumProfileEntry, SpectrumMoverEntry } from "@/lib/api";
import { useSidebar, buildContextWeights } from "@/lib/store";

/**
 * Shift spectrum — top-K principal axes of Δ = D' − D, computed via SVD.
 *
 * Complements the Rankings panel (which answers "where does this context
 * point?") with the orthogonal question: "how many independent ways is the
 * context rewriting the cloud, and which descriptors move along each axis?"
 *
 *   σ₁/σ₂ near 1   → multi-axial / polarizing context
 *   σ₁/σ₂ ≫ 1     → narrow / single-direction context
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
      <div className="mb-2 flex items-center justify-between gap-3">
        <div>
          <div className="section-title">Shift spectrum</div>
          <div className="text-[11px] text-ink-400">
            principal axes of Δ = D' − D · how many independent ways the context rewires the cloud
          </div>
        </div>
        {q.data && <HeaderBadges data={q.data} />}
      </div>

      {!hasCtx && (
        <div className="p-6 text-sm text-ink-300">
          Add a context (sentence, weights, audio, image, or alchemist blend) to compute the spectrum.
        </div>
      )}
      {hasCtx && q.isPending && (
        <div className="p-6 text-sm text-ink-300">computing…</div>
      )}
      {q.data && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          {q.data.axes.map((ax, i) => (
            <AxisCard key={i} idx={i} axis={ax} topSigma={q.data.axes[0]?.sigma ?? 1} colorMap={colorMap} />
          ))}
        </div>
      )}
    </div>
  );
}

function HeaderBadges({ data }: { data: ShiftSpectrumResponse }) {
  const dom = data.dominance_ratio;
  const er = data.effective_rank;
  // Heuristic colour: narrow = teal-ish, polarising = warm
  const domLabel =
    dom == null ? "—" :
    dom >= 4 ? "single-axis" :
    dom >= 1.8 ? "primary axis" :
    "multi-axis";
  return (
    <div className="flex items-center gap-1.5">
      <span
        className="pill !text-[10px]"
        title="σ₁/σ₂ — ratio of the largest two singular values.  High = a single direction dominates the shift.  ~1 = several independent rewriting axes."
      >
        σ₁/σ₂ {dom == null ? "—" : dom.toFixed(2)} · {domLabel}
      </span>
      <span
        className="pill !text-[10px]"
        title="Participation ratio of σ² values.  1 = one axis carries everything.  3 = three axes share the load equally."
      >
        eff. rank {er.toFixed(2)}
      </span>
    </div>
  );
}

function AxisCard({
  idx, axis, topSigma, colorMap,
}: {
  idx: number;
  axis: SpectrumAxis;
  topSigma: number;
  colorMap: Record<string, string>;
}) {
  // σ-bar — width proportional to this axis's σ relative to axis 1
  const pct = Math.max(2, Math.min(100, (axis.sigma / Math.max(topSigma, 1e-9)) * 100));
  return (
    <div className="rounded-lg border border-ink-700/70 bg-ink-900/50 p-3">
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <div className="text-[11px] font-medium uppercase tracking-wider text-ink-300">
          Axis {idx + 1}
        </div>
        <div className="font-mono text-[11px] text-ink-200">σ = {axis.sigma.toFixed(3)}</div>
      </div>
      <div className="mb-3 h-1.5 w-full overflow-hidden rounded-full bg-ink-800">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background: "linear-gradient(90deg, #3bbdb0 0%, #5fcfc4 100%)",
          }}
        />
      </div>

      <div className="mb-2">
        <div className="sub-title mb-1">Archetype profile</div>
        <ArchetypeProfile entries={axis.archetype_profile} colorMap={colorMap} />
      </div>

      <div className="grid grid-cols-2 gap-2">
        <MoverList title="+ movers" movers={axis.positive_movers} colorMap={colorMap} sign="+" />
        <MoverList title="− movers" movers={axis.negative_movers} colorMap={colorMap} sign="-" />
      </div>
    </div>
  );
}

function ArchetypeProfile({
  entries, colorMap,
}: {
  entries: SpectrumProfileEntry[];
  colorMap: Record<string, string>;
}) {
  // Render each archetype as a centered horizontal bar: positive alignment
  // extends right, negative extends left.  Width = |alignment|, max ~0.9.
  if (!entries.length) {
    return <div className="text-[11px] text-ink-500">—</div>;
  }
  return (
    <div className="space-y-1">
      {entries.map((e) => {
        const c = colorMap[e.symbol] ?? "#888";
        const abs = Math.min(1, Math.abs(e.alignment));
        const w = `${abs * 50}%`; // half the row, centered
        return (
          <div key={e.symbol} className="flex items-center gap-2 text-[11px]">
            <div className="w-16 truncate text-ink-200" style={{ color: c }}>{e.symbol}</div>
            <div className="relative h-3 flex-1 rounded bg-ink-800/70">
              <div className="absolute left-1/2 top-0 h-full w-px bg-ink-600/60" />
              {e.alignment >= 0 ? (
                <div
                  className="absolute left-1/2 top-0 h-full rounded-r"
                  style={{ width: w, background: c, opacity: 0.85 }}
                />
              ) : (
                <div
                  className="absolute right-1/2 top-0 h-full rounded-l"
                  style={{ width: w, background: c, opacity: 0.85 }}
                />
              )}
            </div>
            <div className="w-12 text-right font-mono text-ink-300">
              {(e.alignment * 100).toFixed(0)}%
            </div>
          </div>
        );
      })}
    </div>
  );
}

function MoverList({
  title, movers, colorMap, sign,
}: {
  title: string;
  movers: SpectrumMoverEntry[];
  colorMap: Record<string, string>;
  sign: "+" | "-";
}) {
  return (
    <div>
      <div className="sub-title mb-1">{title}</div>
      {movers.length === 0 ? (
        <div className="text-[11px] text-ink-500">—</div>
      ) : (
        <ul className="space-y-0.5">
          {movers.slice(0, 6).map((m) => {
            const c = colorMap[m.symbol] ?? "#888";
            return (
              <li key={m.descriptor + sign} className="flex items-center gap-1.5 text-[11px]">
                <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: c }} />
                <span className="truncate text-ink-200">{m.descriptor}</span>
                <span className="ml-auto shrink-0 font-mono text-[10px] text-ink-400">
                  {m.score.toFixed(2)}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
