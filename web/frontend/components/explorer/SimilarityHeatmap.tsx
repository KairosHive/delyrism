"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Plot } from "../plots/Plot";
import { api, SimilarityResponse, SymbolSimilarityResponse } from "@/lib/api";
import { useSidebar, buildContextWeights } from "@/lib/store";

/**
 * Δ-similarity heatmap with two modes:
 *   • "Within symbol"  — per-symbol descriptor × descriptor Δ (original view)
 *   • "Between symbols" — single symbol × symbol Δ over centroid cosines,
 *     answering "does context make symbol A look more like symbol B?"
 *
 * Both views use the same context plumbing (sentence / weights / audio / image
 * overrides + the same shift strategy params).
 */
type Mode = "within" | "between";

export function SimilarityHeatmap() {
  const symbols = useSidebar((s) => s.symbols);
  const sid = useSidebar((s) => s.spaceId);
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
  const audioNonce = useSidebar((s) => s.audioNonce);
  const imageActive = useSidebar((s) => s.imageActive);
  const imageNonce = useSidebar((s) => s.imageNonce);

  const [mode, setMode] = React.useState<Mode>("within");
  const [symbol, setSymbol] = React.useState<string>("");
  React.useEffect(() => {
    if (!symbol && symbols.length) setSymbol(symbols[0]);
  }, [symbols, symbol]);

  const hasContext = !!sentence.trim() || !!weights || audioActive || imageActive;

  const commonBody = {
    space_id: sid,
    sentence: sentence.trim() || null,
    weights,
    strategy, beta, gate, tau,
    within_symbol_softmax: wss,
    gamma,
    pool_type: poolType,
    pool_w: poolW,
    membership_alpha: mAlpha,
  } as const;

  const within = useQuery({
    enabled: !!sid && !!symbol && hasContext && mode === "within",
    queryKey: ["similarity", sid, symbol, sentence, weights, strategy, beta, gate, tau, wss, gamma, poolType, poolW, mAlpha, audioNonce, imageNonce],
    queryFn: () =>
      api.post<SimilarityResponse>("/similarity", {
        ...commonBody,
        symbol,
        order_by_attention: true,
      }),
  });

  const between = useQuery({
    enabled: !!sid && hasContext && mode === "between",
    queryKey: ["similarity-symbols", sid, sentence, weights, strategy, beta, gate, tau, wss, gamma, poolType, poolW, mAlpha, audioNonce, imageNonce],
    queryFn: () =>
      api.post<SymbolSimilarityResponse>("/similarity-symbols", commonBody),
  });

  const q = mode === "within" ? within : between;
  const labels = mode === "within" ? within.data?.descriptors : between.data?.symbols;
  const delta = q.data?.delta;

  return (
    <div className="panel-tight">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="section-title">
            {mode === "within"
              ? "Within-symbol associative increase (Δ)"
              : "Between-symbol centroid drift (Δ)"}
          </div>
          <div className="text-[11px] text-ink-400">
            {mode === "within"
              ? "descriptor × descriptor similarity, after − before. red = strengthens · blue = weakens."
              : "symbol × symbol centroid-cosine, after − before. red = symbols pull together · blue = symbols pull apart."}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="inline-flex overflow-hidden rounded-md border border-ink-700">
            <button
              className={`px-2.5 py-1 text-[11px] transition ${
                mode === "within" ? "bg-accent-600/30 text-ink-50" : "text-ink-300 hover:bg-ink-800"
              }`}
              onClick={() => setMode("within")}
            >
              within symbol
            </button>
            <button
              className={`px-2.5 py-1 text-[11px] transition ${
                mode === "between" ? "bg-accent-600/30 text-ink-50" : "text-ink-300 hover:bg-ink-800"
              }`}
              onClick={() => setMode("between")}
            >
              between symbols
            </button>
          </div>
          {mode === "within" && (
            <select
              className="select-base !w-auto !min-w-[160px]"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
            >
              {symbols.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          )}
        </div>
      </div>

      {!hasContext && (
        <div className="p-6 text-sm text-ink-300">
          Add a context (sentence, symbol weights, audio, or image) to compute the Δ matrix.
        </div>
      )}
      {q.isPending && hasContext && (
        <div className="p-6 text-sm text-ink-300">computing…</div>
      )}
      {delta && labels && (() => {
        const n = labels.length;
        const masked = delta.map((row, i) =>
          row.map((v, j) => (i === j ? null : v)),
        );
        const flat = delta.flat().filter((v, idx) => idx % (n + 1) !== 0);
        const lo = Math.min(...flat);
        const hi = Math.max(...flat);
        return (
          <Plot
            data={[
              {
                type: "heatmap",
                z: masked,
                x: labels,
                y: labels,
                colorscale: [
                  [0.0,  "#3a86ff"],
                  [0.25, "#06d6a0"],
                  [0.5,  "#ffd166"],
                  [0.75, "#f77f00"],
                  [1.0,  "#ef476f"],
                ],
                zmin: lo,
                zmax: hi,
                colorbar: { title: { text: "Δ", side: "right" }, thickness: 12, len: 0.8 },
                hovertemplate: "%{y} ↔ %{x}<br>Δ = %{z:.3f}<extra></extra>",
              },
            ]}
            layout={{
              autosize: true,
              height: Math.max(440, 22 * n + 110),
              margin: { l: 110, r: 24, t: 6, b: 110 },
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              font: { color: "#cad4e0", family: "Inter, system-ui" },
              xaxis: { tickangle: -55, tickfont: { size: 10 }, showgrid: false, ticks: "",
                       scaleanchor: "y", constrain: "domain" },
              yaxis: { autorange: "reversed", tickfont: { size: 10 }, showgrid: false, ticks: "",
                       constrain: "domain" },
            }}
            useResizeHandler
            style={{ width: "100%" }}
            config={{ displaylogo: false, responsive: true }}
          />
        );
      })()}
    </div>
  );
}
