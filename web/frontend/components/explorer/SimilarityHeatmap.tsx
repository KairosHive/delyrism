"use client";
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Plot } from "../plots/Plot";
import { api, SimilarityResponse } from "@/lib/api";
import { useSidebar, buildContextWeights } from "@/lib/store";

/**
 * Within-Symbol Associative Increase (Δ) — one heatmap of (after − before)
 * descriptor-descriptor cosine similarity, per selected symbol.  Mirrors the
 * single matrix the Streamlit app showed at the bottom of the Explorer.
 */
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

  const [symbol, setSymbol] = React.useState<string>("");
  React.useEffect(() => {
    if (!symbol && symbols.length) setSymbol(symbols[0]);
  }, [symbols, symbol]);

  const hasContext = !!sentence.trim() || !!weights || audioActive || imageActive;

  const q = useQuery({
    enabled: !!sid && !!symbol && hasContext,
    queryKey: ["similarity", sid, symbol, sentence, weights, strategy, beta, gate, tau, wss, gamma, poolType, poolW, mAlpha, audioNonce, imageNonce],
    queryFn: () =>
      api.post<SimilarityResponse>("/similarity", {
        space_id: sid,
        symbol,
        sentence: sentence.trim() || null,
        weights,
        strategy, beta, gate, tau,
        within_symbol_softmax: wss,
        gamma,
        pool_type: poolType,
        pool_w: poolW,
        membership_alpha: mAlpha,
        order_by_attention: true,
      }),
  });

  return (
    <div className="panel-tight">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div>
          <div className="section-title">Within-symbol associative increase (Δ)</div>
          <div className="text-[11px] text-ink-400">
            descriptor × descriptor similarity, after − before. red = strengthens · blue = weakens.
          </div>
        </div>
        <select
          className="select-base !w-auto !min-w-[160px]"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
        >
          {symbols.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      {!hasContext && (
        <div className="p-6 text-sm text-ink-300">
          Add a context (sentence, symbol weights, or audio) to compute the Δ matrix.
        </div>
      )}
      {q.isPending && hasContext && (
        <div className="p-6 text-sm text-ink-300">computing…</div>
      )}
      {q.data && (() => {
        // Multi-hue Spectral_r-style scale (matches the old Streamlit look):
        // low/negative = blue, mid = green/yellow, high/positive = red.
        // We keep the diagonal masked (NaN) so it appears as plot bg.
        const n = q.data.descriptors.length;
        const masked = q.data.delta.map((row, i) =>
          row.map((v, j) => (i === j ? null : v)),
        );
        const flat = q.data.delta.flat().filter((v, idx) => idx % (n + 1) !== 0);
        const lo = Math.min(...flat);
        const hi = Math.max(...flat);
        return (
          <Plot
            data={[
              {
                type: "heatmap",
                z: masked,
                x: q.data.descriptors,
                y: q.data.descriptors,
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
              // scaleanchor pins x-units to y-units so each cell is a square
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
