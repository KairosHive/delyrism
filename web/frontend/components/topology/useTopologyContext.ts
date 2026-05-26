"use client";
import { useSidebar, buildContextWeights } from "@/lib/store";

/**
 * Shared hook for every Topology subview.  Returns the payload + query-key
 * suffix to merge into each endpoint call, plus diagnostic flags for the
 * "context overlay" toggle in the tab header.
 *
 *   active   — toggle is on AND there's a context to apply
 *   payload  — fields to spread into the POST body (`use_context: true`
 *              + every shift parameter)
 *   keyTail  — array to spread into the TanStack queryKey so context
 *              changes invalidate the cache properly
 *   summary  — human-readable description of the active context for the
 *              header badge ("a quiet grief · gate · β=1.2")
 */
export function useTopologyContext() {
  const show = useSidebar((s) => s.topologyShowContext);
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

  const hasContext =
    !!sentence.trim() ||
    !!weights ||
    audioActive ||
    imageActive ||
    alchemistActive;

  const active = show && hasContext;

  const payload: Record<string, unknown> = active
    ? {
        use_context: true,
        sentence: sentence.trim() || null,
        weights,
        strategy,
        beta,
        gate,
        tau,
        within_symbol_softmax: wss,
        gamma,
        pool_type: poolType,
        pool_w: poolW,
        membership_alpha: mAlpha,
      }
    : {};

  const keyTail = active
    ? ["ctx", sentence, weights, strategy, beta, gate, tau, wss, gamma, poolType, poolW, mAlpha, audioNonce, imageNonce, alchemistNonce] as const
    : ["intrinsic"] as const;

  // Human label for the header badge
  let sourceLabel: string;
  if (alchemistActive) sourceLabel = "alchemist blend";
  else if (audioActive) sourceLabel = "audio override";
  else if (imageActive) sourceLabel = "image override";
  else if (sentence.trim()) {
    const s = sentence.trim();
    sourceLabel = `"${s.length > 38 ? s.slice(0, 38) + "…" : s}"`;
  } else if (weights) sourceLabel = "symbol weights";
  else sourceLabel = "—";

  const strategyLabel = `${strategy}${strategy === "gate" || strategy === "hybrid" ? "·" + gate : ""}`;
  const summary = `${sourceLabel} · ${strategyLabel} · β=${beta.toFixed(1)}`;

  return { show, active, hasContext, payload, keyTail, summary };
}
