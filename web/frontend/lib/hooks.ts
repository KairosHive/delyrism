// Query hooks shared across Explorer panels.  Each hook reads the relevant
// slice of sidebar state and only refires when those pieces actually change.

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api } from "./api";
import {
  ProposeResponse,
  AmbiguityResponse,
  Reduce2DResponse,
  ShiftResponse,
  DeltaGraphResponse,
  SubgraphResponse,
  AttentionResponse,
} from "./api";
import { useSidebar, buildContextWeights } from "./store";

const sentenceOrNull = (s: string) => (s.trim() ? s.trim() : null);

export function useRankings() {
  const sid = useSidebar((s) => s.spaceId);
  const sentence = useSidebar((s) => s.contextSentence);
  const weights = useSidebar(buildContextWeights);
  const tau = useSidebar((s) => s.tau);
  const alpha = useSidebar((s) => s.alpha);
  const lam = useSidebar((s) => s.lambda);
  const usePPR = useSidebar((s) => s.usePPR);
  const blindSpot = useSidebar((s) => s.blindSpot);
  const topk = useSidebar((s) => s.topk);
  const audioNonce = useSidebar((s) => s.audioNonce);
  const imageNonce = useSidebar((s) => s.imageNonce);
  const morphNonce = useSidebar((s) => s.morphNonce);

  return useQuery({
    enabled: !!sid,
    placeholderData: keepPreviousData,
    queryKey: ["propose", sid, sentence, weights, tau, alpha, lam, usePPR, blindSpot, topk, audioNonce, imageNonce, morphNonce],
    queryFn: () =>
      api.post<ProposeResponse>("/propose", {
        space_id: sid,
        sentence: sentenceOrNull(sentence),
        weights,
        tau, alpha, lam, use_ppr: usePPR, blind_spot: blindSpot, topk,
      }),
  });
}

export function useAmbiguity() {
  const sid = useSidebar((s) => s.spaceId);
  return useQuery({
    enabled: !!sid,
    queryKey: ["ambiguity", sid],
    queryFn: () => api.post<AmbiguityResponse>("/ambiguity", { space_id: sid, tau: 0.5, k: 10 }),
  });
}

export function useReduce2D() {
  const sid = useSidebar((s) => s.spaceId);
  const method = useSidebar((s) => s.reducer);
  const cent = useSidebar((s) => s.includeCentroids);
  const ncent = useSidebar((s) => s.normalizeCentroids);
  return useQuery({
    enabled: !!sid,
    placeholderData: keepPreviousData,
    queryKey: ["reduce2d", sid, method, cent, ncent],
    queryFn: () =>
      api.post<Reduce2DResponse>("/reduce-2d", {
        space_id: sid,
        method,
        include_centroids: cent,
        normalize_centroids: ncent,
      }),
  });
}

function shiftPayload() {
  const s = useSidebar.getState();
  return {
    space_id: s.spaceId,
    sentence: sentenceOrNull(s.contextSentence),
    weights: buildContextWeights(s),
    strategy: s.strategy,
    beta: s.beta,
    gate: s.gate,
    tau: s.shiftTau,
    within_symbol_softmax: s.withinSymbolSoftmax,
    gamma: s.gamma,
    prompt_template: "{sent}, {desc}",
    pool_type: s.poolType,
    pool_w: s.poolW,
    membership_alpha: s.membershipAlpha,
    reducer: s.reducer,
  } as const;
}

/** Build a full /delta-graph request payload from the current sidebar
 *  state.  Exported so the Story Generator can pass the SAME params into
 *  motif extraction — otherwise the words in the generated story come
 *  from a default-parametrized Δ-graph that doesn't match what the user
 *  sees in Explorer. */
export function deltaGraphPayload() {
  const s = useSidebar.getState();
  return {
    ...shiftPayload(),
    top_abs_edges: s.topAbsEdges,
    min_abs_delta: s.minAbsDelta,
    within_symbol: s.withinSymbolEdges,
    connected_only: s.connectedOnly,
    sym_filter: s.symbolFilter.length ? s.symbolFilter : null,
    only_symbol: null,
    sign_filter: s.deltaSign === "both" ? null : s.deltaSign,
  } as const;
}

export function useShift() {
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
  const reducer = useSidebar((s) => s.reducer);
  const audioActive = useSidebar((s) => s.audioActive);
  const imageActive = useSidebar((s) => s.imageActive);
  const morphActive = useSidebar((s) => s.morphActive);
  const audioNonce = useSidebar((s) => s.audioNonce);
  const imageNonce = useSidebar((s) => s.imageNonce);
  const morphNonce = useSidebar((s) => s.morphNonce);

  return useQuery({
    // any override (audio / image / morphing blend) counts as context — the
    // engine's ctx_vec() honors them even when sentence is empty
    enabled: !!sid && (!!sentence.trim() || !!weights || audioActive || imageActive || morphActive),
    placeholderData: keepPreviousData,
    queryKey: ["shift", sid, sentence, weights, strategy, beta, gate, tau, wss, gamma, poolType, poolW, mAlpha, reducer, audioNonce, imageNonce, morphNonce],
    queryFn: () => api.post<ShiftResponse>("/shift", shiftPayload()),
  });
}

export function useDeltaGraph() {
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
  const topAbs = useSidebar((s) => s.topAbsEdges);
  const minAbs = useSidebar((s) => s.minAbsDelta);
  const withinSym = useSidebar((s) => s.withinSymbolEdges);
  const conn = useSidebar((s) => s.connectedOnly);
  const symFilter = useSidebar((s) => s.symbolFilter);
  const deltaSign = useSidebar((s) => s.deltaSign);
  const audioActive = useSidebar((s) => s.audioActive);
  const imageActive = useSidebar((s) => s.imageActive);
  const morphActive = useSidebar((s) => s.morphActive);
  const audioNonce = useSidebar((s) => s.audioNonce);
  const imageNonce = useSidebar((s) => s.imageNonce);
  const morphNonce = useSidebar((s) => s.morphNonce);

  return useQuery({
    enabled: !!sid && (!!sentence.trim() || !!weights || audioActive || imageActive || morphActive),
    placeholderData: keepPreviousData,
    queryKey: ["delta-graph", sid, sentence, weights, strategy, beta, gate, tau, wss, gamma, poolType, poolW, mAlpha, topAbs, minAbs, withinSym, conn, symFilter, deltaSign, audioNonce, imageNonce, morphNonce],
    queryFn: () =>
      api.post<DeltaGraphResponse>("/delta-graph", {
        ...shiftPayload(),
        top_abs_edges: topAbs,
        min_abs_delta: minAbs,
        within_symbol: withinSym,
        connected_only: conn,
        sym_filter: symFilter.length ? symFilter : null,
        only_symbol: null,
        sign_filter: deltaSign === "both" ? null : deltaSign,
      }),
  });
}

export function useSubgraph() {
  const sid = useSidebar((s) => s.spaceId);
  const sentence = useSidebar((s) => s.contextSentence);
  const ts = useSidebar((s) => s.subTopSymbols);
  const td = useSidebar((s) => s.subTopDescriptors);
  const method = useSidebar((s) => s.subMethod);
  const alpha = useSidebar((s) => s.subAlpha);
  const tau = useSidebar((s) => s.subTau);
  const audioActive = useSidebar((s) => s.audioActive);
  const imageActive = useSidebar((s) => s.imageActive);
  const morphActive = useSidebar((s) => s.morphActive);
  const audioNonce = useSidebar((s) => s.audioNonce);
  const imageNonce = useSidebar((s) => s.imageNonce);
  const morphNonce = useSidebar((s) => s.morphNonce);
  return useQuery({
    // subgraph needs *some* form of context — sentence or any active override
    enabled: !!sid && (!!sentence.trim() || audioActive || imageActive || morphActive),
    placeholderData: keepPreviousData,
    queryKey: ["subgraph", sid, sentence, ts, td, method, alpha, tau, audioNonce, imageNonce, morphNonce],
    queryFn: () =>
      api.post<SubgraphResponse>("/subgraph", {
        space_id: sid,
        sentence: sentence.trim(),
        topk_symbols: ts,
        topk_desc: td,
        method,
        alpha,
        tau,
      }),
  });
}

export function useAttention(symbol: string | null) {
  const sid = useSidebar((s) => s.spaceId);
  const sentence = useSidebar((s) => s.contextSentence);
  const weights = useSidebar(buildContextWeights);
  const tau = useSidebar((s) => s.tau);
  const audioNonce = useSidebar((s) => s.audioNonce);
  const imageNonce = useSidebar((s) => s.imageNonce);
  const morphNonce = useSidebar((s) => s.morphNonce);
  return useQuery({
    enabled: !!sid && !!symbol,
    placeholderData: keepPreviousData,
    queryKey: ["attention", sid, symbol, sentence, weights, tau, audioNonce, imageNonce, morphNonce],
    queryFn: () =>
      api.post<AttentionResponse>("/attention", {
        space_id: sid,
        symbol,
        sentence: sentenceOrNull(sentence),
        weights,
        tau,
      }),
  });
}
