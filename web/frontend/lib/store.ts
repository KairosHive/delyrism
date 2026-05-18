// Single Zustand store for all sidebar / control state.
// Updates here trigger TanStack Query refetches via the keys we derive from
// these fields.  Keep the schema close to the FastAPI ShiftRequest /
// ProposeRequest / Reduce2DRequest payloads.

import { create } from "zustand";

export type Reducer2D = "umap" | "tsne" | "pca";
export type Strategy = "gate" | "reembed" | "hybrid" | "pooling";
export type Gate = "relu" | "cos" | "softmax" | "uniform";

export interface SidebarState {
  // identity (set after POST /spaces)
  spaceId: string | null;
  symbols: string[];
  colorMap: Record<string, string>;

  // ----- Symbolic structure -----
  presetName: string | null;
  symbolMapJson: string; // editable text

  // ----- Context Options -----
  contextSentence: string;
  selectedContextSymbols: string[];
  symbolWeights: Record<string, number>;
  // dual-context (Alchemist mode)
  alchemistMode: boolean;
  contextSentenceB: string;

  // ----- Embedding Model -----
  embedderBackend: string;
  embedderModel: string; // optional override
  embedderPooling: "eos" | "mean" | "cls" | "last";
  qwenInstruction: string;
  qwenContextMode: "none" | "global" | "per-descriptor";
  qwenGlobalContext: string;

  // descriptor graph
  descriptorThreshold: number;

  // ----- Semantic Map -----
  reducer: Reducer2D;
  drawHulls: boolean;
  includeCentroids: boolean;
  normalizeCentroids: boolean;
  showArrows: boolean;

  // ----- Ranking -----
  tau: number;
  alpha: number;
  lambda: number;
  usePPR: boolean;
  blindSpot: boolean;
  topk: number;

  // ----- Contextual Subgraph -----
  subTopSymbols: number;
  subTopDescriptors: number;
  subMethod: "ppr" | "softmax";
  subTau: number;
  subAlpha: number;
  subThreshold: number;

  // ----- Δ Graph -----
  strategy: Strategy;
  gate: Gate;
  beta: number;
  gamma: number;
  poolType: "avg" | "max" | "min";
  poolW: number;
  membershipAlpha: number;
  shiftTau: number;
  withinSymbolSoftmax: boolean;

  topAbsEdges: number;
  minAbsDelta: number;
  withinSymbolEdges: boolean;
  connectedOnly: boolean;
  symbolFilter: string[];

  // ----- audio context override -----
  // The backend stores a context vector on the cached space; we mirror just
  // enough state here to know it's active and to invalidate dependent queries.
  audioActive: boolean;
  audioNonce: number;        // bumped each time the override changes
  audioMaxSeconds: number;   // upload/record cap

  // ----- selected drill-down -----
  selectedSymbol: string | null;

  // ----- setter (generic) -----
  set: <K extends keyof SidebarState>(k: K, v: SidebarState[K]) => void;
  setBulk: (patch: Partial<SidebarState>) => void;
  setWeight: (symbol: string, value: number) => void;
}

export const useSidebar = create<SidebarState>((set) => ({
  spaceId: null,
  symbols: [],
  colorMap: {},

  presetName: "elements",
  symbolMapJson: "",

  contextSentence: "",
  selectedContextSymbols: [],
  symbolWeights: {},
  alchemistMode: false,
  contextSentenceB: "",

  embedderBackend: "cloudflare-bge-m3",
  embedderModel: "",
  embedderPooling: "eos",
  qwenInstruction: "",
  qwenContextMode: "none",
  qwenGlobalContext: "",

  descriptorThreshold: 0.2,

  reducer: "umap",
  drawHulls: true,
  includeCentroids: true,
  normalizeCentroids: false,
  showArrows: true,

  tau: 0.3,
  alpha: 0.8,
  lambda: 0.6,
  usePPR: true,
  blindSpot: false,
  topk: 20,

  subTopSymbols: 3,
  subTopDescriptors: 3,
  subMethod: "ppr",
  subTau: 0.1,
  subAlpha: 0.85,
  subThreshold: 0.1,

  strategy: "gate",
  gate: "relu",
  beta: 1.2,
  gamma: 0.5,
  poolType: "avg",
  poolW: 0.7,
  membershipAlpha: 0.0,
  shiftTau: 0.3,
  withinSymbolSoftmax: true,

  topAbsEdges: 30,
  minAbsDelta: 0.01,
  withinSymbolEdges: false,
  connectedOnly: true,
  symbolFilter: [],

  selectedSymbol: null,

  audioActive: false,
  audioNonce: 0,
  audioMaxSeconds: 10,

  set: (k, v) => set({ [k]: v } as any),
  setBulk: (patch) => set(patch as any),
  setWeight: (symbol, value) =>
    set((s) => ({ symbolWeights: { ...s.symbolWeights, [symbol]: value } })),
}));

// helpers to build request payloads from state ------------------

export function buildContextWeights(s: SidebarState) {
  const w: Record<string, number> = {};
  for (const sym of s.selectedContextSymbols) {
    const v = s.symbolWeights[sym];
    if (typeof v === "number" && v > 0) w[sym] = v;
  }
  return Object.keys(w).length ? w : undefined;
}
