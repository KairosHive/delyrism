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
  // dual-context (Alchemist mode) — A and B sentences are blended server-side
  // into a single override vector (slider drives `alchemistBlend`, 0=A, 1=B).
  // `alchemistActive` mirrors backend override state; `alchemistNonce` is
  // bumped on every blend change so dependent queries refetch.
  alchemistMode: boolean;
  contextSentenceB: string;
  alchemistBlend: number;
  alchemistActive: boolean;
  alchemistNonce: number;

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
  // When a context is active, colour descriptor dots by the 2D length of
  // their shift arrow (how strongly the context moved them).  Overrides
  // per-symbol palette colouring while on.
  pullHeatmap: boolean;

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
  deltaSign: "up" | "down" | "both"; // strengthens / weakens / both

  // ----- Story Generator -----
  // Persisted across tab switches so settings + generated story aren't lost
  // when the user flips back to Explorer.  All fields mirror the StoryRequest
  // payload sent to /story/generate.
  storyModel: string;
  storyTone: string;
  storyLanguage: "English" | "Français" | "Español";
  storyPov: "first" | "third";
  storyTense: "present" | "past" | "future";
  storyForm: "prose" | "short-story" | "poem" | "myth" | "incantation" | "vignette";
  storyLengthWords: number;
  storyTemperature: number;
  storyTopP: number;
  storyPositiveOnly: boolean;
  // Anchor archetype: "" → none, "auto" → top-ranked, "EARTH"/"WATER"/… → explicit.
  storyAnchor: string;
  // Motif controls — number of motif words and where they come from.
  storyMotifDensity: number;
  storyMotifSource: "delta-graph" | "top-attention" | "mixed" | "transformation" | "cycle";
  // Sub-modes for the topology-driven sources.  Only used when
  // storyMotifSource is the matching value.
  storyTransformationMode: "emergence" | "fading" | "becoming";
  storyCycleDim: "h1" | "h2";
  // Last-generated story and motifs (so they survive tab navigation).
  storyResult: { story: string; motifs: string[]; model: string; auto_target?: string | null } | null;
  storyError: string | null;

  // ----- audio context override -----
  // The backend stores a context vector on the cached space; we mirror just
  // enough state here to know it's active and to invalidate dependent queries.
  audioActive: boolean;
  audioNonce: number;        // bumped each time the override changes
  audioMaxSeconds: number;   // upload/record cap

  // ----- image context override (vision-LLM shim) -----
  // Same single-slot context_override on the backend — image and audio are
  // mutually exclusive.  The description is what the vision LLM "read" from
  // the image; surfaced in the UI for interpretability.
  imageActive: boolean;
  imageNonce: number;
  imageDescription: string;
  imageThumbnail: string | null;   // object-URL for the selected file

  // ----- topology tab — context overlay -----
  // When true, every Topology subview re-runs PH on the context-shifted
  // D' (using the same shift strategy + sentence as Explorer) instead of
  // the intrinsic space.D.  Lets the user toggle "what shape does this
  // archetype have?" vs "what shape does it take under THIS context?"
  topologyShowContext: boolean;

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
  alchemistBlend: 0.5,
  alchemistActive: false,
  alchemistNonce: 0,

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
  pullHeatmap: false,

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
  // default to strengthens-only — most users want the positive associations
  deltaSign: "up",

  topologyShowContext: false,

  selectedSymbol: null,

  audioActive: false,
  audioNonce: 0,
  audioMaxSeconds: 10,

  imageActive: false,
  imageNonce: 0,
  imageDescription: "",
  imageThumbnail: null,

  storyModel: "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
  storyTone: "dreamy",
  storyLanguage: "English",
  storyPov: "third",
  storyTense: "present",
  storyForm: "prose",
  storyLengthWords: 180,
  storyTemperature: 0.85,
  storyTopP: 0.9,
  storyPositiveOnly: true,
  storyAnchor: "",
  storyMotifDensity: 12,
  storyMotifSource: "delta-graph",
  storyTransformationMode: "becoming",
  storyCycleDim: "h1",
  storyResult: null,
  storyError: null,

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
