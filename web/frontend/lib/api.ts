// Thin fetch client for the FastAPI backend.
// All routes accept JSON bodies; responses are JSON.  Errors throw with the
// HTTP status + the detail string the backend returns.

// In production the FastAPI process serves both the API and the static
// export at the same origin, so BASE stays empty and fetch resolves to
// relative URLs.  In local dev the frontend (Next.js) runs on :3000 and
// the backend (uvicorn) on :8000, so we need an absolute base — Next.js
// inlines NODE_ENV at build time, which lets us pick the right default
// automatically without the developer setting any env var.
//
// We default to 127.0.0.1 (not localhost) because on Windows the browser
// may resolve `localhost` to ::1 (IPv6) while uvicorn binds to IPv4 only
// by default → ERR_CONNECTION_REFUSED.  127.0.0.1 is unambiguous.
//
// Override with NEXT_PUBLIC_API_BASE if the backend lives anywhere else.
const BASE =
  process.env.NEXT_PUBLIC_API_BASE
  || (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "");

// Per-route timing log so the UI can surface "what was slow on the last call".
// Keyed by path; updated on every successful request.
export interface Timing {
  serverMs: number;
  totalMs: number;
  at: number;
}
const _timings: Record<string, Timing> = {};
const _listeners = new Set<() => void>();
export function subscribeTimings(fn: () => void): () => void {
  _listeners.add(fn);
  // Wrap the delete (returns boolean) in a void-returning closure so this is
  // compatible with React.useEffect's cleanup-function contract under strict
  // TS (`next build` fails otherwise).
  return () => {
    _listeners.delete(fn);
  };
}
export function getTimings(): Record<string, Timing> {
  return _timings;
}
function recordTiming(path: string, res: Response, totalMs: number) {
  const hdr = res.headers.get("X-Server-Ms");
  const serverMs = hdr ? parseFloat(hdr) : NaN;
  _timings[path] = { serverMs: Number.isFinite(serverMs) ? serverMs : totalMs, totalMs, at: Date.now() };
  _listeners.forEach((fn) => fn());
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const t0 = performance.now();
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  recordTiming(path, res, performance.now() - t0);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = (j as any).detail ?? JSON.stringify(j);
    } catch {}
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) }),
  upload: async <T>(path: string, form: FormData): Promise<T> => {
    const t0 = performance.now();
    const res = await fetch(`${BASE}${path}`, { method: "POST", body: form });
    recordTiming(path, res, performance.now() - t0);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json() as Promise<T>;
  },
};

// ---------- typed helpers (mirror schemas.py) ----------

export type SymbolMap = Record<string, string[]>;

export interface EmbedderConfig {
  backend: string;
  model?: string | null;
  pooling?: "eos" | "mean" | "cls" | "last";
  default_instruction?: string | null;
  default_context?: string | null;
}

export interface SpaceConfig {
  symbols: SymbolMap;
  embedder: EmbedderConfig;
  descriptor_threshold?: number;
  contextual_embeddings?: boolean;
  palette?: string;
}

export interface SpaceCreateResponse {
  space_id: string;
  symbols: string[];
  descriptors: string[];
  owners: Record<string, string>;
  embedding_dim: number;
  color_map: Record<string, string>;
}

export interface ProposalRow {
  symbol: string;
  score: number;
  coherence: number;
  pagerank: number;
}

export interface ProposeResponse { rows: ProposalRow[]; }

export interface AttentionResponse {
  symbol: string;
  descriptors: string[];
  weights: number[];
}

export interface AmbiguityRow {
  symbol: string;
  dispersion: number;
  leakage: number;
  entropy: number;
}

export interface AmbiguityResponse { rows: AmbiguityRow[]; }

export interface Point2D {
  x: number; y: number; label: string; symbol: string;
  kind: "descriptor" | "centroid";
}

export interface Reduce2DResponse { points: Point2D[]; }

export interface ShiftArrow {
  descriptor: string; symbol: string;
  x0: number; y0: number; x1: number; y1: number;
}

export interface ShiftCentroid { symbol: string; x: number; y: number; }

export interface ShiftResponse { arrows: ShiftArrow[]; centroids: ShiftCentroid[]; }

export interface DeltaNode { id: string; symbol: string; color: string; }
export interface DeltaEdge {
  source: string; target: string; delta: number;
  sign: "up" | "down"; abs_delta: number;
}
export interface DeltaGraphResponse { nodes: DeltaNode[]; edges: DeltaEdge[]; }

export interface SubgraphNode {
  id: string; kind: "symbol" | "descriptor"; symbol: string;
  color: string; score: number;
}
export interface SubgraphEdge { source: string; target: string; weight: number; }
export interface SubgraphResponse { nodes: SubgraphNode[]; edges: SubgraphEdge[]; }

export interface BackendsResponse {
  embedders: { id: string; label: string; remote: boolean; audio: boolean; dim: number }[];
}

export interface SimilarityResponse {
  symbol: string;
  descriptors: string[];
  before: number[][];
  after: number[][];
  delta: number[][];
}

export interface SymbolSimilarityResponse {
  symbols: string[];
  before: number[][];
  after: number[][];
  delta: number[][];
}

export interface MigrationEntry {
  descriptor: string;
  from_archetype: string;
  to_archetype: string;
  sim_before_from: number;
  sim_before_to: number;
  sim_after_from: number;
  sim_after_to: number;
  score: number;
}

export interface IdentityEntry {
  descriptor: string;
  owner: string;            // home archetype (drives the chip colour)
  score: number;
}

export interface ArchetypeIdentityCard {
  symbol: string;
  before: IdentityEntry[];
  after: IdentityEntry[];
  emerged: string[];        // names of descriptors new in `after`
  faded: string[];          // names of descriptors that dropped out
}

export interface TransformationsResponse {
  migrations: MigrationEntry[];
  identities: ArchetypeIdentityCard[];
}

export interface StoryResponse {
  story: string;
  motifs: string[];
  model: string;
  /** When the backend auto-picked the target archetype (no anchor was
   *  set + motif_source is "transformation" or "cycle"), the chosen
   *  symbol is echoed back here so the UI can show "auto-picked: FIRE". */
  auto_target?: string | null;
}


// ───────────────────────── topology (persistent homology) ─────────────────────────

export interface TopologySummaryEntry {
  symbol: string;
  h0_cohesion: number;
  h0_outlier: number;
  h1_sum: number;
  h1_max: number;
  h1_count: number;
  h2_sum: number;
  h2_max: number;
  h2_count: number;
  topo_score: number;
}

export interface PCAPoint {
  word: string;
  symbol: string;
  x: number;
  y: number;
}

export interface SetQualityMetrics {
  coverage_h1: number;
  coverage_h2: number;
  richness_mean: number;
  cohesion_balance: number;       // [0, 1] higher = more even tightness
  separation_tightness: number;   // [0, ~2] higher = archetypes more distinct
  count_balance: number;          // [0, 1] higher = balanced descriptor counts
  focus: number;                  // (0, 1] higher = tighter per-symbol clouds
}

export interface TopologySummaryResponse {
  entries: TopologySummaryEntry[];
  points: PCAPoint[];
  ripser_available: boolean;
  set_quality: SetQualityMetrics | null;
}

export interface PersistencePoint {
  dim: 0 | 1 | 2;
  birth: number;
  death: number;
  is_infinite: boolean;
}

export interface PersistenceDiagramResponse {
  symbol: string;
  points: PersistencePoint[];
  max_finite_death: number;
  ripser_available: boolean;
}

export interface AllDiagramsEntry {
  symbol: string;
  points: PersistencePoint[];
  h0_finite: number;
  h1_total: number;
  h1_persistent: number;
  h2_total: number;
  h2_persistent: number;
  max_persistence_h1: number;
  max_persistence_h2: number;
}

export interface AllDiagramsResponse {
  entries: AllDiagramsEntry[];
  max_finite_death: number;
  ripser_available: boolean;
}

export interface CycleVertex {
  word: string;
  index: number;
  x: number;
  y: number;
  home_symbol?: string | null;
}

export interface PersistentCycle {
  dim: 1 | 2;
  birth: number;
  death: number;
  persistence: number;
  vertices: CycleVertex[];
}

export interface TopologyCyclesResponse {
  symbol: string;
  cycles: PersistentCycle[];
  descriptors: CycleVertex[];
  ripser_available: boolean;
}

export interface SynergyEntry {
  a: string;
  b: string;
  synergy_h1: number;
  synergy_h2: number;
  sum_h1_union: number;
  sum_h2_union: number;
}

export interface TopologySynergyResponse {
  symbols: string[];
  entries: SynergyEntry[];
  ripser_available: boolean;
}

export interface PairCycle {
  dim: 1 | 2;
  birth: number;
  death: number;
  persistence: number;
  mix: "pure_a" | "pure_b" | "mixed";
  cross_fraction: number;
  vertices: CycleVertex[];   // each carries home_symbol
}

export interface PairCyclesResponse {
  a: string;
  b: string;
  cycles: PairCycle[];
  descriptors: CycleVertex[];
  ripser_available: boolean;
}

export interface WordCatalystEntry {
  word: string;
  delta_h1: number;
  delta_h2: number;
  cycle_weight: number;
  composite: number;
}

export interface WordCatalystResponse {
  symbol: string;
  entries: WordCatalystEntry[];
  h1_baseline: number;
  h2_baseline: number;
  ripser_available: boolean;
}
