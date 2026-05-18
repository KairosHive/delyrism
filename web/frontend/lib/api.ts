// Thin fetch client for the FastAPI backend.
// All routes accept JSON bodies; responses are JSON.  Errors throw with the
// HTTP status + the detail string the backend returns.

// Empty BASE → fetch falls back to the current origin (production, where
// FastAPI serves the static export at the same host).  In local dev, set
// NEXT_PUBLIC_API_BASE=http://localhost:8000 to point at the standalone
// uvicorn server.
const BASE = process.env.NEXT_PUBLIC_API_BASE || "";

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

export interface StoryResponse {
  story: string;
  motifs: string[];
  model: string;
}
