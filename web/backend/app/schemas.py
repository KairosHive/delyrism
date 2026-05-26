"""Pydantic schemas for the Delyrism HTTP API.

Mirrors the kwargs accepted by the underlying engine in delyrism/delyrism.py so
clients can pass them through with no translation.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field


# ----------------------- shared types -----------------------

SymbolMap = Dict[str, List[str]]


class EmbedderConfig(BaseModel):
    backend: str = "cloudflare-bge-m3"
    model: Optional[str] = None
    pooling: Literal["eos", "mean", "cls", "last"] = "eos"
    default_instruction: Optional[str] = None
    default_context: Optional[str] = None


class SpaceConfig(BaseModel):
    """Identifies a SymbolSpace on the server. The server caches the instance
    keyed by the canonical hash of (symbols, embedder, descriptor_threshold)."""

    symbols: SymbolMap
    embedder: EmbedderConfig
    descriptor_threshold: float = 0.2
    contextual_embeddings: bool = False
    palette: str = "AuroraPop"  # one of CHIC_PALETTES in delyrism/delyrism.py


class SpaceCreateResponse(BaseModel):
    space_id: str
    symbols: List[str]
    descriptors: List[str]
    owners: Dict[str, str]
    embedding_dim: int
    color_map: Dict[str, str]


# ----------------------- proposal -----------------------

class ProposeRequest(BaseModel):
    space_id: str
    sentence: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    topk: int = 5
    tau: float = 0.3
    lam: float = 0.6
    alpha: float = 0.85
    use_ppr: bool = True
    blind_spot: bool = False


class ProposalRow(BaseModel):
    symbol: str
    score: float
    coherence: float
    pagerank: float


class ProposeResponse(BaseModel):
    rows: List[ProposalRow]


# ----------------------- attention -----------------------

class AttentionRequest(BaseModel):
    space_id: str
    symbol: str
    sentence: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    tau: float = 0.3


class AttentionResponse(BaseModel):
    symbol: str
    descriptors: List[str]
    weights: List[float]


# ----------------------- ambiguity -----------------------

class AmbiguityRequest(BaseModel):
    space_id: str
    tau: float = 0.5
    k: int = 10


class AmbiguityRow(BaseModel):
    symbol: str
    dispersion: float
    leakage: float
    entropy: float


class AmbiguityResponse(BaseModel):
    rows: List[AmbiguityRow]


# ----------------------- 2d projection -----------------------

class Reduce2DRequest(BaseModel):
    space_id: str
    method: Literal["umap", "tsne", "pca", "auto"] = "umap"
    include_centroids: bool = True
    normalize_centroids: bool = False


class Point2D(BaseModel):
    x: float
    y: float
    label: str
    symbol: str
    kind: Literal["descriptor", "centroid"]


class Reduce2DResponse(BaseModel):
    points: List[Point2D]


# ----------------------- shifted matrix / arrows -----------------------

class ShiftRequest(BaseModel):
    space_id: str
    sentence: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    strategy: Literal["gate", "reembed", "hybrid", "pooling"] = "gate"
    beta: float = 0.6
    gate: Literal["relu", "cos", "softmax", "uniform"] = "relu"
    tau: float = 0.3
    within_symbol_softmax: bool = False
    gamma: float = 0.5
    prompt_template: str = "{sent}, {desc}"
    pool_type: Literal["avg", "max", "min"] = "avg"
    pool_w: float = 0.7
    membership_alpha: float = 0.0
    # Reducer used to project the arrow endpoints — must match whatever the
    # client is rendering in MeaningSpace so the arrows live in the same frame.
    reducer: Literal["umap", "tsne", "pca"] = "umap"


class ShiftArrow(BaseModel):
    descriptor: str
    symbol: str
    x0: float
    y0: float
    x1: float
    y1: float


class ShiftCentroid(BaseModel):
    symbol: str
    x: float
    y: float


class ShiftResponse(BaseModel):
    arrows: List[ShiftArrow]
    centroids: List[ShiftCentroid] = []


# ----------------------- delta graph -----------------------

class DeltaGraphRequest(ShiftRequest):
    top_abs_edges: int = 30
    min_abs_delta: float = 0.01
    within_symbol: bool = False
    only_symbol: Optional[str] = None
    sym_filter: Optional[List[str]] = None
    connected_only: bool = True
    # "up" → strengthens only, "down" → weakens only, None → both
    sign_filter: Optional[Literal["up", "down"]] = None


class DeltaNode(BaseModel):
    id: str
    symbol: str
    color: str


class DeltaEdge(BaseModel):
    source: str
    target: str
    delta: float
    sign: Literal["up", "down"]
    abs_delta: float


class DeltaGraphResponse(BaseModel):
    nodes: List[DeltaNode]
    edges: List[DeltaEdge]


# ----------------------- per-symbol similarity matrices -----------------------

class SimilarityRequest(ShiftRequest):
    symbol: str
    order_by_attention: bool = True


class SimilarityResponse(BaseModel):
    symbol: str
    descriptors: List[str]
    before: List[List[float]]
    after: List[List[float]]
    delta: List[List[float]]


class SymbolSimilarityRequest(ShiftRequest):
    """Same context inputs as the per-symbol matrix; no `symbol` field needed —
    the response is a single symbol×symbol matrix."""
    pass


class SymbolSimilarityResponse(BaseModel):
    symbols: List[str]
    before: List[List[float]]
    after: List[List[float]]
    delta: List[List[float]]


# ---------------- Contextual transformations (migrations + identity) ----------------

class TransformationsRequest(ShiftRequest):
    # How many top descriptors per archetype to show on each side of an
    # identity card.  6 is a good fit for the typical UI width.
    topk: int = 6
    # Migrations below this combined score are filtered out as noise.
    min_migration_score: float = 0.04


class MigrationEntry(BaseModel):
    descriptor: str
    from_archetype: str
    to_archetype: str
    # Cosines to original centroids — diagnostic for the migration.
    sim_before_from: float
    sim_before_to: float
    sim_after_from: float
    sim_after_to: float
    # Gain at destination + loss at source.
    score: float


class IdentityEntry(BaseModel):
    descriptor: str
    owner: str  # original archetype — drives the color in the UI
    score: float


class ArchetypeIdentityCard(BaseModel):
    symbol: str
    before: List[IdentityEntry]   # original top-K, sorted by cosine to fixed centroid
    after: List[IdentityEntry]    # top-K under context (foreign descriptors allowed)
    emerged: List[str]            # descriptors in `after` but not in `before`
    faded: List[str]              # descriptors in `before` but not in `after`


class TransformationsResponse(BaseModel):
    migrations: List[MigrationEntry]
    identities: List[ArchetypeIdentityCard]


# ----------------------- contextual subgraph -----------------------

class SubgraphRequest(BaseModel):
    space_id: str
    sentence: str
    topk_symbols: int = 3
    topk_desc: int = 3
    method: Literal["ppr", "softmax"] = "ppr"
    alpha: float = 0.85
    tau: float = 0.1


class SubgraphNode(BaseModel):
    id: str
    kind: Literal["symbol", "descriptor"]
    symbol: str
    color: str
    score: float


class SubgraphEdge(BaseModel):
    source: str
    target: str
    weight: float


class SubgraphResponse(BaseModel):
    nodes: List[SubgraphNode]
    edges: List[SubgraphEdge]


# ----------------------- context audio -----------------------

class EncodeAudioResponse(BaseModel):
    vector: List[float]
    dim: int


class EncodeImageResponse(BaseModel):
    vector: List[float]
    dim: int
    # The vision-LLM's reading of the image — exposed to the UI so the user
    # can see what symbolic field the engine "saw" in their image.
    description: str
    # Which CF vision model produced the description.
    model: str


# ----------------------- story generator -----------------------

class StoryRequest(BaseModel):
    space_id: str
    sentence: Optional[str] = None
    weights: Optional[Dict[str, float]] = None

    provider: Literal["cloudflare", "local"] = "cloudflare"
    model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"

    tone: str = "dreamy"
    language: Literal["English", "Français", "Español"] = "English"
    pov: Literal["first", "third"] = "third"
    tense: Literal["present", "past", "future"] = "present"
    # Output form / shape — separate from tone (which is register).
    form: Literal["prose", "short-story", "poem", "myth", "incantation", "vignette"] = "prose"
    length_words: int = 180
    temperature: float = 0.85
    top_p: float = 0.9
    positive_delta_only: bool = True

    # ---- explorer-aware story controls ----
    # Anchor archetype — pin one symbol as the story's center.
    #   None / "" → no anchor
    #   "auto"    → backend picks the top-ranked symbol via propose()
    #   "EARTH"   → use that specific symbol
    anchor_archetype: Optional[str] = None
    # How many motif words to weave explicitly into the prompt.
    motif_density: int = 12
    # Where motifs come from:
    #   "delta-graph"   — words from the Δ-graph (current behavior)
    #   "top-attention" — top-attended descriptors of the anchor (or top-ranked
    #                     symbol if no anchor), via conditioned_symbol()
    #   "mixed"         — half-and-half, deduped
    motif_source: Literal["delta-graph", "top-attention", "mixed"] = "delta-graph"

    # delta graph params used to extract motifs
    delta_params: Optional[DeltaGraphRequest] = None


class StoryResponse(BaseModel):
    story: str
    motifs: List[str]
    model: str


# ----------------------- topology (persistent homology) -----------------------

class TopologySummaryEntry(BaseModel):
    symbol: str
    h0_cohesion: float          # median H0 persistence — lower = tighter cluster
    h0_outlier: float           # max H0 persistence — biggest single-point lifetime
    h1_sum: float               # total H1 persistence — loopiness
    h1_max: float
    h1_count: int               # number of H1 cycles with persistence > thr
    h2_sum: float               # total H2 persistence — voidiness
    h2_max: float
    h2_count: int
    topo_score: float           # z-score composite: H1_z + H2_z − H0_cohesion_z


class SetQualityMetrics(BaseModel):
    """Set-level metrics for evaluating an archetype design as a whole.
    Higher generally = better-shaped set, but each metric is interpretable
    on its own so the user can tune their archetype mapping with intent.

    Comparing intrinsic vs context-shifted values tells you whether
    applying context tightens / loosens / reshapes the archetypal field.
    """
    # H1 / H2 mass on the UNION of all descriptors — does the set as a
    # whole have loop / void structure (covers the manifold) or is it
    # just a single blob?
    coverage_h1: float
    coverage_h2: float
    # Mean per-archetype "internal richness" (count of persistent H1+H2
    # features above noise).  Higher = archetypes are multi-faceted, not
    # synonym clusters.
    richness_mean: float
    # 1 − std(H0_cohesion)/mean(H0_cohesion) across archetypes.  Higher
    # = archetypes are similarly tight ("even design"); lower = some
    # archetypes are tight, others diffuse.
    cohesion_balance: float
    # Mean pairwise cosine distance between archetype centroids.  Higher
    # = archetypes occupy distinct regions of the semantic space.
    separation_tightness: float
    # Shannon entropy of descriptor-count distribution across archetypes,
    # normalised to [0, 1] by log2(S).  1 = perfectly balanced counts;
    # 0 = one archetype hoards everything.
    count_balance: float


class TopologySummaryResponse(BaseModel):
    entries: List[TopologySummaryEntry]
    # Joint PCA-2D of all descriptors across all symbols — used by the
    # overview map so every symbol's cloud lives in the same frame.
    points: List["PCAPoint"]
    ripser_available: bool
    # Set-level quality scalars (None if too few symbols / ripser missing)
    set_quality: Optional[SetQualityMetrics] = None


class PCAPoint(BaseModel):
    word: str
    symbol: str
    x: float
    y: float


class PersistencePoint(BaseModel):
    dim: int                    # 0 / 1 / 2
    birth: float
    death: float                # infinity encoded as the diagram's max+1 for plotting
    is_infinite: bool


class PersistenceDiagramResponse(BaseModel):
    symbol: str
    points: List[PersistencePoint]
    max_finite_death: float     # for setting axis limits client-side
    ripser_available: bool


class CycleVertex(BaseModel):
    word: str
    index: int                  # index inside the symbol's descriptor list
    x: float                    # PCA-2D coords for plotting
    y: float
    home_symbol: Optional[str] = None  # only used for pair cycles; None = same as card


class PersistentCycle(BaseModel):
    dim: int                    # 1 or 2
    birth: float
    death: float
    persistence: float
    vertices: List[CycleVertex] # ordered (forms a loop for H1)


class TopologyCyclesResponse(BaseModel):
    symbol: str
    cycles: List[PersistentCycle]
    # ALL of this symbol's descriptors, projected — for drawing context
    # behind the active cycle.
    descriptors: List[CycleVertex]
    ripser_available: bool


class SynergyEntry(BaseModel):
    a: str
    b: str
    synergy_h1: float           # how much H1 mass exists only when A∪B is connected
    synergy_h2: float
    sum_h1_union: float
    sum_h2_union: float


class TopologySynergyResponse(BaseModel):
    symbols: List[str]
    entries: List[SynergyEntry]
    ripser_available: bool


class PairCycle(BaseModel):
    dim: int
    birth: float
    death: float
    persistence: float
    mix: Literal["pure_a", "pure_b", "mixed"]
    cross_fraction: float       # fraction of cycle edges/triangles spanning A↔B
    vertices: List[CycleVertex] # home_symbol filled in for each vertex


class PairCyclesResponse(BaseModel):
    a: str
    b: str
    cycles: List[PairCycle]
    descriptors: List[CycleVertex]  # all A + B points, home_symbol on each
    ripser_available: bool


class WordCatalystEntry(BaseModel):
    word: str
    delta_h1: float             # how much removing this word drops H1_sum
    delta_h2: float
    cycle_weight: float         # vertex-credit from cycle-participation
    composite: float            # delta_h1 + delta_h2 + 0.5 * cycle_weight


class WordCatalystResponse(BaseModel):
    symbol: str
    entries: List[WordCatalystEntry]
    h1_baseline: float
    h2_baseline: float
    ripser_available: bool


class AllDiagramsEntry(BaseModel):
    symbol: str
    points: List[PersistencePoint]
    # Roll-ups for the per-symbol mini-card title and auto-narration.
    h0_finite: int
    h1_total: int
    h1_persistent: int          # H1 features with persistence > 0.02
    h2_total: int
    h2_persistent: int
    max_persistence_h1: float
    max_persistence_h2: float


class AllDiagramsResponse(BaseModel):
    entries: List[AllDiagramsEntry]
    # Shared max for the y-axis so every mini-diagram lives in the same
    # frame — comparison-ready out of the box.
    max_finite_death: float
    ripser_available: bool
