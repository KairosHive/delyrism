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
    length_words: int = 180
    temperature: float = 0.85
    top_p: float = 0.9
    positive_delta_only: bool = True

    # delta graph params used to extract motifs
    delta_params: Optional[DeltaGraphRequest] = None


class StoryResponse(BaseModel):
    story: str
    motifs: List[str]
    model: str
