"""Ranking, attention, ambiguity, and 2D projection endpoints."""
from __future__ import annotations
import numpy as np
from fastapi import APIRouter, HTTPException

from ..schemas import (
    ProposeRequest, ProposeResponse, ProposalRow,
    AttentionRequest, AttentionResponse,
    AmbiguityRequest, AmbiguityResponse, AmbiguityRow,
    Reduce2DRequest, Reduce2DResponse, Point2D,
)
from .. import engine_cache

router = APIRouter(tags=["analysis"])


def _require(space_id: str):
    space = engine_cache.get_space(space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")
    return space


@router.post("/propose", response_model=ProposeResponse)
def propose(req: ProposeRequest) -> ProposeResponse:
    space = _require(req.space_id)
    key = engine_cache.memo_key(req.space_id, "propose", req.model_dump(exclude={"space_id"}))
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    rows_raw = space.propose(
        weights=req.weights,
        sentence=req.sentence,
        topk=req.topk,
        tau=req.tau,
        lam=req.lam,
        alpha=req.alpha,
        use_ppr=req.use_ppr,
        blind_spot=req.blind_spot,
    )
    out = ProposeResponse(rows=[
        ProposalRow(symbol=s, score=float(sc), coherence=float(c), pagerank=float(p))
        for (s, sc, c, p) in rows_raw
    ])
    engine_cache.memo_put(key, out)
    return out


@router.post("/attention", response_model=AttentionResponse)
def attention(req: AttentionRequest) -> AttentionResponse:
    space = _require(req.space_id)
    if req.symbol not in space.symbol_to_idx:
        raise HTTPException(status_code=400, detail=f"unknown symbol '{req.symbol}'")
    _, attn = space.conditioned_symbol(
        symbol=req.symbol,
        weights=req.weights,
        sentence=req.sentence,
        tau=req.tau,
    )
    descs = space.symbols_to_descriptors[req.symbol]
    weights = [float(attn[d]) for d in descs]
    return AttentionResponse(symbol=req.symbol, descriptors=descs, weights=weights)


@router.post("/ambiguity", response_model=AmbiguityResponse)
def ambiguity(req: AmbiguityRequest) -> AmbiguityResponse:
    space = _require(req.space_id)
    key = engine_cache.memo_key(req.space_id, "ambiguity", req.model_dump(exclude={"space_id"}))
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached
    rows = []
    for s in space.symbols:
        rows.append(AmbiguityRow(
            symbol=s,
            dispersion=float(space.dispersion(s)),
            leakage=float(space.leakage(s, k=req.k)),
            entropy=float(space.soft_entropy(s, tau=req.tau)),
        ))
    out = AmbiguityResponse(rows=rows)
    engine_cache.memo_put(key, out)
    return out


@router.post("/reduce-2d", response_model=Reduce2DResponse)
def reduce_2d(req: Reduce2DRequest) -> Reduce2DResponse:
    space = _require(req.space_id)
    key = engine_cache.memo_key(req.space_id, "reduce-2d", req.model_dump(exclude={"space_id"}))
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    method = req.method
    if method == "auto":
        method = "umap"  # match the existing default

    # build the base descriptor projection
    X = space.D
    if req.include_centroids:
        # project centroids in the same space by stacking and re-fitting
        centroids = np.stack([space.symbol_centroids[s] for s in space.symbols])
        if req.normalize_centroids:
            centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
        X_full = np.vstack([X, centroids])
    else:
        X_full = X

    if method == "pca":
        # Reuse the PCA already fit at SymbolSpace construction time.
        # Centroids project linearly through the same basis.
        Y = space._pca.transform(X_full)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        Y = TSNE(
            n_components=2,
            metric="cosine",
            random_state=42,
            init="random",
            perplexity=min(20, max(2, X_full.shape[0] // 4)),
        ).fit_transform(X_full)
    else:
        try:
            # Reuse the cached UMAP layout — same model that /shift uses, so
            # the meaning-space dots stay put when context changes (only the
            # shifted arrow tips move).
            model, Y_desc, Y_cent = engine_cache.get_umap_layout(space)
            Y = np.vstack([Y_desc, Y_cent]) if req.include_centroids else Y_desc
        except ImportError:
            Y = space._pca.transform(X_full)

    pts = []
    n_desc = len(space.descriptors)
    for i, name in enumerate(space.descriptors):
        pts.append(Point2D(
            x=float(Y[i, 0]),
            y=float(Y[i, 1]),
            label=name,
            symbol=space.owner[name],
            kind="descriptor",
        ))
    if req.include_centroids:
        for j, sym in enumerate(space.symbols):
            row = Y[n_desc + j]
            pts.append(Point2D(
                x=float(row[0]),
                y=float(row[1]),
                label=sym,
                symbol=sym,
                kind="centroid",
            ))

    out = Reduce2DResponse(points=pts)
    engine_cache.memo_put(key, out)
    return out
