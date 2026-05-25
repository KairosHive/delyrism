"""Context shift, delta graph, and contextual subgraph endpoints."""
from __future__ import annotations
import numpy as np
import networkx as nx
from fastapi import APIRouter, HTTPException

from ..schemas import (
    ShiftRequest, ShiftResponse, ShiftArrow, ShiftCentroid,
    DeltaGraphRequest, DeltaGraphResponse, DeltaNode, DeltaEdge,
    SubgraphRequest, SubgraphResponse, SubgraphNode, SubgraphEdge,
    SimilarityRequest, SimilarityResponse,
    SymbolSimilarityRequest, SymbolSimilarityResponse,
    SpectrumRequest, ShiftSpectrumResponse, SpectrumAxis,
    SpectrumProfileEntry, SpectrumMoverEntry,
)
from .. import engine_cache
from ..util import to_hex
from delyrism import context_delta_graph
from delyrism.delyrism import softmax

router = APIRouter(tags=["delta"])


def _require(space_id: str):
    space = engine_cache.get_space(space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")
    return space


@router.post("/shift", response_model=ShiftResponse)
def shift_arrows(req: ShiftRequest) -> ShiftResponse:
    """Return 2D arrows showing how each descriptor moves under the chosen
    context-shift strategy.

    Both endpoints are projected through the *same* reducer the caller is using
    in /reduce-2d, by fitting it on the joint matrix [D ; D_shifted].  That way
    the arrows live in the same frame as the dots in MeaningSpace, regardless
    of whether the user picked UMAP, t-SNE or PCA.
    """
    space = _require(req.space_id)
    key = engine_cache.memo_key(req.space_id, "shift-arrows", req.model_dump(exclude={"space_id"}))
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    import numpy as np

    D1 = space.make_shifted_matrix(
        weights=req.weights,
        sentence=req.sentence,
        strategy=req.strategy,
        beta=req.beta,
        gate=req.gate,
        tau=req.tau,
        within_symbol_softmax=req.within_symbol_softmax,
        gamma=req.gamma,
        prompt_template=req.prompt_template,
        pool_type=req.pool_type,
        pool_w=req.pool_w,
        membership_alpha=req.membership_alpha,
    )
    n = space.D.shape[0]
    centroids = np.stack([space.symbol_centroids[s] for s in space.symbols])
    n_c = centroids.shape[0]

    if req.reducer == "pca":
        # PCA is linear, so we fit on the *original* descriptor cloud and
        # transform both shifted descriptors and centroids through that same
        # basis.  This keeps arrows anchored to dots and avoids the "comet
        # trail" where the first PC ends up aligned with the global shift
        # direction (which happens when you fit jointly on [D ; D_shifted]).
        pca = space._pca  # already fit on space.D at construction time
        Y0 = pca.transform(space.D)
        Y1 = pca.transform(D1)
        Yc = pca.transform(centroids)
    elif req.reducer == "tsne":
        # t-SNE has no stable .transform — joint fit is the only option.
        from sklearn.manifold import TSNE
        X = np.vstack([space.D, D1, centroids])
        Y = TSNE(
            n_components=2,
            metric="cosine",
            random_state=42,
            init="random",
            perplexity=min(20, max(2, X.shape[0] // 4)),
        ).fit_transform(X)
        Y0 = Y[:n]; Y1 = Y[n : 2 * n]; Yc = Y[2 * n : 2 * n + n_c]
    else:  # umap
        try:
            # Use the cached UMAP layout (fit once on [D ; centroids]) and
            # only project the shifted descriptors through model.transform().
            # This keeps dots and centroids fixed across context changes —
            # only arrow tips move, which is what the user expects from a
            # "context shift" visualization.
            model, Y0, Yc = engine_cache.get_umap_layout(space)
            Y1 = model.transform(D1)
        except ImportError:
            pca = space._pca
            Y0 = pca.transform(space.D); Y1 = pca.transform(D1); Yc = pca.transform(centroids)

    arrows = [
        ShiftArrow(
            descriptor=d,
            symbol=space.owner[d],
            x0=float(Y0[i, 0]), y0=float(Y0[i, 1]),
            x1=float(Y1[i, 0]), y1=float(Y1[i, 1]),
        )
        for i, d in enumerate(space.descriptors)
    ]
    centroids_out = [
        ShiftCentroid(symbol=s, x=float(Yc[j, 0]), y=float(Yc[j, 1]))
        for j, s in enumerate(space.symbols)
    ]
    out = ShiftResponse(arrows=arrows, centroids=centroids_out)
    engine_cache.memo_put(key, out)
    return out


@router.post("/delta-graph", response_model=DeltaGraphResponse)
def delta_graph(req: DeltaGraphRequest) -> DeltaGraphResponse:
    space = _require(req.space_id)
    key = engine_cache.memo_key(req.space_id, "delta-graph", req.model_dump(exclude={"space_id"}))
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    G = context_delta_graph(
        space,
        sentence=req.sentence,
        weights=req.weights,
        strategy=req.strategy,
        beta=req.beta,
        gate=req.gate,
        tau=req.tau,
        within_symbol_softmax=req.within_symbol_softmax,
        gamma=req.gamma,
        prompt_template=req.prompt_template,
        top_abs_edges=req.top_abs_edges,
        sym_filter=req.sym_filter,
        min_abs_delta=req.min_abs_delta,
        within_symbol=req.within_symbol,
        only_symbol=req.only_symbol,
        connected_only=req.connected_only,
        pool_type=req.pool_type,
        pool_w=req.pool_w,
        membership_alpha=req.membership_alpha,
        sign_filter=req.sign_filter,
    )
    cmap = {k: to_hex(v) for k, v in space.get_symbol_color_dict(palette=engine_cache.get_palette(space)).items()}
    nodes = [
        DeltaNode(id=n, symbol=space.owner.get(n, ""), color=cmap.get(space.owner.get(n, ""), "#888"))
        for n in G.nodes()
    ]
    edges = []
    for u, v, attrs in G.edges(data=True):
        edges.append(DeltaEdge(
            source=u,
            target=v,
            delta=float(attrs["delta"]),
            sign=attrs["sign"],
            abs_delta=float(attrs["abs_delta"]),
        ))
    out = DeltaGraphResponse(nodes=nodes, edges=edges)
    engine_cache.memo_put(key, out)
    return out


@router.post("/subgraph", response_model=SubgraphResponse)
def contextual_subgraph(req: SubgraphRequest) -> SubgraphResponse:
    """Top-K symbols + their top-M descriptors under the given context.
    Mirrors plot_contextual_subgraph_colored's data, but returns a graph-shaped
    payload the frontend renders with sigma/force-graph."""
    space = _require(req.space_id)
    key = engine_cache.memo_key(req.space_id, "subgraph", req.model_dump(exclude={"space_id"}))
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    sent_vec = space.ctx_vec(sentence=req.sentence)
    sims = space.D @ sent_vec
    pers = {f"D:{d}": float(w) for d, w in zip(space.descriptors, softmax(sims, tau=req.tau))}
    pr = nx.pagerank(space.G, alpha=req.alpha, personalization=pers, weight="weight")
    sym_scores = {n[2:]: v for n, v in pr.items() if n.startswith("S:")}

    if req.method == "softmax":
        # rank symbols by direct attention over their descriptors
        for s in space.symbols:
            idx = space.symbol_to_idx[s]
            sym_scores[s] = float(softmax(sims[idx], tau=req.tau).sum())

    top_syms = sorted(sym_scores.items(), key=lambda kv: kv[1], reverse=True)[: req.topk_symbols]
    cmap = {k: to_hex(v) for k, v in space.get_symbol_color_dict(palette=engine_cache.get_palette(space)).items()}

    # Pick top symbols + their top descriptors first, then take the induced
    # subgraph of space.G over those node names.  This mirrors
    # `plot_contextual_subgraph_colored` and brings back the descriptor↔descriptor
    # cosine edges that connect clusters across symbols (without them the
    # graph reads as a set of isolated stars).
    sym_set = {f"S:{s}" for s, _ in top_syms}
    desc_set: set[str] = set()
    sym_of_desc: dict[str, str] = {}
    desc_scores: dict[str, float] = {}

    for sym, _ in top_syms:
        descs = space.symbols_to_descriptors[sym]
        if req.method == "softmax":
            idx = space.symbol_to_idx[sym]
            local = softmax(sims[idx], tau=req.tau)
            ranked = sorted(zip(descs, local), key=lambda kv: kv[1], reverse=True)[: req.topk_desc]
        else:
            ranked = sorted(
                ((d, pr.get(f"D:{d}", 0.0)) for d in descs),
                key=lambda kv: kv[1],
                reverse=True,
            )[: req.topk_desc]
        for d, dscore in ranked:
            did = f"D:{d}"
            desc_set.add(did)
            sym_of_desc[did] = sym
            desc_scores[did] = float(dscore)

    all_node_ids = sym_set | desc_set
    subG = space.G.subgraph(all_node_ids).copy()

    nodes: list[SubgraphNode] = []
    for sym, sc in top_syms:
        nid = f"S:{sym}"
        if nid in subG:
            nodes.append(SubgraphNode(
                id=nid, kind="symbol", symbol=sym,
                color=cmap.get(sym, "#888"), score=float(sc),
            ))
    for did in desc_set:
        if did in subG:
            sym = sym_of_desc[did]
            nodes.append(SubgraphNode(
                id=did, kind="descriptor", symbol=sym,
                color=cmap.get(sym, "#888"), score=desc_scores[did],
            ))

    edges: list[SubgraphEdge] = [
        SubgraphEdge(source=u, target=v, weight=float(d.get("weight", 1.0)))
        for u, v, d in subG.edges(data=True)
    ]

    out = SubgraphResponse(nodes=nodes, edges=edges)
    engine_cache.memo_put(key, out)
    return out


@router.post("/similarity", response_model=SimilarityResponse)
def similarity_matrices(req: SimilarityRequest) -> SimilarityResponse:
    """Per-symbol Before/After/Δ descriptor similarity matrices.
    Mirrors `SymbolSpace.descriptor_similarity_matrices` and feeds the
    Streamlit-style 3-panel heatmap."""
    space = _require(req.space_id)
    if req.symbol not in space.symbol_to_idx:
        raise HTTPException(status_code=400, detail=f"unknown symbol '{req.symbol}'")
    key = engine_cache.memo_key(req.space_id, "similarity", req.model_dump(exclude={"space_id"}))
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    mats = space.descriptor_similarity_matrices(
        weights=req.weights,
        sentence=req.sentence,
        strategy=req.strategy,
        beta=req.beta,
        gate=req.gate,
        tau=req.tau,
        within_symbol_softmax=req.within_symbol_softmax,
        order_by_attention=req.order_by_attention,
        gamma=req.gamma,
        prompt_template=req.prompt_template,
        pool_type=req.pool_type,
        pool_w=req.pool_w,
        membership_alpha=req.membership_alpha,
    )
    block = mats[req.symbol]
    out = SimilarityResponse(
        symbol=req.symbol,
        descriptors=list(block["descriptors"]),
        before=block["S_before"].tolist(),
        after=block["S_after"].tolist(),
        delta=block["S_delta"].tolist(),
    )
    engine_cache.memo_put(key, out)
    return out


@router.post("/similarity-symbols", response_model=SymbolSimilarityResponse)
def symbol_centroid_similarity(req: SymbolSimilarityRequest) -> SymbolSimilarityResponse:
    """Symbol-by-symbol centroid-cosine matrices (Before / After / Δ).

    Same context plumbing as /similarity, but rolled up: for each symbol we
    take the L2-normalized mean of its descriptor vectors before and after
    the shift, then form the S×S cosine matrix.  Answers 'does context make
    symbol A look more like symbol B?' rather than 'do these two descriptors
    inside A look more alike?'.
    """
    space = _require(req.space_id)
    key = engine_cache.memo_key(req.space_id, "similarity-symbols", req.model_dump(exclude={"space_id"}))
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    D_after = space.make_shifted_matrix(
        weights=req.weights,
        sentence=req.sentence,
        strategy=req.strategy,
        beta=req.beta,
        gate=req.gate,
        tau=req.tau,
        within_symbol_softmax=req.within_symbol_softmax,
        gamma=req.gamma,
        prompt_template=req.prompt_template,
        pool_type=req.pool_type,
        pool_w=req.pool_w,
        membership_alpha=req.membership_alpha,
    )

    symbols = list(space.symbols)

    def _centroid_matrix(D: np.ndarray) -> np.ndarray:
        # L2-normalized mean per symbol → S×dim, then cosine via the gram matrix.
        rows = []
        for s in symbols:
            idx = space.symbol_to_idx[s]
            c = D[idx].mean(axis=0) if len(idx) else np.zeros(D.shape[1], dtype=D.dtype)
            n = float(np.linalg.norm(c))
            rows.append(c / n if n > 1e-9 else c)
        C = np.stack(rows)
        return C @ C.T

    S_before = _centroid_matrix(space.D)
    S_after = _centroid_matrix(D_after)
    S_delta = S_after - S_before

    out = SymbolSimilarityResponse(
        symbols=symbols,
        before=S_before.tolist(),
        after=S_after.tolist(),
        delta=S_delta.tolist(),
    )
    engine_cache.memo_put(key, out)
    return out


@router.post("/shift-spectrum", response_model=ShiftSpectrumResponse)
def shift_spectrum(req: SpectrumRequest) -> ShiftSpectrumResponse:
    """Top-K principal axes of the context-induced shift Δ = D' − D.

    Decomposes the per-descriptor displacement field into orthogonal
    'rewriting axes', each interpretable as a direction in embedding space
    that some descriptors are dragged along.  For each axis we surface:
      - σ_k (magnitude of motion along the axis)
      - archetype profile (which symbol centroids the axis points toward)
      - top positive movers / top negative movers (descriptors moving along
        +axis vs −axis)

    σ₁ / σ₂ is the headline scalar: high values mean 'narrow context, one
    direction of pull'; values near 1 mean 'multi-axis / polarizing context'.
    """
    space = _require(req.space_id)
    key = engine_cache.memo_key(req.space_id, "shift-spectrum", req.model_dump(exclude={"space_id"}))
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    D0 = space.D
    D1 = space.make_shifted_matrix(
        weights=req.weights,
        sentence=req.sentence,
        strategy=req.strategy,
        beta=req.beta,
        gate=req.gate,
        tau=req.tau,
        within_symbol_softmax=req.within_symbol_softmax,
        gamma=req.gamma,
        prompt_template=req.prompt_template,
        pool_type=req.pool_type,
        pool_w=req.pool_w,
        membership_alpha=req.membership_alpha,
    )
    delta = D1 - D0  # N × d
    # Thin SVD — Δ = U · diag(s) · Vᵀ.  U is N × r, Vᵀ is r × d, where
    # r = min(N, d).  Singular values are sorted descending by numpy.
    U, s, Vt = np.linalg.svd(delta, full_matrices=False)

    # Participation ratio: smooth scalar in [1, r] saying 'how many axes are
    # actually doing the work'.  Single-axis context → ~1; uniform across k
    # axes → ~k.
    s_sq = s ** 2
    energy = float(s_sq.sum())
    if energy > 1e-18:
        effective_rank = float(energy * energy / float((s_sq ** 2).sum() + 1e-18))
    else:
        effective_rank = 0.0

    # σ₁ / σ₂ — high = narrow / single-direction context, ~1 = polarizing.
    if s.shape[0] >= 2 and s[1] > 1e-9:
        dominance = float(s[0] / s[1])
    else:
        dominance = None

    K = min(int(req.topk), int(s.shape[0]))

    # Pre-compute L2-normalized symbol centroids for the archetype-profile dot.
    symbols = list(space.symbols)
    if symbols:
        cents = np.stack([space.symbol_centroids[sym] for sym in symbols])
        cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
    else:
        cents = np.zeros((0, D0.shape[1]), dtype=D0.dtype)

    descriptor_names = list(space.descriptors)
    owners = [space.owner.get(d, "") for d in descriptor_names]

    axes_out: list[SpectrumAxis] = []
    for k in range(K):
        sigma_k = float(s[k])
        u_k = U[:, k]            # N — descriptor loadings
        v_k = Vt[k, :]           # d — direction in embedding space
        v_norm = float(np.linalg.norm(v_k))
        v_unit = v_k / (v_norm + 1e-9)

        # Archetype profile: signed cosine of axis direction with each centroid.
        if cents.shape[0] > 0:
            alignments = cents @ v_unit            # S
            prof_order = np.argsort(-np.abs(alignments))[:6]
            profile = [
                SpectrumProfileEntry(symbol=symbols[i], alignment=float(alignments[i]))
                for i in prof_order
            ]
        else:
            profile = []

        # Mover contributions along this axis = σ · u_k.  Same units as the
        # original Δ — i.e. cosine displacement per descriptor.
        contrib = sigma_k * u_k
        # Top 8 each side, dropping the wrong-signed entries.
        pos_order = np.argsort(-contrib)[:8]
        neg_order = np.argsort(contrib)[:8]
        positive_movers = [
            SpectrumMoverEntry(
                descriptor=descriptor_names[i],
                symbol=owners[i],
                score=float(contrib[i]),
            )
            for i in pos_order if contrib[i] > 1e-6
        ]
        negative_movers = [
            SpectrumMoverEntry(
                descriptor=descriptor_names[i],
                symbol=owners[i],
                score=float(contrib[i]),
            )
            for i in neg_order if contrib[i] < -1e-6
        ]

        axes_out.append(SpectrumAxis(
            sigma=sigma_k,
            archetype_profile=profile,
            positive_movers=positive_movers,
            negative_movers=negative_movers,
        ))

    out = ShiftSpectrumResponse(
        sigma=[float(x) for x in s[: max(K, 6)]],
        axes=axes_out,
        dominance_ratio=dominance,
        effective_rank=effective_rank,
    )
    engine_cache.memo_put(key, out)
    return out
