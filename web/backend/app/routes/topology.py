"""Persistent homology endpoints — drives the Topology tab.

Performance notes:
  - All endpoints share a process-wide per-symbol PH cache, keyed on
    (space_id, symbol, hash(X.bytes)).  So /summary, /diagrams-all,
    /cycles, /catalysts, /synergy, /pair-cycles all reuse a single
    ripser run per (space, symbol, current D).
  - /summary parallelises across symbols via ThreadPoolExecutor.
    Ripser releases the GIL inside its C++ Vietoris–Rips core, so
    threading gives ~N× speedup for N CPU cores on cold cache.


Wraps `delyrism/ph.py` and exposes:
  /topology/summary               TopoScore + joint PCA-2D for the overview map
  /topology/diagrams/{symbol}     full persistence diagram (H0/H1/H2)
  /topology/cycles/{symbol}       top persistent cycles + PCA coords
  /topology/synergy               pairwise H1/H2 synergy matrix
  /topology/pair-cycles?a&b       mixed/pure cycles between two symbols
  /topology/catalysts/{symbol}    word-level LOO + cycle-participation impact

Every endpoint is memoised on (space_id, op, params).  None of these
depend on the current context — PH measures the *shape* of the
unconditioned descriptor cloud — so the cache lives as long as the
SymbolSpace itself.

Ripser is the fast path.  If it's missing on the host, the endpoints
return 503 with a clear message (the cheap H0-via-MST routes could in
principle stand alone, but the user-facing value is the H1/H2 surface
that requires ripser).
"""
from __future__ import annotations
import hashlib
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ..schemas import (
    TopologySummaryEntry, TopologySummaryResponse, PCAPoint, SetQualityMetrics,
    PersistenceDiagramResponse, PersistencePoint,
    AllDiagramsEntry, AllDiagramsResponse,
    TopologyCyclesResponse, PersistentCycle, CycleVertex,
    TopologySynergyResponse, SynergyEntry,
    PairCyclesResponse, PairCycle,
    WordCatalystResponse, WordCatalystEntry,
)
from .. import engine_cache

router = APIRouter(prefix="/topology", tags=["topology"])


# ---------- utilities -------------------------------------------------------

def _require(space_id: str):
    space = engine_cache.get_space(space_id)
    if space is None:
        raise HTTPException(status_code=404, detail="unknown space_id")
    return space


def _ripser_or_503():
    """Lazy import; raise a clear HTTPException if ripser isn't available."""
    try:
        from ripser import ripser  # noqa: F401
        return True
    except Exception:
        raise HTTPException(
            status_code=503,
            detail=(
                "Persistent-homology endpoints require the `ripser` package. "
                "Install it (pip install ripser) and restart the backend."
            ),
        )


def _ripser_available() -> bool:
    try:
        import ripser  # noqa: F401
        return True
    except Exception:
        return False


def _pca2d(X: np.ndarray) -> np.ndarray:
    """Joint PCA-2D, mean-centered.  Returns N×2."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def _row_norm(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _sum_finite(dgm: np.ndarray) -> float:
    if dgm.size == 0:
        return 0.0
    m = np.isfinite(dgm[:, 1])
    return float(np.sum(dgm[m, 1] - dgm[m, 0])) if m.any() else 0.0


def _max_finite(dgm: np.ndarray) -> float:
    if dgm.size == 0:
        return 0.0
    m = np.isfinite(dgm[:, 1])
    if not m.any():
        return 0.0
    return float(np.max(dgm[m, 1] - dgm[m, 0]))


def _symbol_embeddings_from(space, D: np.ndarray) -> dict:
    """{symbol: N_s × d numpy array of its descriptor vectors} built from
    the *given* descriptor matrix D — could be the original space.D or a
    context-shifted D' from `engine_cache.get_or_compute_shifted_matrix`.
    """
    out = {}
    for s in space.symbols:
        idx = space.symbol_to_idx[s]
        if len(idx) >= 4:  # PH needs at least a few points to be meaningful
            out[s] = _row_norm(D[idx])
    return out


def _symbol_embeddings(space) -> dict:
    return _symbol_embeddings_from(space, space.D)


def _symbol_words(space) -> dict:
    return {s: list(space.symbols_to_descriptors[s]) for s in space.symbols}


def _cache_key(space_id: str, op: str, params: dict | None = None) -> str:
    return engine_cache.memo_key(space_id, f"topology:{op}", params or {})


def _shift_params_from_body(body: dict) -> dict:
    """Pull the standard shift-matrix parameter set out of a request body.
    Mirrors the schema the Explorer's shiftPayload sends — same defaults so
    a missing field collapses to the engine's standard behaviour."""
    return {
        "weights": body.get("weights"),
        "sentence": body.get("sentence"),
        "strategy": body.get("strategy", "gate"),
        "beta": body.get("beta", 0.6),
        "gate": body.get("gate", "relu"),
        "tau": body.get("tau", 0.3),
        "within_symbol_softmax": body.get("within_symbol_softmax", False),
        "gamma": body.get("gamma", 0.5),
        "prompt_template": body.get("prompt_template", "{sent}, {desc}"),
        "pool_type": body.get("pool_type", "avg"),
        "pool_w": body.get("pool_w", 0.7),
        "membership_alpha": body.get("membership_alpha", 0.0),
    }


# ─────── per-symbol PH cache shared across all endpoints ───────────────
# Computing ripser(X, maxdim=2, do_cocycles=True) is the hot path for the
# whole tab.  Cache the raw output by (space_id, symbol, X-bytes hash) so
# /summary, /diagrams-all, /cycles, /catalysts, /synergy and /pair-cycles
# all reuse a single computation per (space, symbol, current D).

_ph_cache: dict[str, dict] = {}
_ph_lock = threading.Lock()
_PH_CACHE_MAX = 256


def _x_fingerprint(X: np.ndarray) -> str:
    # First 16 hex of sha256 is plenty for collision-avoidance here.
    return hashlib.sha256(X.tobytes()).hexdigest()[:16]


def invalidate_for_space(space_id: str) -> None:
    """Drop every cached PH entry belonging to a given space.  Called when
    the space is rebuilt or its context_override changes — though most
    topology endpoints don't honour context_override, the cache key
    includes X.bytes so a new build automatically misses anyway."""
    prefix = f"{space_id}:"
    with _ph_lock:
        for k in list(_ph_cache.keys()):
            if k.startswith(prefix):
                _ph_cache.pop(k, None)


def _get_ph(space_id: str, symbol: str, X: np.ndarray) -> dict:
    """Cached ripser(maxdim=2, do_cocycles=True) on `X`.

    Always computes with cocycles — the extra cost is small and lets the
    cycles / pair-cycles / catalysts endpoints share this cache instead of
    needing their own separate computation.
    """
    fp = _x_fingerprint(X)
    key = f"{space_id}:{symbol}:{fp}"
    with _ph_lock:
        cached = _ph_cache.get(key)
        if cached is not None:
            return cached
    from ripser import ripser
    out = ripser(X, maxdim=2, metric="cosine", do_cocycles=True)
    with _ph_lock:
        if len(_ph_cache) > _PH_CACHE_MAX:
            # FIFO-ish eviction
            for k in list(_ph_cache.keys())[: _PH_CACHE_MAX // 4]:
                _ph_cache.pop(k, None)
        _ph_cache[key] = out
    return out


def _resolve_D_and_key(
    space, space_id: str, body: dict, op: str, extra: dict | None = None,
) -> tuple[np.ndarray, str, bool]:
    """Return (D matrix, cache_key, used_context).

    use_context=True in the body re-runs the same context-shift pipeline
    Explorer uses (gate/reembed/pooling/hybrid, β, γ, τ, etc.) and returns
    the shifted D' for the topology endpoint to consume.  The shift params
    fold into the cache key so every (op, shift_params) combination memoises
    independently.

    use_context=False (default) returns the original space.D — the
    "intrinsic shape" reading.
    """
    use_ctx = bool(body.get("use_context", False))
    base = dict(extra or {})
    if not use_ctx:
        D = space.D
        params = base
    else:
        shift_params = _shift_params_from_body(body)
        D = engine_cache.get_or_compute_shifted_matrix(space_id, space, shift_params)
        params = {**base, **shift_params, "_ctx": True}
    key = _cache_key(space_id, op, params)
    return D, key, use_ctx


# ---------- /topology/summary -----------------------------------------------

@router.post("/summary", response_model=TopologySummaryResponse)
def topology_summary(req: dict):
    """Per-symbol TopoScore + a joint PCA-2D layout of every descriptor.

    Returned in one shot so the Overview view doesn't need to roundtrip
    for the layout.  PCA is on the *unit-normalised* original descriptors
    — same frame everywhere PH is used in this tab.
    """
    space_id = req.get("space_id")
    if not space_id:
        raise HTTPException(status_code=400, detail="space_id required")
    space = _require(space_id)

    D, key, _ = _resolve_D_and_key(space, space_id, req, "summary")
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    have_ripser = _ripser_available()
    sym_emb = _symbol_embeddings_from(space, D)

    entries: list[TopologySummaryEntry] = []
    if have_ripser:
        # Parallelise across symbols — ripser releases the GIL inside its
        # C++ VR core so ThreadPoolExecutor gets real wall-clock speedup.
        # `_get_ph` is process-wide cached so warm calls are instant.
        def _one(item):
            sym, X = item
            return sym, _get_ph(space_id, sym, X)
        items = list(sym_emb.items())
        max_workers = min(8, max(1, len(items)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            raw = list(ex.map(_one, items))

        rows = []
        for sym, ph_out in raw:
            dgms = ph_out["dgms"]
            H0, H1, H2 = dgms[0], dgms[1], dgms[2]
            # cohesion + outlier from H0
            if H0.size and np.isfinite(H0[:, 1]).any():
                p0 = H0[np.isfinite(H0[:, 1])]
                pers0 = p0[:, 1] - p0[:, 0]
                coh = float(np.median(pers0))
                out = float(np.max(pers0))
            else:
                coh, out = 0.0, 0.0
            h1_sum = _sum_finite(H1)
            h1_max = _max_finite(H1)
            h2_sum = _sum_finite(H2)
            h2_max = _max_finite(H2)
            thr = 0.02
            h1_count = int(np.sum(np.where(np.isfinite(H1[:, 1] if H1.size else 0), (H1[:, 1] - H1[:, 0] if H1.size else 0), 0) > thr)) if H1.size else 0
            h2_count = int(np.sum(np.where(np.isfinite(H2[:, 1] if H2.size else 0), (H2[:, 1] - H2[:, 0] if H2.size else 0), 0) > thr)) if H2.size else 0
            rows.append((sym, coh, out, h1_sum, h1_max, h1_count, h2_sum, h2_max, h2_count))

        # z-score composite
        import statistics
        if len(rows) >= 2:
            cohs = [r[1] for r in rows]; h1s = [r[3] for r in rows]; h2s = [r[6] for r in rows]
            def zsc(xs):
                m = statistics.mean(xs); sd = statistics.pstdev(xs) or 1e-9
                return [(x - m) / sd for x in xs]
            coh_z = zsc(cohs); h1_z = zsc(h1s); h2_z = zsc(h2s)
        else:
            coh_z = [0.0] * len(rows); h1_z = [0.0] * len(rows); h2_z = [0.0] * len(rows)

        for (r, cz, h1z, h2z) in zip(rows, coh_z, h1_z, h2_z):
            (sym, coh, out, h1_sum, h1_max, h1_count, h2_sum, h2_max, h2_count) = r
            entries.append(TopologySummaryEntry(
                symbol=sym,
                h0_cohesion=coh, h0_outlier=out,
                h1_sum=h1_sum, h1_max=h1_max, h1_count=h1_count,
                h2_sum=h2_sum, h2_max=h2_max, h2_count=h2_count,
                topo_score=float(h1z + h2z - cz),
            ))
    else:
        # Cheap fallback: just H0 cohesion via MST + flag the missing dependency
        from delyrism.ph import h0_bar_lengths_from_mst
        for sym, X in sym_emb.items():
            try:
                lens = h0_bar_lengths_from_mst(X, metric="cosine")
                coh = float(np.median(lens)) if lens.size else 0.0
                out = float(np.max(lens)) if lens.size else 0.0
            except Exception:
                coh, out = 0.0, 0.0
            entries.append(TopologySummaryEntry(
                symbol=sym, h0_cohesion=coh, h0_outlier=out,
                h1_sum=0.0, h1_max=0.0, h1_count=0,
                h2_sum=0.0, h2_max=0.0, h2_count=0, topo_score=0.0,
            ))

    # Joint PCA-2D over all descriptors (uses the same D that drove PH —
    # original or shifted depending on use_context).
    Z = _pca2d(_row_norm(D))
    pts = [
        PCAPoint(word=d, symbol=space.owner[d], x=float(Z[i, 0]), y=float(Z[i, 1]))
        for i, d in enumerate(space.descriptors)
    ]

    # ─── set-level quality scalars ───
    set_q: Optional[SetQualityMetrics] = None
    if have_ripser and len(entries) >= 2 and sym_emb:
        # Coverage — PH on the UNION of all descriptors.
        union_X = _row_norm(np.vstack(list(sym_emb.values())))
        union_dgms = _get_ph(space_id, "__union__", union_X)["dgms"]
        coverage_h1 = _sum_finite(union_dgms[1])
        coverage_h2 = _sum_finite(union_dgms[2])

        # Internal richness — mean of (H1_persistent + H2_persistent) per symbol.
        richness_mean = float(np.mean([e.h1_count + e.h2_count for e in entries]))

        # Cohesion balance — 1 − std/mean of h0_cohesion across symbols.
        coh_vals = np.array([e.h0_cohesion for e in entries], dtype=float)
        if coh_vals.mean() > 1e-9:
            cohesion_balance = float(max(0.0, min(1.0, 1.0 - coh_vals.std() / coh_vals.mean())))
        else:
            cohesion_balance = 0.0

        # Separation tightness — mean pairwise cosine distance between
        # archetype centroids on the unit sphere.
        cents = np.stack([X.mean(axis=0) for X in sym_emb.values()])
        cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
        sim_mat = cents @ cents.T
        n = len(cents)
        if n >= 2:
            iu = np.triu_indices(n, k=1)
            separation_tightness = float(np.mean(1.0 - sim_mat[iu]))
        else:
            separation_tightness = 0.0

        # Count balance — Shannon entropy of descriptor counts, normalised.
        counts = np.array([len(X) for X in sym_emb.values()], dtype=float)
        if counts.sum() > 0 and n >= 2:
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            ent = float(-np.sum(probs * np.log2(probs)))
            count_balance = ent / float(np.log2(n))
        else:
            count_balance = 0.0

        set_q = SetQualityMetrics(
            coverage_h1=coverage_h1,
            coverage_h2=coverage_h2,
            richness_mean=richness_mean,
            cohesion_balance=cohesion_balance,
            separation_tightness=separation_tightness,
            count_balance=count_balance,
        )

    out = TopologySummaryResponse(entries=entries, points=pts,
                                  ripser_available=have_ripser,
                                  set_quality=set_q)
    engine_cache.memo_put(key, out)
    return out


# ---------- /topology/diagrams/{symbol} -------------------------------------

@router.post("/diagrams", response_model=PersistenceDiagramResponse)
def topology_diagram(req: dict):
    space_id = req.get("space_id"); symbol = req.get("symbol")
    if not space_id or not symbol:
        raise HTTPException(status_code=400, detail="space_id and symbol required")
    space = _require(space_id)
    _ripser_or_503()

    D, key, _ = _resolve_D_and_key(space, space_id, req, "diagram", {"symbol": symbol})
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    sym_emb = _symbol_embeddings_from(space, D)
    if symbol not in sym_emb:
        raise HTTPException(status_code=400, detail=f"symbol '{symbol}' has too few descriptors for PH")
    X = sym_emb[symbol]
    dgms = _get_ph(space_id, symbol, X)["dgms"]

    pts: list[PersistencePoint] = []
    max_finite = 0.0
    for d, dgm in enumerate(dgms[:3]):
        if not dgm.size:
            continue
        for birth, death in dgm:
            is_inf = not np.isfinite(death)
            b = float(birth); de = float(death) if not is_inf else float("inf")
            if not is_inf and de > max_finite:
                max_finite = de
            pts.append(PersistencePoint(dim=d, birth=b, death=de if not is_inf else 0.0, is_infinite=is_inf))
    # second pass: replace +inf with max_finite * 1.1 so the JSON is finite
    inf_value = (max_finite if max_finite > 0 else 1.0) * 1.1
    pts = [
        PersistencePoint(dim=p.dim, birth=p.birth,
                         death=inf_value if p.is_infinite else p.death,
                         is_infinite=p.is_infinite)
        for p in pts
    ]
    out = PersistenceDiagramResponse(symbol=symbol, points=pts,
                                     max_finite_death=float(max_finite),
                                     ripser_available=True)
    engine_cache.memo_put(key, out)
    return out


# ---------- /topology/diagrams-all ------------------------------------------

@router.post("/diagrams-all", response_model=AllDiagramsResponse)
def topology_diagrams_all(req: dict):
    """All symbols' persistence diagrams in one shot — drives the
    small-multiples grid in the Diagrams sub-view.  Shared axis frame
    (max_finite_death) so the minis are visually comparable.
    """
    space_id = req.get("space_id")
    if not space_id:
        raise HTTPException(status_code=400, detail="space_id required")
    space = _require(space_id)
    _ripser_or_503()

    D, key, _ = _resolve_D_and_key(space, space_id, req, "diagrams-all")
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    sym_emb = _symbol_embeddings_from(space, D)

    # First pass: compute all PH (parallel + cached) and find the global
    # max-finite-death so the minis share an axis frame.
    def _one(item):
        sym, X = item
        return sym, _get_ph(space_id, sym, X)["dgms"]
    items = list(sym_emb.items())
    max_workers = min(8, max(1, len(items)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        raw_pairs = list(ex.map(_one, items))
    raw: dict[str, list] = {sym: dgms for sym, dgms in raw_pairs}
    max_finite = 0.0
    for dgms in raw.values():
        for dgm in dgms:
            if not dgm.size:
                continue
            mask = np.isfinite(dgm[:, 1])
            if mask.any():
                m = float(np.max(dgm[mask, 1]))
                if m > max_finite:
                    max_finite = m

    inf_value = (max_finite if max_finite > 0 else 1.0) * 1.1
    PERS_THR = 0.02

    entries: list[AllDiagramsEntry] = []
    for sym, dgms in raw.items():
        pts: list[PersistencePoint] = []
        h0_finite = 0
        h1_total = 0; h1_pers_count = 0; max_h1 = 0.0
        h2_total = 0; h2_pers_count = 0; max_h2 = 0.0
        for d, dgm in enumerate(dgms[:3]):
            if not dgm.size:
                continue
            for birth, death in dgm:
                is_inf = not np.isfinite(death)
                b = float(birth)
                de = inf_value if is_inf else float(death)
                pts.append(PersistencePoint(dim=d, birth=b, death=de, is_infinite=is_inf))
                if d == 0 and not is_inf:
                    h0_finite += 1
                elif d == 1:
                    h1_total += 1
                    if not is_inf:
                        p = float(death) - b
                        if p > PERS_THR:
                            h1_pers_count += 1
                        if p > max_h1:
                            max_h1 = p
                elif d == 2:
                    h2_total += 1
                    if not is_inf:
                        p = float(death) - b
                        if p > PERS_THR:
                            h2_pers_count += 1
                        if p > max_h2:
                            max_h2 = p
        entries.append(AllDiagramsEntry(
            symbol=sym, points=pts,
            h0_finite=h0_finite,
            h1_total=h1_total, h1_persistent=h1_pers_count,
            h2_total=h2_total, h2_persistent=h2_pers_count,
            max_persistence_h1=max_h1, max_persistence_h2=max_h2,
        ))

    out = AllDiagramsResponse(entries=entries, max_finite_death=max_finite,
                              ripser_available=True)
    engine_cache.memo_put(key, out)
    return out


# ---------- /topology/cycles/{symbol} ---------------------------------------

@router.post("/cycles", response_model=TopologyCyclesResponse)
def topology_cycles(req: dict):
    """Top persistent H1 + H2 cycles for one symbol's descriptor cloud.

    Each cycle carries the ordered vertices (for H1, this traces a loop
    in semantic space; for H2, the triangle vertices of the void),
    pre-projected to PCA-2D so the frontend can draw them on a shared
    scatter without any client-side computation.
    """
    space_id = req.get("space_id"); symbol = req.get("symbol")
    top_h1 = int(req.get("top_h1", 6))
    top_h2 = int(req.get("top_h2", 3))
    if not space_id or not symbol:
        raise HTTPException(status_code=400, detail="space_id and symbol required")
    space = _require(space_id)
    _ripser_or_503()

    D, key, _ = _resolve_D_and_key(space, space_id, req, "cycles",
                                    {"symbol": symbol, "k1": top_h1, "k2": top_h2})
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    sym_emb = _symbol_embeddings_from(space, D)
    if symbol not in sym_emb:
        raise HTTPException(status_code=400, detail=f"symbol '{symbol}' has too few descriptors for PH")
    X = sym_emb[symbol]
    words = list(space.symbols_to_descriptors[symbol])
    # Defensive — drop duplicates by index alignment
    words = words[:X.shape[0]]

    out_r = _get_ph(space_id, symbol, X)
    H1, H2 = out_r["dgms"][1], out_r["dgms"][2]
    coc1 = out_r.get("cocycles", [[], [], []])[1] if "cocycles" in out_r else []
    coc2 = out_r.get("cocycles", [[], [], []])[2] if "cocycles" in out_r else []

    # PCA-2D for this symbol's descriptors
    Z = _pca2d(X)
    descriptors = [
        CycleVertex(word=words[i], index=i, x=float(Z[i, 0]), y=float(Z[i, 1]))
        for i in range(X.shape[0])
    ]

    def _vertices_ordered_h1(cyc) -> list[int]:
        """Given an H1 cocycle (list of [i, j, coeff]), return an *ordered*
        traversal of the loop's vertices.  Greedy: start at the lowest-index
        vertex, walk along edges.  Falls back to set order if the edges
        don't form a single loop."""
        if cyc is None or len(cyc) == 0:
            return []
        edges: dict[int, list[int]] = {}
        for row in cyc:
            i, j = int(row[0]), int(row[1])
            edges.setdefault(i, []).append(j)
            edges.setdefault(j, []).append(i)
        if not edges:
            return []
        start = min(edges.keys())
        path = [start]
        visited = {start}
        cur = start
        while True:
            nxt = None
            for v in edges.get(cur, []):
                if v not in visited:
                    nxt = v; break
            if nxt is None:
                break
            path.append(nxt); visited.add(nxt); cur = nxt
        # close the loop visually by appending start at the end? frontend
        # will handle that — we return open path of distinct vertices
        # plus any unvisited vertices appended (rare)
        for v in edges.keys():
            if v not in visited:
                path.append(v)
        return path

    def _vertices_h2(cyc) -> list[int]:
        verts = set()
        for row in cyc:
            verts.add(int(row[0])); verts.add(int(row[1])); verts.add(int(row[2]))
        return sorted(verts)

    cycles: list[PersistentCycle] = []
    # H1
    if H1.size:
        pers = np.where(np.isfinite(H1[:, 1]), H1[:, 1] - H1[:, 0], 0.0)
        order = np.argsort(pers)[::-1][:top_h1]
        for idx in order:
            if pers[idx] <= 0:
                continue
            cyc = coc1[idx] if idx < len(coc1) else []
            verts = _vertices_ordered_h1(cyc)
            if not verts:
                continue
            cycles.append(PersistentCycle(
                dim=1,
                birth=float(H1[idx, 0]),
                death=float(H1[idx, 1] if np.isfinite(H1[idx, 1]) else 0.0),
                persistence=float(pers[idx]),
                vertices=[CycleVertex(word=words[v], index=v,
                                      x=float(Z[v, 0]), y=float(Z[v, 1]))
                          for v in verts if v < len(words)],
            ))
    # H2
    if H2.size:
        pers = np.where(np.isfinite(H2[:, 1]), H2[:, 1] - H2[:, 0], 0.0)
        order = np.argsort(pers)[::-1][:top_h2]
        for idx in order:
            if pers[idx] <= 0:
                continue
            cyc = coc2[idx] if idx < len(coc2) else []
            verts = _vertices_h2(cyc)
            if not verts:
                continue
            cycles.append(PersistentCycle(
                dim=2,
                birth=float(H2[idx, 0]),
                death=float(H2[idx, 1] if np.isfinite(H2[idx, 1]) else 0.0),
                persistence=float(pers[idx]),
                vertices=[CycleVertex(word=words[v], index=v,
                                      x=float(Z[v, 0]), y=float(Z[v, 1]))
                          for v in verts if v < len(words)],
            ))

    out = TopologyCyclesResponse(symbol=symbol, cycles=cycles, descriptors=descriptors,
                                 ripser_available=True)
    engine_cache.memo_put(key, out)
    return out


# ---------- /topology/synergy -----------------------------------------------

@router.post("/synergy", response_model=TopologySynergyResponse)
def topology_synergy(req: dict):
    """Pairwise synergy_H1 / synergy_H2 for every (a, b) symbol pair.

    Synergy = PH(A ∪ B) − PH(A ∪ B without cross-edges).  Captures how
    much loop / void mass *requires* the two symbols to interact —
    structure that wouldn't exist if you analysed them separately.
    """
    space_id = req.get("space_id")
    if not space_id:
        raise HTTPException(status_code=400, detail="space_id required")
    space = _require(space_id)
    _ripser_or_503()

    D, key, _ = _resolve_D_and_key(space, space_id, req, "synergy")
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    from ripser import ripser
    sym_emb = _symbol_embeddings_from(space, D)
    syms = list(sym_emb.keys())

    def _one_pair(pair):
        a, b = pair
        A = sym_emb[a]; B = sym_emb[b]
        X = _row_norm(np.vstack([A, B]))
        nA = len(A)
        # union PH (full point cloud)
        d_union = ripser(X, maxdim=2, metric="cosine")["dgms"]
        sumH1_u = _sum_finite(d_union[1])
        sumH2_u = _sum_finite(d_union[2])
        # union with cross-edges blocked — precomputed cosine distance.
        S = X @ X.T
        np.clip(S, -1.0, 1.0, out=S)
        Dm = 1.0 - S
        np.fill_diagonal(Dm, 0.0)
        big = 1e9
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if (i < nA) != (j < nA):
                    Dm[i, j] = Dm[j, i] = big
        d_nc = ripser(Dm, maxdim=2, metric="precomputed")["dgms"]
        sumH1_nc = _sum_finite(d_nc[1])
        sumH2_nc = _sum_finite(d_nc[2])
        return SynergyEntry(
            a=a, b=b,
            synergy_h1=float(sumH1_u - sumH1_nc),
            synergy_h2=float(sumH2_u - sumH2_nc),
            sum_h1_union=float(sumH1_u),
            sum_h2_union=float(sumH2_u),
        )

    pairs = list(itertools.combinations(syms, 2))
    max_workers = min(8, max(1, len(pairs)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        entries = list(ex.map(_one_pair, pairs))

    out = TopologySynergyResponse(symbols=syms, entries=entries, ripser_available=True)
    engine_cache.memo_put(key, out)
    return out


# ---------- /topology/pair-cycles -------------------------------------------

@router.post("/pair-cycles", response_model=PairCyclesResponse)
def topology_pair_cycles(req: dict):
    """*Bridge* cycles in A ∪ B — loops/voids that REQUIRE both symbols
    to close.  Pure-A and pure-B cycles are filtered out by default
    because they're identical to the cycles you'd see in the Cycles tab
    for symbol A or B alone (set include_pure=true to override).
    """
    space_id = req.get("space_id")
    a = req.get("a"); b = req.get("b")
    top_h1 = int(req.get("top_h1", 8))
    top_h2 = int(req.get("top_h2", 4))
    include_pure = bool(req.get("include_pure", False))
    if not space_id or not a or not b:
        raise HTTPException(status_code=400, detail="space_id, a, b required")
    space = _require(space_id)
    _ripser_or_503()

    D, key, _ = _resolve_D_and_key(space, space_id, req, "pair-cycles",
                                    {"a": a, "b": b, "k1": top_h1, "k2": top_h2,
                                     "include_pure": include_pure})
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    sym_emb = _symbol_embeddings_from(space, D)
    if a not in sym_emb or b not in sym_emb:
        raise HTTPException(status_code=400, detail="both symbols must have ≥4 descriptors")
    A = sym_emb[a]; B = sym_emb[b]
    wordsA = list(space.symbols_to_descriptors[a])[:A.shape[0]]
    wordsB = list(space.symbols_to_descriptors[b])[:B.shape[0]]
    X = _row_norm(np.vstack([A, B]))
    nA = len(A)
    labels = np.array([0] * nA + [1] * len(B))
    words = wordsA + wordsB
    homes = [a] * nA + [b] * len(B)

    Z = _pca2d(X)
    descriptors = [
        CycleVertex(word=words[i], index=i, x=float(Z[i, 0]), y=float(Z[i, 1]),
                    home_symbol=homes[i])
        for i in range(len(X))
    ]

    from ripser import ripser
    out_r = ripser(X, maxdim=2, metric="cosine", do_cocycles=True)
    H1, H2 = out_r["dgms"][1], out_r["dgms"][2]
    coc1 = out_r.get("cocycles", [[], [], []])[1] if "cocycles" in out_r else []
    coc2 = out_r.get("cocycles", [[], [], []])[2] if "cocycles" in out_r else []

    def _h1_walk(cyc):
        if cyc is None or len(cyc) == 0:
            return []
        edges = {}
        for row in cyc:
            i, j = int(row[0]), int(row[1])
            edges.setdefault(i, []).append(j)
            edges.setdefault(j, []).append(i)
        if not edges:
            return []
        start = min(edges.keys())
        path = [start]; visited = {start}; cur = start
        while True:
            nxt = None
            for v in edges.get(cur, []):
                if v not in visited:
                    nxt = v; break
            if nxt is None:
                break
            path.append(nxt); visited.add(nxt); cur = nxt
        for v in edges.keys():
            if v not in visited:
                path.append(v)
        return path

    def _mix(verts: list[int]) -> tuple[str, float]:
        if not verts:
            return "mixed", 0.0
        labs = labels[np.array(verts, dtype=int)]
        n0 = int(np.sum(labs == 0)); n1 = int(np.sum(labs == 1))
        if n1 == 0:
            return "pure_a", 0.0
        if n0 == 0:
            return "pure_b", 0.0
        return "mixed", float(min(n0, n1) / max(n0, n1))

    cycles: list[PairCycle] = []

    if H1.size:
        pers = np.where(np.isfinite(H1[:, 1]), H1[:, 1] - H1[:, 0], 0.0)
        order = np.argsort(pers)[::-1][:top_h1]
        for idx in order:
            if pers[idx] <= 0:
                continue
            cyc = coc1[idx] if idx < len(coc1) else []
            verts = _h1_walk(cyc)
            if not verts:
                continue
            mix_label, _ = _mix(verts)
            if not include_pure and mix_label != "mixed":
                continue
            # cross-edge fraction (H1)
            total = 0; cross = 0
            for row in cyc:
                i, j = int(row[0]), int(row[1]); total += 1
                if labels[i] != labels[j]:
                    cross += 1
            cross_frac = float(cross / total) if total > 0 else 0.0
            cycles.append(PairCycle(
                dim=1, birth=float(H1[idx, 0]),
                death=float(H1[idx, 1] if np.isfinite(H1[idx, 1]) else 0.0),
                persistence=float(pers[idx]), mix=mix_label,
                cross_fraction=cross_frac,
                vertices=[CycleVertex(word=words[v], index=v,
                                      x=float(Z[v, 0]), y=float(Z[v, 1]),
                                      home_symbol=homes[v])
                          for v in verts if v < len(words)],
            ))

    if H2.size:
        pers = np.where(np.isfinite(H2[:, 1]), H2[:, 1] - H2[:, 0], 0.0)
        order = np.argsort(pers)[::-1][:top_h2]
        for idx in order:
            if pers[idx] <= 0:
                continue
            cyc = coc2[idx] if idx < len(coc2) else []
            verts = set()
            cross_t = 0; total_t = 0
            for row in cyc:
                i, j, k = int(row[0]), int(row[1]), int(row[2])
                verts.update([i, j, k]); total_t += 1
                if len({labels[i], labels[j], labels[k]}) >= 2:
                    cross_t += 1
            verts = sorted(verts)
            if not verts:
                continue
            mix_label, _ = _mix(verts)
            if not include_pure and mix_label != "mixed":
                continue
            cycles.append(PairCycle(
                dim=2, birth=float(H2[idx, 0]),
                death=float(H2[idx, 1] if np.isfinite(H2[idx, 1]) else 0.0),
                persistence=float(pers[idx]), mix=mix_label,
                cross_fraction=float(cross_t / total_t) if total_t > 0 else 0.0,
                vertices=[CycleVertex(word=words[v], index=v,
                                      x=float(Z[v, 0]), y=float(Z[v, 1]),
                                      home_symbol=homes[v])
                          for v in verts if v < len(words)],
            ))

    out = PairCyclesResponse(a=a, b=b, cycles=cycles, descriptors=descriptors,
                             ripser_available=True)
    engine_cache.memo_put(key, out)
    return out


# ---------- /topology/catalysts/{symbol} ------------------------------------

@router.post("/catalysts", response_model=WordCatalystResponse)
def topology_catalysts(req: dict):
    """Per-word topological criticality for one symbol.

    Combines:
      - leave-one-out: how much H1_sum / H2_sum drops if you delete this word
      - cycle participation: vertex-credit from top persistent cocycles

    The composite scores surface descriptors that are *holding the
    topology together* — remove them and loops collapse.
    """
    space_id = req.get("space_id"); symbol = req.get("symbol")
    if not space_id or not symbol:
        raise HTTPException(status_code=400, detail="space_id and symbol required")
    space = _require(space_id)
    _ripser_or_503()

    D, key, _ = _resolve_D_and_key(space, space_id, req, "catalysts", {"symbol": symbol})
    cached = engine_cache.memo_get(key)
    if cached is not None:
        return cached

    from ripser import ripser
    sym_emb = _symbol_embeddings_from(space, D)
    if symbol not in sym_emb:
        raise HTTPException(status_code=400, detail=f"symbol '{symbol}' has too few descriptors for PH")
    X = sym_emb[symbol]
    words = list(space.symbols_to_descriptors[symbol])[:X.shape[0]]

    # baseline — shared across endpoints via _get_ph
    base = _get_ph(space_id, symbol, X)
    H1 = base["dgms"][1]; H2 = base["dgms"][2]
    base_h1 = _sum_finite(H1); base_h2 = _sum_finite(H2)
    coc1 = base.get("cocycles", [[], [], []])[1] if "cocycles" in base else []
    coc2 = base.get("cocycles", [[], [], []])[2] if "cocycles" in base else []

    # cycle-participation weights
    cycle_w = np.zeros(len(words), dtype=float)
    def _add_weights(H, C, dim, topk):
        if not H.size:
            return
        pers = np.where(np.isfinite(H[:, 1]), H[:, 1] - H[:, 0], 0.0)
        order = np.argsort(pers)[::-1][:topk]
        for idx in order:
            cyc = C[idx] if idx < len(C) else []
            verts = set()
            if dim == 1:
                for row in cyc:
                    verts.add(int(row[0])); verts.add(int(row[1]))
            else:
                for row in cyc:
                    verts.update([int(row[0]), int(row[1]), int(row[2])])
            for v in verts:
                if v < len(words):
                    cycle_w[v] += float(pers[idx])
    _add_weights(H1, coc1, 1, topk=6)
    _add_weights(H2, coc2, 2, topk=4)

    # LOO — N ripser calls.  Parallelised across the deleted-index list so
    # bigger symbols don't take N× the per-ripser-call time.
    def _loo_one(i):
        Xi = np.delete(X, i, axis=0)
        if Xi.shape[0] < 4:
            return i, 0.0, 0.0
        dgmi = ripser(Xi, maxdim=2, metric="cosine")["dgms"]
        return i, base_h1 - _sum_finite(dgmi[1]), base_h2 - _sum_finite(dgmi[2])

    delta_h1 = np.zeros(len(words), dtype=float)
    delta_h2 = np.zeros(len(words), dtype=float)
    with ThreadPoolExecutor(max_workers=min(8, max(1, X.shape[0]))) as ex:
        for i, d1, d2 in ex.map(_loo_one, range(X.shape[0])):
            delta_h1[i] = d1
            delta_h2[i] = d2

    composite = delta_h1 + delta_h2 + 0.5 * cycle_w

    entries: list[WordCatalystEntry] = []
    for i in range(len(words)):
        entries.append(WordCatalystEntry(
            word=words[i],
            delta_h1=float(delta_h1[i]), delta_h2=float(delta_h2[i]),
            cycle_weight=float(cycle_w[i]),
            composite=float(composite[i]),
        ))
    entries.sort(key=lambda e: -e.composite)

    out = WordCatalystResponse(symbol=symbol, entries=entries,
                               h1_baseline=base_h1, h2_baseline=base_h2,
                               ripser_available=True)
    engine_cache.memo_put(key, out)
    return out
