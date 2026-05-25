"""Server-side caches for the Delyrism engine.

The engine wraps three expensive operations:
- TextEmbedder construction (loads models, ~seconds to minutes)
- SymbolSpace construction (encodes every descriptor, builds graph)
- propose / make_shifted_matrix / context_delta_graph (pure functions of the
  space + sliders — cheap once the space is built, but worth memoizing for the
  steady-state slider-twiddling case where the same params repeat).

Cache keys are JSON-canonical hashes so identical requests collapse onto one
result.  The cache is in-process; this server is intended to run as one or two
workers since the engine holds large numpy state.
"""
from __future__ import annotations
import hashlib
import json
import sys
import os
import threading
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

# Make the existing `delyrism/` package importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from delyrism import SymbolSpace, TextEmbedder  # noqa: E402


# The engine knows only these primitive backend ids:
_ENGINE_BACKENDS = {"sentence-transformer", "qwen2", "qwen3", "cloudflare", "audioclip", "clap"}

# UI-facing Cloudflare presets — same mapping the Streamlit app uses.  Each UI
# id collapses to backend="cloudflare" + a specific @cf/... model.
_CF_PRESETS = {
    "cloudflare-bge-base":        "@cf/baai/bge-base-en-v1.5",
    "cloudflare-bge-large":       "@cf/baai/bge-large-en-v1.5",
    "cloudflare-bge-m3":          "@cf/baai/bge-m3",
    "cloudflare-embeddinggemma":  "@cf/google/embeddinggemma-300m",
    "cloudflare-qwen3":           "@cf/qwen/qwen3-embedding-0.6b",
}


def _resolve_backend(backend: str, model: Any) -> tuple[str, Any]:
    """Translate a UI backend id (e.g. 'cloudflare-bge-m3') into the
    (engine_backend, model) pair the TextEmbedder constructor expects."""
    if backend in _CF_PRESETS:
        # User-supplied model overrides the preset; otherwise use the preset.
        return "cloudflare", (model or _CF_PRESETS[backend])
    if backend not in _ENGINE_BACKENDS:
        raise ValueError(
            f"Unknown embedder backend '{backend}'. "
            f"Known UI ids: {sorted({*_ENGINE_BACKENDS, *_CF_PRESETS})}"
        )
    return backend, model


def _hash(payload: Any) -> str:
    s = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ---- embedder cache ---------------------------------------------------------

_embedder_cache: Dict[str, TextEmbedder] = {}
_embedder_lock = threading.Lock()


def _add_encode_cache(emb: TextEmbedder, max_entries: int = 512) -> None:
    """Wrap `emb.encode` so single-text calls are deduplicated AND
    single-flight across concurrent callers.

    Why both:
    * Every context-touching endpoint (propose / attention / subgraph /
      delta / shift / similarity) ultimately calls
      `embedder.encode([sentence])`.  With a remote backend (Cloudflare)
      each call is a 200–500 ms round-trip.
    * On a sentence change, 6 endpoints fire in parallel — without
      single-flight all six race past a plain "check cache, miss, fetch"
      and we send 6 identical embed requests, blowing through the rate
      limit.

    The single-flight pattern: the first caller for an unseen text becomes
    the leader and does the upstream call; everyone else for the same text
    blocks on a `threading.Event` until the leader publishes the result.
    One CF request per unique sentence regardless of how many endpoints
    ask for it.

    Multi-text batches (descriptor encoding, reembed strategy) bypass the
    cache entirely and call the original implementation unchanged.
    """
    if getattr(emb, "_encode_cached", False):
        return
    original = emb.encode
    cache: Dict[Tuple[str, str, str], Any] = {}
    in_flight: Dict[Tuple[str, str, str], threading.Event] = {}
    lock = threading.Lock()

    def wrapped(texts, instruction=None, context=None, batch_size: int = 32, **kw):
        if isinstance(texts, list) and len(texts) == 1 and not kw and context is None:
            key = (texts[0], instruction or "", "")
            with lock:
                if key in cache:
                    return cache[key]
                ev = in_flight.get(key)
                if ev is not None:
                    leader = False
                else:
                    ev = threading.Event()
                    in_flight[key] = ev
                    leader = True

            if not leader:
                # wait for the leader to finish; cap at 60s so a stuck
                # request can't hang us forever
                ev.wait(timeout=60)
                cached = cache.get(key)
                if cached is not None:
                    return cached
                # leader failed and cleared its event without populating
                # the cache — fall through and try ourselves
                pass

            try:
                v = original(texts, instruction=instruction, context=context, batch_size=batch_size)
            except Exception:
                with lock:
                    in_flight.pop(key, None)
                    ev.set()
                raise
            with lock:
                if len(cache) >= max_entries:
                    for k in list(cache.keys())[: max_entries // 4]:
                        cache.pop(k, None)
                cache[key] = v
                in_flight.pop(key, None)
                ev.set()
            return v
        return original(texts, instruction=instruction, context=context, batch_size=batch_size, **kw)

    emb.encode = wrapped  # type: ignore[assignment]
    emb._encode_cached = True  # type: ignore[attr-defined]


def get_embedder(cfg: Dict[str, Any]) -> Tuple[str, TextEmbedder]:
    """Return a (fingerprint, embedder) pair. Reuses cached instance if possible.

    Resolves UI backend ids (e.g. 'cloudflare-bge-m3') to the engine's
    (backend, model) pair before instantiation.  The cache fingerprint is
    computed on the *original* cfg so identical UI requests still collapse to
    one instance even though the constructor args were rewritten."""
    fp = _hash(cfg)
    with _embedder_lock:
        emb = _embedder_cache.get(fp)
        if emb is None:
            engine_backend, engine_model = _resolve_backend(
                cfg.get("backend", "cloudflare-bge-m3"),
                cfg.get("model"),
            )
            emb = TextEmbedder(
                backend=engine_backend,
                model=engine_model,
                pooling=cfg.get("pooling", "eos"),
                default_instruction=cfg.get("default_instruction"),
                default_context=cfg.get("default_context"),
            )
            _add_encode_cache(emb)
            _embedder_cache[fp] = emb
    return fp, emb


# ---- space cache -----------------------------------------------------------

_space_cache: Dict[str, SymbolSpace] = {}
_space_lock = threading.Lock()


def make_space_id(symbols: Dict[str, list], embedder_fp: str, descriptor_threshold: float, contextual_embeddings: bool) -> str:
    return _hash({
        "symbols": symbols,
        "emb": embedder_fp,
        "thr": round(float(descriptor_threshold), 6),
        "ctx_emb": bool(contextual_embeddings),
    })


def get_or_build_space(
    symbols: Dict[str, list],
    embedder_cfg: Dict[str, Any],
    descriptor_threshold: float = 0.2,
    contextual_embeddings: bool = False,
    palette: str = "AuroraPop",
) -> Tuple[str, SymbolSpace]:
    emb_fp, embedder = get_embedder(embedder_cfg)
    sid = make_space_id(symbols, emb_fp, descriptor_threshold, contextual_embeddings)
    with _space_lock:
        sp = _space_cache.get(sid)
        if sp is None:
            sp = SymbolSpace(
                symbols_to_descriptors=symbols,
                embedder=embedder,
                descriptor_threshold=float(descriptor_threshold),
                contextual_embeddings=bool(contextual_embeddings),
            )
            _space_cache[sid] = sp
        # Stash the requested palette on the cached space so every endpoint
        # (delta-graph, subgraph, …) renders with the same colors the
        # /spaces response just returned to the client.  Last writer wins —
        # if the user rebuilds with a different palette we update in place.
        setattr(sp, "_palette", palette)
    return sid, sp


def get_palette(space) -> str:
    return getattr(space, "_palette", "AuroraPop")


# ---- UMAP model cache --------------------------------------------------------
# UMAP is the only reducer that's both (a) slow to fit and (b) supports
# .transform() for projecting new points into an existing embedding.  We fit
# once on [descriptors ; centroids] when /shift or /reduce-2d first needs it,
# then reuse the model so future calls only do the cheap projection step.
# This is what makes "dots stay still, only arrows move" possible — both
# endpoints read from the same cached layout.

def get_umap_layout(space):
    """Return (umap_model, Y_descriptors, Y_centroids).

    Lazily fits a UMAP model on the concatenation of the space's descriptor
    matrix and the symbol centroids, caches it on the space instance, and
    returns the projected positions.  Subsequent calls reuse the cached
    model — including for shifted descriptor matrices via `model.transform()`.
    """
    import numpy as np
    cached = getattr(space, "_umap_layout", None)
    if cached is not None:
        return cached

    import umap
    centroids = np.stack([space.symbol_centroids[s] for s in space.symbols])
    n = space.D.shape[0]
    joint = np.vstack([space.D, centroids])

    model = umap.UMAP(
        n_neighbors=min(20, max(2, joint.shape[0] - 1)),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    Y = model.fit_transform(joint)
    layout = (model, Y[:n], Y[n:])
    space._umap_layout = layout  # type: ignore[attr-defined]
    return layout


def get_space(space_id: str) -> Optional[SymbolSpace]:
    return _space_cache.get(space_id)


# ---- result memo  -----------------------------------------------------------
# Wrap repeated calls (same space + same params) with a small LRU.

@lru_cache(maxsize=256)
def _memo_propose(space_id: str, payload_hash: str):
    raise RuntimeError("Use memo_call instead")


_result_cache: Dict[str, Any] = {}
_result_lock = threading.Lock()
_RESULT_CACHE_MAX = 512


def memo_get(key: str) -> Any:
    return _result_cache.get(key)


def memo_put(key: str, value: Any) -> None:
    with _result_lock:
        if len(_result_cache) > _RESULT_CACHE_MAX:
            # crude FIFO eviction
            for k in list(_result_cache.keys())[: _RESULT_CACHE_MAX // 4]:
                _result_cache.pop(k, None)
        _result_cache[key] = value


def memo_key(space_id: str, op: str, params: Dict[str, Any]) -> str:
    return f"{space_id}:{op}:{_hash(params)}"


def invalidate_space(space_id: str) -> None:
    """Drop a space and any cached results referencing it."""
    with _space_lock:
        _space_cache.pop(space_id, None)
    with _result_lock:
        for k in list(_result_cache.keys()):
            if k.startswith(f"{space_id}:"):
                _result_cache.pop(k, None)


def invalidate_results(space_id: str) -> None:
    """Drop only the result memo for a space — keep the SymbolSpace itself.

    Used when an in-memory aspect of the space changes that ISN'T part of
    any result-cache key — most importantly `context_override` (set by
    /context/set-override).  Without this, /propose etc. keep returning
    the previously-cached "no context" result after audio is applied.
    """
    with _result_lock:
        for k in list(_result_cache.keys()):
            if k.startswith(f"{space_id}:"):
                _result_cache.pop(k, None)
    # Shift-matrix cache is also keyed on (space_id, params) and depends on
    # context_override via ctx_vec — same invalidation contract.
    with _shift_lock:
        for k in list(_shift_cache.keys()):
            if k.startswith(f"{space_id}:"):
                _shift_cache.pop(k, None)


# ---- shifted-matrix cache ---------------------------------------------------
# /shift, /delta-graph, /similarity, /similarity-symbols, /shift-spectrum all
# call SymbolSpace.make_shifted_matrix with the same params on every keystroke
# (after the debounced textarea settles, all five panels refire in parallel).
# Without sharing they each recompute the same N×d matrix.  Cheap for `gate`
# strategy but the per-endpoint duplicated work still costs ~5×.  For
# `reembed` / `hybrid` it's N encoder calls × 5 endpoints = catastrophic.
#
# Cache a numpy ndarray keyed on the shift params.  Same invalidation rules
# as the result memo (cleared when context_override changes).

_shift_cache: Dict[str, Any] = {}
_shift_lock = threading.Lock()
_SHIFT_CACHE_MAX = 64  # ~50 MB at d=1024, N=400, float32


def get_or_compute_shifted_matrix(space_id: str, space, params: Dict[str, Any]):
    """Return the cached D' for these exact params, or compute + memoize.

    `params` is the kwargs dict for SymbolSpace.make_shifted_matrix —
    weights / sentence / strategy / beta / gate / tau / etc.  Keying on the
    full dict means any change in any param invalidates the cache; identical
    payloads collapse onto one computation.
    """
    key = f"{space_id}:shift:{_hash(params)}"
    with _shift_lock:
        cached = _shift_cache.get(key)
        if cached is not None:
            return cached
    D = space.make_shifted_matrix(**params)
    with _shift_lock:
        if len(_shift_cache) > _SHIFT_CACHE_MAX:
            for k in list(_shift_cache.keys())[: _SHIFT_CACHE_MAX // 4]:
                _shift_cache.pop(k, None)
        _shift_cache[key] = D
    return D


def cache_stats() -> Dict[str, int]:
    return {
        "embedders": len(_embedder_cache),
        "spaces": len(_space_cache),
        "results": len(_result_cache),
    }
