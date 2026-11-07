"""Lakota symbol affinity for the TouchDesigner background, for any space.

The background tiles Lakota symbols (td/assets/lakota).  Which ones, and in
what shares, follows the words the live graph activates — whatever symbols
the space itself was built from.  Each Lakota symbol is a prototype: the mean
embedding of its descriptors in delyrism/structures/lakota.json, embedded by
the space's own text embedder.  Every descriptor of the space gets a soft
assignment over the Lakota symbols (a softmax over its standardised cosines,
so the spread is the same for every embedder); a descriptor of a space symbol
that *is* a Lakota symbol (the Lakota space) belongs to it outright.

Per OSC frame the activation of Lakota symbol k is

    Σ_i w_i · A[i, k]      w_i = relevance (query) or size (no query), alive nodes

plus, for a text / movement query, the query's own assignment (blended in
with the same mass as the nodes), so a word that isn't in the graph still
pulls its Lakota symbols.  Computed once per space (one batch of ~140 texts).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_PATH = Path(__file__).resolve().parents[3] / "delyrism" / "structures" / "lakota.json"
LAKOTA: Dict[str, List[str]] = json.loads(_PATH.read_text(encoding="utf-8"))
NAMES: List[str] = list(LAKOTA)
_UPPER = {n.upper(): k for k, n in enumerate(NAMES)}
SHARPNESS = 2.5        # softmax gain on the standardised cosines
_LOCK = threading.Lock()


def _unit_rows(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def _soft(C: np.ndarray) -> np.ndarray:
    """Rows of cosines → rows of shares (z-scored per row, then softmax)."""
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))
    z = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-9)
    e = np.exp(SHARPNESS * (z - z.max(1, keepdims=True)))
    return e / e.sum(1, keepdims=True)


def prototypes(space) -> np.ndarray:
    """K × dim unit prototypes of the Lakota symbols in the space's text embedder (cached)."""
    P = getattr(space, "_lakota_protos", None)
    if P is not None:
        return P
    with _LOCK:
        P = getattr(space, "_lakota_protos", None)
        if P is None:
            texts = [d for n in NAMES for d in LAKOTA[n]]
            E = _unit_rows(space.embedder.encode(texts))
            P, o = [], 0
            for n in NAMES:
                m = len(LAKOTA[n])
                P.append(E[o:o + m].mean(0))
                o += m
            P = _unit_rows(np.stack(P))
            space._lakota_protos = P
    return P


def affinity(space) -> np.ndarray:
    """n_descriptors × K soft assignment of every descriptor of the space (cached)."""
    A = getattr(space, "_lakota_aff", None)
    if A is not None:
        return A
    P = prototypes(space)
    A = _soft(_unit_rows(space.D) @ P.T).astype(np.float32)
    for i, d in enumerate(space.descriptors):            # Lakota space: its own symbol, outright
        k = _UPPER.get(str(space.owner.get(d, "")).upper())
        if k is not None:
            A[i] = 0.0
            A[i, k] = 1.0
    space._lakota_aff = A
    return A


def node_rows(space, ids: List[str]) -> Dict[str, np.ndarray]:
    """{node id: K shares} for the graph's nodes."""
    A = affinity(space)
    return {n: A[space.desc_idx[n]] for n in ids if n in space.desc_idx}


def query_shares(space, vq: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """K shares for a query vector in the space's text embedding, or None."""
    if vq is None:
        return None
    v = np.asarray(vq, dtype=np.float32).ravel()
    P = prototypes(space)
    if v.shape[0] != P.shape[1] or float(np.linalg.norm(v)) < 1e-9:
        return None
    return _soft(P @ (v / np.linalg.norm(v)))[0].astype(np.float32)
