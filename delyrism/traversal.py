"""
traversal.py — generative symbol construction by walking the context Δ graph
===========================================================================

The Explorer's Δ-graph is a *snapshot*: give the engine a context, it shows
which descriptor pairs moved closer (or apart).  This module turns that
snapshot into a *process*.  A `DeltaWalker` sits on one descriptor, looks at
the Δ-field induced by the current context, and steps to the descriptor the
context is pulling it toward.  The trail it leaves — an ordered set of
descriptors drawn from several archetypes — is a *new symbol*, assembled on
the fly.  Seal it, hand it a name, and it is a `{symbol: [descriptors]}` entry
like any preset.

The walker is deliberately transport-agnostic: it knows nothing about HTTP,
WebSockets, OSC or the browser.  You feed it a context vector (from text,
from a 2D pointer over the layout, from a blend of archetype anchors) and
ask it for candidates / steps / frames.  `web/backend/app/routes/realtime.py`
wraps it in a WebSocket; `td_realtime.py` can wrap it in OSC.

Core loop
---------
    walker = DeltaWalker(space)
    walker.set_context(walker.context_from_text("a slow fire under ice"))
    walker.compute_field()                 # Δ = D1·D1ᵀ − D·Dᵀ  under that context
    for _ in range(walker.params.max_len):
        walker.step()                      # argmax / softmax over Δ-candidates
    sym = walker.symbol()                  # {'descriptors': [...], 'composition': {...}, ...}

Scoring
-------
For a path p_0 … p_T (current node p_T) and candidate j:

    score_j = Σ_t w_t · S[p_t, j]                   Δ pull (sign-filtered)
            + coherence · Σ_t w_t · C0[p_t, j]      base cosine, keeps the walk continuous
            − diversity · Σ_t w_t · [owner_j = owner_{p_t}]   cross-archetype pressure

with recency weights w_t ∝ memory^(T−t) (memory=0 → only the current node,
memory=1 → the whole path counts equally).  Visited descriptors (by *name*,
since presets repeat words across archetypes) are masked out.  The top-K by
score are the candidates; selection is a range-normalised softmax at
`temperature` (→ 0 is greedy).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence
import math

import numpy as np

from .delyrism import l2_normalize


# ----------------------------------------------------------------------------
#  parameters
# ----------------------------------------------------------------------------

@dataclass
class WalkerParams:
    # --- context-shift (same semantics as SymbolSpace.make_shifted_matrix) ---
    # Only the cheap, closed-form strategies make sense at interactive rates.
    strategy: str = "gate"            # gate | pooling
    beta: float = 1.2
    gate: str = "relu"                # relu | cos | softmax | uniform
    tau: float = 0.3
    within_symbol_softmax: bool = True
    pool_type: str = "avg"
    pool_w: float = 0.7
    membership_alpha: float = 0.0

    # --- traversal ---
    sign: str = "up"                  # up (strengthening) | down (weakening) | abs
    temperature: float = 0.25         # 0 → greedy
    memory: float = 0.5               # recency decay over the path
    diversity: float = 0.35           # penalty for staying inside one archetype
    coherence: float = 0.15           # weight of base cosine in the score
    n_candidates: int = 6
    max_len: int = 8
    auto_seal: bool = True            # seal + start a new symbol at max_len
    self_weight: float = 0.0          # autopoiesis: blend path centroid into context

    # --- field export ---
    hot_edges: int = 90               # strongest |Δ| edges returned per frame
    seed: int = 7

    def update(self, patch: Dict) -> "WalkerParams":
        for k, v in patch.items():
            if hasattr(self, k) and v is not None:
                cur = getattr(self, k)
                if isinstance(cur, bool):
                    setattr(self, k, bool(v))
                elif isinstance(cur, int):
                    setattr(self, k, int(v))
                elif isinstance(cur, float):
                    setattr(self, k, float(v))
                else:
                    setattr(self, k, v)
        return self

    def to_dict(self) -> Dict:
        return asdict(self)


# ----------------------------------------------------------------------------
#  walker
# ----------------------------------------------------------------------------

class DeltaWalker:
    """Stateful traversal of the context Δ-field over one SymbolSpace."""

    def __init__(self, space, params: Optional[WalkerParams] = None,
                 positions: Optional[np.ndarray] = None,
                 centroid_positions: Optional[np.ndarray] = None):
        self.space = space
        self.params = params or WalkerParams()

        self.D = np.asarray(space.D, dtype=np.float32)
        self.N, self.dim = self.D.shape
        self.names: List[str] = list(space.descriptors)
        self.symbols: List[str] = list(space.symbols)
        sym_index = {s: i for i, s in enumerate(self.symbols)}
        self.owner_idx = np.array([sym_index[space.owner[d]] for d in self.names], dtype=np.int32)
        # duplicate words across archetypes → mask by name
        self.name_groups: Dict[str, List[int]] = {}
        for i, n in enumerate(self.names):
            self.name_groups.setdefault(n, []).append(i)

        cents = np.stack([space.symbol_centroids[s] for s in self.symbols]).astype(np.float32)
        self.centroids = l2_normalize(cents, axis=1)

        self.C0 = self.D @ self.D.T
        np.fill_diagonal(self.C0, 0.0)

        # 2D layout (unit square) — drives pointer → context and the field view
        if positions is None:
            positions = space._pca.transform(space.D)
            centroid_positions = space._pca.transform(cents)
        self.P, self.PC = self._normalise_layout(np.asarray(positions, dtype=np.float32),
                                                 None if centroid_positions is None
                                                 else np.asarray(centroid_positions, dtype=np.float32))

        self.rng = np.random.default_rng(self.params.seed)
        self._text_cache: Dict[str, np.ndarray] = {}

        # context state
        self.ctx_external = np.zeros(self.dim, dtype=np.float32)
        self.ctx = np.zeros(self.dim, dtype=np.float32)
        self.ctx_point: Optional[List[float]] = None

        # field state
        self.D1 = self.D
        self.Delta = np.zeros_like(self.C0)
        self.align = np.zeros(self.N, dtype=np.float32)
        self._field_version = 0

        self.sealed: List[Dict] = []
        self.reset()

    # ------------------------------------------------------------------ layout

    @staticmethod
    def _normalise_layout(P: np.ndarray, PC: Optional[np.ndarray]):
        allp = P if PC is None else np.vstack([P, PC])
        lo = allp.min(axis=0)
        hi = allp.max(axis=0)
        span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
        pad = 0.06
        f = lambda X: pad + (1 - 2 * pad) * (X - lo) / span
        return f(P), (None if PC is None else f(PC))

    def layout(self) -> Dict:
        return {
            "nodes": [
                {"id": self.names[i], "symbol": self.symbols[self.owner_idx[i]],
                 "x": float(self.P[i, 0]), "y": float(self.P[i, 1])}
                for i in range(self.N)
            ],
            "centroids": [
                {"symbol": s, "x": float(self.PC[k, 0]), "y": float(self.PC[k, 1])}
                for k, s in enumerate(self.symbols)
            ] if self.PC is not None else [],
            "symbols": self.symbols,
        }

    # ----------------------------------------------------------------- context

    def context_from_text(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        v = self._text_cache.get(text)
        if v is None:
            v = self.space._sent_vec(text, ignore_override=True).astype(np.float32)
            self._text_cache[text] = v
        return v

    def context_from_point(self, u: float, v: float, bandwidth: float = 0.08) -> np.ndarray:
        """Gaussian kernel over descriptor layout positions around (u, v) ∈ [0,1]²."""
        d2 = ((self.P - np.array([u, v], dtype=np.float32)) ** 2).sum(axis=1)
        w = np.exp(-d2 / (2.0 * max(bandwidth, 1e-3) ** 2))
        s = float(w.sum())
        if s < 1e-6:
            return np.zeros(self.dim, dtype=np.float32)
        vec = (w[:, None] * self.D).sum(axis=0) / s
        n = float(np.linalg.norm(vec))
        return (vec / n).astype(np.float32) if n > 1e-9 else np.zeros(self.dim, dtype=np.float32)

    def context_from_anchors(self, weights: Dict[str, float]) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tot = 0.0
        for k, s in enumerate(self.symbols):
            w = float(weights.get(s, 0.0))
            if w > 0:
                vec += w * self.centroids[k]
                tot += w
        if tot <= 1e-9:
            return np.zeros(self.dim, dtype=np.float32)
        n = float(np.linalg.norm(vec))
        return vec / n if n > 1e-9 else np.zeros(self.dim, dtype=np.float32)

    @staticmethod
    def blend(vectors: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
        acc = None
        for v, w in zip(vectors, weights):
            if v is None or w <= 0 or float(np.linalg.norm(v)) < 1e-9:
                continue
            acc = w * v if acc is None else acc + w * v
        if acc is None:
            return None
        n = float(np.linalg.norm(acc))
        return (acc / n).astype(np.float32) if n > 1e-9 else None

    def set_context(self, vec: Optional[np.ndarray], point: Optional[Sequence[float]] = None) -> None:
        self.ctx_external = (np.zeros(self.dim, dtype=np.float32) if vec is None
                             else np.asarray(vec, dtype=np.float32))
        self.ctx_point = None if point is None else [float(point[0]), float(point[1])]

    def _effective_context(self) -> np.ndarray:
        sw = float(self.params.self_weight)
        if sw > 0 and self.path:
            selfvec = self.D[self.path].mean(axis=0)
            v = self.blend([self.ctx_external, selfvec], [1.0 - sw, sw])
            if v is not None:
                return v
        return self.ctx_external

    def has_context(self) -> bool:
        return float(np.linalg.norm(self.ctx)) > 1e-6

    # ------------------------------------------------------------------- field

    def _shift(self, vctx: np.ndarray) -> np.ndarray:
        p = self.params
        if float(np.linalg.norm(vctx)) < 1e-6:
            return self.D
        if p.strategy == "pooling":
            D1 = self.space._pool_with_context(self.D, vctx, pool_type=p.pool_type, pool_w=p.pool_w)
        else:
            sims = self.D @ vctx
            if p.gate == "relu":
                g = np.maximum(0.0, sims)
            elif p.gate == "cos":
                g = sims
            elif p.gate == "softmax":
                from .delyrism import softmax
                if p.within_symbol_softmax:
                    g = np.zeros_like(sims)
                    for k in range(len(self.symbols)):
                        idx = np.where(self.owner_idx == k)[0]
                        if len(idx):
                            g[idx] = softmax(sims[idx], tau=p.tau)
                else:
                    g = softmax(sims, tau=p.tau)
            else:
                g = np.ones_like(sims)
            D1 = l2_normalize(self.D + (p.beta * g[:, None]) * vctx[None, :], axis=1)
        if float(p.membership_alpha) > 0:
            D1 = self.space._apply_membership_blend(self.D, D1, float(p.membership_alpha))
        return D1.astype(np.float32)

    def compute_field(self) -> None:
        """Recompute D1, Δ and per-node alignment for the current context."""
        self.ctx = self._effective_context()
        self.D1 = self._shift(self.ctx)
        if self.D1 is self.D:
            self.Delta = np.zeros_like(self.C0)
            self.align = np.zeros(self.N, dtype=np.float32)
        else:
            C1 = self.D1 @ self.D1.T
            np.fill_diagonal(C1, 0.0)
            self.Delta = C1 - self.C0
            self.align = (self.D1 @ self.ctx).astype(np.float32)
        self._field_version += 1

    def _signed(self) -> np.ndarray:
        s = self.params.sign
        if s == "down":
            return -self.Delta
        if s == "abs":
            return np.abs(self.Delta)
        return self.Delta

    # ---------------------------------------------------------------- walking

    def reset(self, start: Optional[int] = None) -> None:
        self.path: List[int] = []
        self.gains: List[float] = []
        self.visited_names: set = set()
        self._cand_cache = None
        if start is not None:
            self._append(int(start), 0.0)

    def _append(self, idx: int, gain: float) -> None:
        self.path.append(idx)
        self.gains.append(float(gain))
        self.visited_names.add(self.names[idx])
        self._cand_cache = None

    def undo(self) -> None:
        if not self.path:
            return
        idx = self.path.pop()
        self.gains.pop()
        if self.names[idx] not in {self.names[i] for i in self.path}:
            self.visited_names.discard(self.names[idx])
        self._cand_cache = None

    def _recency_weights(self) -> np.ndarray:
        T = len(self.path)
        m = float(np.clip(self.params.memory, 0.0, 1.0))
        if T == 0:
            return np.zeros(0, dtype=np.float32)
        if m <= 1e-6:
            w = np.zeros(T, dtype=np.float32); w[-1] = 1.0
            return w
        w = m ** np.arange(T - 1, -1, -1, dtype=np.float32)
        return w / w.sum()

    def _mask(self) -> np.ndarray:
        mask = np.zeros(self.N, dtype=bool)
        for n in self.visited_names:
            for i in self.name_groups.get(n, ()):
                mask[i] = True
        return mask

    def scores(self) -> np.ndarray:
        """Raw candidate score for every descriptor (−inf for masked)."""
        p = self.params
        if not self.path:
            # opening move: the descriptor the context lights up most;
            # without context, nothing to prefer → flat (random pick)
            s = self.align.copy() if self.has_context() else np.zeros(self.N, dtype=np.float32)
        else:
            w = self._recency_weights()
            rows = np.asarray(self.path)
            S = self._signed()
            pull = (w[:, None] * S[rows]).sum(axis=0)
            base = (w[:, None] * self.C0[rows]).sum(axis=0)
            same = (w[:, None] * (self.owner_idx[None, :] == self.owner_idx[rows][:, None])).sum(axis=0)
            coh = float(p.coherence) if self.has_context() else max(float(p.coherence), 1.0)
            s = pull + coh * base - float(p.diversity) * 0.05 * same
        s = s.astype(np.float64)
        s[self._mask()] = -np.inf
        return s

    def candidates(self) -> List[Dict]:
        if self._cand_cache is not None and self._cand_cache[0] == self._field_version:
            return self._cand_cache[1]
        s = self.scores()
        k = max(1, int(self.params.n_candidates))
        finite = np.isfinite(s)
        if not finite.any():
            return []
        order = np.argsort(-s)[:k]
        order = [int(i) for i in order if np.isfinite(s[i])]
        vals = s[order]
        T = max(float(self.params.temperature), 0.0)
        if len(vals) == 1 or T < 1e-3:
            probs = np.zeros(len(vals)); probs[0] = 1.0
        else:
            rng = float(vals.max() - vals.min())
            z = (vals - vals.max()) / (rng + 1e-9) / T
            e = np.exp(z - z.max())
            probs = e / e.sum()
        cur = self.path[-1] if self.path else None
        out = []
        for i, sc, pr in zip(order, vals, probs):
            out.append({
                "i": i,
                "id": self.names[i],
                "symbol": self.symbols[self.owner_idx[i]],
                "score": float(sc),
                "p": float(pr),
                "delta": float(self.Delta[cur, i]) if cur is not None else float(self.align[i]),
            })
        self._cand_cache = (self._field_version, out)
        return out

    def step(self, choice: Optional[int] = None) -> Dict:
        """Advance one node.  `choice` is a descriptor index (must be a live
        candidate) or None to let the walker sample.  Returns an event dict."""
        cands = self.candidates()
        if not cands:
            return {"event": "exhausted"}
        if choice is not None:
            pick = next((c for c in cands if c["i"] == int(choice)), None)
            if pick is None:
                # allow any unvisited node when the user insists
                if self._mask()[int(choice)]:
                    return {"event": "invalid", "detail": "already visited"}
                i = int(choice)
                cur = self.path[-1] if self.path else None
                gain = float(self.Delta[cur, i]) if cur is not None else float(self.align[i])
                self._append(i, gain)
                return self._after_step(i, forced=True)
            self._append(pick["i"], pick["delta"])
            return self._after_step(pick["i"], forced=True)
        probs = np.array([c["p"] for c in cands])
        j = int(self.rng.choice(len(cands), p=probs / probs.sum()))
        self._append(cands[j]["i"], cands[j]["delta"])
        return self._after_step(cands[j]["i"], forced=False)

    def _after_step(self, i: int, forced: bool) -> Dict:
        ev = {"event": "step", "i": i, "id": self.names[i], "forced": forced, "len": len(self.path)}
        if len(self.path) >= int(self.params.max_len) and self.params.auto_seal:
            ev["sealed"] = self.seal()
        if float(self.params.self_weight) > 0:
            self.compute_field()  # autopoiesis: the symbol-so-far re-shapes the field
        return ev

    # ---------------------------------------------------------------- symbols

    def symbol(self) -> Dict:
        """Summary of the symbol under construction."""
        if not self.path:
            return {"descriptors": [], "owners": [], "gains": [], "composition": {},
                    "coherence": 0.0, "context_fit": 0.0, "nearest": [], "name": "", "pairs": []}
        idx = np.asarray(self.path)
        owners = [self.symbols[self.owner_idx[i]] for i in idx]
        comp: Dict[str, float] = {}
        for o in owners:
            comp[o] = comp.get(o, 0.0) + 1.0
        tot = sum(comp.values())
        comp = {k: v / tot for k, v in sorted(comp.items(), key=lambda kv: -kv[1])}
        sub = self.C0[np.ix_(idx, idx)]
        n = len(idx)
        coherence = float(sub.sum() / (n * (n - 1))) if n > 1 else 1.0
        fit = float(self.align[idx].mean()) if self.has_context() else 0.0
        centroid = l2_normalize(self.D[idx].mean(axis=0, keepdims=True), axis=1)[0]
        near = self.centroids @ centroid
        order = np.argsort(-near)[:3]
        nearest = [{"symbol": self.symbols[k], "sim": float(near[k])} for k in order]
        # default name: the two strongest-gain descriptors, hyphenated
        g = np.asarray(self.gains)
        top = [self.names[idx[k]] for k in np.argsort(-g)[:2]] if n > 1 else [self.names[idx[0]]]
        name = "-".join(t.split()[0].lower() for t in top)
        return {
            "descriptors": [self.names[i] for i in idx],
            "owners": owners,
            "gains": [float(x) for x in self.gains],
            "composition": comp,
            "coherence": coherence,
            "context_fit": fit,
            "nearest": nearest,
            "name": name,
            "pairs": np.round(sub, 4).tolist(),
        }

    def seal(self, name: Optional[str] = None) -> Optional[Dict]:
        """Freeze the current path as a finished symbol and start a new one."""
        if not self.path:
            return None
        sym = self.symbol()
        if name:
            sym["name"] = name
        sym["index"] = len(self.sealed)
        sym["path"] = list(self.path)
        self.sealed.append(sym)
        self.reset()
        return sym

    def drop_sealed(self, index: int) -> None:
        self.sealed = [s for s in self.sealed if s.get("index") != index]

    # ------------------------------------------------------------------ frame

    def hot_edges(self) -> List[List[float]]:
        k = int(self.params.hot_edges)
        if k <= 0 or not self.has_context():
            return []
        S = self._signed()
        iu = np.triu_indices(self.N, k=1)
        vals = S[iu]
        if vals.size == 0:
            return []
        k = min(k, vals.size)
        part = np.argpartition(-vals, k - 1)[:k]
        part = part[np.argsort(-vals[part])]
        out = []
        for t in part:
            v = float(vals[t])
            if v <= 0:
                break
            out.append([int(iu[0][t]), int(iu[1][t]), round(v, 5)])
        return out

    def frame(self) -> Dict:
        heat = self.align
        if self.has_context():
            lo, hi = float(heat.min()), float(heat.max())
            heat = (heat - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(heat)
        return {
            "type": "frame",
            "ctx_point": self.ctx_point,
            "has_context": self.has_context(),
            "path": list(self.path),
            "gains": [round(float(g), 5) for g in self.gains],
            "candidates": self.candidates(),
            "heat": np.round(heat, 3).tolist(),
            "hot_edges": self.hot_edges(),
            "symbol": self.symbol(),
            "sealed_count": len(self.sealed),
            "params": self.params.to_dict(),
        }
