"""A Live session: what was asked of the network over time, and what it did.

`SessionRecorder` keeps one snapshot a second while a session runs (the query's relevance over
every descriptor, its kind and text) and every event (context changes, queries, heard and spoken
words, movement descriptions, Δ-graph changes).  `analyse` turns a recording into the panels of
the session view:

  trajectory   where the query's attention sat on the space's 2D map, second by second
               (relevance-weighted position of its top descriptors), dwell points, the context's
               own path
  territory    each descriptor's share of time among the top 5, the share of the network
               reached, the descriptors never reached, each symbol's share of time on top and
               its share in each minute
  topology     in sliding windows, how many separate groups the attended descriptors form
               (single-linkage at the network's own neighbour scale, H0) and how many loops they
               close (H1 persistence, ripser when installed), with the moments these change; and
               the Δ-graph's edges born and dying at each context change
  phases       the session cut where its relevance profile shifts (binary segmentation), each
               phase named by its top words and symbol, and a recurrence matrix (correlation of
               relevance profiles between moments) with the returns it shows
  archetypes   the visited descriptors clustered into session archetypes, a structure to export
  fingerprint  symbol time shares, diversity, drift, recurrence: one glyph per session

    rec = SessionRecorder(space_meta)
    rec.snapshot(t, rel, kind, query)        # every second
    rec.event(t, "context", "a slow fire")   # as they happen
    result = analyse(rec.to_raw(), D, layout)
"""
from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

TOP_K = 5


# ----------------------------------------------------------------------------
#  recording
# ----------------------------------------------------------------------------

class SessionRecorder:
    def __init__(self, space_id: str, descriptors: Sequence[str], owners: Dict[str, str],
                 symbols: Sequence[str], name: str = ""):
        self.space_id = space_id
        self.descriptors = list(descriptors)
        self.owners = dict(owners)
        self.symbols = list(symbols)
        self.name = name
        self.started = time.time()
        self.t0 = time.perf_counter()
        self.t: List[float] = []
        self.rel: List[Optional[np.ndarray]] = []
        self.kind: List[str] = []
        self.query: List[str] = []
        self.events: List[Dict[str, Any]] = []
        self.ended: Optional[float] = None

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def snapshot(self, rel: Optional[np.ndarray], kind: str, query: str, t: Optional[float] = None) -> None:
        """One moment: the relevance over every descriptor (None: no query)."""
        self.t.append(self.now() if t is None else float(t))
        self.rel.append(None if rel is None else np.asarray(rel, dtype=np.float32))
        self.kind.append(kind)
        self.query.append(query)

    def event(self, kind: str, text: str = "", t: Optional[float] = None, **extra) -> None:
        self.events.append({"t": round(self.now() if t is None else float(t), 3), "type": kind, "text": text, **extra})

    def stop(self) -> None:
        self.ended = self.now()

    def to_raw(self) -> Dict[str, Any]:
        return {
            "format": "delyrism-session/1", "space_id": self.space_id, "name": self.name,
            "started": self.started, "duration": round(self.ended if self.ended is not None else self.now(), 3),
            "descriptors": self.descriptors, "owners": self.owners, "symbols": self.symbols,
            "t": [round(x, 3) for x in self.t], "kind": self.kind, "query": self.query,
            # a moment with no query is null (valid JSON, unlike NaN)
            "rel": [None if r is None else np.round(r, 5).tolist() for r in self.rel], "events": self.events,
        }


# ----------------------------------------------------------------------------
#  analysis
# ----------------------------------------------------------------------------

def _zrows(R: np.ndarray) -> np.ndarray:
    return (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-9)


def _attention_xy(R: np.ndarray, XY: np.ndarray, temp: float = 0.6) -> np.ndarray:
    """Relevance-weighted map position per row: softmax over z-scored relevance (top few dominate)."""
    Z = _zrows(R) / temp
    W = np.exp(Z - Z.max(1, keepdims=True))
    W /= W.sum(1, keepdims=True)
    return W @ XY


def _symbol_profile(R: np.ndarray, descriptors, owners, symbols) -> np.ndarray:
    cols = {s: [i for i, d in enumerate(descriptors) if owners.get(d) == s] for s in symbols}
    return np.stack([R[:, cols[s]].mean(1) if cols[s] else np.full(len(R), -np.inf) for s in symbols], 1)


def _segments(Z: np.ndarray, min_len: int, max_phases: int = 8, min_gain: float = 0.06) -> List[int]:
    """Binary segmentation of rows of Z (n × d) by within-segment sum of squares.
    Returns the start index of each phase (0 first)."""
    n = len(Z)
    if n < 2 * min_len:
        return [0]
    cs = np.vstack([np.zeros(Z.shape[1]), np.cumsum(Z, 0)])
    cs2 = np.concatenate([[0.0], np.cumsum((Z ** 2).sum(1))])

    def sse(a, b):
        s = cs[b] - cs[a]
        return float(cs2[b] - cs2[a] - (s @ s) / max(b - a, 1))

    total = sse(0, n)
    cuts = [0, n]
    while len(cuts) - 1 < max_phases:
        best = None
        for a, b in zip(cuts[:-1], cuts[1:]):
            base = sse(a, b)
            for c in range(a + min_len, b - min_len + 1):
                gain = base - sse(a, c) - sse(c, b)
                if best is None or gain > best[0]:
                    best = (gain, c)
        if best is None or best[0] < min_gain * total:
            break
        cuts = sorted(cuts + [best[1]])
    return cuts[:-1]


def _clusters_and_loops(V: np.ndarray, scale: float) -> tuple:
    """(groups at `scale` by single linkage, persistent loops) of unit vectors V, cosine distance."""
    if len(V) < 2:
        return len(V), 0
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    d = np.clip(pdist(V, "cosine"), 0, 2)
    groups = int(fcluster(linkage(d, "single"), scale, "distance").max())
    loops = 0
    if len(V) >= 4:
        try:
            import warnings
            from ripser import ripser
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")          # "more columns than rows": expected for embeddings
                h1 = ripser(V, maxdim=1, metric="cosine")["dgms"][1]
            if h1.size:
                loops = int(((h1[:, 1] - h1[:, 0]) > 0.02).sum())
        except ImportError:
            pass
    return groups, loops


def analyse(raw: Dict[str, Any], D: np.ndarray, XY: np.ndarray, colors: Optional[Dict[str, str]] = None,
            window: Optional[float] = None, stride: Optional[float] = None) -> Dict[str, Any]:
    """The session view from a recording.  D: descriptor vectors (rows = raw["descriptors"]),
    XY: their positions on the space's 2D map.  Topology windows default to a quarter of the
    session, at most 30 s, sliding by a third of a window."""
    descriptors, owners, symbols = raw["descriptors"], raw["owners"], raw["symbols"]
    N = len(descriptors)
    t = np.asarray(raw["t"], dtype=float)
    R_all = (np.array([[np.nan] * N if row is None else row for row in raw["rel"]], dtype=float).reshape(len(t), N)
             if len(t) else np.zeros((0, N)))
    active = ~np.isnan(R_all).any(1) if len(t) else np.zeros(0, bool)
    R, ta = R_all[active], t[active]
    D = np.asarray(D, dtype=float)
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    XY = np.asarray(XY, dtype=float)
    span = np.ptp(XY, 0).max() or 1.0
    XYn = (XY - XY.min(0)) / span                                  # map units 0…1
    colors = colors or {}
    out: Dict[str, Any] = {
        "summary": {"duration": raw["duration"], "snapshots": int(len(t)), "active_share": round(float(active.mean()), 3) if len(t) else 0.0,
                    "kinds": sorted(set(k for k, a in zip(raw["kind"], active) if a)), "name": raw.get("name", "")},
        "layout": [{"id": d, "symbol": owners.get(d, ""), "x": round(float(x), 4), "y": round(float(y), 4),
                    "color": colors.get(owners.get(d, ""), "#888")} for d, (x, y) in zip(descriptors, XYn)],
        "symbols": [{"id": s, "color": colors.get(s, "#888")} for s in symbols],
    }
    # E · the session's words, in order (typed and heard queries, spoken words, movement descriptions)
    out["words"] = [{"t": e["t"], "type": e["type"], "text": e["text"]} for e in raw["events"]
                    if e["type"] in ("query", "heard", "spoken", "motion", "context") and str(e.get("text") or "").strip()]
    if len(R) < 2:
        out["empty"] = True
        return out

    # ---- A · trajectory --------------------------------------------------------------
    P = _attention_xy(R, XYn)
    step = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    dt = np.r_[1.0, np.clip(np.diff(ta), 1e-3, None)]
    speed = step / dt
    top = np.argsort(-R, 1)[:, :TOP_K]
    # dwells: stretches where attention stays within a small radius of where it settled
    dwells, i, radius = [], 0, 0.08
    min_dwell = max(4.0, float(ta[-1] - ta[0]) / 30)
    while i < len(P):
        j = i
        while j + 1 < len(P) and np.linalg.norm(P[j + 1] - P[i:j + 2].mean(0)) < radius:
            j += 1
        if ta[j] - ta[i] >= min_dwell:
            w = np.bincount(top[i:j + 1].ravel(), minlength=N)
            dwells.append({"t0": round(float(ta[i]), 1), "t1": round(float(ta[j]), 1),
                           "x": round(float(P[i:j + 1, 0].mean()), 4), "y": round(float(P[i:j + 1, 1].mean()), 4),
                           "words": [descriptors[k] for k in np.argsort(-w)[:3]]})
            i = j + 1
        else:
            i += 1
    ctx_path = []
    for e in raw["events"]:                    # the context's own place: its relevance, recorded with the event
        if e["type"] == "context" and e.get("rel") and len(e["rel"]) == N:
            x, y = _attention_xy(np.asarray([e["rel"]], dtype=float), XYn)[0]
            ctx_path.append({"t": e["t"], "text": e["text"], "x": round(float(x), 4), "y": round(float(y), 4)})
    S_now = _symbol_profile(R, descriptors, owners, symbols).argmax(1)       # the leading symbol, moment by moment
    out["trajectory"] = {"path": [{"t": round(float(a), 2), "x": round(float(x), 4), "y": round(float(y), 4),
                                   "kind": raw["kind"][k], "s": int(S_now[i]), "top": [descriptors[j] for j in top[i, :3]],
                                   "q": raw["query"][k]}
                                  for i, (k, a, (x, y)) in enumerate(zip(np.flatnonzero(active), ta, P))],
                         "dwells": dwells, "context_path": ctx_path,
                         "distance": round(float(step.sum()), 3)}

    # ---- B · territory ---------------------------------------------------------------
    visits = np.bincount(top.ravel(), minlength=N) / float(len(R))      # share of moments in the top 5
    S = _symbol_profile(R, descriptors, owners, symbols)
    top_sym = S.argmax(1)
    sym_share = np.bincount(top_sym, minlength=len(symbols)) / float(len(R))
    edges = np.arange(0.0, max(float(ta[-1]), 60.0) + 60.0, 60.0)
    stream = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ta >= a) & (ta < b)
        if m.any():
            stream.append({"t0": float(a), "t1": float(b),
                           "shares": [round(float(x), 3) for x in np.bincount(top_sym[m], minlength=len(symbols)) / m.sum()]})
    out["territory"] = {"visits": {d: round(float(v), 4) for d, v in zip(descriptors, visits) if v > 0},
                        "reached": round(float((visits > 0).mean()), 3),
                        "never": [d for d, v in zip(descriptors, visits) if v == 0],
                        "symbol_share": {s: round(float(x), 3) for s, x in zip(symbols, sym_share)},
                        "stream": stream}

    # ---- C · topology over time --------------------------------------------------------
    span_t = float(ta[-1] - ta[0]) if len(ta) > 1 else 0.0
    window = window or float(np.clip(span_t / 4, 6.0, 30.0))
    stride = stride or window / 3
    nn = np.sort(np.clip(1 - D @ D.T, 0, 2), 1)[:, 1]               # each descriptor's nearest-neighbour distance
    scale = float(np.median(nn) * 1.5)
    wins, prev = [], None
    events = []
    starts = np.arange(float(ta[0]), max(float(ta[-1]) - window, float(ta[0])) + stride, stride)
    for a in starts:
        m = (ta >= a) & (ta < a + window)
        if m.sum() < 3:
            continue
        rows = np.unique(top[m].ravel())
        g, l = _clusters_and_loops(D[rows], scale)
        wins.append({"t0": round(float(a), 1), "t1": round(float(a + window), 1), "visited": int(len(rows)),
                     "groups": g, "loops": l})
        if prev is not None and (g != prev[0] or l != prev[1]):
            parts = []
            if g != prev[0]:
                parts.append(f"attention {'splits into' if g > prev[0] else 'gathers into'} {g} group{'s' * (g != 1)}")
            if l != prev[1]:
                parts.append(f"{'a loop forms' if l > prev[1] else 'a loop closes'} ({l} now)")
            events.append({"t": round(float(a + window / 2), 1), "text": ", ".join(parts)})
        prev = (g, l)
    graph_events = []
    graphs = [e for e in raw["events"] if e["type"] == "graph"]
    for e0, e1 in zip(graphs[:-1], graphs[1:]):
        a0 = {tuple(x[:2]): x[2] for x in e0.get("edges", [])}
        a1 = {tuple(x[:2]): x[2] for x in e1.get("edges", [])}
        born = sorted(set(a1) - set(a0), key=lambda k: -abs(a1[k]))
        died = sorted(set(a0) - set(a1), key=lambda k: -abs(a0[k]))
        graph_events.append({"t": e1["t"], "born": len(born), "died": len(died), "kept": len(set(a0) & set(a1)),
                             "top_born": [f"{u} — {v}" for u, v in born[:3]], "top_died": [f"{u} — {v}" for u, v in died[:3]]})
    out["topology"] = {"windows": wins, "events": events, "graph_events": graph_events, "scale": round(scale, 3)}

    # ---- D · phases and returns --------------------------------------------------------
    Z = _zrows(R)
    k = max(3, len(Z) // 60)
    Zs = np.stack([Z[max(0, i - k):i + k + 1].mean(0) for i in range(len(Z))])     # a few seconds of smoothing
    starts_i = _segments(Zs, min_len=max(4, len(Z) // 20))
    phases = []
    for a, b in zip(starts_i, starts_i[1:] + [len(Z)]):
        m = R[a:b].mean(0)
        sm = np.bincount(top_sym[a:b], minlength=len(symbols))
        phases.append({"t0": round(float(ta[a]), 1), "t1": round(float(ta[b - 1]), 1),
                       "words": [descriptors[j] for j in np.argsort(-m)[:4]], "symbol": symbols[int(sm.argmax())]})
    nb = min(150, len(Z))
    bins = np.array_split(np.arange(len(Z)), nb)
    B = np.stack([Z[ix].mean(0) for ix in bins])
    B = (B - B.mean(1, keepdims=True)) / (np.linalg.norm(B - B.mean(1, keepdims=True), axis=1, keepdims=True) + 1e-9)
    M = B @ B.T
    tb = np.array([ta[ix].mean() for ix in bins])
    # a return: a moment that resembles an earlier one (at least 15 % of the session before, 15 s
    # minimum), after the session had left it
    gap = max(15.0, 0.15 * float(tb[-1] - tb[0]))
    returns = []
    for i in range(nb):
        for j in range(i):
            if tb[i] - tb[j] >= gap and M[i, j] > 0.8 and M[j, j + 1:i].min(initial=1.0) < 0.3:
                if not returns or abs(returns[-1]["t"] - tb[i]) > 20:
                    returns.append({"t": round(float(tb[i]), 1), "to": round(float(tb[j]), 1), "r": round(float(M[i, j]), 2)})
                break
    out["phases"] = {"phases": phases, "recurrence": {"t": [round(float(x), 1) for x in tb], "M": np.round(M, 2).tolist()},
                     "returns": returns[:12]}

    # ---- G · session archetypes ---------------------------------------------------------
    visited = np.flatnonzero(visits > 0)
    arche = {}
    if len(visited) >= 6:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
        V = D[visited]
        Lk = linkage(pdist(V, "cosine"), "average")
        best = None
        for kk in range(3, min(7, len(visited) // 2) + 1):
            lab = fcluster(Lk, kk, "maxclust")
            if len(np.unique(lab)) < 2:
                continue
            try:
                from sklearn.metrics import silhouette_score
                sc = silhouette_score(V, lab, metric="cosine")
            except Exception:
                sc = -kk
            if best is None or sc > best[0]:
                best = (sc, lab)
        if best is not None:
            for c in np.unique(best[1]):
                members = visited[best[1] == c]
                if len(members) < 2:
                    continue                      # a lone word is not an archetype
                members = members[np.argsort(-visits[members])]
                name = descriptors[members[0]].upper()
                arche[name] = [descriptors[j] for j in members[:12]]
    out["archetypes"] = arche

    # ---- H · fingerprint ----------------------------------------------------------------
    p = visits[visits > 0] / visits[visits > 0].sum()
    eff = float(np.exp(-(p * np.log(p)).sum()))
    iu = np.tril_indices(nb, -1)
    far = (tb[:, None] - tb[None, :])[iu] >= gap
    out["fingerprint"] = {"symbols": symbols, "symbol_share": [round(float(x), 3) for x in sym_share],
                          "diversity": round(eff / N, 3), "drift": round(float(np.mean(speed[1:])) if len(speed) > 1 else 0.0, 4),
                          "recurrence": round(float((M[iu][far] > 0.8).mean()) if far.any() else 0.0, 3),
                          "phases": len(phases), "duration": raw["duration"]}
    return out


def to_json(obj: Any) -> str:
    """Strict JSON (NaN is refused: browsers cannot parse it)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
