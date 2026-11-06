"""Server-side twin of the Live tab's spring layout.

The browser lays the live Δ-graph out in `web/frontend/lib/springSim.ts`
(physics) + `components/realtime/LiveDeltaGraph.tsx` (relevance → rest
lengths).  That layout only exists inside the page's requestAnimationFrame
loop, which stops whenever the browser window is hidden — useless as a feed
for an external renderer such as TouchDesigner.  This module runs the same
model on the server, one instance per live socket, so the OSC stream keeps
flowing no matter what the browser is doing.

Same forces, same constants, same knobs (the browser forwards focus,
contraction, stiffness, damping, repulsion and disentangle as `shape`
messages).  Positions are not bit-identical to the browser's — new nodes are
seeded with random jitter on both sides — but the dynamics are.

Browser sync: the browser streams its own node positions (`layout` socket
messages).  While those are fresh the server does no physics and simply
forwards them, so TouchDesigner shows exactly the motion the user sees —
including slow settling when the page renders below ~20 fps (the browser
clamps its step to 50 ms, so its simulation then runs slower than real time)
and dragged nodes.  When the browser goes quiet (tab hidden, throttled) this
sim continues from the last synced state, so the output never freezes.

On top of the browser model, nodes and edges carry a `fade` envelope: they
fade in over FADE_IN seconds when they appear and, instead of vanishing when
the context drops them, they stay for FADE_OUT seconds (no longer exerting
forces, drifting to a stop) while fading out.  A particle system downstream
can use that to avoid popping.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# LiveDeltaGraph.tsx
BASE_DIST = 80.0
SHORT_DIST = 34.0
# springSim.ts
SUBSTEP_HZ = 240.0
SLEEP_ENERGY = 0.25   # mean v² (px²/s²) below which the sim counts as quiet
SLEEP_AFTER = 0.5     # seconds of quiet before it sleeps

FADE_IN = 0.6
FADE_OUT = 0.9

SHAPE_DEFAULTS: Dict[str, float] = dict(
    # liveGraph.ts store defaults
    focus=1.6, contraction=0.7, damping=8.0, stiffness=40.0, repulsion=40000.0, unlinked=6000.0,
    # springSim.ts params not exposed in the UI
    center=2.0, radial=6.0,
)


def hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = (h or "#888888").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return (0.53, 0.53, 0.53)


class LiveSim:
    """Spring-mass layout of the live Δ-graph with appear / disappear fades.

    Node rows and edge rows are numpy arrays.  Row order is stable between
    graph updates (survivors keep their row, newcomers are appended) and only
    changes when a fully faded row is dropped; `order_version` bumps whenever
    it does, so consumers know when to re-read labels.
    """

    def __init__(self) -> None:
        self.shape: Dict[str, float] = dict(SHAPE_DEFAULTS)
        self.relevance: Dict[str, float] = {}
        self.query_active = False
        self.graph_seq = 0
        self.order_version = 0
        self.awake = True
        self._sleep_timer = 0.0
        self.client_t = -1e9          # perf_counter of the last browser layout sync
        self.client_fps = 0.0
        self._uid_of: Dict[str, int] = {}
        self._rng = random.Random()
        self._set_nodes_empty()
        self._set_edges_empty()

    # ------------------------------------------------------------------ state
    def _set_nodes_empty(self) -> None:
        self.ids: List[str] = []
        self.symbols: List[str] = []
        self.symbols_of: Dict[str, str] = {}
        z = np.zeros(0)
        self.uid = np.zeros(0, dtype=np.int32)
        self.sym = np.zeros(0, dtype=np.int32)
        self.x, self.y, self.vx, self.vy = z.copy(), z.copy(), z.copy(), z.copy()
        self.size, self.pull, self.rel, self.fade = z.copy(), z.copy(), z.copy(), z.copy()
        self.rgb = np.zeros((0, 3))
        self.alive = np.zeros(0, dtype=bool)

    def _set_edges_empty(self) -> None:
        self.e_key: List[Tuple[str, str]] = []
        z = np.zeros(0)
        self.e_a = np.zeros(0, dtype=np.int64)
        self.e_b = np.zeros(0, dtype=np.int64)
        self.e_abs, self.e_delta, self.e_weight = z.copy(), z.copy(), z.copy()
        self.e_rest, self.e_k, self.e_overlap, self.e_fade = z.copy(), z.copy(), z.copy(), z.copy()
        self.e_sign = np.zeros(0, dtype=np.int32)
        self.e_alive = np.zeros(0, dtype=bool)

    def wake(self) -> None:
        self.awake = True
        self._sleep_timer = 0.0

    @property
    def fading(self) -> bool:
        """True while any fade envelope is still moving."""
        return bool(
            np.any(self.alive & (self.fade < 1.0)) or np.any(~self.alive)
            or np.any(self.e_alive & (self.e_fade < 1.0)) or np.any(~self.e_alive)
        )

    # ------------------------------------------------------------- inputs
    def sync_from_client(self, ids: Sequence[str], xy: Sequence[float], v: Sequence[float],
                         awake: bool, fps: float = 0.0) -> None:
        """Adopt the browser's layout (flat [x0, y0, x1, y1, …] in sim px).

        The browser is what the user looks at; while it streams, TouchDesigner
        should show exactly that motion.  Nodes the browser doesn't have
        (fading-out rows, or a graph update it hasn't ingested yet) keep their
        server state."""
        row = {nid: i for i, nid in enumerate(self.ids)}
        n = min(len(ids), len(xy) // 2)
        has_v = len(v) >= 2 * n
        for k in range(n):
            i = row.get(ids[k])
            if i is None or not self.alive[i]:
                continue
            self.x[i] = float(xy[2 * k]); self.y[i] = float(xy[2 * k + 1])
            if has_v:
                self.vx[i] = float(v[2 * k]); self.vy[i] = float(v[2 * k + 1])
        self.awake = bool(awake)
        self._sleep_timer = 0.0
        self.client_t = time.perf_counter()
        self.client_fps = float(fps)

    def client_fresh(self, now: float, max_age: float = 0.5) -> bool:
        """True while browser layout syncs are arriving (the browser sends at least
        every ~0.4 s, even when its simulation sleeps)."""
        return now - self.client_t < max_age

    def set_shape(self, patch: Dict[str, Any]) -> None:
        for k, v in patch.items():
            if k in SHAPE_DEFAULTS and v is not None:
                self.shape[k] = float(v)
        self.apply_rest()

    def set_relevance(self, relevance: Dict[str, float], active: bool) -> None:
        self.relevance = dict(relevance or {})
        self.query_active = bool(active)
        self.apply_rest()

    def set_graph(self, nodes: Sequence[Dict[str, Any]], edges: Sequence[Dict[str, Any]],
                  symbols: Sequence[str]) -> None:
        """Ingest a `graph` payload.  Survivors keep position and velocity,
        newcomers spawn beside their archetype siblings (else on a
        phyllotaxis spiral), dropped nodes / edges start fading out."""
        self.graph_seq += 1
        self.symbols = list(symbols)
        sym_index = {s: i for i, s in enumerate(self.symbols)}
        new_ids = [n["id"] for n in nodes]
        new_set = set(new_ids)

        # Σ|Δ| per node, min-max normalised (LiveDeltaGraph `weight`)
        wsum: Dict[str, float] = {}
        for e in edges:
            wsum[e["source"]] = wsum.get(e["source"], 0.0) + float(e["abs_delta"])
            wsum[e["target"]] = wsum.get(e["target"], 0.0) + float(e["abs_delta"])
        wv = list(wsum.values())
        w_lo = min(wv) if wv else 0.0
        w_hi = max(wv) if wv else 1.0
        w_r = (w_hi - w_lo) if (w_hi - w_lo) > 1e-12 else 1.0

        # archetype centroids of the current live layout (seed for newcomers)
        by_sym: Dict[str, List[float]] = {}
        for i, nid in enumerate(self.ids):
            if self.alive[i]:
                s = self.symbols_of.get(nid, "")
                acc = by_sym.setdefault(s, [0.0, 0.0, 0.0])
                acc[0] += self.x[i]; acc[1] += self.y[i]; acc[2] += 1

        old_row = {nid: i for i, nid in enumerate(self.ids)}
        rows: List[Dict[str, Any]] = []
        # 1) everything already on screen, in its current order
        for i, nid in enumerate(self.ids):
            rows.append(dict(id=nid, x=self.x[i], y=self.y[i], vx=self.vx[i], vy=self.vy[i],
                             fade=self.fade[i], alive=nid in new_set, uid=int(self.uid[i]),
                             size=self.size[i], pull=self.pull[i], rgb=self.rgb[i].tolist(),
                             symbol=self.symbols_of.get(nid, "")))
        # 2) newcomers
        golden = math.pi * (3 - math.sqrt(5))
        payload = {n["id"]: n for n in nodes}
        for j, nid in enumerate(new_ids):
            if nid in old_row:
                continue
            n = payload[nid]
            acc = by_sym.get(n.get("symbol", ""))
            if acc and acc[2]:
                x = acc[0] / acc[2] + (self._rng.random() - 0.5) * 30
                y = acc[1] / acc[2] + (self._rng.random() - 0.5) * 30
            else:
                r, th = 12 * math.sqrt(j + 1), j * golden
                x, y = r * math.cos(th), r * math.sin(th)
            if nid not in self._uid_of:
                self._uid_of[nid] = len(self._uid_of)
            rows.append(dict(id=nid, x=x, y=y, vx=0.0, vy=0.0, fade=0.0, alive=True,
                             uid=self._uid_of[nid], size=0.0, pull=0.0, rgb=[0.5, 0.5, 0.5],
                             symbol=n.get("symbol", "")))
        # 3) refresh per-node attributes from the payload
        for r in rows:
            n = payload.get(r["id"])
            if n is None:
                continue
            r["size"] = (wsum.get(r["id"], 0.0) - w_lo) / w_r
            r["pull"] = float(n.get("pull", 0.0))
            r["rgb"] = list(hex_to_rgb(n.get("color", "#888888")))
            r["symbol"] = n.get("symbol", "")
        self._load_rows(rows, sym_index)

        # ---- edges -----------------------------------------------------------
        index = {nid: i for i, nid in enumerate(self.ids)}
        ev = [float(e["abs_delta"]) for e in edges]
        e_lo = min(ev) if ev else 0.0
        e_hi = max(ev) if ev else 1.0
        e_r = (e_hi - e_lo) if (e_hi - e_lo) > 1e-12 else 1.0
        old_edges = {k: i for i, k in enumerate(self.e_key)}
        new_keys: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for e in edges:
            if e["source"] in index and e["target"] in index:
                new_keys[(e["source"], e["target"])] = e
        erows: List[Dict[str, Any]] = []
        for i, k in enumerate(self.e_key):          # existing, in order
            if k[0] not in index or k[1] not in index:
                continue                              # an endpoint was dropped entirely
            e = new_keys.get(k)
            if e is None:
                erows.append(dict(key=k, abs=self.e_abs[i], delta=self.e_delta[i], sign=int(self.e_sign[i]),
                                  weight=self.e_weight[i], fade=self.e_fade[i], alive=False))
            else:
                erows.append(dict(key=k, abs=float(e["abs_delta"]), delta=float(e["delta"]),
                                  sign=1 if e.get("sign") == "up" else -1,
                                  weight=(float(e["abs_delta"]) - e_lo) / e_r, fade=self.e_fade[i], alive=True))
        for k, e in new_keys.items():                 # newcomers
            if k in old_edges:
                continue
            erows.append(dict(key=k, abs=float(e["abs_delta"]), delta=float(e["delta"]),
                              sign=1 if e.get("sign") == "up" else -1,
                              weight=(float(e["abs_delta"]) - e_lo) / e_r, fade=0.0, alive=True))
        self._load_edges(erows, index)
        self.apply_rest()

    # ---------------------------------------------------------- row storage
    def _load_rows(self, rows: List[Dict[str, Any]], sym_index: Dict[str, int]) -> None:
        old_ids = self.ids
        self.ids = [r["id"] for r in rows]
        self.symbols_of = {r["id"]: r["symbol"] for r in rows}
        self.sym = np.asarray([sym_index.get(r["symbol"], -1) for r in rows], dtype=np.int32)
        self.uid = np.asarray([r["uid"] for r in rows], dtype=np.int32)
        f = lambda k: np.asarray([float(r[k]) for r in rows], dtype=np.float64)  # noqa: E731
        self.x, self.y, self.vx, self.vy = f("x"), f("y"), f("vx"), f("vy")
        self.size, self.pull, self.fade = f("size"), f("pull"), f("fade")
        self.rel = np.zeros(len(rows))
        self.rgb = np.asarray([r["rgb"] for r in rows], dtype=np.float64).reshape(-1, 3)
        self.alive = np.asarray([bool(r["alive"]) for r in rows], dtype=bool)
        if self.ids != old_ids:
            self.order_version += 1
        self.wake()

    def _load_edges(self, erows: List[Dict[str, Any]], index: Dict[str, int]) -> None:
        self.e_key = [r["key"] for r in erows]
        self.e_a = np.asarray([index[k[0]] for k in self.e_key], dtype=np.int64)
        self.e_b = np.asarray([index[k[1]] for k in self.e_key], dtype=np.int64)
        f = lambda k: np.asarray([float(r[k]) for r in erows], dtype=np.float64)  # noqa: E731
        self.e_abs, self.e_delta, self.e_weight, self.e_fade = f("abs"), f("delta"), f("weight"), f("fade")
        self.e_sign = np.asarray([r["sign"] for r in erows], dtype=np.int32)
        self.e_alive = np.asarray([bool(r["alive"]) for r in erows], dtype=bool)
        n = len(erows)
        self.e_rest = np.full(n, BASE_DIST)
        self.e_k = np.ones(n)
        self.e_overlap = np.ones(n)

    def _compact(self) -> None:
        """Drop nodes / edges whose fade-out has finished."""
        keep_e = self.e_alive | (self.e_fade > 0.0)
        keep_n = self.alive | (self.fade > 0.0)
        if keep_n.all() and keep_e.all():
            return
        # edges whose endpoint disappears go too
        keep_e &= keep_n[self.e_a] & keep_n[self.e_b] if len(self.e_a) else keep_e
        remap = np.cumsum(keep_n) - 1
        for name in ("e_abs", "e_delta", "e_weight", "e_fade", "e_sign", "e_alive", "e_rest", "e_k", "e_overlap"):
            setattr(self, name, getattr(self, name)[keep_e])
        self.e_key = [k for k, m in zip(self.e_key, keep_e) if m]
        self.e_a = remap[self.e_a[keep_e]]
        self.e_b = remap[self.e_b[keep_e]]
        if not keep_n.all():
            for name in ("uid", "x", "y", "vx", "vy", "size", "pull", "rel", "fade", "rgb", "alive", "sym"):
                setattr(self, name, getattr(self, name)[keep_n])
            self.ids = [i for i, m in zip(self.ids, keep_n) if m]
            self.order_version += 1

    # ------------------------------------------------- relevance → geometry
    def apply_rest(self) -> None:
        """LiveDeltaGraph.applyRest: sharpened relevance sets each spring's
        rest length and stiffness; the radial spring follows relevance."""
        g = self.shape["focus"]
        rel = np.asarray([self.relevance.get(i, 0.0) for i in self.ids], dtype=np.float64)
        self.rel = np.power(np.clip(rel, 0.0, 1.0), g) if len(rel) else rel
        if len(self.e_a):
            ov = np.sqrt(np.maximum(0.0, self.rel[self.e_a] * self.rel[self.e_b]))
            self.e_overlap = ov
            long = BASE_DIST + 140.0 * self.shape["contraction"]
            if self.query_active:
                self.e_rest = long + (SHORT_DIST - long) * ov
                self.e_k = 0.6 + 0.8 * ov
            else:
                self.e_rest = np.full(len(ov), BASE_DIST)
                self.e_k = np.ones(len(ov))
        self.wake()

    # ---------------------------------------------------------------- step
    def step(self, dt: float, physics: bool = True) -> None:
        """Advance by dt.  physics=False (browser layout is driving): only the
        fade envelopes run and fading-out nodes coast to a stop under drag."""
        dt = float(min(0.05, max(0.0, dt)))
        if dt <= 0.0:
            return
        self._step_fades(dt)
        n = len(self.ids)
        if not physics:
            dead = ~self.alive
            if dead.any():
                drag = math.exp(-self.shape["damping"] * dt)
                self.vx[dead] *= drag; self.vy[dead] *= drag
                self.x[dead] += self.vx[dead] * dt; self.y[dead] += self.vy[dead] * dt
            return
        if not self.awake or n == 0:
            return
        p = self.shape
        sub = max(1, math.ceil(dt * SUBSTEP_HZ))
        h = dt / sub
        drag = math.exp(-p["damping"] * h)
        live = self.alive
        A = np.flatnonzero(live)
        # live edges only (between live nodes) exert spring forces
        em = self.e_alive & live[self.e_a] & live[self.e_b] if len(self.e_a) else np.zeros(0, dtype=bool)
        ea, eb = self.e_a[em], self.e_b[em]
        rest, kk = self.e_rest[em], self.e_k[em]
        # adjacency among live nodes (for the "disentangle" term)
        m = len(A)
        pos_of = np.full(n, -1, dtype=np.int64)
        pos_of[A] = np.arange(m)
        adj = np.zeros((m, m), dtype=bool)
        if len(ea):
            adj[pos_of[ea], pos_of[eb]] = True
            adj[pos_of[eb], pos_of[ea]] = True
        not_adj = ~adj
        np.fill_diagonal(not_adj, False)
        parity = np.where(np.arange(m) % 2 == 1, 1.0, -1.0)[:, None]
        energy = 0.0
        for _ in range(sub):
            ax = np.zeros(n)
            ay = np.zeros(n)
            # springs (symmetric)
            if len(ea):
                dx = self.x[eb] - self.x[ea]
                dy = self.y[eb] - self.y[ea]
                d = np.hypot(dx, dy)
                tiny = d < 1e-6
                dx = np.where(tiny, 1e-3, dx); dy = np.where(tiny, 0.0, dy); d = np.where(tiny, 1e-3, d)
                f = p["stiffness"] * kk * (d - rest) / d
                np.add.at(ax, ea, f * dx); np.add.at(ay, ea, f * dy)
                np.add.at(ax, eb, -f * dx); np.add.at(ay, eb, -f * dy)
            # repulsion (symmetric, all live pairs)
            if m > 1:
                X, Y = self.x[A], self.y[A]
                DX = X[None, :] - X[:, None]     # x_j − x_i
                DY = Y[None, :] - Y[:, None]
                D2 = DX * DX + DY * DY
                tiny = D2 < 1e-6
                np.fill_diagonal(tiny, False)
                if tiny.any():
                    DX = np.where(tiny, parity * 1e-2, DX)
                    DY = np.where(tiny, 1e-2, DY)
                    D2 = np.where(tiny, 2e-4, D2)
                np.fill_diagonal(D2, 1.0)
                Dd = np.sqrt(D2)
                F = p["repulsion"] / (D2 + 400.0) / Dd
                if p["unlinked"] > 0:
                    F = F + not_adj * (p["unlinked"] / (D2 + 2500.0))
                np.fill_diagonal(F, 0.0)
                ax[A] -= (F * DX).sum(axis=1)
                ay[A] -= (F * DY).sum(axis=1)
            # central forces (torque-free about the origin)
            ax[A] -= p["center"] * self.x[A]
            ay[A] -= p["center"] * self.y[A]
            if self.query_active:
                r = np.hypot(self.x[A], self.y[A])
                r = np.where(r == 0, 1e-6, r)
                target = 40.0 + (1.0 - self.rel[A]) * 240.0
                f = p["radial"] * (target - r) / r
                ax[A] += f * self.x[A]
                ay[A] += f * self.y[A]
            # integrate (semi-implicit Euler + exact linear drag); fading-out
            # nodes feel no force and just coast to a stop
            self.vx = (self.vx + ax * h) * drag
            self.vy = (self.vy + ay * h) * drag
            self.x += self.vx * h
            self.y += self.vy * h
            energy = float((self.vx[A] ** 2 + self.vy[A] ** 2).sum())
        if energy / max(1, m) < SLEEP_ENERGY:
            self._sleep_timer += dt
            if self._sleep_timer > SLEEP_AFTER:
                self.awake = False
                self.vx[:] = 0.0
                self.vy[:] = 0.0
        else:
            self._sleep_timer = 0.0

    def _step_fades(self, dt: float) -> None:
        if len(self.fade):
            self.fade = np.where(self.alive, np.minimum(1.0, self.fade + dt / FADE_IN),
                                 np.maximum(0.0, self.fade - dt / FADE_OUT))
        if len(self.e_fade):
            self.e_fade = np.where(self.e_alive, np.minimum(1.0, self.e_fade + dt / FADE_IN),
                                   np.maximum(0.0, self.e_fade - dt / FADE_OUT))
        self._compact()

    # -------------------------------------------------------------- output
    def bounds(self) -> Tuple[float, float, float, float]:
        if not len(self.x):
            return (0.0, 0.0, 0.0, 0.0)
        return (float(self.x.min()), float(self.y.min()), float(self.x.max()), float(self.y.max()))
