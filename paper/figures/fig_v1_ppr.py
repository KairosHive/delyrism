"""V1 Figure 3 — PPR contextual subgraph (force-directed, clean).

Preserved from the original NeurIPS 2025 submission (PLAN.md §4.5).  V2 keeps
this because PPR answers a *different* question than the Δ-graph — propagation
of relevance through the bipartite symbol/descriptor graph, vs change in
pairwise coupling.

Data selection mirrors the running app's `POST /subgraph`
(web/backend/app/routes/delta.py): personalised PageRank over `space.G`,
top-K symbols, their top-M descriptors, then the induced subgraph — which
brings back the descriptor↔descriptor cosine edges that web the clusters
together.  Rendering mirrors the app's force-graph
(web/frontend/components/explorer/Subgraph.tsx):

  • symbol nodes  — rounded squares, full symbol colour, fixed size
  • descriptor    — circles, size ∝ normalised PR score, lightened colour
  • edges         — faint grey, width ∝ edge weight
  • labels        — placed just below each node with a white halo so they stay
                    legible over edges (matches the app's double-draw text)

Output: paper/v2/figures/fig_v1_ppr.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch

from _setup import CONTEXTS, CONTEXT_LABELS, build_space, save_fig, set_paper_style


def _lighten(color, amount: float = 0.45):
    """Mirror lighten_color()/lighten() in the engine + frontend."""
    r, g, b, a = mcolors.to_rgba(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--context", default="C1",
                    help="Context id (C1/C2/C3/C_A/C_B/C_scene) or a literal sentence.")
    ap.add_argument("--topk-symbols", type=int, default=3)
    ap.add_argument("--topk-desc", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--no-degree-norm", action="store_true",
                    help="Rank symbols by raw PPR (biased toward high-degree "
                         "symbols like EARTH, 19 descriptors). Default divides "
                         "each symbol's PPR by its descriptor count.")
    ap.add_argument("--k", type=float, default=1.5,
                    help="spring_layout optimal distance (larger = more spread).")
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    sentence = CONTEXTS.get(args.context, args.context)
    label = CONTEXT_LABELS.get(args.context, "")
    print(f"[fig_v1_ppr] context: {sentence!r}")

    # ── PPR data (mirrors POST /subgraph) ────────────────────────────────────
    from delyrism.delyrism import softmax

    sent_vec = space.ctx_vec(sentence=sentence)
    sims = space.D @ sent_vec
    pers = {f"D:{d}": float(w) for d, w in zip(space.descriptors, softmax(sims, tau=args.tau))}
    pr = nx.pagerank(space.G, alpha=args.alpha, personalization=pers, weight="weight")
    # Degree-normalize symbol scores: raw PPR over the bipartite symbol↔descriptor
    # graph rewards high-degree symbols (EARTH has 19 descriptors → wins almost
    # any context). Dividing by descriptor count surfaces the context-specific
    # symbol instead of the largest one. See _explore_morph_semantic.py.
    raw_scores = {n[2:]: v for n, v in pr.items() if n.startswith("S:")}
    if args.no_degree_norm:
        sym_scores = raw_scores
    else:
        sym_scores = {s: v / len(space.symbols_to_descriptors[s])
                      for s, v in raw_scores.items()}
    top_syms = sorted(sym_scores.items(), key=lambda kv: kv[1],
                      reverse=True)[:args.topk_symbols]

    symbols = [s for s, _ in top_syms]
    symbol_to_desc: dict[str, list[str]] = {}
    desc_score: dict[str, float] = {}
    for sym, _ in top_syms:
        descs = space.symbols_to_descriptors[sym]
        ranked = sorted(((d, pr.get(f"D:{d}", 0.0)) for d in descs),
                        key=lambda kv: kv[1], reverse=True)[:args.topk_desc]
        symbol_to_desc[sym] = [d for d, _ in ranked]
        for d, sc in ranked:
            desc_score[d] = float(sc)
        print(f"[fig_v1_ppr]   {sym:<12} score={dict(top_syms)[sym]:.6f}  "
              f"| {symbol_to_desc[sym]}")

    descriptors = [d for s in symbols for d in symbol_to_desc[s]]
    cdict = space.get_symbol_color_dict(palette="Nord")

    # ── Induced subgraph + spring layout ─────────────────────────────────────
    nodes = [f"S:{s}" for s in symbols] + [f"D:{d}" for d in descriptors]
    subG = space.G.subgraph(nodes).copy()
    pos = nx.spring_layout(subG, seed=42, k=args.k, iterations=400)

    # Normalise descriptor scores → marker size (frontend: val = 3 + norm*6)
    dvals = np.array([desc_score[d] for d in descriptors], dtype=float)
    dlo, dhi = dvals.min(), dvals.max()
    dn = (dvals - dlo) / (dhi - dlo + 1e-12) if dhi - dlo > 1e-12 else np.full_like(dvals, 0.5)
    desc_size = {d: 170 + 560 * dn[i] for i, d in enumerate(descriptors)}

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    ax.axis("off")
    ax.set_aspect("equal")
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]

    # ── Edges: faint grey, width ∝ weight (frontend: max(0.5, w*4)) ──────────
    for u, v, d in subG.edges(data=True):
        w = float(d.get("weight", 1.0))
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], "-", color="#9aa0a6",
                lw=max(0.5, min(3.2, w * 3.0)),
                alpha=0.45, zorder=1, solid_capstyle="round")

    # ── Descriptor nodes (circles, size ∝ score, lightened colour) ───────────
    for s in symbols:
        for d in symbol_to_desc[s]:
            x, y = pos[f"D:{d}"]
            ax.scatter([x], [y], s=desc_size[d], marker="o",
                       color=_lighten(cdict.get(s, "0.5"), 0.45),
                       edgecolors="black", linewidths=0.5, zorder=3)

    # ── Symbol nodes (rounded squares, full colour, fixed size) ──────────────
    # Square half-side in data coords; FancyBboxPatch gives the rounded look
    # the frontend draws with roundRect.
    span = max(np.ptp([p[0] for p in pos.values()]),
               np.ptp([p[1] for p in pos.values()]), 1e-6)
    half = 0.052 * span
    for s in symbols:
        x, y = pos[f"S:{s}"]
        ax.add_patch(FancyBboxPatch(
            (x - half, y - half), 2 * half, 2 * half,
            boxstyle="round,pad=0,rounding_size=" + str(half * 0.45),
            facecolor=cdict.get(s, "0.5"), edgecolor="black",
            linewidth=1.3, zorder=4, mutation_aspect=1.0,
        ))

    # ── Labels: below each node, dark text + white halo ──────────────────────
    yspan = np.ptp([p[1] for p in pos.values()]) or 1.0
    for s in symbols:
        x, y = pos[f"S:{s}"]
        ax.text(x, y - half - 0.012 * yspan, s, ha="center", va="top",
                fontsize=10.5, fontweight="bold", color="#161616",
                path_effects=halo, zorder=6)
    for d in descriptors:
        x, y = pos[f"D:{d}"]
        ax.text(x, y - 0.045 * yspan, d, ha="center", va="top",
                fontsize=7.8, color="#222222", path_effects=halo, zorder=6)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.30 * span, max(xs) + 0.30 * span)
    ax.set_ylim(min(ys) - 0.22 * span, max(ys) + 0.18 * span)

    fig.suptitle(f"PPR contextual subgraph — {label or args.context}",
                 fontsize=12, y=0.965)
    ax.set_title(f"“{sentence}”", fontsize=8, color="0.4", style="italic", pad=4)

    save_fig(fig, "fig_v1_ppr")
    plt.close(fig)


if __name__ == "__main__":
    main()
