"""Figure 3c — Symbol-weight morphing: a CONTROLLED morph axis.

Companion to fig03b (text-context morph).  fig03b blends two *sentence* vectors;
that path is a chord through embedding space whose intermediate points are not
real sentences, so the axis itself has no clean semantics.  Here the morph axis
is a blend between two named SYMBOL centroids:

    vctx(α) = (1-α)·centroid[SYM_A] + α·centroid[SYM_B]

so α literally means "how much SYM_A vs SYM_B am I asserting".  The endpoints are
crisp, named priors; the informative content is how the *descriptor field* and
the *other* symbols' relevance reorganize in between — handoffs, and especially
BRIDGE symbols that dominate only mid-morph (visible at neither endpoint).

Readout is identical to fig03b: degree-normalized personalized-PageRank symbol
scores (raw PPR / #descriptors, to remove the bipartite high-degree bias).  This
keeps the two figures directly comparable — only the morph axis differs.

The two endpoint symbols are drawn with a dashed line; emergent symbols are solid,
so a bridge symbol that is NOT an endpoint stands out immediately.

Output: paper/v2/figures/fig03c_symbol_morph.{pdf,png}
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from _setup import build_space, save_fig, set_paper_style

# Contrasting symbol pairs chosen to span distinct registers of the kit:
#   ground ↔ storm,  earthly ↔ cosmic,  home/safety ↔ power/danger,  honor ↔ storm
SYMBOL_PAIRS = [
    ("EARTH", "THUNDER"),
    ("EARTH", "STAR"),
    ("HOUSE", "LIGHTNING"),
    ("FEATHER", "THUNDER"),
]

ALPHA_STEPS = 21
PPR_ALPHA = 0.85     # PageRank teleport (graph diffusion), NOT the morph alpha
PPR_TAU = 0.1        # softmax temperature for descriptor personalization


def _blend_centroids(space, sym_a, sym_b, alpha):
    """Mirror _weight_vec for weights {A:1-α, B:α}: blend symbol centroids."""
    return (1.0 - alpha) * space.symbol_centroids[sym_a] \
        + alpha * space.symbol_centroids[sym_b]


def _ppr_symbol_scores(space):
    """Degree-normalized PPR symbol scores for the current context override."""
    from delyrism.delyrism import softmax

    vctx = space.context_override
    pers = {f"D:{d}": float(w)
            for d, w in zip(space.descriptors, softmax(space.D @ vctx, tau=PPR_TAU))}
    pr = nx.pagerank(space.G, alpha=PPR_ALPHA, personalization=pers, weight="weight")
    raw = {n[2:]: v for n, v in pr.items() if n.startswith("S:")}
    return {s: raw.get(s, 0.0) / len(space.symbols_to_descriptors[s])
            for s in space.symbols}


def _sweep(space, sym_a, sym_b, alphas):
    syms = list(space.symbols)
    traj = {s: [] for s in syms}
    for alpha in alphas:
        space.set_context_vec(_blend_centroids(space, sym_a, sym_b, float(alpha)))
        sc = _ppr_symbol_scores(space)
        space.set_context_vec(None)
        for s in syms:
            traj[s].append(sc[s])
    return {s: np.array(v) for s, v in traj.items()}


def _runs(seq):
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--steps", type=int, default=ALPHA_STEPS)
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)
    cdict = space.get_symbol_color_dict(palette="Nord")

    alphas = np.linspace(0.0, 1.0, args.steps)
    trajectories = []
    t0 = time.time()
    for sa, sb in SYMBOL_PAIRS:
        print(f"[fig03c] morph  {sa} -> {sb}")
        trajectories.append((sa, sb, _sweep(space, sa, sb, alphas)))
    print(f"[fig03c] swept {len(SYMBOL_PAIRS)} pairs in {time.time() - t0:.1f}s")

    n_cols = len(SYMBOL_PAIRS)
    fig = plt.figure(figsize=(3.5 * n_cols, 4.6))
    gs = fig.add_gridspec(
        2, n_cols, height_ratios=[0.10, 1.0],
        hspace=0.06, wspace=0.26,
        top=0.80, bottom=0.13, left=0.06, right=0.99,
    )

    syms = list(space.symbols)

    for col, (sa, sb, traj) in enumerate(trajectories):
        endpoints = {sa, sb}
        M = np.vstack([traj[s] for s in syms])
        winners = [syms[i] for i in M.argmax(axis=0)]
        ever_top3 = set()
        for j in range(M.shape[1]):
            ever_top3.update(syms[i] for i in np.argsort(M[:, j])[-3:])

        # rank-1 dominance stripe
        ax_s = fig.add_subplot(gs[0, col])
        for j, w in enumerate(winners):
            ax_s.axvspan(alphas[j] - 0.5 / (len(alphas) - 1),
                         alphas[j] + 0.5 / (len(alphas) - 1),
                         color=cdict.get(w, "0.5"), lw=0)
        ax_s.set_xlim(0, 1)
        ax_s.set_yticks([]); ax_s.set_xticks([])
        for sp in ax_s.spines.values():
            sp.set_visible(False)
        ax_s.set_title(f"{sa} $\\to$ {sb}", fontsize=10, fontweight="bold", pad=12)
        ax_s.set_ylabel("rank-1", fontsize=6.5, rotation=0, ha="right", va="center",
                        labelpad=2)

        # trajectory panel — endpoints dashed, emergent/bridge symbols solid
        ax = fig.add_subplot(gs[1, col])
        for i, s in enumerate(syms):
            if s in ever_top3:
                ls = "--" if s in endpoints else "-"
                ax.plot(alphas, M[i], color=cdict.get(s, "0.4"), lw=1.9,
                        ls=ls, zorder=3)
                end = -1 if M[i, -1] >= M[i, 0] else 0
                ax.text(alphas[end] + (0.01 if end == -1 else -0.01), M[i, end], s,
                        ha="left" if end == -1 else "right", va="center",
                        fontsize=6.8, fontweight="bold", color=cdict.get(s, "0.4"),
                        zorder=4)
            else:
                ax.plot(alphas, M[i], color="0.7", lw=0.6, alpha=0.5, zorder=1)

        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$\alpha\;$ (A $\to$ B)", fontsize=8, labelpad=2)
        if col == 0:
            ax.set_ylabel("degree-normalized PPR score", fontsize=8)
        ax.margins(x=0.12)
        ax.set_title(" → ".join(_runs(winners)), fontsize=6.8, color="0.35", pad=3)

    fig.suptitle(
        "Symbol-weight morphing — degree-normalized PPR reorganization "
        "(dashed = endpoint symbol, solid = emergent/bridge)",
        fontsize=10.5, y=0.95,
    )

    save_fig(fig, "fig03c_symbol_morph")
    plt.close(fig)


if __name__ == "__main__":
    main()
