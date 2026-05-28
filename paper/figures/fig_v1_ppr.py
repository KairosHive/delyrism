"""V1 Figure 3 — PPR contextual subgraph.

Preserved from the original NeurIPS 2025 submission (PLAN.md §4.5).  V2 keeps
this because PPR answers a *different* question than the Δ-graph — propagation
of relevance through the bipartite graph, vs change in pairwise coupling.

Output: paper/v2/figures/fig_v1_ppr.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from _setup import CONTEXTS, build_space, save_fig, set_paper_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--context", default="C1",
                    help="Placeholder context id (C1/C2/C3/C_A/C_B/C_scene) or a literal sentence.")
    ap.add_argument("--topk-symbols", type=int, default=5)
    ap.add_argument("--topk-desc", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--normalize", action="store_true",
                    help="Subtract baseline (uniform-personalization) PPR.")
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    sentence = CONTEXTS.get(args.context, args.context)
    print(f"[fig_v1_ppr] context: {sentence!r}")

    from delyrism.delyrism import plot_contextual_subgraph_colored

    # The legacy plot_contextual_subgraph_colored creates its own figure via
    # plt.figure(figsize=...) and calls plt.show() implicitly via tight_layout.
    # We swallow plt.show, let it run, then grab the current figure.
    plot_contextual_subgraph_colored(
        space,
        context_sentence=sentence,
        topk_symbols=args.topk_symbols,
        topk_desc=args.topk_desc,
        method="ppr",
        alpha=args.alpha,
        tau=args.tau,
        normalize=args.normalize,
        global_color_map=space.get_symbol_color_dict(palette="Nord"),
        figsize=(9.0, 6.5),
    )
    fig = plt.gcf()
    fig.suptitle(f"PPR contextual subgraph  —  context: {sentence[:60]}…",
                 fontsize=10, y=1.01)
    save_fig(fig, "fig_v1_ppr")
    plt.close(fig)


if __name__ == "__main__":
    main()
