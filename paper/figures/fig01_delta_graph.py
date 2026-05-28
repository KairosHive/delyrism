"""Figure 1 — Context as relational rewiring (PLAN.md §6, Analysis 1).

Three side-by-side Δ-graphs for the Lakota Shape Kit under contexts C1/C2/C3.
Each panel: signed pairwise-coupling changes (strengthened = red, weakened =
blue), nodes coloured by archetype, sized by total |Δ| incident edge weight.

Output: paper/v2/figures/fig01_delta_graph.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Patch

from _setup import (
    CONTEXTS, CONTEXT_LABELS,
    build_space, save_fig, set_paper_style,
)

# Match the backend's defaults exactly — same shift strategy as the running app.
SHIFT_KW = dict(
    strategy="gate",
    gate="relu",
    beta=1.2,
    tau=0.3,
    within_symbol_softmax=True,
    gamma=0.5,
    pool_type="avg", pool_w=0.7,
    membership_alpha=0.0,
)

# Δ-graph rendering parameters
TOP_EDGES = 40            # top-|Δ| edges retained per context
MIN_ABS_DELTA = 0.01
WITHIN_SYMBOL = False     # cross-symbol edges allowed — they're the most interesting
CONNECTED_ONLY = True     # drop isolates


def _delta_graph_panel(ax, G, color_dict, *, title: str, layout_pos=None):
    """Render a Δ-graph onto the given axes."""
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, "(no edges above threshold)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="0.5")
        ax.set_axis_off()
        ax.set_title(title)
        return None

    pos = layout_pos if layout_pos is not None else nx.spring_layout(G, seed=42, k=0.55)

    # Node size from total |Δ| incident weight
    abs_sum = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        a = float(d.get("abs_delta", 0.0))
        abs_sum[u] += a
        abs_sum[v] += a
    arr = np.array(list(abs_sum.values()), dtype=float)
    if arr.max() - arr.min() < 1e-9:
        node_norm = {n: 0.5 for n in G.nodes()}
    else:
        lo, hi = arr.min(), arr.max()
        node_norm = {n: (abs_sum[n] - lo) / (hi - lo) for n in G.nodes()}
    sizes = [60 + 280 * node_norm[n] for n in G.nodes()]

    # Edge widths from |Δ|, min-max normalised
    edges = list(G.edges(data=True))
    abs_d = np.array([float(d.get("abs_delta", 0.0)) for _, _, d in edges])
    lo, hi = (abs_d.min(), abs_d.max()) if len(abs_d) else (0.0, 1.0)
    if hi - lo < 1e-9:
        widths = np.full_like(abs_d, 1.0)
    else:
        widths = 0.5 + 2.6 * (abs_d - lo) / (hi - lo)

    up_idx = [i for i, (_, _, d) in enumerate(edges) if d["delta"] > 0]
    dn_idx = [i for i, (_, _, d) in enumerate(edges) if d["delta"] < 0]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=[(edges[i][0], edges[i][1]) for i in up_idx],
        edge_color="#b3262a", width=[widths[i] for i in up_idx], alpha=0.80,
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=[(edges[i][0], edges[i][1]) for i in dn_idx],
        edge_color="#2f5d8f", width=[widths[i] for i in dn_idx], alpha=0.60,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=list(G.nodes()),
        node_color=[color_dict.get(G.nodes[n]["symbol"], "0.5") for n in G.nodes()],
        edgecolors="black", linewidths=0.25,
        node_size=sizes, alpha=0.92,
    )
    # Light descriptor labels (top-N by node weight) so the panel stays readable
    top_for_labels = sorted(G.nodes(), key=lambda n: -abs_sum[n])[:8]
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={n: n for n in top_for_labels},
        font_size=6.2, font_color="black",
    )
    ax.set_title(title)
    ax.set_axis_off()
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    # Symbol colour palette (consistent across panels)
    cdict = space.get_symbol_color_dict(palette="Nord")

    from delyrism import context_delta_graph

    context_ids = ["C1", "C2", "C3"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    for ax, cid in zip(axes, context_ids):
        sentence = CONTEXTS[cid]
        label = CONTEXT_LABELS[cid]
        print(f"[fig01] {cid}: {sentence!r}")
        G = context_delta_graph(
            space,
            sentence=sentence,
            top_abs_edges=TOP_EDGES,
            min_abs_delta=MIN_ABS_DELTA,
            within_symbol=WITHIN_SYMBOL,
            connected_only=CONNECTED_ONLY,
            **SHIFT_KW,
        )
        n_up = sum(1 for _, _, d in G.edges(data=True) if d["delta"] > 0)
        n_dn = sum(1 for _, _, d in G.edges(data=True) if d["delta"] < 0)
        print(f"[fig01]   nodes={len(G.nodes())} edges={len(G.edges())} "
              f"(+{n_up} / −{n_dn})")
        _delta_graph_panel(
            ax, G, cdict,
            title=f"{cid} — {label}\n+{n_up} strengthened   −{n_dn} weakened",
        )

    # Shared legend below
    syms_present = list(space.symbols)
    handles = [Patch(facecolor=cdict.get(s, "0.5"), edgecolor="black", label=s)
               for s in syms_present]
    fig.legend(
        handles=handles, ncol=min(len(syms_present), 10),
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        frameon=False, fontsize=7.5,
    )
    fig.suptitle(
        "Context as relational rewiring — top-|Δ| descriptor-pair couplings "
        f"(top-{TOP_EDGES} edges per panel)",
        fontsize=11, y=1.01,
    )
    plt.subplots_adjust(top=0.92, bottom=0.10, wspace=0.02)

    save_fig(fig, "fig01_delta_graph")
    plt.close(fig)


if __name__ == "__main__":
    main()
