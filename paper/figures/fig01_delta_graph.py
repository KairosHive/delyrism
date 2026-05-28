"""Figure 1 — Context as relational rewiring (PLAN.md §6, Analysis 1).

Two-row figure for contexts C1 / C2 / C3:

  Top row    — 10×10 between-symbol centroid-drift heatmap.  Replicates the
               engine's `/similarity-symbols` endpoint exactly:
                   M = cos(C_a', C_b') - cos(C_a, C_b)
               where C_a is the L2-normalized centroid of symbol a's descriptor
               vectors, before and after the context shift.  Diverging colour
               (red = symbols pull together, blue = symbols pull apart).
  Bottom row — Δ-graph of top-K descriptor pairs (no sign filter — both
               strengthening and weakening edges shown), matching the engine's
               default `/delta-graph` call.

Output: paper/v2/figures/fig01_delta_graph.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Match the frontend's 5-stop perceptual colour ramp for Δ heatmaps
# (web/frontend/components/explorer/SimilarityHeatmap.tsx).
DELTA_CMAP = LinearSegmentedColormap.from_list("delta", [
    (0.00, "#3a86ff"),   # blue        — symbols pull together least
    (0.25, "#06d6a0"),   # green
    (0.50, "#ffd166"),   # yellow
    (0.75, "#f77f00"),   # orange
    (1.00, "#ef476f"),   # red         — symbols pull together most
])

from _setup import (
    CONTEXTS, CONTEXT_LABELS,
    build_space, save_fig, set_paper_style,
)

# Mirror the running app's default shift parameters exactly.
SHIFT_KW = dict(
    strategy="gate", gate="relu", beta=1.2, tau=0.3,
    within_symbol_softmax=True, gamma=0.5,
    pool_type="avg", pool_w=0.7, membership_alpha=0.0,
)

TOP_EDGES = 28              # match the UI's typical top-N feel
MIN_ABS_DELTA = 0.005
WITHIN_SYMBOL = False
CONNECTED_ONLY = True
SIGN_FILTER = "up"          # strengthening edges only


def _l2_normalize_rows(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def _centroid_matrix(D, symbol_to_idx, symbols):
    """S×S cosine matrix of L2-normalized symbol centroids."""
    rows = []
    for s in symbols:
        idx = symbol_to_idx[s]
        c = D[idx].mean(axis=0)
        n = float(np.linalg.norm(c))
        rows.append(c / n if n > 1e-9 else c)
    C = np.stack(rows)
    return C @ C.T


def _draw_centroid_drift_matrix(ax, M, syms, *, vmin: float, vmax: float, title: str):
    """Heatmap of symbol-centroid drift.  Mask the diagonal (always 0).

    Colour spans the data range (min → max) on a 5-stop perceptual gradient,
    matching the running app's heatmap so the figure replicates what users
    see online.  Blue means "pulled together least", red means "pulled
    together most" — both are positive when context compresses the cloud.
    """
    M_disp = M.copy()
    np.fill_diagonal(M_disp, np.nan)
    im = ax.imshow(M_disp, cmap=DELTA_CMAP, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(syms)))
    ax.set_yticks(range(len(syms)))
    ax.set_xticklabels(syms, rotation=45, ha="right", fontsize=7.2)
    ax.set_yticklabels(syms, fontsize=7.2)
    ax.set_title(title, fontsize=9)
    # Light grid for legibility
    n = len(syms)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#fff", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    # Diagonal as light grey "self" cells
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   facecolor="0.85", edgecolor="white", linewidth=0.6, zorder=2))
    return im


def _draw_delta_graph(ax, G, color_dict, *, title: str):
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, "(no edges above threshold)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="0.5")
        ax.set_axis_off()
        ax.set_title(title, fontsize=9)
        return

    pos = nx.spring_layout(G, seed=42, k=0.7, iterations=80)

    abs_sum = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        a = float(d.get("abs_delta", 0.0))
        abs_sum[u] += a; abs_sum[v] += a
    arr = np.array(list(abs_sum.values()))
    if arr.max() - arr.min() < 1e-9:
        node_norm = {n: 0.5 for n in G.nodes()}
    else:
        lo, hi = arr.min(), arr.max()
        node_norm = {n: (abs_sum[n] - lo) / (hi - lo) for n in G.nodes()}
    sizes = [90 + 320 * node_norm[n] for n in G.nodes()]

    edges = list(G.edges(data=True))
    abs_d = np.array([float(d.get("abs_delta", 0.0)) for _, _, d in edges])
    lo, hi = (abs_d.min(), abs_d.max()) if len(abs_d) else (0.0, 1.0)
    widths = (np.full_like(abs_d, 1.5) if (hi - lo < 1e-9)
              else 0.6 + 2.6 * (abs_d - lo) / (hi - lo))

    up_edges = [(e[0], e[1]) for e in edges if e[2]["delta"] > 0]
    dn_edges = [(e[0], e[1]) for e in edges if e[2]["delta"] < 0]
    up_w = [widths[i] for i, e in enumerate(edges) if e[2]["delta"] > 0]
    dn_w = [widths[i] for i, e in enumerate(edges) if e[2]["delta"] < 0]

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=up_edges,
                           edge_color="#b3262a", width=up_w, alpha=0.82)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=dn_edges,
                           edge_color="#2f5d8f", width=dn_w, alpha=0.62)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=list(G.nodes()),
        node_color=[color_dict.get(G.nodes[n]["symbol"], "0.5") for n in G.nodes()],
        edgecolors="black", linewidths=0.25,
        node_size=sizes, alpha=0.93,
    )
    n_labels = min(10, len(G.nodes()))
    top_for_labels = sorted(G.nodes(), key=lambda n: -abs_sum[n])[:n_labels]
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: n for n in top_for_labels},
                            font_size=6.4, font_color="black")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)
    cdict = space.get_symbol_color_dict(palette="Nord")
    syms = list(space.symbols)

    from delyrism import context_delta_graph

    context_ids = ["C1", "C2", "C3"]

    # ─── Centroid-drift matrices ───────────────────────────────────────────
    S_before = _centroid_matrix(space.D, space.symbol_to_idx, syms)
    matrices: dict[str, np.ndarray] = {}
    for cid in context_ids:
        sentence = CONTEXTS[cid]
        print(f"[fig01] {cid}: {sentence!r}")
        D1 = space.make_shifted_matrix(sentence=sentence, **SHIFT_KW)
        S_after = _centroid_matrix(D1, space.symbol_to_idx, syms)
        matrices[cid] = S_after - S_before

    # Shared colour range across panels — span the actual data range (off-
    # diagonal min → max) like the running app, so the perceptual gradient
    # spreads the variation visibly.
    all_offdiag = np.concatenate([
        m[~np.eye(len(syms), dtype=bool)] for m in matrices.values()
    ])
    vmin = float(all_offdiag.min())
    vmax = float(all_offdiag.max())
    print(f"[fig01] matrix colour range: [{vmin:+.3f}, {vmax:+.3f}]")
    for cid, M in matrices.items():
        off = M[~np.eye(len(syms), dtype=bool)]
        print(f"[fig01]   {cid} Δ off-diagonal range "
              f"[{off.min():+.3f}, {off.max():+.3f}]   mean {off.mean():+.4f}")

    # ─── Δ-graphs (no sign filter, like the UI default) ───────────────────
    graphs: dict[str, "nx.Graph"] = {}
    for cid in context_ids:
        G = context_delta_graph(
            space,
            sentence=CONTEXTS[cid],
            top_abs_edges=TOP_EDGES,
            min_abs_delta=MIN_ABS_DELTA,
            within_symbol=WITHIN_SYMBOL,
            connected_only=CONNECTED_ONLY,
            sign_filter=SIGN_FILTER,
            **SHIFT_KW,
        )
        graphs[cid] = G
        n_up = sum(1 for _, _, d in G.edges(data=True) if d["delta"] > 0)
        n_dn = sum(1 for _, _, d in G.edges(data=True) if d["delta"] < 0)
        print(f"[fig01]   {cid} graph: {len(G.nodes())} nodes, "
              f"+{n_up}/-{n_dn} edges")

    # ─── Render ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.0, 1.15],
        hspace=0.30, wspace=0.22,
        top=0.92, bottom=0.06, left=0.05, right=0.94,
    )

    last_im = None
    for col, cid in enumerate(context_ids):
        ax_top = fig.add_subplot(gs[0, col])
        last_im = _draw_centroid_drift_matrix(
            ax_top, matrices[cid], syms,
            vmin=vmin, vmax=vmax,
            title=f"{cid} — {CONTEXT_LABELS[cid]}\n"
                  "(centroid drift: cos(C_a',C_b') − cos(C_a,C_b))",
        )

        ax_bot = fig.add_subplot(gs[1, col])
        G = graphs[cid]
        n_up = sum(1 for _, _, d in G.edges(data=True) if d["delta"] > 0)
        n_dn = sum(1 for _, _, d in G.edges(data=True) if d["delta"] < 0)
        _draw_delta_graph(
            ax_bot, G, cdict,
            title=f"top-{len(G.edges())} strengthened descriptor pairs",
        )

    # Shared colorbar for the matrix row
    if last_im is not None:
        cax = fig.add_axes([0.955, 0.55, 0.012, 0.34])
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label("symbol-pair Δ\n(red: pull together, blue: pull apart)", fontsize=7.5)

    # Symbol legend for the graph row
    handles = [Patch(facecolor=cdict.get(s, "0.5"), edgecolor="black", label=s)
               for s in syms]
    fig.legend(
        handles=handles, ncol=min(len(syms), 10),
        loc="lower center", bbox_to_anchor=(0.5, -0.005),
        frameon=False, fontsize=7.4,
    )

    fig.suptitle(
        "Context as relational rewiring — symbol-centroid drift (top) and "
        "top-K descriptor pairs (bottom) under three contexts",
        fontsize=11, y=0.985,
    )

    save_fig(fig, "fig01_delta_graph")
    plt.close(fig)


if __name__ == "__main__":
    main()
