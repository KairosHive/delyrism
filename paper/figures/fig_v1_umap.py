"""V1 Figure 1 — UMAP map of the symbolic field + ambiguity metrics.

Preserved from the original NeurIPS 2025 submission (PLAN.md §4.5).  This
figure stays in v2 largely intact as the opening empirical figure.

Output: paper/v2/figures/fig_v1_umap.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from _setup import build_space, save_fig, set_paper_style


def _ambiguity_metrics(space):
    """Compute dispersion / leakage / inter-symbolic entropy per symbol."""
    rows = []
    for s in space.symbols:
        try:
            disp = float(space.dispersion(s))
        except Exception:
            disp = float("nan")
        try:
            leak = float(space.leakage(s, k=min(10, len(space.descriptors) - 1)))
        except Exception:
            leak = float("nan")
        try:
            ent = float(space.soft_entropy(s, tau=0.5))
        except Exception:
            ent = float("nan")
        rows.append((s, disp, leak, ent))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    cdict = space.get_symbol_color_dict(palette="Nord")

    # UMAP reduction
    try:
        from umap import UMAP
    except ImportError:
        raise SystemExit("umap-learn required — `pip install umap-learn`")
    reducer = UMAP(n_neighbors=15, n_components=2, metric="cosine", random_state=42)
    Z = reducer.fit_transform(space.D)

    # Ambiguity metrics
    rows = _ambiguity_metrics(space)
    rows.sort(key=lambda r: -r[1])      # sort by dispersion desc
    syms_sorted = [r[0] for r in rows]
    disps = [r[1] for r in rows]
    leaks = [r[2] for r in rows]
    ents  = [r[3] for r in rows]

    # ─── Render ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12.0, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.20)

    # (a) UMAP scatter
    ax_a = fig.add_subplot(gs[0, 0])
    for s in space.symbols:
        idx = space.symbol_to_idx[s]
        ax_a.scatter(Z[idx, 0], Z[idx, 1],
                     color=cdict.get(s, "0.5"),
                     edgecolors="black", linewidths=0.2,
                     s=22, alpha=0.85, label=s)
        # centroid as star
        cx, cy = Z[idx, 0].mean(), Z[idx, 1].mean()
        ax_a.scatter([cx], [cy], marker="*", s=140,
                     color=cdict.get(s, "0.5"),
                     edgecolors="black", linewidths=0.5, zorder=4)
    ax_a.set_xlabel("UMAP 1")
    ax_a.set_ylabel("UMAP 2")
    ax_a.set_title("(a) Symbolic field — UMAP projection")
    handles = [Patch(facecolor=cdict.get(s, "0.5"), edgecolor="black", label=s)
               for s in space.symbols]
    ax_a.legend(handles=handles, ncol=2, fontsize=6.8, loc="best", frameon=False)

    # (b) ambiguity metrics — grouped bar
    ax_b = fig.add_subplot(gs[0, 1])
    y = np.arange(len(syms_sorted))
    w = 0.27
    ax_b.barh(y - w, disps, w, label="dispersion",
              color="#b3262a", edgecolor="black", linewidth=0.3)
    ax_b.barh(y,     leaks, w, label="leakage",
              color="#2f5d8f", edgecolor="black", linewidth=0.3)
    ax_b.barh(y + w, ents,  w, label="inter-symbolic entropy",
              color="#3a6b4f", edgecolor="black", linewidth=0.3)
    ax_b.set_yticks(y); ax_b.set_yticklabels(syms_sorted, fontsize=7.5)
    ax_b.invert_yaxis()
    ax_b.set_title("(b) Ambiguity metrics (sorted by dispersion)")
    ax_b.legend(loc="lower right", fontsize=7.5)

    fig.suptitle("Symbolic field at rest — UMAP + per-symbol ambiguity metrics",
                 fontsize=11, y=1.02)
    save_fig(fig, "fig_v1_umap")
    plt.close(fig)


if __name__ == "__main__":
    main()
