"""V1 Figure 1a — Symbol descriptor map (UMAP).

Uses the engine's own `space.plot_map(...)` so the figure is produced by
exactly the same code path that drives the running app — same UMAP reducer,
same convex hulls, same centroid stars, same tab20 palette.

Output: paper/v2/figures/fig_v1_umap.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from _setup import build_space, save_fig, set_paper_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--method", default="umap", choices=["umap", "pca", "tsne"])
    ap.add_argument("--n-neighbors", type=int, default=15)
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    # plot_map calls plt.show() internally; we render then grab the figure.
    space.plot_map(
        method=args.method,
        n_neighbors=args.n_neighbors,
        with_hulls=True,
        include_centroids=True,
        normalize_centroids=False,
        figsize=(8.5, 6.0),
        title="Symbol descriptor map  (Lakota Shape Kit · UMAP)",
    )
    fig = plt.gcf()
    save_fig(fig, "fig_v1_umap")
    plt.close(fig)


if __name__ == "__main__":
    main()
