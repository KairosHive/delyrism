"""V1 Figure — Star plots of mean prediction scores per root symbol.

Mirrors the v1 paper logic exactly:
  • For each ROOT symbol s:
      • For each PARTNER p ≠ s:
          • Sample N_ITER weight pairs (w_s, w_p) ∼ Dirichlet(1, 1)
          • For each pair, run space.propose(weights={s: w_s, p: w_p})
          • For every OTHER symbol q (not in the pair), record q's score
  • Each star panel shows: for each non-root symbol q, the *mean* score
    accumulated over all (partner, sample) combinations involving the root.
  • Bars at angular positions, coloured per symbol.

This answers a different question than the Δ-graph:
  Δ-graph asks:    "Under THIS context, which descriptor pairs strengthen?"
  Star plots ask:  "When this symbol is involved in the context, which
                    OTHER symbols become salient on average?"

Also produces a companion radial-violin figure (full distribution per symbol
across ALL contexts), called fig_v1_starplots_radial.

Output:
  paper/v2/figures/fig_v1_starplots.{pdf,png}
  paper/v2/figures/fig_v1_starplots_radial.{pdf,png}
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from _setup import build_space, save_fig, set_paper_style


N_ITER = 10
PROPOSE_KW = dict(tau=0.3, lam=0.6, alpha=0.85, use_ppr=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-iter", type=int, default=N_ITER)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)
    syms = list(space.symbols)
    N = len(syms)
    rng = np.random.default_rng(args.seed)

    # Per-symbol colours — tab20 (matches v1 code)
    palette = plt.cm.tab20(np.linspace(0, 1, N))
    sym_color = {s: palette[i] for i, s in enumerate(syms)}

    # ──────────────────────────────────────────────────────────────────────
    # Main bootstrapping loop.
    #
    # all_results[root][q] = mean score for q across (partner, sample)
    # pairs where root was in the context but q was not.
    # all_scores_pool[q] = full pool of q-scores across ALL context pairs
    # (any root, any partner not = q), used for the radial-violin figure.
    # ──────────────────────────────────────────────────────────────────────
    all_results: dict[str, dict[str, float]] = {}
    all_scores_pool: dict[str, list[float]] = {s: [] for s in syms}

    print(f"[starplots] {N * (N - 1)} (root, partner) pairs × "
          f"{args.n_iter} samples = {N * (N - 1) * args.n_iter} "
          f"propose() calls")
    t0 = time.time()

    for context_root in syms:
        others = [s for s in syms if s != context_root]
        symbol_score_dict: dict[str, list[float]] = {s: [] for s in others}

        for ctx_partner in others:
            ctx_pair = [context_root, ctx_partner]
            weights_samples = rng.dirichlet([1, 1], size=args.n_iter)
            for w in weights_samples:
                weights_ctx = dict(zip(ctx_pair, w))
                preds = space.propose(weights=weights_ctx,
                                      topk=len(syms), **PROPOSE_KW)
                score_dict = {s: score for s, score, _coh, _pr in preds}
                for s in syms:
                    if s not in ctx_pair:
                        symbol_score_dict[s].append(score_dict[s])
                        all_scores_pool[s].append(score_dict[s])

        all_results[context_root] = {
            s: float(np.mean(symbol_score_dict[s])) if symbol_score_dict[s] else 0.0
            for s in others
        }
        print(f"[starplots]   {context_root:12s}  done   "
              f"(elapsed {time.time() - t0:.1f}s)")

    # ──────────────────────────────────────────────────────────────────────
    # Figure 1 — star bar plots, one polar subplot per root
    # ──────────────────────────────────────────────────────────────────────
    ncols = int(np.ceil(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, subplot_kw=dict(polar=True),
        figsize=(2.7 * ncols, 2.7 * nrows),
    )
    axs_flat = axs.flatten() if N > 1 else [axs]

    # Determine shared radial limit across panels
    r_max = max(
        max(all_results[r].values()) for r in syms if all_results[r]
    ) * 1.08
    r_max = max(r_max, 1e-3)

    angles = np.linspace(0, 2 * np.pi, N - 1, endpoint=False)

    for i, root in enumerate(syms):
        ax = axs_flat[i]
        others = [s for s in syms if s != root]
        means = [all_results[root][s] for s in others]
        bar_colors = [sym_color[s] for s in others]
        ax.bar(
            angles, means,
            width=2 * np.pi / (N - 1) * 0.85,
            align="center",
            alpha=0.85, color=bar_colors,
            edgecolor="black", linewidth=0.5,
        )
        ax.set_xticks(angles)
        ax.set_xticklabels(others, fontsize=6.5)
        ax.set_yticklabels([])
        ax.set_title(root, va="bottom", fontsize=10, pad=10)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.grid(False)
        ax.set_ylim(0, r_max)

    # Hide unused cells
    for idx in range(N, nrows * ncols):
        fig.delaxes(axs_flat[idx])

    fig.suptitle(
        "Star plots of mean prediction scores for each root symbol\n"
        f"(across all context-pair partners, bootstrapped n_iter={args.n_iter})",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    save_fig(fig, "fig4_starplots")
    plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────
    # Figure 2 — radial violin of full distributions, one violin per symbol
    # ──────────────────────────────────────────────────────────────────────
    angles2 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    fig2 = plt.figure(figsize=(6.5, 6.5))
    ax2 = fig2.add_subplot(111, polar=True)

    # Shared radial range
    pool_all = np.concatenate([np.asarray(all_scores_pool[s])
                               for s in syms if all_scores_pool[s]])
    r_max2 = float(pool_all.max()) * 1.08

    for i, s in enumerate(syms):
        data = np.asarray(all_scores_pool[s])
        if len(data) < 2:
            continue
        angle = angles2[i]
        kde = gaussian_kde(data)
        r_grid = np.linspace(0, r_max2, 200)
        dens = kde(r_grid)
        dens = dens / dens.max() * (2 * np.pi / N) * 0.40
        # Two-sided violin
        ax2.plot(angle + dens, r_grid, color=sym_color[s], lw=1.4)
        ax2.plot(angle - dens, r_grid, color=sym_color[s], lw=1.4)
        ax2.fill_betweenx(r_grid, angle - dens, angle + dens,
                          color=sym_color[s], alpha=0.55)
        # Mean tick
        mean = float(np.mean(data))
        ax2.plot([angle - 0.04, angle + 0.04], [mean, mean],
                 color="black", lw=2.0, zorder=6)

    ax2.set_xticks(angles2)
    ax2.set_xticklabels(syms, fontsize=9)
    ax2.set_yticklabels([])
    ax2.set_ylim(0, r_max2)
    ax2.set_title(
        "Radial violin — distribution of prediction scores per symbol\n"
        f"(across every (root, partner) context pair, n_iter={args.n_iter})",
        fontsize=10, pad=22,
    )
    ax2.grid(False)
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    plt.tight_layout()
    save_fig(fig2, "fig_v1_starplots_radial")
    plt.close(fig2)


if __name__ == "__main__":
    main()
