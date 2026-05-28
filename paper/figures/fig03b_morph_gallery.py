"""Figure 3b — Morphing gallery: four A↔B context pairs side by side.

Companion to fig03 (which analyses one pair in depth).  This figure shows
that the U-shaped phase transition in (coverage_h1, focus) generalizes
across multiple Black Elk Speaks vision morphs, not just the C_A ↔ C_B
pair.

Each column is one (A, B) morph; the four pairs systematically cover the
three vision-register contexts plus the visionary-call ↔ vision-loss pair:

  col 1  C_A ↔ C_B    sacred voice calling  ↔  the dream's end
  col 2  C1  ↔ C3     the sacred hoop       ↔  the dream's end
  col 3  C1  ↔ C2     the sacred hoop       ↔  thunder voice & dawn horse
  col 4  C2  ↔ C3     thunder voice & dawn horse ↔ the dream's end

Top row    — phase portrait in (coverage_h1, focus), α-coloured trajectory.
Bottom row — coverage_h1 and focus as a function of α.

Output: paper/v2/figures/fig03b_morph_gallery.{pdf,png}
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from _setup import (
    CONTEXTS, CONTEXT_LABELS,
    build_space, save_fig, set_paper_style,
)

SHIFT_KW = dict(
    strategy="gate", gate="relu", beta=1.2, tau=0.3,
    within_symbol_softmax=True, gamma=0.5,
    pool_type="avg", pool_w=0.7, membership_alpha=0.0,
)

# Default morphing pairs — each draws on the existing CONTEXTS dict so swap-
# ping any context propagates here too.
MORPH_PAIRS = [
    ("C_A", "C_B"),
    ("C1", "C3"),
    ("C1", "C2"),
    ("C2", "C3"),
]

ALPHA_STEPS = 21


def _ripser():
    try:
        from ripser import ripser
        return ripser
    except ImportError as e:
        raise SystemExit(f"ripser required — `pip install ripser`  ({e})")


def _row_norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _ph_dgms(X, *, maxdim=2, thresh=1.0):
    return _ripser()(X, maxdim=maxdim, metric="cosine",
                     thresh=thresh, do_cocycles=False)["dgms"]


def _sum_finite(dgm):
    if dgm.size == 0:
        return 0.0
    fin = np.isfinite(dgm[:, 1])
    if not fin.any():
        return 0.0
    return float(np.sum(dgm[fin, 1] - dgm[fin, 0]))


def _blend_vec(space, sent_a, sent_b, alpha):
    v_a = space.embedder.encode([sent_a])[0]
    v_b = space.embedder.encode([sent_b])[0]
    return (1.0 - alpha) * v_a + alpha * v_b


def _sweep(space, sent_a, sent_b, alphas):
    """Return (coverage_h1, focus) trajectories over α."""
    coverage_h1 = []
    focus_vals  = []
    for alpha in alphas:
        v = _blend_vec(space, sent_a, sent_b, float(alpha))
        space.set_context_vec(v)
        D1 = space.make_shifted_matrix(**SHIFT_KW)
        space.set_context_vec(None)

        # Union PH (coverage_h1)
        union_X = _row_norm(D1)
        if union_X.shape[0] > 150:
            rng = np.random.default_rng(42)
            sel = rng.choice(union_X.shape[0], 150, replace=False)
            union_X = union_X[sel]
        union_dgms = _ph_dgms(union_X, maxdim=2, thresh=1.0)
        coverage_h1.append(_sum_finite(union_dgms[1]))

        # Focus from per-symbol H0 median bar-length
        cohs = []
        for idx in space.symbol_to_idx.values():
            if len(idx) < 4:
                continue
            X = _row_norm(D1[idx])
            d0 = _ph_dgms(X, maxdim=0, thresh=1.0)[0]
            fin = np.isfinite(d0[:, 1])
            if fin.any():
                cohs.append(float(np.median(d0[fin, 1] - d0[fin, 0])))
        focus_vals.append(
            1.0 / (1.0 + float(np.mean(cohs))) if cohs else 0.5
        )
    return np.array(coverage_h1), np.array(focus_vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--steps", type=int, default=ALPHA_STEPS)
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    alphas = np.linspace(0.0, 1.0, args.steps)
    trajectories: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    t0 = time.time()
    for ca, cb in MORPH_PAIRS:
        sent_a = CONTEXTS[ca]; sent_b = CONTEXTS[cb]
        lab_a  = CONTEXT_LABELS[ca]; lab_b = CONTEXT_LABELS[cb]
        print(f"[fig03b] morph  {ca} ↔ {cb}   ({lab_a}  ↔  {lab_b})")
        cov, foc = _sweep(space, sent_a, sent_b, alphas)
        trajectories.append((ca, cb, cov, foc))
        print(f"[fig03b]   done  elapsed {time.time() - t0:.1f}s   "
              f"cov range [{cov.min():.3f}, {cov.max():.3f}]   "
              f"focus range [{foc.min():.4f}, {foc.max():.4f}]")

    # ── Render — 2 rows × n_pairs cols ──────────────────────────────────
    n_cols = len(MORPH_PAIRS)
    fig = plt.figure(figsize=(3.4 * n_cols, 6.6))
    gs = fig.add_gridspec(
        2, n_cols,
        height_ratios=[1.0, 1.0],
        hspace=0.40, wspace=0.35,
        top=0.90, bottom=0.10, left=0.06, right=0.97,
    )

    for col, (ca, cb, cov, foc) in enumerate(trajectories):
        lab_a = CONTEXT_LABELS[ca]; lab_b = CONTEXT_LABELS[cb]

        # Top row — phase portrait (α-coloured trajectory)
        ax_top = fig.add_subplot(gs[0, col])
        pts = ax_top.scatter(cov, foc, c=alphas, cmap="viridis",
                             s=36, edgecolors="black", linewidths=0.4, zorder=3)
        ax_top.plot(cov, foc, "-", color="0.5", lw=0.8, alpha=0.7, zorder=2)
        ax_top.annotate(
            "α=0", xy=(cov[0], foc[0]),
            xytext=(7, 7), textcoords="offset points",
            fontsize=7, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.5),
        )
        ax_top.annotate(
            "α=1", xy=(cov[-1], foc[-1]),
            xytext=(-7, -7), textcoords="offset points",
            fontsize=7, ha="right", va="top",
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.5),
        )
        ax_top.set_xlabel("coverage_h1", fontsize=8)
        ax_top.set_ylabel("focus", fontsize=8)
        ax_top.set_title(
            f"{ca} ↔ {cb}\n{lab_a}  ↔  {lab_b}",
            fontsize=8.5,
        )
        if col == n_cols - 1:
            cb_axes = plt.colorbar(pts, ax=ax_top, shrink=0.7, pad=0.04)
            cb_axes.set_label("α", fontsize=7.5)

        # Bottom row — coverage_h1 + focus over α (dual axis)
        ax_bot = fig.add_subplot(gs[1, col])
        ax_bot.plot(alphas, cov, "-o", markersize=3, color="#b3262a",
                    label="coverage_h1")
        ax_bot.set_xlabel("α (A → B)", fontsize=8)
        ax_bot.set_ylabel("coverage_h1", color="#b3262a", fontsize=8)
        ax_bot.tick_params(axis="y", labelcolor="#b3262a")
        ax_bot2 = ax_bot.twinx()
        ax_bot2.plot(alphas, foc, "-o", markersize=3, color="#2f5d8f",
                     label="focus")
        ax_bot2.set_ylabel("focus", color="#2f5d8f", fontsize=8)
        ax_bot2.tick_params(axis="y", labelcolor="#2f5d8f")
        ax_bot.set_xlim(0, 1)

    fig.suptitle(
        "Morphing gallery — phase portraits and α-trajectories across "
        "four context pairs from Black Elk Speaks",
        fontsize=11, y=0.96,
    )

    save_fig(fig, "fig03b_morph_gallery")
    plt.close(fig)


if __name__ == "__main__":
    main()
