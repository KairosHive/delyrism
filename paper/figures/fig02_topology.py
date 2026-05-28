"""Figure 2 — Topology of a symbolic field: intrinsic vs context-induced
(PLAN.md §6, Analysis 2).

Three panels:
  (a) Persistence diagrams (H0/H1/H2) on the intrinsic descriptor matrix D
      and on the context-shifted D' under C3.
  (b) Per-symbol H1+H2 persistent count, intrinsic vs context-induced.
  (c) Set-level metrics bar chart (coverage_h1/h2, focus, separation).

Output: paper/v2/figures/fig02_topology.{pdf,png}
"""
from __future__ import annotations

import argparse

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

PERS_THR = 0.02            # below this is considered noise
DGM_COLORS = {0: "#444", 1: "#b3262a", 2: "#2f5d8f"}


def _ripser():
    try:
        from ripser import ripser  # noqa: F401
        return ripser
    except ImportError as e:
        raise SystemExit(
            "ripser is required for fig02 — `pip install ripser`\n"
            f"  ({e})"
        )


def _row_norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _ph_dgms(X, *, maxdim=2, thresh=1.0):
    ripser_fn = _ripser()
    return ripser_fn(X, maxdim=maxdim, metric="cosine",
                     thresh=thresh, do_cocycles=False)["dgms"]


def _sum_finite(dgm):
    if dgm.size == 0:
        return 0.0
    fin = np.isfinite(dgm[:, 1])
    if not fin.any():
        return 0.0
    return float(np.sum(dgm[fin, 1] - dgm[fin, 0]))


def _persistent_count(dgm, thr=PERS_THR):
    if dgm.size == 0:
        return 0
    fin = np.isfinite(dgm[:, 1])
    if not fin.any():
        return 0
    return int(np.sum((dgm[fin, 1] - dgm[fin, 0]) > thr))


def _draw_diagram(ax, dgms_intrinsic, dgms_ctx, title):
    """One persistence-diagram panel comparing intrinsic vs context."""
    max_finite = 0.0
    for d in (dgms_intrinsic, dgms_ctx):
        for dim, dgm in enumerate(d[:3]):
            if dgm.size:
                fin = np.isfinite(dgm[:, 1])
                if fin.any():
                    max_finite = max(max_finite, float(dgm[fin, 1].max()))
    if max_finite <= 0:
        max_finite = 1.0
    inf_y = max_finite * 1.08

    for dim, dgm in enumerate(dgms_intrinsic[:3]):
        if not dgm.size:
            continue
        births = dgm[:, 0]; deaths = np.where(np.isfinite(dgm[:, 1]), dgm[:, 1], inf_y)
        ax.scatter(births, deaths, s=22, c=DGM_COLORS[dim],
                   marker="o", alpha=0.55, label=f"H{dim} intrinsic",
                   edgecolors="none")
    for dim, dgm in enumerate(dgms_ctx[:3]):
        if not dgm.size:
            continue
        births = dgm[:, 0]; deaths = np.where(np.isfinite(dgm[:, 1]), dgm[:, 1], inf_y)
        ax.scatter(births, deaths, s=22, c=DGM_COLORS[dim],
                   marker="x", alpha=0.85, label=f"H{dim} context",
                   linewidths=1.2)

    ax.plot([0, inf_y], [0, inf_y], "--", color="0.7", lw=0.7)
    ax.axhline(inf_y, color="0.85", lw=0.5)
    ax.set_xlim(-0.02, max_finite * 1.08)
    ax.set_ylim(-0.02, inf_y * 1.05)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=6.5, ncol=2)


def _per_symbol_counts(space, D_use):
    out = {}
    for s in space.symbols:
        idx = space.symbol_to_idx[s]
        if len(idx) < 4:
            out[s] = (0, 0)
            continue
        X = _row_norm(D_use[idx])
        dgms = _ph_dgms(X, maxdim=2, thresh=1.0)
        out[s] = (_persistent_count(dgms[1]), _persistent_count(dgms[2]))
    return out


def _set_metrics(D_use, sym_to_idx):
    """Replicates the backend's set-quality metrics (coverage / focus / separation)."""
    parts = []
    for s, idx in sym_to_idx.items():
        if len(idx) >= 4:
            parts.append(_row_norm(D_use[idx]))
    if len(parts) < 2:
        return None
    union_X = _row_norm(np.vstack(parts))
    # Subsample if huge (we won't be here for 10 symbols × ~15 descriptors)
    if union_X.shape[0] > 150:
        rng = np.random.default_rng(42)
        idx = rng.choice(union_X.shape[0], 150, replace=False)
        union_X = union_X[idx]
    dgms = _ph_dgms(union_X, maxdim=2, thresh=1.0)
    cov_h1 = _sum_finite(dgms[1])
    cov_h2 = _sum_finite(dgms[2])

    # Per-symbol H0 cohesion (median per-symbol H0 bar length)
    cohs = []
    for idx in sym_to_idx.values():
        if len(idx) < 4:
            continue
        X = _row_norm(D_use[idx])
        d0 = _ph_dgms(X, maxdim=0, thresh=1.0)[0]
        fin = np.isfinite(d0[:, 1])
        if fin.any():
            cohs.append(float(np.median(d0[fin, 1] - d0[fin, 0])))
    mean_coh = float(np.mean(cohs)) if cohs else 0.0
    focus = 1.0 / (1.0 + mean_coh)

    # Centroid pairwise cosine-distance separation
    cents = np.stack([_row_norm(D_use[idx]).mean(0) for idx in sym_to_idx.values() if len(idx) >= 4])
    cents = _row_norm(cents)
    n = len(cents)
    iu = np.triu_indices(n, 1)
    separation = float(np.mean(1.0 - (cents @ cents.T)[iu]))

    return {
        "coverage_h1": cov_h1,
        "coverage_h2": cov_h2,
        "focus": focus,
        "separation": separation,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--context", default="C3",
                    help="Which placeholder context to overlay (default C3).")
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    # Shifted matrix D' under chosen context
    sentence = CONTEXTS[args.context]
    print(f"[fig02] context: {args.context} = {sentence!r}")
    D_intrinsic = space.D
    D_shifted = space.make_shifted_matrix(sentence=sentence, **SHIFT_KW)

    # ─── Panel A: full-field persistence diagrams ───────────────────────────
    print("[fig02] computing union PH (intrinsic + context)…")
    union_X_intr = _row_norm(D_intrinsic)
    union_X_ctx  = _row_norm(D_shifted)
    dgms_intr = _ph_dgms(union_X_intr, maxdim=2, thresh=1.0)
    dgms_ctx  = _ph_dgms(union_X_ctx,  maxdim=2, thresh=1.0)

    # ─── Panel B: per-symbol H1+H2 counts intrinsic vs context ──────────────
    print("[fig02] computing per-symbol PH (intrinsic + context)…")
    counts_intr = _per_symbol_counts(space, D_intrinsic)
    counts_ctx  = _per_symbol_counts(space, D_shifted)

    # ─── Panel C: set-level metrics ─────────────────────────────────────────
    print("[fig02] computing set-level metrics…")
    metrics_intr = _set_metrics(D_intrinsic, space.symbol_to_idx)
    metrics_ctx  = _set_metrics(D_shifted,  space.symbol_to_idx)

    # ─── Render ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.0], wspace=0.32,
                          top=0.86, bottom=0.18, left=0.05, right=0.97)

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_diagram(ax_a, dgms_intr, dgms_ctx,
                  title=f"(a) Field PH — intrinsic vs '{CONTEXT_LABELS[args.context]}'")

    ax_b = fig.add_subplot(gs[0, 1])
    syms = list(space.symbols)
    h_intr = np.array([counts_intr[s][0] + counts_intr[s][1] for s in syms])
    h_ctx  = np.array([counts_ctx[s][0]  + counts_ctx[s][1]  for s in syms])
    x = np.arange(len(syms))
    w = 0.36
    ax_b.bar(x - w/2, h_intr, w, label="intrinsic", color="0.55", edgecolor="black", linewidth=0.4)
    ax_b.bar(x + w/2, h_ctx, w, label="under context", color="#b3262a", edgecolor="black", linewidth=0.4)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(syms, rotation=35, ha="right", fontsize=7.5)
    ax_b.set_ylabel("persistent H1 + H2 (count)")
    ax_b.set_title("(b) Per-symbol topological richness")
    ax_b.legend(loc="upper right", fontsize=7.5)

    ax_c = fig.add_subplot(gs[0, 2])
    if metrics_intr is None or metrics_ctx is None:
        ax_c.text(0.5, 0.5, "(too few symbols)", ha="center", va="center", transform=ax_c.transAxes)
        ax_c.set_axis_off()
    else:
        names = ["coverage_h1", "coverage_h2", "focus", "separation"]
        vals_intr = [metrics_intr[k] for k in names]
        vals_ctx  = [metrics_ctx[k]  for k in names]
        y = np.arange(len(names))
        ax_c.barh(y - 0.18, vals_intr, 0.35, label="intrinsic", color="0.55", edgecolor="black", linewidth=0.4)
        ax_c.barh(y + 0.18, vals_ctx, 0.35, label="under context", color="#b3262a", edgecolor="black", linewidth=0.4)
        ax_c.set_yticks(y); ax_c.set_yticklabels(names, fontsize=8)
        ax_c.invert_yaxis()
        ax_c.set_xlabel("metric value")
        ax_c.set_title("(c) Set-level quality metrics")
        ax_c.legend(loc="lower right", fontsize=7.5)

    fig.suptitle(
        f"Topology of the symbolic field — intrinsic vs context-induced "
        f"(context: {CONTEXT_LABELS[args.context]})",
        fontsize=11, y=1.02,
    )
    save_fig(fig, "fig02_topology")
    plt.close(fig)


if __name__ == "__main__":
    main()
