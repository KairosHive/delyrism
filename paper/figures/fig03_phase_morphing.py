"""Figure 3 — Phase transitions in continuous morphing (PLAN.md §6, Analysis 3).

Sweep blend α ∈ [0, 1] between contexts C_A and C_B.  Track:
  • per-symbol H1+H2 count as a function of α
  • migration events (descriptors switching nearest archetype)
  • Δ-edge sign flip events (top-K edge signs reversing between adjacent α)
  • set-level coverage_h1 + focus

Three-panel figure:
  (a) per-symbol H1+H2 trajectories (line per symbol)
  (b) migration count + Δ-edge sign flips per α-step (event chart)
  (c) coverage_h1 and focus over α

Output: paper/v2/figures/fig03_phase_morphing.{pdf,png}
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

TOP_EDGES_PER_STEP = 60
ALPHA_STEPS = 21          # 0.00, 0.05, …, 1.00


def _ripser():
    try:
        from ripser import ripser
        return ripser
    except ImportError as e:
        raise SystemExit(
            "ripser is required for fig03 — `pip install ripser`\n"
            f"  ({e})"
        )


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


def _persistent_count(dgm, thr=0.02):
    if dgm.size == 0:
        return 0
    fin = np.isfinite(dgm[:, 1])
    if not fin.any():
        return 0
    return int(np.sum((dgm[fin, 1] - dgm[fin, 0]) > thr))


def _blend_context_vec(space, sent_a, sent_b, alpha):
    """Encode A and B with the embedder; return the linear blend vector."""
    v_a = space.embedder.encode([sent_a])[0]
    v_b = space.embedder.encode([sent_b])[0]
    return (1.0 - alpha) * v_a + alpha * v_b


def _nearest_archetype(space, D_use):
    """For each descriptor row, return the archetype with closest centroid."""
    cents = np.stack([_row_norm(D_use[space.symbol_to_idx[s]]).mean(0) for s in space.symbols])
    cents = _row_norm(cents)
    sims = _row_norm(D_use) @ cents.T
    arg = sims.argmax(1)
    return [space.symbols[i] for i in arg]


def _top_edges(D_use, top_n=TOP_EDGES_PER_STEP):
    """Return (edge_keys, sign) for top-|Δ|-against-intrinsic edges.

    Edge key is a frozenset((i, j)) so it's hashable / comparable.  We compute
    Δ = D'D'^T − DD^T against the *intrinsic* D, then pick top-N |Δ| in the
    upper triangle.
    """
    return None  # filled inline below — kept here for documentation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--steps", type=int, default=ALPHA_STEPS)
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    sent_a = CONTEXTS["C_A"]; sent_b = CONTEXTS["C_B"]
    lab_a  = CONTEXT_LABELS["C_A"]; lab_b = CONTEXT_LABELS["C_B"]
    print(f"[fig03] A = {sent_a!r}")
    print(f"[fig03] B = {sent_b!r}")

    alphas = np.linspace(0.0, 1.0, args.steps)

    # Pre-compute intrinsic objects we re-use
    D0 = space.D
    C0 = D0 @ D0.T

    per_sym_h12 = {s: [] for s in space.symbols}
    coverage_h1 = []
    focus_vals  = []
    migrations  = []           # count of descriptors that changed archetype vs intrinsic

    # Pre-compute upper-triangle indices once
    n_desc = D0.shape[0]
    tri_i, tri_j = np.triu_indices(n_desc, k=1)
    # Store Δ for every pair at every α — we'll pick the most-varying ones for panel (a).
    # Shape: (n_alphas, n_pairs).  For 140 descriptors × 21 α this is ~204K floats, fine.
    delta_trace = np.zeros((len(alphas), len(tri_i)), dtype=np.float32)

    # Baseline nearest archetype on intrinsic field
    base_nearest = _nearest_archetype(space, D0)

    for k, alpha in enumerate(alphas):
        # Blend context vector
        v = _blend_context_vec(space, sent_a, sent_b, float(alpha))

        # Push into the engine as an override so make_shifted_matrix uses it
        space.set_context_vec(v)
        D1 = space.make_shifted_matrix(**SHIFT_KW)
        space.set_context_vec(None)

        # Per-symbol PH
        h12_total = 0
        for s in space.symbols:
            idx = space.symbol_to_idx[s]
            if len(idx) < 4:
                per_sym_h12[s].append(0)
                continue
            X = _row_norm(D1[idx])
            dgms = _ph_dgms(X, maxdim=2, thresh=1.0)
            cnt = _persistent_count(dgms[1]) + _persistent_count(dgms[2])
            per_sym_h12[s].append(cnt)
            h12_total += cnt

        # Set-level union PH (coverage_h1 + focus)
        union_X = _row_norm(D1)
        if union_X.shape[0] > 150:
            rng = np.random.default_rng(42)
            sel = rng.choice(union_X.shape[0], 150, replace=False)
            union_X = union_X[sel]
        union_dgms = _ph_dgms(union_X, maxdim=2, thresh=1.0)
        coverage_h1.append(_sum_finite(union_dgms[1]))

        # Focus: 1 / (1 + mean per-symbol H0 cohesion)
        cohs = []
        for idx in space.symbol_to_idx.values():
            if len(idx) < 4:
                continue
            X = _row_norm(D1[idx])
            d0 = _ph_dgms(X, maxdim=0, thresh=1.0)[0]
            fin = np.isfinite(d0[:, 1])
            if fin.any():
                cohs.append(float(np.median(d0[fin, 1] - d0[fin, 0])))
        focus_vals.append(1.0 / (1.0 + float(np.mean(cohs)))) if cohs else focus_vals.append(0.5)

        # Migrations vs intrinsic
        nearest = _nearest_archetype(space, D1)
        migrations.append(sum(int(a != b) for a, b in zip(nearest, base_nearest)))

        # Δ matrix at this α — store the upper-triangle values for the
        # descriptor-pair trajectory panel below.
        Delta = D1 @ D1.T - C0
        np.fill_diagonal(Delta, 0.0)
        delta_trace[k, :] = Delta[tri_i, tri_j]

        print(f"[fig03] α={alpha:.2f}  H1+H2={h12_total:3d}  "
              f"migrations={migrations[-1]:2d}  "
              f"cov_h1={coverage_h1[-1]:.3f}  focus={focus_vals[-1]:.3f}")

    # ─── Render ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.36,
                          top=0.86, bottom=0.18, left=0.06, right=0.96)
    cdict = space.get_symbol_color_dict(palette="Nord")

    # ── (a) Top-varying descriptor-pair Δ trajectories across α ─────────────
    # Pick the pairs whose Δ varies most across the morph — these are the
    # couplings that "phase-transition" rather than staying constant.  We use
    # std over α to rank, then take the top-N.  Plot each as a coloured line.
    pair_std = delta_trace.std(axis=0)
    top_pair_idx = np.argsort(-pair_std)[:8]
    ax_a = fig.add_subplot(gs[0, 0])
    for rank, idx in enumerate(top_pair_idx):
        i, j = int(tri_i[idx]), int(tri_j[idx])
        di, dj = space.descriptors[i], space.descriptors[j]
        si, sj = space.owner[di], space.owner[dj]
        # Colour line by the "stronger" symbol in the pair (alphabetically tie-break)
        color = cdict.get(si if si <= sj else sj, "0.5")
        label = f"{di[:16]} ↔ {dj[:16]}"
        ax_a.plot(alphas, delta_trace[:, idx], "-",
                  color=color, lw=1.4, alpha=0.85, label=label)
    ax_a.axhline(0, color="0.7", lw=0.5)
    ax_a.set_xlabel("α (A → B)")
    ax_a.set_ylabel("Δ coupling   (D'D'ᵀ − DDᵀ)")
    ax_a.set_title(f"(a) Most-varying descriptor pairs  ({lab_a} → {lab_b})",
                   fontsize=9)
    ax_a.legend(ncol=1, fontsize=6.0, loc="best", frameon=False,
                handlelength=1.4, handletextpad=0.4)
    ax_a.set_xlim(0, 1)

    # ── (b) Phase portrait — coverage_h1 vs focus, parametric over α ───────
    # The trajectory in (cov, focus) phase-space.  Loops or sharp turns
    # indicate phase transitions; a clean curve means the morph is smooth.
    ax_b = fig.add_subplot(gs[0, 1])
    cov = np.array(coverage_h1); foc = np.array(focus_vals)
    # Colour the trajectory by α so we can read direction
    pts = ax_b.scatter(cov, foc, c=alphas, cmap="viridis",
                       s=42, edgecolors="black", linewidths=0.4, zorder=3)
    # Connect consecutive points
    ax_b.plot(cov, foc, "-", color="0.5", lw=0.8, alpha=0.7, zorder=2)
    # Mark endpoints
    ax_b.annotate(f"α=0\n({lab_a})", xy=(cov[0], foc[0]),
                  xytext=(8, 8), textcoords="offset points",
                  fontsize=7.5, ha="left", va="bottom",
                  arrowprops=dict(arrowstyle="-", color="0.4", lw=0.5))
    ax_b.annotate(f"α=1\n({lab_b})", xy=(cov[-1], foc[-1]),
                  xytext=(-8, -8), textcoords="offset points",
                  fontsize=7.5, ha="right", va="top",
                  arrowprops=dict(arrowstyle="-", color="0.4", lw=0.5))
    ax_b.set_xlabel("coverage_h1")
    ax_b.set_ylabel("focus")
    ax_b.set_title("(b) Phase portrait  (cov_h1, focus)", fontsize=9)
    cb = plt.colorbar(pts, ax=ax_b, shrink=0.7, pad=0.02)
    cb.set_label("α", fontsize=7.5)

    # ── (c) coverage_h1 + focus over α (unchanged — this is the clean panel) ─
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.plot(alphas, coverage_h1, "-o", markersize=3, color="#b3262a",
              label="coverage_h1")
    ax_c.set_xlabel("α (A → B)")
    ax_c.set_ylabel("coverage_h1", color="#b3262a")
    ax_c.tick_params(axis="y", labelcolor="#b3262a")
    ax_c2 = ax_c.twinx()
    ax_c2.plot(alphas, focus_vals, "-o", markersize=3, color="#2f5d8f", label="focus")
    ax_c2.set_ylabel("focus", color="#2f5d8f")
    ax_c2.tick_params(axis="y", labelcolor="#2f5d8f")
    ax_c.set_title("(c) Set-level signals over α", fontsize=9)
    ax_c.set_xlim(0, 1)

    fig.suptitle(
        f"Phase transitions across continuous morphing  ({lab_a} ↔ {lab_b})",
        fontsize=11, y=1.02,
    )
    save_fig(fig, "fig03_phase_morphing")
    plt.close(fig)


if __name__ == "__main__":
    main()
