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
    edge_flips  = []           # count of top-edge signs flipped vs previous step

    prev_signs: dict[frozenset, int] | None = None

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

        # Top-edge sign flips vs previous step
        Delta = D1 @ D1.T - C0
        np.fill_diagonal(Delta, 0.0)
        tri_i, tri_j = np.triu_indices_from(Delta, k=1)
        vals = Delta[tri_i, tri_j]
        order = np.argsort(-np.abs(vals))[:TOP_EDGES_PER_STEP]
        signs = {
            frozenset((int(tri_i[k]), int(tri_j[k]))): int(np.sign(vals[k]))
            for k in order
        }
        if prev_signs is None:
            edge_flips.append(0)
        else:
            common = set(prev_signs.keys()) & set(signs.keys())
            flips = sum(1 for e in common if prev_signs[e] * signs[e] < 0)
            # Edges that left or entered the top-N count as "structural change" too,
            # but at half weight so true sign flips dominate.
            churn = len(prev_signs.keys() ^ signs.keys())
            edge_flips.append(flips + churn // 2)
        prev_signs = signs

        print(f"[fig03] α={alpha:.2f}  H1+H2={h12_total:3d}  "
              f"migrations={migrations[-1]:2d}  "
              f"edge-churn={edge_flips[-1]:2d}")

    # ─── Render ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.36,
                          top=0.86, bottom=0.18, left=0.06, right=0.96)

    # (a) field-total topology over α — most per-symbol lines stay near
    # 1 under heavy compression, so the meaningful signal is the SUM over
    # symbols.  Plot the field total, with thin colored ticks at the bottom
    # showing which symbols contribute non-zero at each α.
    ax_a = fig.add_subplot(gs[0, 0])
    cdict = space.get_symbol_color_dict(palette="Nord")
    field_total = np.array([sum(per_sym_h12[s][k] for s in space.symbols)
                            for k in range(len(alphas))])
    ax_a.plot(alphas, field_total, "-o", markersize=4, lw=1.6,
              color="black", label="field total H1+H2")
    # Mark per-symbol contributions as colored ticks (stacked) at the bottom
    bar_y_base = -0.5
    for s in space.symbols:
        vals = np.array(per_sym_h12[s])
        active = vals > 0
        if active.any():
            ax_a.scatter(alphas[active], np.full(active.sum(), bar_y_base),
                         color=cdict.get(s, "0.5"), s=22,
                         marker="s", label=s if vals.max() > 0 else None,
                         edgecolors="none")
            bar_y_base -= 0.45
    ax_a.set_xlabel("α (A → B)")
    ax_a.set_ylabel("persistent H1 + H2 (count)")
    ax_a.set_title(f"(a) Field topology over morphing  ({lab_a} → {lab_b})",
                   fontsize=9)
    handles = [plt.Line2D([0], [0], color="black", marker="o", label="field total")]
    handles += [plt.Line2D([0], [0], color=cdict.get(s, "0.5"), marker="s",
                           linestyle="", label=s)
                for s in space.symbols if max(per_sym_h12[s]) > 0]
    ax_a.legend(handles=handles, ncol=2, fontsize=6.4, loc="upper left",
                frameon=False)
    ax_a.set_xlim(0, 1)
    ax_a.axhline(0, color="0.85", lw=0.5)

    # (b) migration + edge-churn events
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.bar(alphas, migrations, width=0.04, color="#b3262a", alpha=0.85,
             label="migrations vs intrinsic")
    ax_b.set_xlabel("α (A → B)")
    ax_b.set_ylabel("migrations (count)", color="#b3262a")
    ax_b.tick_params(axis="y", labelcolor="#b3262a")
    ax_b2 = ax_b.twinx()
    ax_b2.plot(alphas, edge_flips, "-o", color="#2f5d8f", markersize=3,
               lw=1.0, label="top-edge churn vs prev")
    ax_b2.set_ylabel("top-edge churn (count)", color="#2f5d8f")
    ax_b2.tick_params(axis="y", labelcolor="#2f5d8f")
    ax_b.set_title("(b) Phase-transition events")
    ax_b.set_xlim(0, 1)

    # (c) coverage_h1 + focus
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
    ax_c.set_title("(c) Set-level signals")
    ax_c.set_xlim(0, 1)

    fig.suptitle(
        f"Phase transitions across continuous morphing  ({lab_a} ↔ {lab_b})",
        fontsize=11, y=1.02,
    )
    save_fig(fig, "fig03_phase_morphing")
    plt.close(fig)


if __name__ == "__main__":
    main()
