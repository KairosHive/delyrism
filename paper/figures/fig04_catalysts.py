"""Figure 4 — Catalysts and structural integrity (PLAN.md §6, Analysis 4).

For each archetype, leave-one-out PH: how much persistent H1+H2 mass collapses
when each descriptor is removed?  The top-3 catalysts per archetype are the
*load-bearing* descriptors — those holding the relational shape together.

No context — purely intrinsic analysis.

Output:
  paper/v2/figures/fig04_catalysts.{pdf,png}
  paper/v2/figures/fig04_catalysts_table.csv   ← top-3 per archetype
"""
from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from _setup import OUTDIR, build_space, save_fig, set_paper_style


def _ripser():
    try:
        from ripser import ripser
        return ripser
    except ImportError as e:
        raise SystemExit(
            "ripser is required for fig04 — `pip install ripser`\n"
            f"  ({e})"
        )


def _row_norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _ph_dgms(X, *, maxdim=2):
    return _ripser()(X, maxdim=maxdim, metric="cosine", do_cocycles=False)["dgms"]


def _sum_finite(dgm):
    if dgm.size == 0:
        return 0.0
    fin = np.isfinite(dgm[:, 1])
    if not fin.any():
        return 0.0
    return float(np.sum(dgm[fin, 1] - dgm[fin, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--top", type=int, default=3, help="catalysts per archetype")
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    # Compute LOO catalysts per archetype
    table: list[dict] = []
    bar_data: list[tuple[str, float, str]] = []   # (archetype, importance, descriptor)

    for s in space.symbols:
        idx = space.symbol_to_idx[s]
        if len(idx) < 5:
            print(f"[fig04] skipping {s}: only {len(idx)} descriptors")
            continue
        X = _row_norm(space.D[idx])
        descs = list(space.symbols_to_descriptors[s])

        base = _ph_dgms(X, maxdim=2)
        base_mass = _sum_finite(base[1]) + _sum_finite(base[2])

        impacts = []
        for i in range(len(idx)):
            Xi = np.delete(X, i, axis=0)
            if Xi.shape[0] < 4:
                impacts.append(0.0); continue
            dgmi = _ph_dgms(Xi, maxdim=2)
            loo_mass = _sum_finite(dgmi[1]) + _sum_finite(dgmi[2])
            impacts.append(base_mass - loo_mass)

        rank = np.argsort(impacts)[::-1]
        print(f"[fig04] {s}: base H1+H2 mass = {base_mass:.3f}, top catalysts:")
        for r in rank[:args.top]:
            print(f"        {descs[r]:30s}  Δ = {impacts[r]:+.3f}")
            table.append({
                "archetype": s,
                "descriptor": descs[r],
                "impact": float(impacts[r]),
                "base_mass": float(base_mass),
            })
            bar_data.append((s, float(impacts[r]), descs[r]))

    # ─── Render single horizontal bar chart, grouped by archetype ─────────
    cdict = space.get_symbol_color_dict(palette="Nord")

    # Sort by archetype order, then by impact desc within archetype
    sym_order = {s: i for i, s in enumerate(space.symbols)}
    bar_data.sort(key=lambda r: (sym_order[r[0]], -r[1]))

    labels = [f"{r[2]}  ({r[0]})" for r in bar_data]
    impacts = [r[1] for r in bar_data]
    colors  = [cdict.get(r[0], "0.5") for r in bar_data]

    n = len(bar_data)
    fig_h = max(4.0, 0.20 * n + 1.2)
    fig, ax = plt.subplots(figsize=(7.8, fig_h))
    y = np.arange(n)
    ax.barh(y, impacts, color=colors, edgecolor="black", linewidth=0.3, alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Δ persistent (H1 + H2)  →  load-bearing")
    ax.set_title(f"Top-{args.top} catalysts per archetype  (Lakota Shape Kit, intrinsic)")
    ax.axvline(0, color="0.7", lw=0.6)
    save_fig(fig, "fig04_catalysts")
    plt.close(fig)

    # Save CSV companion
    csv_path = OUTDIR / "fig04_catalysts_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["archetype", "descriptor", "impact", "base_mass"])
        w.writeheader()
        w.writerows(table)
    print(f"[fig04] saved table → {csv_path.name}")


if __name__ == "__main__":
    main()
