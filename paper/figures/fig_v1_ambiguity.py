"""V1 Figure 1b — Per-symbol ambiguity metrics.

Side-by-side grouped bars (Dispersion / Leakage / Inter-symbolic Entropy)
sorted by dispersion descending.  Each symbol gets its tab20 colour; the
three metrics use darken/base/lighten variants of the same colour, which
makes the per-symbol identity readable across the three bars while still
distinguishing the three metrics.

Each metric is min-max normalised then rescaled to [0.1, 0.9] so all three
have a visible bar regardless of raw scale.  This matches the v1 paper's
visualisation choice.

Uses the new z-scored soft_entropy (engine default since the UI update);
see delyrism.delyrism.soft_entropy — `zscore_logits=True` re-spreads cosine
similarities so the entropy actually discriminates between symbols rather
than collapsing to uniform.

Output: paper/v2/figures/fig_v1_ambiguity.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from _setup import build_space, save_fig, set_paper_style


def _darken(rgba, factor: float = 0.6):
    return tuple(factor * np.array(rgba[:3]))


def _lighten(rgba, factor: float = 0.6):
    return tuple((1 - factor) + factor * np.array(rgba[:3]))


def _rescale(values: np.ndarray, lo: float = 0.1, hi: float = 0.9) -> np.ndarray:
    """Min-max normalise to [0, 1] then squeeze to [lo, hi]."""
    v = np.asarray(values, dtype=float)
    mn, mx = float(v.min()), float(v.max())
    if mx - mn < 1e-12:
        return np.full_like(v, (lo + hi) / 2.0)
    z = (v - mn) / (mx - mn)
    return lo + (hi - lo) * z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--soft-entropy-tau", type=float, default=0.5,
                    help="Temperature for the soft archetype-assignment "
                         "softmax inside soft_entropy.")
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    symbols = list(space.symbols)
    disp_raw = np.array([space.dispersion(s) for s in symbols])
    leak_raw = np.array([space.leakage(s, k=min(10, len(space.descriptors) - 1))
                         for s in symbols])
    entr_raw = np.array([space.soft_entropy(s, tau=args.soft_entropy_tau)
                         for s in symbols])

    # Sort by dispersion desc
    sort_idx = np.argsort(disp_raw)[::-1]
    syms_sorted = [symbols[i] for i in sort_idx]
    disp = _rescale(disp_raw[sort_idx])
    leak = _rescale(leak_raw[sort_idx])
    entr = _rescale(entr_raw[sort_idx])

    # Print the raw + rescaled table for the record
    print("\n[Ambiguity metrics — raw and rescaled to [0.1, 0.9]]")
    print(f"{'Symbol':<14} | {'disp raw':>10}  {'disp':>6} | "
          f"{'leak raw':>10}  {'leak':>6} | {'ent raw':>10}  {'ent':>6}")
    print("-" * 78)
    for i_sorted, s in enumerate(syms_sorted):
        i_orig = sort_idx[i_sorted]
        print(f"{s:<14} | {disp_raw[i_orig]:10.3f}  {disp[i_sorted]:6.3f} | "
              f"{leak_raw[i_orig]:10.3f}  {leak[i_sorted]:6.3f} | "
              f"{entr_raw[i_orig]:10.3f}  {entr[i_sorted]:6.3f}")

    # ── Render — grouped bars per symbol ─────────────────────────────────
    n = len(syms_sorted)
    x = np.arange(n)
    w = 0.22

    # Per-symbol tab20 colour (in sorted order)
    base = plt.cm.tab20(np.linspace(0, 1, n))
    base = base[sort_idx]

    disp_colors = [_darken(c) for c in base]
    leak_colors = [c for c in base]
    entr_colors = [_lighten(c, factor=0.8) for c in base]

    fig, ax = plt.subplots(figsize=(max(5.0, 0.55 * n + 1.5), 3.6))
    ax.bar(x - w, disp, width=w, color=disp_colors,
           edgecolor="black", linewidth=0.3, label="Dispersion")
    ax.bar(x,     leak, width=w, color=leak_colors,
           edgecolor="black", linewidth=0.3, label="Leakage")
    ax.bar(x + w, entr, width=w, color=entr_colors,
           edgecolor="black", linewidth=0.3, alpha=0.85,
           label="Inter-symbolic entropy")

    ax.set_xticks(x)
    ax.set_xticklabels(syms_sorted, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("metric value (rescaled to [0.1, 0.9])")
    ax.set_ylim(0, 1.05)
    ax.set_title("Symbol ambiguity metrics  (sorted by dispersion)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    plt.tight_layout()
    save_fig(fig, "fig_v1_ambiguity")
    plt.close(fig)


if __name__ == "__main__":
    main()
