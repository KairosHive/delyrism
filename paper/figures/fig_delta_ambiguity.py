"""Figure — THE READOUT KNOWS WHEN IT DOESN'T KNOW: confidence and ambiguity.

The Δ symbol profile carries its own uncertainty signal: the MARGIN between
the top two symbol responses (ws_z top-1 − top-2).  Three findings on the
30-probe oblique battery + 6 designed-ambiguous phrases:

(A) The margin separates the readout's own hits from its misses (AUC ≈ 0.78),
    and DESIGNED-AMBIGUOUS phrases (written so two symbols fit equally) land
    in the same low-margin region as the misses — low confidence tracks
    genuine underdetermination, not just error.
(B) Selective answering: ranking probes by margin and abstaining on the least
    confident raises top-1 accuracy from 67% (full coverage) to ~90% (top 10).
(C) What confidence looks like: the symbol profile of a high-margin correct
    probe (peaked) vs a low-margin miss (spread over semantically close
    symbols).

Honest scope note: a stricter minimal-pair test (ambiguous phrase vs two
disambiguated rewrites of the same scene) did NOT reliably separate — the
readout's resolution is coarser than minimal pairs.  The margin works at the
population level shown here.

Output: paper/v2/figures/fig_delta_ambiguity.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from _setup import build_space, save_fig, set_paper_style
from _delta_common import DeltaReadout
from fig_delta_probe_atlas import PROBES as REF_PROBES, short_sym

# Phrases written so that TWO symbols of the kit fit equally well.
AMBIGUOUS = [
    ("flash in the sky",    "a sudden light crossed the night sky"),                          # LIGHTNING/STAR
    ("holding everything",  "it held everything the people needed"),                          # BAG/EARTH
    ("drifting high above", "something drifted high above the camp"),                         # CLOUDS/FEATHER
    ("a great voice",       "a great voice filled the air and everyone stopped to listen"),   # THUNDER/HOUSE
    ("quick at the water",  "something quick and light moved at the edge of the water"),      # DRAGONFLY/HORSE TRACK
    ("kept at the center",  "the most precious things were kept safe at the center"),         # HOUSE/BAG
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)
    dr = DeltaReadout(space)
    R = dr.fit_reference([p[2] for p in REF_PROBES])
    intended = [p[1] for p in REF_PROBES]
    n = len(REF_PROBES)

    def profile_margin(z):
        o = np.argsort(z)[::-1]
        return float(z[o[0]] - z[o[1]]), int(o[0])

    Z = np.vstack([dr.ws_z(r) for r in R])
    margins, top1 = np.zeros(n), np.zeros(n, dtype=int)
    for k in range(n):
        margins[k], top1[k] = profile_margin(Z[k])
    correct = np.array([dr.syms[top1[k]] == intended[k] for k in range(n)])

    Za = np.vstack([dr.ws_z(dr.fp(p)) for _, p in AMBIGUOUS])
    m_amb = np.array([profile_margin(Za[k])[0] for k in range(len(AMBIGUOUS))])

    def _auc(pos, neg):
        allv = np.concatenate([pos, neg])
        ranks = allv.argsort().argsort() + 1
        return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                     / (len(pos) * len(neg)))

    auc = _auc(margins[correct], margins[~correct])
    print(f"[ambiguity] margin: correct μ={margins[correct].mean():.2f} "
          f"({correct.sum()})  miss μ={margins[~correct].mean():.2f} "
          f"({(~correct).sum()})  ambiguous μ={m_amb.mean():.2f}  AUC={auc:.2f}")

    # (B) coverage–accuracy by margin
    order = np.argsort(margins)[::-1]
    coverage = np.arange(1, n + 1) / n
    cum_acc = np.cumsum(correct[order]) / np.arange(1, n + 1)
    for cov in (10, 15, 20, 30):
        print(f"[ambiguity] top-{cov} most confident: acc={cum_acc[cov-1]:.0%}")

    # (C) exemplars: best-margin correct, worst-margin miss
    k_hi = int(order[np.argmax(correct[order])])           # highest-margin correct
    miss_order = [k for k in np.argsort(margins) if not correct[k]]
    k_lo = int(miss_order[0])                              # lowest-margin miss

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.0, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.85, 1.0, 0.95, 0.95],
                          wspace=0.36, top=0.80, bottom=0.20, left=0.06,
                          right=0.99)

    # (A) margin strips by group
    axA = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(0)
    groups = [("correct", margins[correct], "#1a9850"),
              ("miss", margins[~correct], "#b2182b"),
              ("designed-\nambiguous", m_amb, "#7b3294")]
    for gx, (name, vals, color) in enumerate(groups):
        jx = rng.uniform(-0.08, 0.08, len(vals))
        axA.scatter(np.full(len(vals), gx) + jx, vals, s=26, color=color,
                    edgecolors="k", linewidths=0.4, alpha=0.85, zorder=3)
        axA.hlines(vals.mean(), gx - 0.2, gx + 0.2, color=color, lw=2)
    axA.set_xticks(range(3))
    axA.set_xticklabels([f"{g[0]}\nμ={g[1].mean():.2f}" for g in groups],
                        fontsize=7)
    axA.set_xlim(-0.5, 2.5)
    axA.set_ylabel("confidence margin (ws$_z$ top-1 − top-2)", fontsize=8)
    axA.set_title(f"(A) The margin separates hits from\nmisses (AUC {auc:.2f}); "
                  f"ambiguous phrases\nland with the misses", fontsize=8.3, pad=5)

    # (B) coverage–accuracy curve
    axB = fig.add_subplot(gs[0, 1])
    axB.step(coverage, cum_acc, where="post", color="#2166ac", lw=1.5)
    axB.axhline(cum_acc[-1], color="0.6", lw=0.8, ls=":")
    axB.annotate(f"answer everything: {cum_acc[-1]:.0%}",
                 (0.36, cum_acc[-1]), fontsize=6.5, color="0.35",
                 xytext=(0, 5), textcoords="offset points")
    for cov in (10, 20):
        axB.scatter([cov / n], [cum_acc[cov - 1]], s=30, color="#2166ac",
                    edgecolors="k", linewidths=0.5, zorder=3)
        axB.annotate(f"top {cov}: {cum_acc[cov-1]:.0%}",
                     (cov / n, cum_acc[cov - 1]), fontsize=6.5,
                     xytext=(6, 6), textcoords="offset points")
    axB.set_xlabel("coverage (fraction of probes answered)", fontsize=8)
    axB.set_ylabel("top-1 accuracy on answered probes", fontsize=8)
    axB.set_ylim(0.55, 1.02)
    axB.tick_params(labelsize=6.5)
    axB.set_title("(B) Selective answering — abstaining on\nlow-margin probes raises accuracy",
                  fontsize=8.3, pad=5)

    # (C) exemplar profiles
    for col, (k, tag, color) in enumerate([
        (k_hi, "high margin, correct", "#1a9850"),
        (k_lo, "low margin, miss", "#b2182b"),
    ]):
        ax = fig.add_subplot(gs[0, 2 + col])
        z = Z[k] - np.nanmean(Z[k])
        x = np.arange(len(dr.syms))
        bars = ax.bar(x, z, 0.7, color="0.75")
        ti = dr.sidx[intended[k]]
        bars[ti].set_color(color)
        if top1[k] != ti:
            bars[top1[k]].set_color("#fdae61")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([short_sym(s) for s in dr.syms], fontsize=5.5,
                           rotation=90)
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel("centered ws$_z$", fontsize=7.5)
        phr = REF_PROBES[k][2]
        ax.set_title(f"(C{col+1}) {tag}  (margin {margins[k]:.2f})\n"
                     f"“{phr[:46]}…”\nintended {short_sym(intended[k])}"
                     + ("" if correct[k] else f" — read as {short_sym(dr.syms[top1[k]])}"),
                     fontsize=7.2, pad=4)

    fig.suptitle("The readout knows when it doesn't know — the Δ profile margin as "
                 "an uncertainty signal", fontsize=12, y=0.97)
    save_fig(fig, "fig_delta_ambiguity")
    plt.close(fig)


if __name__ == "__main__":
    main()
