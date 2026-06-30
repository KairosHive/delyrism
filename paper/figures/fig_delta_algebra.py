"""Figure — CONTEXT ALGEBRA: the Δ readout under subtraction, negation, intensity.

Extends the additivity result (fig_delta_probe_atlas B/C) into a small algebra
of contexts, all through the same calibrated readout:

(A) SUBTRACTION.  If Δ is additive, then res Δ_AB − res Δ_A ≈ res Δ_B.  Tested
    on the 10 compound pairs vs mismatched (shuffled) controls.
(B) NEGATION.  Affirmed vs negated versions of the same event ("rain swept
    across the camp" / "no rain came and the ground stayed dry").  Honest
    diagnostic: sentence embedders are famously weak at negation — does the
    readout inherit that?  We report the target-symbol response affirmed vs
    negated and the residual-fingerprint cosine between the two.
(C) INTENSITY INVARIANCE.  Four graded versions of the same event (faint →
    overwhelming).  Honest finding: described intensity is NOT reliably encoded
    in Δ magnitude (mean dose–response Spearman ρ ≈ 0.1) — instead the readout
    is intensity-INVARIANT: all four grades of a theme produce nearly the same
    rewiring direction (high within-theme residual cosine), while different
    themes are well separated.  Δ reads what a context is ABOUT, not how big
    the described event is.

Calibration from the 30-probe reference battery (fig_delta_probe_atlas.PROBES).
Output: paper/v2/figures/fig_delta_algebra.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from _setup import build_space, save_fig, set_paper_style
from _delta_common import DeltaReadout, cos
from fig_delta_probe_atlas import PROBES as REF_PROBES, COMPOSE, short_sym

# (theme, target_symbol, affirmed, negated) — targets fixed BY DESIGN (we wrote
# the events), not picked post hoc.
NEGATION = [
    ("clouds",    "CLOUDS",      "a grey ceiling drew over the camp and dimmed the light",
                                 "nothing at all covered the sky and the light stayed sharp"),
    ("horses",    "HORSE TRACK", "a band of horses galloped past the camp",
                                 "not a single horse was left anywhere on the plain"),
    ("lightning", "LIGHTNING",   "lightning struck again and again across the ridge",
                                 "no lightning came and the sky stayed dark and still"),
    ("stars",     "STAR",        "the clear night was crowded with bright sharp lights overhead",
                                 "no lights at all could be seen in the night"),
    ("home",      "HOUSE",       "the lodge was full of family, food, and warmth",
                                 "no one was left in the lodge and the fire was out"),
    ("bundle",    "BAG",         "she carried a full bundle of remedies on her back",
                                 "she carried nothing at all on the road"),
]

# (theme, target_symbol, [4 graded phrases, faint → overwhelming])
INTENSITY = [
    ("thunder", "THUNDER", [
        "a faint rumble murmured far off at the edge of hearing",
        "a low rumble rolled in from the distance",
        "loud rolling crashes broke directly overhead",
        "a deafening roar shook the ground and rattled every lodge pole",
    ]),
    ("stars", "STAR", [
        "a single dim point of light showed in the dark",
        "a few quiet lights appeared as the sky darkened",
        "the clear night was scattered with bright points of light",
        "the whole sky blazed, crowded edge to edge with brilliant lights",
    ]),
    ("horses", "HORSE TRACK", [
        "one tired pony walked slowly into camp",
        "a few mounts trotted past the lodges",
        "a band of riders ran their animals hard across the flat",
        "a vast herd stampeded past in a storm of dust and drumming hooves",
    ]),
    ("rain", "CLOUDS", [
        "a thin mist softened the morning light",
        "low grey banks drifted in and the light dimmed",
        "heavy banks rolled overhead and the first drops fell",
        "black masses swallowed the sky and rain fell in blinding sheets",
    ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)
    dr = DeltaReadout(space)
    dr.fit_reference([p[2] for p in REF_PROBES])

    # ── (A) subtraction ───────────────────────────────────────────────────────
    rA = [dr.res(dr.fp(c[0])) for c in COMPOSE]
    rB = [dr.res(dr.fp(c[1])) for c in COMPOSE]
    rAB = [dr.res(dr.fp(c[2])) for c in COMPOSE]
    nC = len(COMPOSE)
    # both directions: (AB − A) vs B  and  (AB − B) vs A
    sub_match = ([cos(rAB[i] - rA[i], rB[i]) for i in range(nC)]
                 + [cos(rAB[i] - rB[i], rA[i]) for i in range(nC)])
    sub_shuf = ([cos(rAB[i] - rA[i], rB[(i + 1) % nC]) for i in range(nC)]
                + [cos(rAB[i] - rB[i], rA[(i + 1) % nC]) for i in range(nC)])
    print(f"[algebra] subtraction matched μ={np.mean(sub_match):.2f} "
          f"shuffled μ={np.mean(sub_shuf):.2f}")

    # Selectivity: ws_z centered per sentence (removes the global-strength
    # offset so the value reads "how much does THIS symbol stand out").
    def zc(fp_):
        z = dr.ws_z(fp_)
        return z - np.nanmean(z)

    # ── (B) negation ──────────────────────────────────────────────────────────
    neg_rows = []
    for theme, tsym, aff, neg in NEGATION:
        fa, fn = dr.fp(aff), dr.fp(neg)
        za, zn = zc(fa), zc(fn)
        t = dr.sidx[tsym]
        c = cos(dr.res(fa), dr.res(fn))
        neg_rows.append((theme, tsym, za[t], zn[t], c))
        print(f"[algebra] negation {theme:<10} target={tsym:<12} "
              f"aff z={za[t]:+.2f}  neg z={zn[t]:+.2f}  cos(res)={c:.2f}")

    # ── (C) intensity invariance ──────────────────────────────────────────────
    # Dose–response is a NULL (Δ tracks topical relevance, not described
    # magnitude) — what holds instead is invariance: grades of one theme give
    # nearly the same rewiring direction; different themes do not.
    theme_res, rhos = [], []
    for theme, tsym, grades in INTENSITY:
        t = dr.sidx[tsym]
        fps = [dr.fp(g) for g in grades]
        ws = np.array([dr.ws(f)[t] for f in fps])
        rhos.append(spearmanr(np.arange(len(grades)), ws).statistic)
        theme_res.append([dr.res(f) for f in fps])
    within, between = [], []
    nT, nG = len(INTENSITY), 4
    for a in range(nT):
        for g1 in range(nG):
            for g2 in range(g1 + 1, nG):
                within.append(cos(theme_res[a][g1], theme_res[a][g2]))
        for b in range(a + 1, nT):
            for g1 in range(nG):
                for g2 in range(nG):
                    between.append(cos(theme_res[a][g1], theme_res[b][g2]))
    print(f"[algebra] intensity invariance: within-theme μ={np.mean(within):.2f}  "
          f"between-theme μ={np.mean(between):.2f}  "
          f"(dose–response mean ρ={np.mean(rhos):.2f} — null)")

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.1, 1.05], wspace=0.34,
                          top=0.82, bottom=0.16, left=0.06, right=0.985)

    # (A) subtraction strip
    axA = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(0)
    nS = len(sub_match)
    jx = rng.uniform(-0.05, 0.05, nS)
    for i in range(nS):
        axA.plot([0, 1], [sub_match[i], sub_shuf[i]], color="0.85", lw=0.7, zorder=1)
    axA.scatter(np.zeros(nS) + jx, sub_match, s=30, color="#b2182b",
                edgecolors="k", linewidths=0.4, zorder=3)
    axA.scatter(np.ones(nS) + jx, sub_shuf, s=30, color="#2166ac",
                edgecolors="k", linewidths=0.4, zorder=3)
    axA.hlines(np.mean(sub_match), -0.16, 0.16, color="#b2182b", lw=2)
    axA.hlines(np.mean(sub_shuf), 0.84, 1.16, color="#2166ac", lw=2)
    axA.axhline(0, color="0.6", lw=0.6, ls=":")
    axA.set_xticks([0, 1])
    axA.set_xticklabels([f"matched\nμ={np.mean(sub_match):.2f}",
                         f"shuffled\nμ={np.mean(sub_shuf):.2f}"], fontsize=7.5)
    axA.set_xlim(-0.4, 1.4)
    axA.set_ylabel("cos(res Δ$_{AB}$ − res Δ$_A$,  res Δ$_B$)", fontsize=8)
    axA.set_title("(A) Subtraction — removing one part\nrecovers the other  (10 pairs × 2 directions)",
                  fontsize=8.5, pad=5)

    # (B) negation paired dots
    axB = fig.add_subplot(gs[0, 1])
    xs = np.arange(len(neg_rows))
    allv = [v for _, _, za, zn, _ in neg_rows for v in (za, zn)]
    axB.set_ylim(min(allv) - 0.30, max(allv) + 0.18)
    for i, (theme, tsym, za, zn, c) in enumerate(neg_rows):
        axB.plot([i, i], [za, zn], color="0.7", lw=1.0, zorder=1)
        axB.scatter([i], [za], s=42, color="#b2182b", edgecolors="k",
                    linewidths=0.4, zorder=3)
        axB.scatter([i], [zn], s=42, color="#2166ac", edgecolors="k",
                    linewidths=0.4, zorder=3)
        axB.annotate(f"cos={c:.2f}", (i, min(za, zn) - 0.12), fontsize=5.8,
                     ha="center", va="top", color="0.35")
    axB.axhline(0, color="0.6", lw=0.6, ls=":")
    axB.scatter([], [], s=38, color="#b2182b", edgecolors="k", linewidths=0.4,
                label="affirmed")
    axB.scatter([], [], s=38, color="#2166ac", edgecolors="k", linewidths=0.4,
                label="negated")
    axB.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    axB.set_xticks(xs)
    axB.set_xticklabels([f"{t}\n→{short_sym(s)}" for t, s, *_ in neg_rows],
                        fontsize=6.5)
    axB.set_ylabel("target-symbol selectivity (centered ws$_z$)", fontsize=8)
    axB.set_title("(B) Negation — does \"no rain came\" undo\nthe rewiring of \"rain swept in\"?",
                  fontsize=8.5, pad=5)

    # (C) intensity invariance strip — within-theme vs between-theme residual cos
    axC = fig.add_subplot(gs[0, 2])
    jw = rng.uniform(-0.06, 0.06, len(within))
    jb = rng.uniform(-0.06, 0.06, len(between))
    axC.scatter(np.zeros(len(within)) + jw, within, s=18, color="#b2182b",
                edgecolors="k", linewidths=0.3, alpha=0.8, zorder=3)
    axC.scatter(np.ones(len(between)) + jb, between, s=18, color="#2166ac",
                edgecolors="k", linewidths=0.3, alpha=0.6, zorder=3)
    axC.hlines(np.mean(within), -0.18, 0.18, color="#b2182b", lw=2)
    axC.hlines(np.mean(between), 0.82, 1.18, color="#2166ac", lw=2)
    axC.axhline(0, color="0.6", lw=0.6, ls=":")
    axC.set_xticks([0, 1])
    axC.set_xticklabels([f"same theme,\ndifferent intensity\nμ={np.mean(within):.2f}",
                         f"different\nthemes\nμ={np.mean(between):.2f}"], fontsize=7)
    axC.set_xlim(-0.45, 1.45)
    axC.set_ylabel("cos(res Δ, res Δ)", fontsize=8)
    axC.set_title(f"(C) Intensity invariance — faint or overwhelming,\n"
                  f"the same event rewires the same way  "
                  f"(dose–response ρ = {np.mean(rhos):.2f}, null)",
                  fontsize=8.5, pad=5)

    fig.suptitle("Context algebra — the calibrated Δ readout under subtraction, negation, and intensity",
                 fontsize=12, y=0.96)
    save_fig(fig, "fig_delta_algebra")
    plt.close(fig)


if __name__ == "__main__":
    main()
