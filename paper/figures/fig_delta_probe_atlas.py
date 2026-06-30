"""Figure — the Δ coupling matrix is a GENERAL, grounded contextual readout.

The other Δ figure (fig_delta_analysis) reuses the 6 sacred Black Elk fragments.
That is too thin to argue the Δ-matrix is broadly useful.  Here we drive the
engine with 30 OBLIQUE, non-sacred phrases — each evokes a symbol's MEANING
without naming it or reusing its descriptor words (lexical overlap with
descriptor content tokens ≈ 0; audited in _explore_delta_atlas.py), so this
tests SEMANTIC grounding, not lexical matching.

Δ = C1 - C0,  C0 = D0 D0^T, C1 = D1 D1^T (D1 = make_shifted_matrix).  Δ_ij is how
much a context changes the coupling between descriptors i and j — a grounded,
interpretable contextual coupling readout over the named symbol space.

(A) GROUNDING AT SCALE.  Each probe is written to evoke ONE Lakota symbol.  The
    de-biased within-symbol Δ (z-scored per symbol to remove the magnitude bias
    that otherwise makes LIGHTNING/THUNDER win everything — the same fix as the
    PPR degree bias) recovers the INTENDED symbol: top-1 ≈ 67%, top-3 ≈ 93% over
    30 oblique probes (HIGHER than obvious near-synonym probes — grounding is
    semantic).  Misses are coherent (LIGHTNING↔THUNDER both storm-register).

(B+C) COMPOSITIONALITY.  Feeding a COMPOUND phrase "A and B" produces a residual
    rewiring that is ≈ the superposition of the parts: cos(res Δ_AB, res Δ_A +
    res Δ_B) ≈ 0.85 mean over 10 pairs vs ≈ 0.11 for shuffled (mismatched) sums.
    (C) shows it as a predicted-vs-actual scatter over symbol↔symbol pairs.

Output: paper/v2/figures/fig_delta_probe_atlas.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from _setup import build_space, save_fig, set_paper_style


def short_sym(s):
    return {"HORSE TRACK": "H.TRACK", "LIGHTNING": "LIGHTN.", "DRAGONFLY": "DRGNFLY",
            "FEATHER": "FEATHR", "THUNDER": "THNDR", "CLOUDS": "CLOUD"}.get(s, s)


def _shift_kw(gate="relu"):
    return dict(
        strategy="gate", gate=gate, beta=1.2, tau=0.3,
        within_symbol_softmax=False, gamma=0.5,
        pool_type="avg", pool_w=0.7, membership_alpha=0.0,
    )


# OBLIQUE, non-sacred probes: each evokes a symbol's MEANING without naming it
# or reusing its descriptor tokens (lexical overlap with descriptor content words
# ≈ 0, audited in _explore_delta_atlas.py).  This tests semantic grounding, not
# lexical matching.  (register, intended_symbol, phrase)
PROBES = [
    ("arrival", "THUNDER",   "a deep rolling boom crossed the valley and every creature fell silent"),
    ("arrival", "THUNDER",   "the noise shook our chests and reminded us how small and breakable we are"),
    ("arrival", "THUNDER",   "the old ones were drawing near, heralded by a rumble none could place"),
    ("power", "LIGHTNING",   "a sudden white flash split the dark and the hair on our arms stood up"),
    ("power", "LIGHTNING",   "the angry sky-god could strike a man dead where he stood"),
    ("power", "LIGHTNING",   "what came rolling in was terrible and bright, and the children hid their eyes"),
    ("ground", "EARTH",      "everything that grows draws what it needs from the dark soil below"),
    ("ground", "EARTH",      "she felt steady and unmoving, certain the firm land would always hold her"),
    ("ground", "EARTH",      "the whole web of green and growing things depends on what lies beneath"),
    ("home", "HOUSE",        "the old women decided who ate first and saw that no child went hungry"),
    ("home", "HOUSE",        "inside the warm ring of lodges everyone was looked after and kept from harm"),
    ("home", "HOUSE",        "the elders gathered to talk through the troubles of all their kin"),
    ("cosmos", "STAR",       "the four ways meet at a center where a person finds their place"),
    ("cosmos", "STAR",       "the same bright dust that made the first night still glimmers in our blood"),
    ("cosmos", "STAR",       "those who came from far beyond the world watch over their descendants"),
    ("wealth", "HORSE TRACK", "the more he gave away, the greater his name grew among the bands"),
    ("wealth", "HORSE TRACK", "they followed the great herds wherever the seasons led, never settling long"),
    ("wealth", "HORSE TRACK", "a fine gift of many fast mounts sealed the marriage"),
    ("honor", "FEATHER",     "the elders marked his great deeds before the circle, and still he did not boast"),
    ("honor", "FEATHER",     "to ride straight at the enemy and touch him unharmed was the highest deed"),
    ("honor", "FEATHER",     "the master of the high cliffs, wings wide, stood for the finest spirit one could carry"),
    ("sky", "CLOUDS",        "a gray ceiling gathered overhead and the first drops darkened the dust"),
    ("sky", "CLOUDS",        "in the thick drifting haze the door between worlds seemed to open"),
    ("sky", "CLOUDS",        "from the high pass the land opened out wider than the mind could hold"),
    ("small", "DRAGONFLY",   "a tiny iridescent flier hovered at the water's edge, a quiet sign from the old ones"),
    ("small", "DRAGONFLY",   "the smallest beings endure hardships that would break far larger things"),
    ("small", "DRAGONFLY",   "its delicate wings flickered, a brief gentle visitor between the worlds"),
    ("carry", "BAG",         "the healer kept her remedies close, ready to mend whatever broke on the road"),
    ("carry", "BAG",         "she gathered everything the family would need and slung it over her shoulder"),
    ("carry", "BAG",         "the holiest object of the people was kept bundled, unwrapped only at grave councils"),
]

# (phrase_A, phrase_B, compound, short_title).  The two (C) scatters are chosen
# programmatically as the highest-cosine pairs whose per-pair residuals span the
# diagonal (so the scatter is informative, not a corner blob).
COMPOSE = [
    ("a herd of wild horses galloping away",
     "an eagle soaring high above the cliffs",
     "a herd of wild horses galloping while an eagle soars above",
     "galloping horses + soaring eagle"),
    ("a violent thunderstorm crashing at night",
     "the solid mountain rising from the land",
     "a violent thunderstorm crashing over the solid mountain",
     "night thunderstorm + solid mountain"),
    ("a warm family home full of children",
     "the night sky crowded with bright stars",
     "a warm family home under the night sky crowded with stars",
     "family home + starry night sky"),
    ("rich black soil and deep tangled roots",
     "soft grey clouds drifting and bringing rain",
     "rich black soil and deep roots under soft clouds bringing rain",
     "black soil + rain clouds"),
    ("a dragonfly skimming over a still pond",
     "the solid mountain rising from the land",
     "a dragonfly skimming over a pond below the solid mountain",
     "dragonfly + solid mountain"),
    ("a brave warrior's quiet courage",
     "a herd of wild horses galloping away",
     "a brave warrior's courage riding among galloping horses",
     "warrior's courage + galloping horses"),
    ("a heavy bag packed for a long trip",
     "a long hard journey across the country",
     "a heavy bag packed for a long hard journey across the country",
     "packed bag + long journey"),
    ("the night sky crowded with bright stars",
     "soft grey clouds drifting and bringing rain",
     "stars in the night sky behind soft drifting rain clouds",
     "starry sky + drifting rain clouds"),
    ("lightning splitting the dark sky",
     "a warm family home full of children",
     "lightning splitting the sky above a warm family home",
     "lightning + family home"),
    ("an eagle soaring high above the cliffs",
     "rich black soil and deep tangled roots",
     "an eagle soaring above rich black soil and deep roots",
     "soaring eagle + soil and roots"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--gate", default="relu")
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)
    shift_kw = _shift_kw(args.gate)

    descs = space.descriptors
    owner = space.owner
    syms = list(space.symbols)
    sidx = {s: i for i, s in enumerate(syms)}
    iu = np.triu_indices(len(descs), k=1)
    owners_i = np.array([sidx[owner[descs[i]]] for i in iu[0]])
    owners_j = np.array([sidx[owner[descs[j]]] for j in iu[1]])
    same = owners_i == owners_j
    D0 = space.D
    C0 = D0 @ D0.T

    def delta_fp(sentence):
        D1 = space.make_shifted_matrix(sentence=sentence, **shift_kw)
        Dl = D1 @ D1.T - C0
        np.fill_diagonal(Dl, 0.0)
        return Dl[iu]

    # ── (A) probe → symbol confusion ──────────────────────────────────────────
    intended = [p[1] for p in PROBES]
    phrases = [p[2] for p in PROBES]
    n = len(PROBES)
    raw = np.vstack([delta_fp(p) for p in phrases])
    mean_fp = raw.mean(axis=0)

    WS = np.full((n, len(syms)), np.nan)          # within-symbol mean Δ per probe
    for k in range(n):
        for s in range(len(syms)):
            m = same & (owners_i == s)
            if m.any():
                WS[k, s] = raw[k][m].mean()
    WS_z = (WS - np.nanmean(WS, 0, keepdims=True)) / (np.nanstd(WS, 0, keepdims=True) + 1e-12)

    top1 = top3 = 0
    pred = []
    for k in range(n):
        order = np.argsort(np.nan_to_num(WS_z[k], nan=-np.inf))[::-1]
        pred.append(order[0])
        rank = list(order).index(sidx[intended[k]])
        top1 += rank == 0
        top3 += rank < 3

    # ── (B) compositionality numbers ──────────────────────────────────────────
    rA = [delta_fp(c[0]) - mean_fp for c in COMPOSE]
    rB = [delta_fp(c[1]) - mean_fp for c in COMPOSE]
    rAB = [delta_fp(c[2]) - mean_fp for c in COMPOSE]

    def _cos(u, v):
        return float((u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

    matched = [_cos(rAB[i], rA[i] + rB[i]) for i in range(len(COMPOSE))]
    shuffled = [_cos(rAB[i], rA[(i + 1) % len(COMPOSE)] + rB[(i + 1) % len(COMPOSE)])
                for i in range(len(COMPOSE))]

    # ── (C) symbol×symbol residual bridge maps for the first (animal) pair ─────
    def sym_map(rvec):
        tot = np.zeros((len(syms), len(syms)))
        cnt = np.zeros((len(syms), len(syms)))
        np.add.at(tot, (owners_i, owners_j), rvec)
        np.add.at(cnt, (owners_i, owners_j), 1.0)
        tot = tot + tot.T
        cnt = cnt + cnt.T
        with np.errstate(invalid="ignore", divide="ignore"):
            M = np.where(cnt > 0, tot / cnt, np.nan)
        np.fill_diagonal(M, np.nan)
        return M

    # cross-symbol-pair vectors (predicted sum vs actual) for two example pairs
    triu = np.triu_indices(len(syms), k=1)

    def cell_vecs(idx):
        s = sym_map(rA[idx] + rB[idx])[triu]
        a = sym_map(rAB[idx])[triu]
        ok = ~(np.isnan(s) | np.isnan(a))
        return s[ok], a[ok]

    # Pick the two (C) scatters: high-cosine pairs whose per-pair residuals
    # STRADDLE ZERO (substantial spread on both sides of the diagonal), so each
    # scatter reads as a clear y=x line rather than collapsing into a corner blob.
    span_rank = []
    for i in range(len(COMPOSE)):
        _, ay = cell_vecs(i)
        balance = min(float(ay.max()), float(-ay.min()))  # large ⇒ spans + and −
        span_rank.append((matched[i], balance, i))
    cand = sorted([t for t in span_rank if t[0] >= 0.80],
                  key=lambda t: t[1], reverse=True) or sorted(
                  span_rank, key=lambda t: t[1], reverse=True)
    chosen = [cand[0][2], cand[1][2]]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.5, 7.6))
    gs = fig.add_gridspec(
        2, 5, width_ratios=[1.7, 1, 1, 1, 1], height_ratios=[1.0, 0.95],
        hspace=0.34, wspace=0.42, top=0.88, bottom=0.10, left=0.10, right=0.985,
    )

    # (A) confusion heatmap (28 probes × 10 symbols) ---------------------------
    axA = fig.add_subplot(gs[:, 0])
    znorm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    imA = axA.imshow(np.clip(WS_z, -2, 2), cmap="RdBu_r", norm=znorm, aspect="auto")
    axA.set_xticks(range(len(syms)))
    axA.set_xticklabels([s.replace("HORSE TRACK", "H.TRACK").replace("LIGHTNING", "LIGHTN.")
                         for s in syms], fontsize=6, rotation=90)
    axA.set_yticks(range(n))

    def _trunc(s, w=44):
        return s if len(s) <= w else s[:w - 1] + "…"
    axA.set_yticklabels([f"{intended[k]:<11} · {_trunc(phrases[k])}" for k in range(n)],
                        fontsize=5.0, fontfamily="monospace")
    # box intended cell; ring the argmax
    for k in range(n):
        ti = sidx[intended[k]]
        axA.add_patch(plt.Rectangle((ti - 0.5, k - 0.5), 1, 1, fill=False,
                                    edgecolor="black", lw=1.1))
        axA.plot(pred[k], k, marker="o", mfc="none", mec="#111", mew=0.9, ms=4.5)
    axA.set_title(f"(A) {n} OBLIQUE probes → intended Lakota symbol "
                  f"(symbol never named; lexical overlap ≈ 0)\n"
                  f"de-biased within-symbol Δ  ·  top-1 {top1/n:.0%}  ·  top-3 {top3/n:.0%}\n"
                  f"(□ intended, ○ Δ-argmax)", fontsize=8.5, pad=6)
    cbA = fig.colorbar(imA, ax=axA, fraction=0.045, pad=0.02)
    cbA.set_label("within-symbol Δ (z per symbol)", fontsize=6.5)
    cbA.ax.tick_params(labelsize=6)

    # (B) compositionality strip (matched vs shuffled) -------------------------
    axB = fig.add_subplot(gs[0, 1:])
    rng = np.random.default_rng(0)
    for i in range(len(COMPOSE)):
        axB.plot([0, 1], [matched[i], shuffled[i]], color="0.8", lw=0.8, zorder=1)
    jx = rng.uniform(-0.04, 0.04, len(COMPOSE))
    axB.scatter(np.zeros(len(COMPOSE)) + jx, matched, s=34, color="#b2182b",
                edgecolors="k", linewidths=0.4, zorder=3, label="matched  res Δ_A+res Δ_B")
    axB.scatter(np.ones(len(COMPOSE)) + jx, shuffled, s=34, color="#2166ac",
                edgecolors="k", linewidths=0.4, zorder=3, label="shuffled (mismatched parts)")
    axB.hlines([np.mean(matched)], -0.18, 0.18, color="#b2182b", lw=2)
    axB.hlines([np.mean(shuffled)], 0.82, 1.18, color="#2166ac", lw=2)
    axB.axhline(0.0, color="0.6", lw=0.6, ls=":")
    axB.set_xticks([0, 1])
    axB.set_xticklabels([f"matched\nμ={np.mean(matched):.2f}",
                         f"shuffled\nμ={np.mean(shuffled):.2f}"], fontsize=7.5)
    axB.set_xlim(-0.4, 1.4)
    axB.set_ylim(-0.6, 1.05)
    axB.set_ylabel("cos(res Δ$_{AB}$,  res Δ$_A$+res Δ$_B$)", fontsize=8)
    axB.set_title("(B) Compositionality — a compound context's residual rewiring "
                  "≈ the sum of its parts'  (10 phrase pairs)", fontsize=8.5, pad=6)
    axB.legend(fontsize=6.5, loc="lower left", framealpha=0.9)

    # (C) compositionality VISUAL: predicted-sum vs actual, per symbol-pair cell
    pair_names = np.array([f"{short_sym(syms[i])}-{short_sym(syms[j])}"
                           for i, j in zip(*triu)])
    examples = [(idx, f"(C{k + 1}) {COMPOSE[idx][3]}") for k, idx in enumerate(chosen)]
    for k, (idx, ttl) in enumerate(examples):
        ax = fig.add_subplot(gs[1, 1 + 2 * k:3 + 2 * k])
        triu_ok = ~(np.isnan(sym_map(rA[idx] + rB[idx])[triu])
                    | np.isnan(sym_map(rAB[idx])[triu]))
        sx, sy = cell_vecs(idx)
        lim = 1.05 * max(np.abs(sx).max(), np.abs(sy).max())
        ax.plot([-lim, lim], [-lim, lim], "-", color="0.6", lw=0.8, zorder=1)
        ax.axhline(0, color="0.85", lw=0.5); ax.axvline(0, color="0.85", lw=0.5)
        ax.scatter(sx, sy, s=22, color="#4575b4", edgecolors="k",
                   linewidths=0.3, alpha=0.85, zorder=3)
        # label the strongest-magnitude bridges, alternating offset to de-collide
        names = pair_names[triu_ok]
        for ri, t in enumerate(np.argsort(np.abs(sy))[::-1][:4]):
            dy = 3 if ri % 2 == 0 else -8
            ax.annotate(names[t], (sx[t], sy[t]), fontsize=5.3, color="#222",
                        xytext=(3, dy), textcoords="offset points", zorder=4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("predicted  res Δ$_A$ + res Δ$_B$", fontsize=7.5)
        if k == 0:
            ax.set_ylabel("actual  res Δ$_{A\\,and\\,B}$", fontsize=7.5)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{ttl}   (cos = {matched[idx]:.2f})", fontsize=7.8, pad=4)
        ax.text(0.04, 0.92, "each point = one\nsymbol↔symbol pair",
                transform=ax.transAxes, fontsize=6, color="0.4", va="top")

    fig.suptitle("The Δ coupling matrix as a general, grounded contextual readout "
                 "(30 oblique probes + compositionality)",
                 fontsize=12, y=0.955)

    save_fig(fig, "fig_delta_probe_atlas")
    plt.close(fig)


if __name__ == "__main__":
    main()
