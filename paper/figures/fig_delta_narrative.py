"""Figure — NARRATIVE TRAJECTORIES: a story is a path through Δ coupling space.

A 12-step generic narrative (a day on the plains: ride out → storm builds →
strikes → shelter → clearing night sky) is fed to the CALIBRATED Δ readout one
step at a time.  Plain English, written for this figure — NOT a paraphrase of
the sacred Black Elk passages.

(A) SYMBOL TIMELINE — de-biased per-symbol response ws_z over story time.  The
    "hot spot" moves through the symbol set following the arc: HORSE TRACK →
    CLOUDS → THUNDER → LIGHTNING → EARTH/CLOUDS (rain) → HOUSE → STAR.
(B) TRAJECTORY — PCA of the 12 residual fingerprints: the story is a connected
    path through coupling space; points colored by dominant symbol.
(C) BRIDGE DYNAMICS — the symbol↔symbol couplings with the largest range over
    the story, traced step by step (e.g. LIGHTNING↔THUNDER surging at the
    strike, HOUSE couplings rising at shelter).

Calibration (mean fingerprint + per-symbol stats) comes from the fixed 30-probe
oblique reference battery (fig_delta_probe_atlas.PROBES) — the narrative is
read with an instrument calibrated on OTHER contexts.

Output: paper/v2/figures/fig_delta_narrative.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from _setup import build_space, save_fig, set_paper_style
from _delta_common import DeltaReadout
from fig_delta_probe_atlas import PROBES as REF_PROBES, short_sym

# (short gloss, sentence) — a generic day-on-the-plains arc, written plain.
NARRATIVE = [
    ("ride out",    "before first light she saddled her fastest mare and rode out from the camp"),
    ("open grass",  "the herd moved easily over the open grass as the sun climbed"),
    ("grey wall",   "by midday a grey ceiling slid in from the west and the light went dim"),
    ("still air",   "the air went still and heavy, and the animals grew uneasy"),
    ("far rumble",  "a low rumble rolled across the plain from far away"),
    ("white bolt",  "a blinding white bolt split the dark and struck the ridge"),
    ("hard rain",   "rain swept down in sheets and the dry ground drank it in"),
    ("race home",   "she turned for home, racing the wall of water"),
    ("hearth",      "inside the warm lodge the family gathered close and shared the evening meal"),
    ("old tales",   "the grandmothers told stories of long ago while the children slept warm by the fire"),
    ("clearing",    "near morning the air came clean and still, and the wide sky opened"),
    ("night lights","overhead the bright ones shone steady and old, keeping watch over the camp"),
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

    glosses = [g for g, _ in NARRATIVE]
    steps = [s for _, s in NARRATIVE]
    T = len(steps)
    raws = [dr.fp(s) for s in steps]
    Z = np.vstack([dr.ws_z(r) for r in raws])          # (T, n_syms)
    # Step-centered display: remove each step's global offset (overall context
    # strength) so the heatmap shows WHICH symbols a step favors.  argmax is
    # unchanged by per-step centering.
    Zc = Z - Z.mean(axis=1, keepdims=True)
    res = np.vstack([dr.res(r) for r in raws])         # (T, n_pairs)
    doms = Z.argmax(axis=1)

    # diagnostics
    print("[narrative] top-3 symbols per step (step-centered z):")
    for t in range(T):
        order = np.argsort(Zc[t])[::-1][:3]
        tops = "  ".join(f"{dr.syms[o]}({Zc[t, o]:+.2f})" for o in order)
        print(f"  {t+1:>2} {glosses[t]:<12} -> {tops}")

    # (B) PCA of residual fingerprints
    Xc = res - res.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = U[:, :2] * S[:2]
    evr = S[:2] ** 2 / (S ** 2).sum()

    # (C) top-varying symbol bridges over the story.  Center each step across
    # all symbol pairs first — otherwise every trace just follows the story's
    # global Δ magnitude (storm in/out) and all lines co-move.
    maps = np.stack([dr.sym_map(r) for r in res])      # (T, n, n)
    triu = np.triu_indices(len(dr.syms), k=1)
    series = maps[:, triu[0], triu[1]]                 # (T, n_pairs_sym)
    series = series - np.nanmean(series, axis=1, keepdims=True)
    rng_ = np.nanmax(series, 0) - np.nanmin(series, 0)
    # Greedy diverse selection: prefer symbol-DISJOINT pairs so the traces
    # cover the whole arc rather than four variations on the storm.
    order_p = np.argsort(np.nan_to_num(rng_, nan=-np.inf))[::-1]
    top, used = [], set()
    for p in order_p:
        a, b = triu[0][p], triu[1][p]
        if a not in used and b not in used:
            top.append(p)
            used |= {a, b}
        if len(top) == 4:
            break
    for p in order_p:  # fill if fewer than 4 disjoint pairs exist
        if len(top) == 4:
            break
        if p not in top:
            top.append(p)

    colors = space.get_symbol_color_dict()
    sym_color = [colors.get(s, "#888") for s in dr.syms]

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.5, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.05], wspace=0.32,
                          top=0.84, bottom=0.20, left=0.075, right=0.985)

    # (A) timeline heatmap: symbols × steps
    axA = fig.add_subplot(gs[0, 0])
    vlim = 1.5
    imA = axA.imshow(np.clip(Zc.T, -vlim, vlim), cmap="RdBu_r",
                     norm=TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim),
                     aspect="auto")
    axA.set_yticks(range(len(dr.syms)))
    axA.set_yticklabels([short_sym(s) for s in dr.syms], fontsize=7)
    axA.set_xticks(range(T))
    axA.set_xticklabels([f"{t+1} {g}" for t, g in enumerate(glosses)],
                        fontsize=6.5, rotation=45, ha="right")
    for t in range(T):  # ring the dominant symbol per step
        axA.add_patch(plt.Rectangle((t - 0.5, doms[t] - 0.5), 1, 1, fill=False,
                                    edgecolor="black", lw=1.1))
    axA.set_title("(A) Symbol timeline — de-biased Δ response per story step\n"
                  "(step-centered ws$_z$;  □ = dominant symbol)", fontsize=8.5, pad=5)
    cbA = fig.colorbar(imA, ax=axA, fraction=0.035, pad=0.02)
    cbA.set_label("ws$_z$ (step-centered)", fontsize=7)
    cbA.ax.tick_params(labelsize=6)

    # (B) trajectory in coupling space
    axB = fig.add_subplot(gs[0, 1])
    axB.plot(P[:, 0], P[:, 1], "-", color="0.75", lw=1.0, zorder=1)
    for t in range(T):
        axB.scatter(P[t, 0], P[t, 1], s=46, color=sym_color[doms[t]],
                    edgecolors="k", linewidths=0.5, zorder=3)
        axB.annotate(str(t + 1), (P[t, 0], P[t, 1]), fontsize=6,
                     xytext=(4, 3), textcoords="offset points")
    # legend: only symbols that dominate at least one step
    seen = []
    for d in doms:
        if d not in seen:
            seen.append(d)
    for d in seen:
        axB.scatter([], [], s=40, color=sym_color[d], edgecolors="k",
                    linewidths=0.5, label=short_sym(dr.syms[d]))
    axB.legend(fontsize=6, loc="best", framealpha=0.9, handletextpad=0.2)
    axB.set_xlabel(f"PC1 ({evr[0]:.0%} var)", fontsize=7.5)
    axB.set_ylabel(f"PC2 ({evr[1]:.0%} var)", fontsize=7.5)
    axB.tick_params(labelsize=6)
    axB.set_title("(B) The story as a path through\nΔ coupling space (PCA of residuals)",
                  fontsize=8.5, pad=5)

    # (C) bridge dynamics
    axC = fig.add_subplot(gs[0, 2])
    line_colors = ["#b2182b", "#2166ac", "#1a9850", "#8c510a"]
    for k, p in enumerate(top):
        a, b = triu[0][p], triu[1][p]
        axC.plot(range(1, T + 1), series[:, p], "-o", ms=3, lw=1.2,
                 color=line_colors[k % 4],
                 label=f"{short_sym(dr.syms[a])}↔{short_sym(dr.syms[b])}")
    axC.axhline(0, color="0.6", lw=0.6, ls=":")
    axC.set_xticks(range(1, T + 1))
    axC.set_xticklabels([str(t + 1) for t in range(T)], fontsize=6)
    axC.set_xlabel("story step", fontsize=7.5)
    axC.set_ylabel("residual coupling (step-centered)", fontsize=7.5)
    axC.tick_params(labelsize=6)
    axC.legend(fontsize=6, loc="best", framealpha=0.9)
    axC.set_title("(C) Bridge dynamics — largest-range\nsymbol↔symbol couplings over the story",
                  fontsize=8.5, pad=5)

    fig.suptitle("Narrative trajectories — a story read step-by-step by the calibrated Δ readout",
                 fontsize=12, y=0.97)
    save_fig(fig, "fig_delta_narrative")
    plt.close(fig)


if __name__ == "__main__":
    main()
