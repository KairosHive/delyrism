"""Figure — Context-induced relational rewiring via the Δ coupling matrix.

The Δ-matrix is the SECOND-ORDER object the engine actually conditions on:

    Δ = C1 - C0,   C0 = D0 D0^T  (baseline descriptor–descriptor coupling),
                   C1 = D1 D1^T  (context-shifted coupling; make_shifted_matrix).

Δ_ij = how much a context changes the coupling between descriptors i and j.
Unlike attention (first-order "which descriptor is relevant"), Δ is relational
and can express a coupling *between two different symbols' descriptors* — a
context-induced bridge that first-order relevance structurally cannot represent.

Three findings, each with a control (see _explore_delta_analysis.py):

(A) DISCRIMINABILITY / VALIDITY.  ~99% of every context's Δ is a context-generic
    mode; the context-SPECIFIC part is the residual after removing the
    across-context mean.  Direction-normalized residual fingerprints are
    discriminative: identical text (C3≡C_B) → cosine 1.00 (ceiling), opposing
    registers → strongly negative, a random context → ~0 (floor).

(B) INTERPRETABILITY + VALUE-ADD.  Aggregating the residual to a symbol×symbol
    matrix shows which SYMBOL PAIRS each context bridges (red) or relatively
    decouples (blue).  The strongest bridge per panel (boxed) reads straight off
    the Black Elk passage:
      C1 (sacred hoop)            EARTH ↔ CLOUDS        (mountain ↔ sky/cosmos;
                                  also EARTH↔HORSE TRACK, BAG↔EARTH)
      C2 (thunder voice & horse)  HORSE TRACK is the hub — bridges to THUNDER
                                  (dawn horse ↔ the Voice) and to FEATHER
      C3 (the dream's end)        HOUSE is the hub — bridges to EARTH, then to
                                  LIGHTNING/THUNDER/STAR (home ↔ the storm)

Note on the gate: on these (anisotropic) qwen3 embeddings every descriptor↔context
cosine is ≥ 0, so gate="relu" and gate="cos" produce *identical* Δ (verified:
‖Δ_relu − Δ_cos‖ = 0).  Raw Δ is therefore strengthening-only; the
"weakening" (blue) below is relative to the cross-context baseline.

Output: paper/v2/figures/fig_delta_analysis.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from _setup import CONTEXTS, CONTEXT_LABELS, build_space, save_fig, set_paper_style


def _shift_kw(gate="relu"):
    return dict(
        strategy="gate", gate=gate, beta=1.2, tau=0.3,
        within_symbol_softmax=False, gamma=0.5,
        pool_type="avg", pool_w=0.7, membership_alpha=0.0,
    )


CTX_IDS = ["C1", "C2", "C3", "C_A", "C_B", "C_scene"]
BRIDGE_CTX = ["C1", "C2", "C3"]   # contexts shown as symbol×symbol bridge maps


def _delta(space, sentence, shift_kw):
    D0 = space.D
    D1 = space.make_shifted_matrix(sentence=sentence, **shift_kw)
    Delta = D1 @ D1.T - D0 @ D0.T
    np.fill_diagonal(Delta, 0.0)
    return Delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--gate", default="relu", help="relu | cos (identical here)")
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

    # ── Δ + residual fingerprints ─────────────────────────────────────────────
    raw = {c: _delta(space, CONTEXTS[c], shift_kw)[iu] for c in CTX_IDS}
    mean_fp = np.mean(np.vstack([raw[c] for c in CTX_IDS]), axis=0)
    res = {c: raw[c] - mean_fp for c in CTX_IDS}
    resn = {c: res[c] / (np.linalg.norm(res[c]) + 1e-12) for c in CTX_IDS}
    shared_frac = np.mean([np.linalg.norm(mean_fp) / (np.linalg.norm(raw[c]) + 1e-12)
                           for c in CTX_IDS])

    S = np.array([[float(resn[a] @ resn[b]) for b in CTX_IDS] for a in CTX_IDS])

    # ── symbol×symbol residual-bridge matrices (mean residual Δ per sym-pair) ──
    # Each context still carries an overall offset (some passages rewire MORE in
    # total than average), which would paint a whole panel one colour and hide
    # the pattern.  We mean-centre each panel over its cross-symbol cells, so the
    # colour shows pairs bridged MORE / LESS than that context's typical pair —
    # i.e. the *distinctive* bridges.
    def sym_matrix(rvec):
        tot = np.zeros((len(syms), len(syms)))
        cnt = np.zeros((len(syms), len(syms)))
        np.add.at(tot, (owners_i, owners_j), rvec)
        np.add.at(cnt, (owners_i, owners_j), 1.0)
        tot = tot + tot.T
        cnt = cnt + cnt.T
        with np.errstate(invalid="ignore", divide="ignore"):
            M = np.where(cnt > 0, tot / cnt, np.nan)
        off = ~np.eye(len(syms), dtype=bool)
        M[off] -= np.nanmean(M[off])     # centre over cross-symbol cells
        np.fill_diagonal(M, np.nan)      # focus on cross-symbol bridges
        return M

    bridges = {c: sym_matrix(res[c]) for c in BRIDGE_CTX}
    bmax = max(np.nanmax(np.abs(b)) for b in bridges.values())

    # cross-symbol fraction among top-|Δ| residual edges + base rate
    base_cross = float((owners_i != owners_j).mean())

    # ── Layout ────────────────────────────────────────────────────────────────
    # Single row: (A) discriminability heatmap on the left, then the three
    # symbol×symbol bridge maps.  width_ratios give (A) a touch more room.
    fig = plt.figure(figsize=(15.0, 4.6))
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1.25, 1, 1, 1],
        wspace=0.30, top=0.80, bottom=0.30, left=0.055, right=0.99,
    )

    # (A) discriminability heatmap ---------------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    imA = axA.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
    axA.set_xticks(range(len(CTX_IDS))); axA.set_yticks(range(len(CTX_IDS)))
    axA.set_xticklabels(CTX_IDS, fontsize=7, rotation=45, ha="right")
    axA.set_yticklabels(CTX_IDS, fontsize=7)
    for i in range(len(CTX_IDS)):
        for j in range(len(CTX_IDS)):
            axA.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                     fontsize=5.6, color="black" if abs(S[i, j]) < 0.6 else "white")
    axA.set_title("(A) Context-specific Δ-fingerprint similarity\n"
                  "identical text C3≡C_B → 1.00;  opposing → negative",
                  fontsize=8.5, pad=6)
    cbA = fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)
    cbA.ax.tick_params(labelsize=6.5)

    # caption strip under (A)
    axA.text(0.0, -0.42,
             f"shared-mode ≈ {shared_frac:.0%} of raw Δ\n"
             f"random-ctx floor ≈ 0.22\n"
             f"cross-symbol base rate {base_cross:.0%}",
             transform=axA.transAxes, fontsize=6.6, color="0.35",
             va="top", linespacing=1.5)

    # (B) symbol×symbol bridge maps --------------------------------------------
    norm = TwoSlopeNorm(vmin=-bmax, vcenter=0.0, vmax=bmax)
    short = [s.replace("HORSE TRACK", "H.TRACK").replace("LIGHTNING", "LIGHTN.")
             for s in syms]
    last_im = None
    for col, c in enumerate(BRIDGE_CTX):
        ax = fig.add_subplot(gs[0, col + 1])
        last_im = ax.imshow(bridges[c], cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(len(syms))); ax.set_yticks(range(len(syms)))
        ax.set_xticklabels(short, fontsize=5.2, rotation=90)
        if col == 0:
            ax.set_yticklabels(short, fontsize=5.2)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"(B) {c} — {CONTEXT_LABELS[c]}", fontsize=8, pad=4)
        # mark the top cross-symbol bridge (ignore NaN diagonal/empty cells)
        M = bridges[c]
        ti, tj = np.unravel_index(np.nanargmax(M), M.shape)
        ax.add_patch(plt.Rectangle((tj - 0.5, ti - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=1.3))

    # (B)-row colorbar, centred under the three bridge maps
    cax = fig.add_axes([0.40, 0.13, 0.45, 0.022])
    cb = fig.colorbar(last_im, cax=cax, orientation="horizontal")
    cb.set_label("symbol↔symbol mean residual Δ coupling   "
                 "(red = context bridges the pair, blue = relatively decouples)",
                 fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    fig.suptitle("Context-induced relational rewiring of the symbol space "
                 "(Δ coupling matrix, Black Elk Speaks)",
                 fontsize=11.5, y=0.97)

    save_fig(fig, "fig_delta_analysis")
    plt.close(fig)


if __name__ == "__main__":
    main()
