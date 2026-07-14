"""Figure — INSIDE THE Δ CONTEXT KERNEL: one global mode hides the content.

The Δ fingerprint induces a kernel between contexts: cos(res Δ_i, res Δ_j) —
"do these two contexts rewire the symbol space the same way?".  Naively this
kernel looks rich, but it is an artifact: a single principal mode carries ~82%
of the residual variance and simply tracks each context's OVERALL activation
strength (r = 0.84 with a pure magnitude-product predictor).  Any descriptor
bank measures that mode the same way — a generic emotion kit's kernel agrees
with the Lakota kit's at r = 0.93, i.e. the raw kernel is kit-independent.

The fix is one standard, label-free operation — remove the top principal
component of the battery residuals ("all-but-the-top", cf. Mu & Viswanath
2018).  After it:
  • the magnitude artifact is gone (r = 0.84 → −0.13),
  • the kernel becomes content-bearing: agreement with sentence-embedding
    similarity rises 0.31 → 0.69, and hierarchical clustering yields coherent
    thematic registers readable directly off the sentences — storm/numinous
    sky, honor-and-deeds, caretaking, growth-and-soil,
  • kit-dependence appears (cross-kit agreement drops 0.93 → 0.76).

No authored labels are used anywhere in the figure: panel (B)'s rows are the
probe sentences themselves, ordered by the kernel's own clustering — the
reader judges coherence directly.  (Diagnostic, not shown: same-intent AUC
improves 0.58 → 0.74 after cleaning; still below raw embeddings at 0.81, so
the claim is "content-bearing readout of rewiring", not "better similarity".)

Output: paper/v2/figures/fig_delta_kernel.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage

from _setup import build_space, save_fig, set_paper_style
from _delta_common import DeltaReadout
from fig_delta_probe_atlas import PROBES as REF_PROBES

# Generic comparison kit (authored, non-community, plain English) — used only
# for the kit-(in)dependence numbers, not displayed.
EMOTION_KIT = {
    "AWE": ["overwhelming wonder", "a vast presence", "feeling small before something great",
            "breath taken away", "trembling amazement", "the sublime"],
    "FEAR": ["danger close at hand", "a racing heart", "the urge to flee",
             "dread of what comes", "hair standing on end", "threat in the dark"],
    "COMFORT": ["safety and warmth", "being cared for", "rest after labor",
                "a full belly", "soft light indoors", "ease among loved ones"],
    "MOTION": ["speed across open ground", "restless travel", "swift passage",
               "bodies in full stride", "momentum that cannot stop", "the rush of going"],
    "QUIET": ["stillness and hush", "calm without wind", "a held breath",
              "silence over the land", "nothing stirring", "peace at dusk"],
    "ABUNDANCE": ["wealth freely given", "more than enough", "gifts passing between hands",
                  "a rich harvest", "generosity rewarded", "plenty for all"],
}


def _kernels(space, phrases, n_remove=1):
    """Return (raw kernel, cleaned kernel, top-PC explained var)."""
    dr = DeltaReadout(space)
    R = dr.fit_reference(phrases)
    Rc = R - R.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    ev1 = float(S[0] ** 2 / (S ** 2).sum())

    def _norm_kernel(X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return Xn @ Xn.T

    K_raw = _norm_kernel(Rc)
    X = Rc - (U[:, :n_remove] * S[:n_remove]) @ Vt[:n_remove]
    K_clean = _norm_kernel(X)
    return K_raw, K_clean, ev1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    set_paper_style()
    phrases = [p[2] for p in REF_PROBES]
    n = len(phrases)
    iu = np.triu_indices(n, k=1)

    space = build_space(backend=args.backend, model=args.model)
    K_raw, K_clean, ev1 = _kernels(space, phrases)
    K_raw_e, K_clean_e, _ = _kernels(
        build_space(backend=args.backend, model=args.model,
                    descriptors=EMOTION_KIT), phrases)

    # magnitude predictor: overall activation strength ‖relu(D·v)‖
    D0 = space.D
    mag = np.array([np.linalg.norm(np.maximum(0, D0 @ space.ctx_vec(sentence=p)))
                    for p in phrases])
    magprod = np.array([(mag[i] - mag.mean()) * (mag[j] - mag.mean())
                        for i, j in zip(*iu)])
    magprod_n = magprod / np.abs(magprod).max()

    # embedding similarity (centered, like everything else)
    V = np.vstack([space.ctx_vec(sentence=p) for p in phrases])
    Vc = V - V.mean(axis=0, keepdims=True)
    Vn = Vc / (np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-12)
    K_emb = Vn @ Vn.T

    kr, kc, kb = K_raw[iu], K_clean[iu], K_emb[iu]
    r_mag_raw = float(np.corrcoef(kr, magprod)[0, 1])
    r_mag_clean = float(np.corrcoef(kc, magprod)[0, 1])
    r_emb_raw = float(np.corrcoef(kr, kb)[0, 1])
    r_emb_clean = float(np.corrcoef(kc, kb)[0, 1])
    r_kit_raw = float(np.corrcoef(kr, K_raw_e[iu])[0, 1])
    r_kit_clean = float(np.corrcoef(kc, K_clean_e[iu])[0, 1])
    print(f"[kernel] top-PC var = {ev1:.0%}")
    print(f"[kernel] corr w/ magnitude product: raw={r_mag_raw:.2f} clean={r_mag_clean:.2f}")
    print(f"[kernel] corr w/ embedding sim:     raw={r_emb_raw:.2f} clean={r_emb_clean:.2f}")
    print(f"[kernel] cross-kit agreement:       raw={r_kit_raw:.2f} clean={r_kit_clean:.2f}")

    # orders: (A) by activation magnitude, (B) by cleaned-kernel clustering
    ord_mag = np.argsort(mag)
    Zc_link = linkage(1 - kc, method="average")
    ord_clu = leaves_list(Zc_link)

    def _trunc(s, w=44):
        return s if len(s) <= w else s[: w - 1] + "…"

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15.0, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.55, 0.90],
                          height_ratios=[1, 1], wspace=0.60, hspace=0.45,
                          top=0.86, bottom=0.09, left=0.03, right=0.985)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    # (A) raw kernel, sorted by activation strength
    axA = fig.add_subplot(gs[:, 0])
    axA.imshow(K_raw[np.ix_(ord_mag, ord_mag)], cmap="RdBu_r", norm=norm)
    axA.set_xticks([]); axA.set_yticks([])
    axA.set_xlabel("probes sorted by activation strength →", fontsize=7)
    axA.set_ylabel("← probes sorted by activation strength", fontsize=7)
    axA.set_title(f"(A) Raw Δ kernel, one mode ({ev1:.0%} of variance)\n"
                  f"tracks overall activation strength (r = {r_mag_raw:.2f});\n"
                  f"kit-independent (cross-kit r = {r_kit_raw:.2f})",
                  fontsize=8.3, pad=5)

    # (B) cleaned kernel as a clustermap: the kernel's own hierarchical
    #     clustering (dendrogram, right) partitions the probes into registers
    #     (boxes), each labelled by a representative probe sentence.
    KREG = 4
    Kc_ord = K_clean[np.ix_(ord_clu, ord_clu)]
    v = float(np.percentile(np.abs(Kc_ord[np.triu_indices(n, 1)]), 92)) or 1.0
    normB = TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)

    gsB = gs[:, 1].subgridspec(1, 2, width_ratios=[1.0, 0.16], wspace=0.015)
    axB = fig.add_subplot(gsB[0, 0])
    axD = fig.add_subplot(gsB[0, 1])

    # dendrogram first: it defines the cluster colours reused for boxes + labels
    thr = Zc_link[-(KREG - 1), 2]
    dn = dendrogram(Zc_link, orientation="right", no_labels=True, ax=axD,
                    color_threshold=thr, above_threshold_color="0.6")
    axD.set_ylim(10 * n, 0)
    axD.axis("off")
    leaf_col = dn["leaves_color_list"]              # one colour per row, in ord_clu order
    bnd = [i for i in range(1, n) if leaf_col[i] != leaf_col[i - 1]]
    starts = np.r_[0, bnd].astype(int)
    ends = np.r_[bnd, n].astype(int)

    imB = axB.imshow(Kc_ord, cmap="RdBu_r", norm=normB,
                     extent=[0, n, n, 0], aspect="auto", interpolation="nearest")
    yt, yl, yc = [], [], []
    for s, e in zip(starts, ends):
        col = leaf_col[s]
        axB.add_patch(Rectangle((s, s), e - s, e - s, fill=False,
                                edgecolor=col, lw=2.4, zorder=3))
        mp = s + int(np.argmax(Kc_ord[s:e, s:e].mean(axis=1)))     # cluster medoid
        yt.append(mp + 0.5)
        yl.append(_trunc(phrases[ord_clu[mp]], 42))
        yc.append(col)
    axB.set_yticks(yt)
    axB.set_yticklabels(yl, fontsize=6.4)
    for lab, col in zip(axB.get_yticklabels(), yc):
        lab.set_color(col)
        lab.set_fontweight("bold")
    axB.tick_params(axis="y", length=0)
    axB.set_xticks([])
    axB.set_xlabel("columns = same probes, same order", fontsize=6.5)
    axB.set_title("(B) Removing that mode lets the kernel's own clustering\n"
                  "split the probes into coherent registers (boxes),\n"
                  "each labelled by a representative probe sentence",
                  fontsize=8.3, pad=5)

    cbB = fig.colorbar(imB, ax=[axB, axD], fraction=0.022, pad=0.02)
    cbB.set_label("cos(res Δ$_i$, res Δ$_j$)", fontsize=6)
    cbB.ax.tick_params(labelsize=5.5)

    # (C1) raw vs magnitude product; (C2) cleaned vs embedding similarity
    for row, (x, y, rr, xl, yl, ttl) in enumerate([
        (magprod_n, kr, r_mag_raw,
         "activation-magnitude product (scaled)", "raw Δ-kernel similarity",
         "(C1) The raw kernel measures magnitude"),
        (kb, kc, r_emb_clean,
         "sentence-embedding similarity", "cleaned Δ-kernel similarity",
         "(C2) The cleaned kernel measures content"),
    ]):
        ax = fig.add_subplot(gs[row, 2])
        ax.scatter(x, y, s=9, color="#4575b4", alpha=0.5, edgecolors="none")
        ax.axhline(0, color="0.85", lw=0.5); ax.axvline(0, color="0.85", lw=0.5)
        ax.set_xlabel(xl, fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{ttl}  (r = {rr:.2f})", fontsize=8.0, pad=4)
        if row == 1:
            ax.text(0.03, 0.97,
                    f"cross-kit agreement:\nraw {r_kit_raw:.2f} → cleaned {r_kit_clean:.2f}\n"
                    f"(kit choice only matters\nafter cleaning)",
                    transform=ax.transAxes, fontsize=5.8, color="0.35", va="top")

    fig.suptitle("Inside the Δ context kernel: removing one global mode turns an "
                 "artifact into a content-bearing similarity", fontsize=12, y=0.965)
    save_fig(fig, "fig_delta_kernel")
    plt.close(fig)


if __name__ == "__main__":
    main()
