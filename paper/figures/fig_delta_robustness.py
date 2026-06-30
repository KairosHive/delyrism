"""Figure — ROBUSTNESS: do the distortions matter, and are they encoder-specific?

Two validation layers, both compute-only:

(A) NULL MODEL.  For each oblique probe we replace its real context-shift with a
    RANDOM shift of equal magnitude (||D'-D||_F matched), and ask whether the
    de-biased readout still recovers the intended symbol.  Real contexts ground
    far above a magnitude-matched random baseline (top-1 67% vs 8%; top-3 93% vs
    31%; chance 10/30%), so the measured distortions are structured signal, not
    an inevitable consequence of moving the descriptors by that much.

(B) MULTI-EMBEDDER.  The grounding battery is re-run on three unrelated encoders
    (Qwen3-0.6B 1024-d; MPNet 768-d; MiniLM 384-d).  Grounding holds across all
    three and the per-probe symbol assignments agree well across encoders, so the
    readout reflects the symbolic system and the probe semantics rather than one
    model's geometry.

Output: paper/v2/figures/fig_delta_robustness.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from _setup import build_space, save_fig, set_paper_style
from _delta_common import DeltaReadout
from fig_delta_probe_atlas import PROBES

EMBEDDERS = [  # (label, backend, model)
    ("Qwen3-0.6B", "qwen3", None),
    ("MPNet", "sentence-transformer", "sentence-transformers/all-mpnet-base-v2"),
    ("MiniLM", "sentence-transformer", "sentence-transformers/all-MiniLM-L6-v2"),
]
N_NULL = 5


def _l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def grounding(dr, R, intended):
    """top-1, top-3 accuracy and the per-probe argmax symbols + ws_z profiles."""
    t1 = t3 = 0
    args, profiles = [], []
    for k in range(len(intended)):
        z = dr.ws_z(R[k]); profiles.append(z)
        o = np.argsort(z)[::-1]
        args.append(dr.syms[o[0]])
        rank = [dr.syms[i] for i in o].index(intended[k])
        t1 += rank == 0; t3 += rank < 3
    return t1 / len(intended), t3 / len(intended), args, np.array(profiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    set_paper_style()

    phrases = [p[2] for p in PROBES]
    intended = [p[1] for p in PROBES]
    n = len(PROBES)

    # ---- (A) null model on the reference embedder (qwen3) ----
    space = build_space(backend="qwen3")
    dr = DeltaReadout(space)
    R = dr.fit_reference(phrases)
    D0, C0, iu = space.D, dr.C0, dr.iu
    rng = np.random.default_rng(0)

    r_t1, r_t3, _, _ = grounding(dr, R, intended)
    r_marg = np.mean([np.diff(np.sort(dr.ws_z(R[k]))[::-1][:2] * [-1, 1]).sum()
                      for k in range(n)])  # top1-top2

    nt1, nt3, nmg = [], [], []
    for k, ph in enumerate(phrases):
        D1 = space.make_shifted_matrix(sentence=ph, **dr.kw)
        fro = np.linalg.norm(D1 - D0)
        for _ in range(N_NULL):
            S = rng.standard_normal(D0.shape); S *= fro / (np.linalg.norm(S) + 1e-12)
            D1n = _l2n(D0 + S)
            raw = (D1n @ D1n.T - C0)[iu]
            z = dr.ws_z(raw); o = np.argsort(z)[::-1]
            nt1.append(dr.syms[o[0]] == intended[k])
            nt3.append(intended[k] in [dr.syms[i] for i in o[:3]])
            nmg.append(z[o[0]] - z[o[1]])
    n_t1, n_t3 = np.mean(nt1), np.mean(nt3)
    print(f"[robust] null: top1 real {r_t1:.0%} vs {n_t1:.0%}; top3 {r_t3:.0%} vs {n_t3:.0%}")

    # ---- (B) multi-embedder ----
    emb_res, emb_args = [], {}
    for label, backend, model in EMBEDDERS:
        sp = build_space(backend=backend, model=model, verbose=False)
        d = DeltaReadout(sp)
        Re = d.fit_reference(phrases)
        t1, t3, ar, _ = grounding(d, Re, intended)
        emb_res.append((label, t1, t3)); emb_args[label] = ar
        print(f"[robust] {label}: top1 {t1:.0%} top3 {t3:.0%}")
    # cross-embedder symbol-assignment agreement (mean over encoder pairs)
    labels = [e[0] for e in EMBEDDERS]
    agrees = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = emb_args[labels[i]], emb_args[labels[j]]
            agrees.append(np.mean([a[k] == b[k] for k in range(n)]))
    mean_agree = np.mean(agrees)
    print(f"[robust] cross-embedder top-1 agreement: {mean_agree:.0%}")

    # ---- figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.07, right=0.985, wspace=0.30)

    # (A) null model grouped bars
    groups = ["top-1", "top-3"]
    x = np.arange(2); w = 0.34
    axA.bar(x - w / 2, [r_t1, r_t3], w, color="#1a9850", edgecolor="k",
            linewidth=0.5, label="real contexts")
    axA.bar(x + w / 2, [n_t1, n_t3], w, color="#b2182b", edgecolor="k",
            linewidth=0.5, label="magnitude-matched random")
    for xi, ch in zip(x, [0.10, 0.30]):
        axA.hlines(ch, xi - 0.42, xi + 0.42, color="0.5", lw=1.0, ls=":")
    axA.text(1.42, 0.30, "chance", fontsize=6, color="0.5", va="center")
    for xi, (rv, nv) in zip(x, [(r_t1, n_t1), (r_t3, n_t3)]):
        axA.text(xi - w / 2, rv + 0.02, f"{rv:.0%}", ha="center", fontsize=7)
        axA.text(xi + w / 2, nv + 0.02, f"{nv:.0%}", ha="center", fontsize=7)
    axA.set_xticks(x); axA.set_xticklabels(groups, fontsize=8)
    axA.set_ylim(0, 1.12); axA.set_ylabel("grounding accuracy", fontsize=8.5)
    axA.legend(fontsize=7, loc="upper left", framealpha=0.9)
    axA.set_title("(A) Null model — real contexts vs a\nmagnitude-matched random shift",
                  fontsize=9, pad=5)

    # (B) multi-embedder grouped bars
    xe = np.arange(len(EMBEDDERS))
    t1s = [r[1] for r in emb_res]; t3s = [r[2] for r in emb_res]
    axB.bar(xe - w / 2, t1s, w, color="#2166ac", edgecolor="k", linewidth=0.5, label="top-1")
    axB.bar(xe + w / 2, t3s, w, color="#92c5de", edgecolor="k", linewidth=0.5, label="top-3")
    for xi, (v1, v3) in zip(xe, zip(t1s, t3s)):
        axB.text(xi - w / 2, v1 + 0.02, f"{v1:.0%}", ha="center", fontsize=7)
        axB.text(xi + w / 2, v3 + 0.02, f"{v3:.0%}", ha="center", fontsize=7)
    axB.hlines(0.10, -0.5, len(EMBEDDERS) - 0.5, color="0.5", lw=1.0, ls=":")
    axB.set_xticks(xe)
    axB.set_xticklabels([f"{e[0]}\n({d})" for e, d in
                         zip(EMBEDDERS, ["1024-d", "768-d", "384-d"])], fontsize=7.5)
    axB.set_ylim(0, 1.05); axB.set_ylabel("grounding accuracy", fontsize=8.5)
    axB.legend(fontsize=7, loc="upper right", framealpha=0.9)
    axB.set_title(f"(B) Multi-embedder — grounding holds across encoders\n"
                  f"(cross-encoder top-1 agreement {mean_agree:.0%})", fontsize=9, pad=5)

    fig.suptitle("Robustness — the distortions are structured signal and not encoder-specific",
                 fontsize=12, y=0.965)
    save_fig(fig, "fig_delta_robustness")
    plt.close(fig)


if __name__ == "__main__":
    main()
