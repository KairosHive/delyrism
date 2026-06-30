"""Figure 3b — Morphing gallery: semantic symbol reorganization across A<->B pairs.

WHAT THIS SHOWS (and why it changed)
------------------------------------
The previous version tracked (coverage_H1, focus) over a linear context blend and
read the U-shape / inverted-U as a "phase transition".  A diagnostic
(paper/figures/_explore_morph_artifact.py, _explore_morph_semantic.py) showed
that signal is a GEOMETRIC ARTIFACT, not semantics:

  • coverage_H1 / focus are a near-deterministic function of the gate drive
    energy  mean_d relu(D @ vctx)  (R^2 = 0.94-1.00; focus is ~100% energy).
  • That energy is structurally inverted-U over ANY blend of two directions,
    because a renormalized average has weaker peak-alignment than its endpoints.
    The U survives SLERP, so it is not an interpolation-speed quirk.

So this figure now tracks a SHIFT-MAGNITUDE-INVARIANT, genuinely semantic signal:
the personalized-PageRank symbol ranking across the morph (same pipeline as
fig_v1_ppr / POST /subgraph), DEGREE-NORMALIZED (PPR score / #descriptors) to
remove the bipartite-graph bias that otherwise makes the highest-degree symbol
(EARTH, 19 descriptors) win at every alpha.  The degree-normalized rank-1 symbol
then *reorganizes* as the context morphs A->B, and the transition is real and
pair-specific:

  C_A->C3     CLOUDS -> THUNDER
  C1->C_scene EARTH -> CLOUDS -> THUNDER
  C2->C_A     THUNDER -> CLOUDS
  C_scene->C3 THUNDER throughout  (both ends are storm-register: correctly flat)

Layout (one column per pair):
  • thin top stripe — degree-normalized rank-1 symbol vs alpha (the transition)
  • main panel      — degree-normalized PPR trajectory per symbol; symbols that
                      ever reach top-3 are bold + coloured (Nord palette, shared
                      with fig_v1_ppr and the app), the rest faint grey.

Output: paper/v2/figures/fig03b_morph_gallery.{pdf,png}
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from _setup import CONTEXTS, CONTEXT_LABELS, build_space, save_fig, set_paper_style

MORPH_PAIRS = [
    ("C_A", "C3"),       # sacred voice calling       <-> the dream's end
    ("C1",  "C_scene"),  # the sacred hoop            <-> horse at dawn (storm + light)
    ("C2",  "C_A"),      # thunder voice & dawn horse <-> sacred voice calling
    ("C_scene", "C3"),   # horse at dawn (storm+light) <-> the dream's end
]

ALPHA_STEPS = 21
PPR_ALPHA = 0.85     # PageRank teleport (graph diffusion), NOT the morph alpha
PPR_TAU = 0.1        # softmax temperature for descriptor personalization


def _blend_vec(space, sent_a, sent_b, alpha):
    v_a = space.embedder.encode([sent_a])[0]
    v_b = space.embedder.encode([sent_b])[0]
    return (1.0 - alpha) * v_a + alpha * v_b


def _ppr_symbol_scores(space):
    """Degree-normalized PPR symbol scores for the CURRENT context override.

    Mirrors POST /subgraph's personalized PageRank, then divides each symbol's
    score by its descriptor count to remove the high-degree bias of the
    bipartite symbol<->descriptor graph (otherwise EARTH always wins).
    """
    from delyrism.delyrism import softmax

    vctx = space.context_override
    pers = {f"D:{d}": float(w)
            for d, w in zip(space.descriptors, softmax(space.D @ vctx, tau=PPR_TAU))}
    pr = nx.pagerank(space.G, alpha=PPR_ALPHA, personalization=pers, weight="weight")
    raw = {n[2:]: v for n, v in pr.items() if n.startswith("S:")}
    return {s: raw.get(s, 0.0) / len(space.symbols_to_descriptors[s])
            for s in space.symbols}


def _sweep(space, sent_a, sent_b, alphas):
    """Return {symbol: trajectory over alpha} of degree-normalized PPR scores."""
    syms = list(space.symbols)
    traj = {s: [] for s in syms}
    for alpha in alphas:
        space.set_context_vec(_blend_vec(space, sent_a, sent_b, float(alpha)))
        sc = _ppr_symbol_scores(space)
        space.set_context_vec(None)
        for s in syms:
            traj[s].append(sc[s])
    return {s: np.array(v) for s, v in traj.items()}


def _runs(seq):
    """Collapse a sequence into ordered unique tokens, e.g. [A,A,B,A]->[A,B,A]."""
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--steps", type=int, default=ALPHA_STEPS)
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)
    cdict = space.get_symbol_color_dict(palette="Nord")

    alphas = np.linspace(0.0, 1.0, args.steps)
    trajectories = []
    t0 = time.time()
    for ca, cb in MORPH_PAIRS:
        print(f"[fig03b] morph  {ca} -> {cb}   "
              f"({CONTEXT_LABELS[ca]}  ->  {CONTEXT_LABELS[cb]})")
        traj = _sweep(space, CONTEXTS[ca], CONTEXTS[cb], alphas)
        trajectories.append((ca, cb, traj))

    print(f"[fig03b] swept {len(MORPH_PAIRS)} pairs in {time.time() - t0:.1f}s")

    # ── Layout: per pair, a thin rank-1 stripe over a trajectory panel ────────
    n_cols = len(MORPH_PAIRS)
    fig = plt.figure(figsize=(3.5 * n_cols, 4.6))
    gs = fig.add_gridspec(
        2, n_cols, height_ratios=[0.10, 1.0],
        hspace=0.06, wspace=0.26,
        top=0.80, bottom=0.13, left=0.06, right=0.99,
    )

    syms = list(space.symbols)

    for col, (ca, cb, traj) in enumerate(trajectories):
        M = np.vstack([traj[s] for s in syms])            # (n_sym, n_alpha)
        winners = [syms[i] for i in M.argmax(axis=0)]     # rank-1 per alpha
        ever_top3 = set()
        for j in range(M.shape[1]):
            ever_top3.update(syms[i] for i in np.argsort(M[:, j])[-3:])

        # ── rank-1 dominance stripe ──────────────────────────────────────────
        ax_s = fig.add_subplot(gs[0, col])
        for j, w in enumerate(winners):
            ax_s.axvspan(alphas[j] - 0.5 / (len(alphas) - 1),
                         alphas[j] + 0.5 / (len(alphas) - 1),
                         color=cdict.get(w, "0.5"), lw=0)
        ax_s.set_xlim(0, 1)
        ax_s.set_yticks([])
        ax_s.set_xticks([])
        for sp in ax_s.spines.values():
            sp.set_visible(False)
        ax_s.set_title(f"${ca}$ $\\to$ ${cb}$", fontsize=10, fontweight="bold", pad=18)
        ax_s.text(0.5, 1.04, f"{CONTEXT_LABELS[ca]}  →  {CONTEXT_LABELS[cb]}",
                  transform=ax_s.transAxes, ha="center", va="bottom",
                  fontsize=6.6, color="0.45", style="italic")
        ax_s.set_ylabel("rank-1", fontsize=6.5, rotation=0, ha="right", va="center",
                        labelpad=2)

        # ── trajectory panel ─────────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, col])
        for i, s in enumerate(syms):
            if s in ever_top3:
                ax.plot(alphas, M[i], color=cdict.get(s, "0.4"), lw=1.9, zorder=3)
                # label at whichever end the line is higher
                end = -1 if M[i, -1] >= M[i, 0] else 0
                ax.text(alphas[end] + (0.01 if end == -1 else -0.01), M[i, end], s,
                        ha="left" if end == -1 else "right", va="center",
                        fontsize=6.8, fontweight="bold", color=cdict.get(s, "0.4"),
                        zorder=4)
            else:
                ax.plot(alphas, M[i], color="0.7", lw=0.6, alpha=0.5, zorder=1)

        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$\alpha\;$ (A $\to$ B)", fontsize=8, labelpad=2)
        if col == 0:
            ax.set_ylabel("degree-normalized PPR score", fontsize=8)
        ax.margins(x=0.12)
        # annotate the rank-1 transition path under the panel
        ax.set_title(" → ".join(_runs(winners)), fontsize=6.8, color="0.35", pad=3)

    fig.suptitle(
        "Morphing gallery — degree-normalized PPR symbol reorganization across "
        "four context pairs (Black Elk Speaks)",
        fontsize=10.5, y=0.95,
    )

    save_fig(fig, "fig03b_morph_gallery")
    plt.close(fig)


if __name__ == "__main__":
    main()
