"""V1 Figure 2 — Context-conditioned attention violins for EARTH.

Preserved from the original NeurIPS 2025 submission (PLAN.md §4.5).  V2 should
pair this with a Δ-graph view of the same context shifts to give attention-level
*and* relational-coupling-level pictures side by side.

Output: paper/v2/figures/fig_v1_attention.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from _setup import build_space, save_fig, set_paper_style


# Four emotion themes from the v1 paper — 10 sentences per theme.
EMOTION_THEMES = {
    "Fear": [
        "Cold sweat clings to my skin as shadows shift.",
        "Every footstep behind me sounds louder than thunder.",
        "I can't breathe; the walls are closing in.",
        "My heart pounds against the silence of the night.",
        "Something is watching from the dark.",
        "The unknown spreads through me like ice.",
        "I clutch the doorframe, knees trembling, voice gone.",
        "A storm is coming and I have nowhere to hide.",
        "Even my own thoughts feel like strangers.",
        "Every sound at the edge of hearing is a threat.",
    ],
    "Love": [
        "Her laughter is sunlight on a slow river.",
        "I hold his hand and the world feels still.",
        "My grandmother's voice carries every winter through.",
        "We share silence the way others share songs.",
        "Coming home to you is the only place I know.",
        "Their joy moves through me like warm rain.",
        "Tenderness has its own gravity.",
        "I would carry your name through any storm.",
        "Every small act of care is a quiet vow.",
        "You are the long answer to a brief question.",
    ],
    "Sadness": [
        "The empty chair holds more than memory.",
        "Rain settles on the window like a slow exhale.",
        "I find old photographs and forget how to breathe.",
        "Grief moves through me without asking.",
        "Some days the weight is the morning itself.",
        "The road home is longer than it used to be.",
        "There is a quiet I cannot name.",
        "Everything I loved has gone somewhere I can't follow.",
        "I am tired in a place sleep does not reach.",
        "The river keeps moving and I cannot.",
    ],
    "Gratitude": [
        "The bread is warm and there is enough.",
        "I keep the morning light in my hands a moment longer.",
        "Tonight I will remember each small kindness by name.",
        "The seeds I planted have come back as a garden.",
        "Even my mistakes have taught me something to carry.",
        "I am held by people I never asked to hold me.",
        "The wind is gentle today and I am paying attention.",
        "My body has carried me through more than I knew.",
        "Each meal we share writes a longer story.",
        "The river thanks me back when I sit beside it.",
    ],
}


def _attention_weights(space, symbol: str, sentence: str, tau: float = 0.2) -> np.ndarray:
    """Softmax(cosine(descriptor, context) / tau) over the symbol's descriptors."""
    idx = space.symbol_to_idx[symbol]
    Xs = space.D[idx]
    v = space.embedder.encode([sentence])[0]
    v = v / (np.linalg.norm(v) + 1e-12)
    sims = Xs @ v
    s = sims / max(tau, 1e-6)
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--symbol", default="EARTH")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="Softmax temperature; lower = sharper peaks. "
                         "0.05 works well with Qwen3-Embedding-0.6B.")
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    if args.symbol not in space.symbols:
        raise SystemExit(f"symbol {args.symbol!r} not in space; available: {space.symbols}")

    descriptors = list(space.symbols_to_descriptors[args.symbol])
    n_desc = len(descriptors)

    # weights[theme] -> shape (n_sentences, n_descriptors)
    weights = {}
    for theme, sentences in EMOTION_THEMES.items():
        W = np.stack([_attention_weights(space, args.symbol, s, tau=args.tau)
                      for s in sentences])
        weights[theme] = W

    # Sort descriptors by median weight across all themes — readable left→right
    all_W = np.concatenate(list(weights.values()), axis=0)
    order = np.argsort(-np.median(all_W, axis=0))
    descriptors = [descriptors[i] for i in order]
    for t in weights:
        weights[t] = weights[t][:, order]

    # ─── Render — one row of violins per theme ────────────────────────────
    theme_names = list(EMOTION_THEMES.keys())
    theme_colors = {"Fear": "#5a4a91", "Love": "#b3262a",
                    "Sadness": "#2f5d8f", "Gratitude": "#3a6b4f"}
    n_themes = len(theme_names)

    fig, axes = plt.subplots(n_themes, 1,
                             figsize=(max(8.5, 0.45 * n_desc + 2.0), 2.0 * n_themes),
                             sharex=True)
    if n_themes == 1:
        axes = [axes]
    for ax, theme in zip(axes, theme_names):
        W = weights[theme]
        parts = ax.violinplot(W, positions=np.arange(n_desc), showmeans=False,
                              showmedians=True, widths=0.85)
        for body in parts["bodies"]:
            body.set_facecolor(theme_colors[theme]); body.set_edgecolor("black")
            body.set_alpha(0.78); body.set_linewidth(0.4)
        for k in ("cmedians", "cbars", "cmins", "cmaxes"):
            if k in parts:
                parts[k].set_color("black"); parts[k].set_linewidth(0.6)
        ax.set_ylabel("attention", fontsize=8)
        ax.set_title(f"{theme}", loc="left", fontsize=9)
        ax.set_ylim(0, max(0.18, float(W.max()) * 1.08))
    axes[-1].set_xticks(np.arange(n_desc))
    axes[-1].set_xticklabels(descriptors, rotation=50, ha="right", fontsize=6.8)

    fig.suptitle(
        f"Context-conditioned attention over {args.symbol} descriptors  "
        f"(four emotional themes; tau={args.tau})",
        fontsize=11, y=1.00,
    )
    save_fig(fig, "fig_v1_attention")
    plt.close(fig)


if __name__ == "__main__":
    main()
