"""V1 Figure 2 — Context-conditioned attention violins for EARTH.

Mirrors the v1 paper's `plot_thematic_first_descriptor_violins`:
  • uses `space.conditioned_symbol(...)` for the attention map (the engine
    function, not manual softmax),
  • one column per emotional theme,
  • descriptors sorted PER COLUMN by their median attention under that theme
    (so each panel's leftmost descriptors are the theme's top facets — this
    is the v1 visual layout that makes per-theme differentiation legible),
  • τ = 0.2 (the v1 default).

Output: paper/v2/figures/fig_v1_attention.{pdf,png}
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _setup import build_space, save_fig, set_paper_style


# Ten-sentence emotional themes — v1-style.
EMOTION_THEMES = {
    "FEAR": [
        "A shadow moves behind the trees; breath turns shallow.",
        "Cold wind rattles the window as footsteps approach.",
        "Eyes watch from the dark water below the boat.",
        "The cave narrows; the light behind fades away.",
        "An animal scream splits the night and then silence.",
        "A sudden howl echoes across the empty field.",
        "The lantern flickers as something brushes past.",
        "Footsteps stop just outside the locked door.",
        "A cold hand grips the back of my neck.",
        "The silence grows heavier with every heartbeat.",
    ],
    "LOVE": [
        "Their hands met by the river at dusk.",
        "Breath synced with a heartbeat under warm light.",
        "A letter folded in a pocket carries a smile.",
        "Two paths converged beside the old willow.",
        "A quiet morning coffee became a promise.",
        "Eyes meet and laughter fills the room.",
        "A gentle touch lingers after goodbye.",
        "Shared secrets whispered beneath the stars.",
        "A song played softly in the kitchen.",
        "The warmth of an embrace on a cold night.",
    ],
    "SADNESS": [
        "Rain kept the chair by the window empty.",
        "Echoes lingered after the song stopped.",
        "Footprints on the shore washed away too fast.",
        "The room remembered the laughter better than we did.",
        "A cracked cup waiting on the shelf.",
        "A faded photograph tucked in a drawer.",
        "The last light in the hallway flickers out.",
        "A suitcase packed but never opened.",
        "The scent of perfume long after she's gone.",
        "A single tear stains the letter unread.",
    ],
    "GRATITUDE": [
        "Warm bread shared at sunrise after a long night.",
        "A stranger's umbrella in sudden rain.",
        "Hands washed clean in the cold river.",
        "A soft blanket after the journey home.",
        "The last apple given without asking.",
        "A smile offered when hope was thin.",
        "A friend's call just when it was needed.",
        "A gentle word that eased the pain.",
        "A door held open on a heavy day.",
        "A memory cherished more with each year.",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--symbol", default="EARTH")
    ap.add_argument("--tau", type=float, default=0.2,
                    help="Softmax temperature for the engine's "
                         "`conditioned_symbol` attention.  v1 used 0.2.")
    ap.add_argument("--top-n", type=int, default=10,
                    help="Limit to the top-N descriptors by overall mean "
                         "attention (None = show all).")
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    if args.symbol not in space.symbols:
        raise SystemExit(f"symbol {args.symbol!r} not in space; "
                         f"available: {space.symbols}")

    descriptors = list(space.symbols_to_descriptors[args.symbol])

    # ── Collect attention weights per sentence × context × descriptor ─────
    rows = []
    for ctx_label, sentences in EMOTION_THEMES.items():
        for sent_ix, sent in enumerate(sentences):
            _, attn = space.conditioned_symbol(
                args.symbol, sentence=sent, tau=args.tau,
            )
            for d in descriptors:
                rows.append({
                    "context": ctx_label,
                    "descriptor": d,
                    "attention": float(attn.get(d, 0.0)),
                    "sentence_ix": sent_ix,
                })
    df = pd.DataFrame(rows)

    # ── Optional global top-N filter ──────────────────────────────────────
    if args.top_n is not None and args.top_n < len(descriptors):
        top_desc = (df.groupby("descriptor")["attention"]
                      .mean()
                      .sort_values(ascending=False)
                      .head(args.top_n)
                      .index.tolist())
        df = df[df["descriptor"].isin(top_desc)]

    # ── Per-context descriptor ordering — by that context's median, desc ──
    ctx_orders = {}
    for ctx in EMOTION_THEMES.keys():
        sub = df[df["context"] == ctx]
        order = (sub.groupby("descriptor")["attention"]
                    .median()
                    .sort_values(ascending=False)
                    .index.tolist())
        ctx_orders[ctx] = order

    # Stable per-descriptor colours via tab20
    unique_desc = sorted(df["descriptor"].unique())
    palette = sns.color_palette("tab20", n_colors=len(unique_desc))
    color_for = {d: palette[i] for i, d in enumerate(unique_desc)}

    # ── Render — one column per theme, shared y-axis ─────────────────────
    theme_names = list(EMOTION_THEMES.keys())
    n_ctx = len(theme_names)
    minval = float(df["attention"].min())
    maxval = float(df["attention"].max())

    fig, axes = plt.subplots(1, n_ctx, figsize=(3.4 * n_ctx, 5.2),
                             sharey=True)
    if n_ctx == 1:
        axes = [axes]

    for ax, ctx in zip(axes, theme_names):
        sub = df[df["context"] == ctx]
        order = ctx_orders[ctx]
        ordered_colors = [color_for[d] for d in order]
        sns.violinplot(
            data=sub, x="descriptor", y="attention", order=order,
            cut=0, density_norm="width", inner="quartile",
            ax=ax, palette=ordered_colors,
            linewidth=1.0, bw_method=0.3, hue="descriptor", legend=False,
        )
        ax.set_title(ctx, fontsize=10)
        ax.set_xlabel("descriptor", fontsize=8)
        ax.set_ylabel("attention weight" if ax is axes[0] else "")
        ax.set_ylim(minval - 0.002, maxval + 0.01)
        ax.tick_params(axis="x", rotation=75)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
            tick.set_fontsize(7)

    fig.suptitle(
        f"{args.symbol} — descriptor attention per thematic context "
        f"(top-{args.top_n}, sorted within each panel; τ={args.tau})",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    save_fig(fig, "fig_v1_attention")
    plt.close(fig)


if __name__ == "__main__":
    main()
