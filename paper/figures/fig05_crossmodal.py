"""Figure 5 — Cross-modal consistency of relational structure
(PLAN.md §6, Analysis 5).

For a single scene (`C_scene`), compute Δ-graphs from three modality probes:
  • text     — the sentence itself, embedded by the text embedder
  • audio    — a recording of the scene, CLAP-embedded, projected into the
               text-embedder space via a small linear bridge (or used directly
               if `--clap-only` is passed)
  • image    — a photograph of the scene, vision-LLM → caption → text-embedder

Compare top-K Δ-edge agreement across modalities (Jaccard, Kendall-τ).

Note
----
Audio and image inputs must be supplied via --audio and --image; without them
this script generates the text-only baseline and prints how to add the others.

Output: paper/v2/figures/fig05_crossmodal.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _setup import (
    CONTEXTS, CONTEXT_LABELS,
    build_space, save_fig, set_paper_style,
)

SHIFT_KW = dict(
    strategy="gate", gate="relu", beta=1.2, tau=0.3,
    within_symbol_softmax=True, gamma=0.5,
    pool_type="avg", pool_w=0.7, membership_alpha=0.0,
)

TOP_EDGES = 60


def _row_norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _delta_edges(D0, D1, top_n=TOP_EDGES):
    """Return list of (i, j, delta) for top-|delta| pairs.  i < j."""
    Delta = D1 @ D1.T - D0 @ D0.T
    np.fill_diagonal(Delta, 0.0)
    tri_i, tri_j = np.triu_indices_from(Delta, k=1)
    vals = Delta[tri_i, tri_j]
    order = np.argsort(-np.abs(vals))[:top_n]
    return [(int(tri_i[k]), int(tri_j[k]), float(vals[k])) for k in order]


def _jaccard(a_keys, b_keys):
    sa = set(a_keys); sb = set(b_keys)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _kendall_tau_top(a, b):
    """Kendall-τ-like rank correlation over the intersection of edge keys."""
    da = {(i, j): r for r, (i, j, _) in enumerate(a)}
    db = {(i, j): r for r, (i, j, _) in enumerate(b)}
    common = sorted(set(da) & set(db))
    if len(common) < 2:
        return 0.0
    ra = np.array([da[k] for k in common])
    rb = np.array([db[k] for k in common])
    n = len(ra)
    concordant = 0; discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign((ra[i] - ra[j]) * (rb[i] - rb[j]))
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom else 0.0


def _embed_audio(space, audio_path: Path) -> np.ndarray:
    """Embed audio via CLAP into the text-embedder's space.

    Uses delyrism's TextEmbedder(backend='clap') as a separate embedder, then
    projects into the text-space by re-encoding CLAP's nearest captions —
    *this is a lightweight bridge* sufficient for the figure; the production
    /context/encode-audio endpoint does the same thing more carefully.
    """
    raise NotImplementedError(
        "Audio modality bridge: wire to delyrism's CLAP backend and project "
        "via the text-embedder.  See /web/backend/app/routes/context.py for the "
        "production version.  Provide --audio PATH and implement here."
    )


def _embed_image_via_caption(space, image_path: Path) -> np.ndarray:
    """Caption the image with a vision-LLM, then encode the caption."""
    raise NotImplementedError(
        "Image modality bridge: caption the image (e.g. via Cloudflare's "
        "@cf/llava-hf/llava-1.5-7b-hf or any local vision-LLM), then encode the "
        "caption through space.embedder.  See /web/backend/app/routes/context.py "
        "for the production version.  Provide --image PATH and implement here."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--audio", type=Path, default=None,
                    help="Path to audio recording of the scene (WAV/MP3).")
    ap.add_argument("--image", type=Path, default=None,
                    help="Path to photograph of the scene.")
    args = ap.parse_args()

    set_paper_style()
    space = build_space(backend=args.backend, model=args.model)

    sentence = CONTEXTS["C_scene"]
    print(f"[fig05] scene context: {sentence!r}")

    # ─── Text modality (always available) ──────────────────────────────────
    D0 = space.D
    D_text = space.make_shifted_matrix(sentence=sentence, **SHIFT_KW)
    edges_text = _delta_edges(D0, D_text, top_n=TOP_EDGES)
    print(f"[fig05] text Δ-edges: {len(edges_text)} (top-{TOP_EDGES})")

    # ─── Audio modality (optional) ────────────────────────────────────────
    edges_audio = None
    if args.audio:
        try:
            v_audio = _embed_audio(space, args.audio)
            space.set_context_vec(v_audio)
            D_audio = space.make_shifted_matrix(**SHIFT_KW)
            space.set_context_vec(None)
            edges_audio = _delta_edges(D0, D_audio, top_n=TOP_EDGES)
            print(f"[fig05] audio Δ-edges: {len(edges_audio)}")
        except NotImplementedError as e:
            print(f"[fig05] audio modality not wired yet: {e}")

    # ─── Image modality (optional) ────────────────────────────────────────
    edges_image = None
    if args.image:
        try:
            v_image = _embed_image_via_caption(space, args.image)
            space.set_context_vec(v_image)
            D_image = space.make_shifted_matrix(**SHIFT_KW)
            space.set_context_vec(None)
            edges_image = _delta_edges(D0, D_image, top_n=TOP_EDGES)
            print(f"[fig05] image Δ-edges: {len(edges_image)}")
        except NotImplementedError as e:
            print(f"[fig05] image modality not wired yet: {e}")

    # ─── Agreement metrics ────────────────────────────────────────────────
    modalities = {"text": edges_text}
    if edges_audio is not None:
        modalities["audio"] = edges_audio
    if edges_image is not None:
        modalities["image"] = edges_image

    names = list(modalities.keys())
    n_mod = len(names)
    jacc = np.eye(n_mod)
    tau  = np.eye(n_mod)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            ka = [(e[0], e[1]) for e in modalities[a]]
            kb = [(e[0], e[1]) for e in modalities[b]]
            jacc[i, j] = jacc[j, i] = _jaccard(ka, kb)
            tau[i, j]  = tau[j, i]  = _kendall_tau_top(modalities[a], modalities[b])

    # ─── Render ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11.0, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0], wspace=0.32)

    # (a) top-N Δ-edges from each modality — node list
    ax_a = fig.add_subplot(gs[0, 0])
    pos_y = 0
    text_lines = []
    for name in names:
        es = modalities[name]
        text_lines.append(f"── {name} ── ({len(es)} edges)")
        for i, j, d in es[:8]:
            text_lines.append(
                f"   {space.descriptors[i]:>20s}  ↔  "
                f"{space.descriptors[j]:<20s}  {d:+.3f}"
            )
        text_lines.append("")
    ax_a.text(0.02, 0.98, "\n".join(text_lines), family="monospace",
              fontsize=6.5, va="top", ha="left", transform=ax_a.transAxes)
    ax_a.set_axis_off()
    ax_a.set_title("(a) Top Δ-edges per modality (first 8)")

    # (b) Jaccard heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    im = ax_b.imshow(jacc, cmap="magma", vmin=0, vmax=1)
    ax_b.set_xticks(range(n_mod)); ax_b.set_xticklabels(names)
    ax_b.set_yticks(range(n_mod)); ax_b.set_yticklabels(names)
    for i in range(n_mod):
        for j in range(n_mod):
            ax_b.text(j, i, f"{jacc[i,j]:.2f}", ha="center", va="center",
                      fontsize=8, color="white" if jacc[i, j] < 0.6 else "black")
    ax_b.set_title("(b) Top-K Jaccard")
    fig.colorbar(im, ax=ax_b, shrink=0.75, label="Jaccard")

    # (c) Kendall-τ heatmap (intersection only)
    ax_c = fig.add_subplot(gs[0, 2])
    im2 = ax_c.imshow(tau, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_c.set_xticks(range(n_mod)); ax_c.set_xticklabels(names)
    ax_c.set_yticks(range(n_mod)); ax_c.set_yticklabels(names)
    for i in range(n_mod):
        for j in range(n_mod):
            ax_c.text(j, i, f"{tau[i,j]:+.2f}", ha="center", va="center",
                      fontsize=8, color="black" if abs(tau[i, j]) < 0.6 else "white")
    ax_c.set_title("(c) Kendall-τ on common edges")
    fig.colorbar(im2, ax=ax_c, shrink=0.75, label="τ")

    fig.suptitle(
        f"Cross-modal consistency of relational rewiring  "
        f"(scene: {CONTEXT_LABELS['C_scene']})",
        fontsize=11, y=1.02,
    )
    save_fig(fig, "fig05_crossmodal")
    plt.close(fig)

    if edges_audio is None and edges_image is None:
        print("\n[fig05] NOTE: only the text modality was wired. To complete this")
        print("        figure, provide --audio <path> and --image <path> and")
        print("        implement _embed_audio() and _embed_image_via_caption().")


if __name__ == "__main__":
    main()
