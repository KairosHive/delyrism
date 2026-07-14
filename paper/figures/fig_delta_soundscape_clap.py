"""Figure — audio drives the Δ readout DIRECTLY through CLAP ("CLAP all the way").

Unlike fig_delta_soundscape.py (which bridges audio -> acoustic labels -> Qwen3),
here the SAME symbolic field is re-encoded with CLAP's text tower and a recording
is used as the context vector straight from CLAP's audio tower. Because CLAP is
trained to align sound with acoustic language, real recordings ground into the
field directly: a thunderclap most strengthens Thunder/Lightning couplings and
its thunder/storm descriptors; a birdsong selects the descriptor ``bird''.

The point: the Δ instrument is genuinely modality-agnostic. Swap the text encoder
for a contrastive audio-text encoder and feed audio; no text bridge is needed.
Sounds with no correspondent in the curated field (there is no Fire/Ocean symbol)
have no target and land diffusely -- shown honestly in panel (A).

Output: paper/v2/figures/fig_delta_soundscape_clap.{pdf,png}
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from _setup import build_space, save_fig, set_paper_style
from _delta_common import DeltaReadout
from fig_delta_probe_atlas import short_sym

AUD = Path(__file__).resolve().parent / "_audio"
# grounded sounds first, diffuse ones last (honest ordering)
ORDER = ["thunder", "birds", "water", "powwow", "drum", "flute", "fire", "ocean"]
NICE = {"thunder": "thunder", "birds": "forest birds", "water": "stream",
        "powwow": "pow-wow", "drum": "drum", "flute": "flute",
        "fire": "campfire", "ocean": "ocean waves"}
EXEMPLARS = ["thunder", "birds"]        # panel (B): distinct, well-grounded subgraphs
EX_COL = {"thunder": "#3a6ea5", "birds": "#52b788"}


def _load_audio(path):
    try:
        import soundfile as sf
        y, sr = sf.read(str(path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y[: int(sr * 8.0)], int(sr)
    except Exception:
        import librosa
        y, _ = librosa.load(str(path), sr=48000, mono=True, duration=8.0)
        return y, 48000


def _short_desc(s, w=17):
    return s if len(s) <= w else s[: w - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.parse_args()
    set_paper_style()

    space = build_space(backend="clap")           # descriptors in CLAP text space
    dr = DeltaReadout(space)
    clap = space.embedder                          # same encoder gives the audio tower
    descs = space.descriptors

    names = [s for s in ORDER if (AUD / f"{s}.wav").exists()]
    raws = []
    for nm in names:
        y, sr = _load_audio(AUD / f"{nm}.wav")
        av = clap.embed_audio_array(y.astype(np.float32), sr).astype(np.float32)
        av /= np.linalg.norm(av) + 1e-9
        space.set_context_vec(av)
        D1 = space.make_shifted_matrix(**dr.kw)
        space.set_context_vec(None)
        Dl = D1 @ D1.T - dr.C0
        np.fill_diagonal(Dl, 0.0)
        raws.append(Dl[dr.iu])
    raws = np.vstack(raws)
    res = raws - raws.mean(axis=0, keepdims=True)          # de-biased across sounds

    # symbol-level response for panel A, double-centered so neither a globally
    # "loud" sound nor an always-active symbol dominates: remove the per-symbol
    # baseline (column mean across sounds) and the per-sound loudness (row mean).
    W = np.vstack([dr.ws(r) for r in raws])                # (n_sounds, n_symbols)
    Wc = W - W.mean(axis=0, keepdims=True)
    Wc = Wc - Wc.mean(axis=1, keepdims=True)
    nS = len(dr.syms)

    # ── layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12.4, 5.15))
    # col1 is a thin spacer so A/colorbar clears B, while B and C stay close.
    gs = fig.add_gridspec(2, 4, width_ratios=[1.08, 0.24, 0.52, 1.24],
                          hspace=0.16, wspace=0.04, top=0.85, bottom=0.135,
                          left=0.125, right=0.99)

    # (A) sound × symbol grounding heatmap
    axA = fig.add_subplot(gs[:, 0])
    vmax = float(np.nanpercentile(np.abs(Wc), 96)) or 1.0
    normA = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = axA.imshow(Wc, cmap="RdBu_r", norm=normA, aspect="auto")
    axA.set_xticks(range(nS))
    axA.set_xticklabels([short_sym(s) for s in dr.syms], fontsize=9.5, rotation=90)
    axA.set_yticks(range(len(names)))
    axA.set_yticklabels([NICE.get(nm, nm) for nm in names], fontsize=10)
    # box only thunder, the one unambiguous a-priori correspondent
    if "thunder" in names:
        r = names.index("thunder")
        c = int(np.argmax(Wc[r]))
        axA.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                    edgecolor="k", lw=1.8))
    axA.set_title("(A) Direct CLAP grounding per recording\n"
                  "(thunder selects Thunder, boxed)",
                  fontsize=11, pad=6)
    cb = fig.colorbar(im, ax=axA, fraction=0.035, pad=0.02)
    cb.ax.set_title("de-biased\nresponse", fontsize=7.5, pad=4)
    cb.ax.tick_params(labelsize=8)

    # (B) exemplar Δ subgraphs from sound: a chord over symbols showing which
    #     couplings each recording strengthens, computed directly from audio
    #     (same style as the bridge figure, but no bridge).
    SYMC = plt.cm.tab10(np.linspace(0, 1, nS))
    sang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, nS, endpoint=False)
    SX, SY = np.cos(sang), np.sin(sang)
    KE = 40                                            # top descriptor-pair edges kept
    for row, nm in enumerate(EXEMPLARS):
        ax = fig.add_subplot(gs[row, 2])
        ax.axis("off")
        ax.set_aspect("equal")
        if nm not in names:
            continue
        rv = res[names.index(nm)]
        Msym = np.zeros((nS, nS))
        for p in np.argsort(np.abs(rv))[::-1][:KE]:
            a, b = int(dr.oi[p]), int(dr.oj[p])
            if a != b:                                 # between-symbol couplings
                Msym[a, b] += abs(rv[p])
                Msym[b, a] += abs(rv[p])
        mx = Msym.max() + 1e-9
        for a in range(nS):
            for b in range(a + 1, nS):
                if Msym[a, b] <= 0:
                    continue
                xs = np.linspace(0, 1, 24)
                bx = (1 - xs) ** 2 * SX[a] + xs ** 2 * SX[b]   # quadratic Bezier via centre
                by = (1 - xs) ** 2 * SY[a] + xs ** 2 * SY[b]
                ax.plot(bx, by, color=EX_COL.get(nm, "#555"),
                        lw=0.4 + 3.6 * Msym[a, b] / mx, alpha=0.62,
                        solid_capstyle="round", zorder=2)
        for s in range(nS):
            ax.scatter(SX[s], SY[s], s=58, color=SYMC[s], edgecolors="k",
                       linewidths=0.5, zorder=3)
            ax.text(1.24 * SX[s], 1.24 * SY[s], short_sym(dr.syms[s]), fontsize=7.2,
                    ha="center", va="center", color="0.2")
        ax.set_xlim(-1.48, 1.48)
        ax.set_ylim(-1.5, 1.42)
        ax.set_title(f"(B{row+1}) {NICE.get(nm, nm)}: Δ subgraph (symbols)",
                     color=EX_COL.get(nm, "k"), fontsize=10, fontweight="bold", pad=1)

    # (C) descriptor-level Δ subgraph (force-directed): the fine structure behind
    #     thunder's symbol chord, drawn as a spring-layout Δ-graph.
    import networkx as nx
    axC = fig.add_subplot(gs[:, 3])
    axC.axis("off")
    axC.set_aspect("equal")
    cnm = "thunder" if "thunder" in names else names[0]
    rv = res[names.index(cnm)]
    ep = np.argsort(rv)[::-1][:12]                     # top strengthened descriptor pairs
    G = nx.Graph()
    for p in ep:
        a, b = descs[int(dr.iu[0][p])], descs[int(dr.iu[1][p])]   # dedupe by descriptor text
        if a == b:
            continue
        w = float(max(rv[p], 1e-6))
        if G.has_edge(a, b):
            G[a][b]["w"] += w
        else:
            G.add_edge(a, b, w=w)
    pos = nx.spring_layout(G, seed=4, k=1.9)
    deg = dict(G.degree())
    ew = np.array([G[u][v]["w"] for u, v in G.edges()])
    ewn = ew / (ew.max() + 1e-9)
    nx.draw_networkx_edges(G, pos, ax=axC, width=1.2 + 6.5 * ewn,
                           edge_color=ewn, edge_cmap=plt.cm.Reds,
                           edge_vmin=-0.15, edge_vmax=1.05, alpha=0.85)
    nx.draw_networkx_nodes(
        G, pos, ax=axC, edgecolors="k", linewidths=0.7,
        node_color=[SYMC[dr.sidx[space.owner[d]]] for d in G.nodes()],
        node_size=[210 + 250 * deg[d] for d in G.nodes()])
    for d, (x, y) in pos.items():
        axC.text(x, y + 0.135, _short_desc(d, 20), fontsize=10.5,
                 ha="center", va="bottom", color="0.05")
    axC.margins(x=0.10, y=0.20)
    axC.set_title(f"(C) {NICE.get(cnm, cnm)}: descriptor-level Δ subgraph\n"
                  "(the couplings behind panel B1)",
                  color=EX_COL.get(cnm, "k"), fontsize=10, fontweight="bold", pad=2)

    fig.suptitle("Audio drives the archetypal field directly through CLAP "
                 "(no text bridge; the Δ instrument is modality-agnostic)",
                 fontsize=13, y=0.965)
    save_fig(fig, "fig_delta_soundscape_clap")
    plt.close(fig)


if __name__ == "__main__":
    main()
