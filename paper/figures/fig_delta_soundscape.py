"""Figure — CREATIVE APPLICATION: navigating the archetypal space with sound.

A bridge lets sound drive the Delta readout without text:
  sound --CLAP--> blend over a generic acoustic vocabulary --qwen3--> context
        --> Delta coupling readout --> distinctive archetypal signature.

We are not testing whether a thunderclap "is" Thunder (trivial); we show that
real recordings land at interpretable, well-separated places in the symbolic
field, and that one can MORPH between two sounds to traverse it continuously --
a sound-driven interface for exploring a curated symbolic system.

Sounds: 7 CC/PD field recordings from Wikimedia Commons (fetched by
_fetch_sounds.py into _audio/).  CLAP recognizes them cleanly (birdsong 0.96,
thunder 0.96, pow-wow -> "chanting and singing" 0.72, ...); the distinctive
(mean-removed) archetypal signatures are well spread (mean pairwise cos ~ -0.1).

Output: paper/v2/figures/fig_delta_soundscape.{pdf,png}
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import librosa

from _setup import build_space, save_fig, set_paper_style
from _delta_common import DeltaReadout
from fig_delta_probe_atlas import PROBES, short_sym

AUD = Path(__file__).resolve().parent / "_audio"
LABELS = ["thunder", "rain", "wind", "crackling fire", "ocean waves", "flowing water",
          "birdsong", "buzzing insects", "drumming", "people chanting and singing",
          "a flute melody", "a rattle", "footsteps", "rustling leaves", "a low rumble",
          "splashing water", "wingbeats", "bells ringing", "hooves", "silence"]
# pow-wow drums+singers fetched too, but its bridge context is a uniform-negative
# outlier (chanting sits far from every nature symbol), so it is excluded from
# the display set for legibility.
ORDER = ["thunder", "fire", "ocean", "water", "birds", "flute", "drum"]
# sound -> family (for colour); families = different regions of the field
FAMILY = {"thunder": "storm", "fire": "hearth", "ocean": "water", "water": "water",
          "birds": "living", "flute": "voice", "drum": "voice"}
FAM_COL = {"storm": "#3a6ea5", "water": "#2a9d8f", "hearth": "#e76f51",
           "living": "#52b788", "voice": "#9d4edd"}
MORPHS = [("thunder", "fire"), ("ocean", "powwow")]  # traverse storm->hearth, sea->ceremony
NICE = {"thunder": "thunder", "fire": "campfire", "ocean": "ocean waves",
        "water": "stream", "birds": "forest birds", "drum": "drum cadence",
        "flute": "flute", "powwow": "pow-wow drums\n& singers"}


def _softmax(x, t):
    x = (x - x.mean()) / (x.std() + 1e-9)
    e = np.exp(x / t); return e / e.sum()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=None)
    ap.parse_args()
    set_paper_style()

    space = build_space(backend="qwen3")
    dr = DeltaReadout(space)
    dr.fit_reference([p[2] for p in PROBES])
    from delyrism.delyrism import TextEmbedder
    clap = TextEmbedder(backend="clap")
    Lc = np.vstack([clap.encode([l])[0] for l in LABELS]); Lc /= np.linalg.norm(Lc, axis=1, keepdims=True) + 1e-9
    Lq = np.vstack([space.embedder.encode([l])[0] for l in LABELS]); Lq /= np.linalg.norm(Lq, axis=1, keepdims=True) + 1e-9

    def ctx_of(wav):
        y, _ = librosa.load(str(wav), sr=48000, mono=True, duration=8.0)
        av = clap.embed_audio_array(y.astype(np.float32), 48000); av /= np.linalg.norm(av) + 1e-9
        wt = _softmax(Lc @ av, 0.5)
        c = (wt[:, None] * Lq).sum(0); return c / (np.linalg.norm(c) + 1e-9), wt

    def sig(ctx):
        """de-biased per-symbol signature for a context vector."""
        space.set_context_vec(ctx)
        D1 = space.make_shifted_matrix(**dr.kw)
        space.set_context_vec(None)
        return dr.ws_z((D1 @ D1.T - dr.C0)[dr.iu])

    names = [s for s in ORDER if (AUD / f"{s}.wav").exists()]
    ctxs, wts = {}, {}
    Z = []
    for nm in names:
        c, wt = ctx_of(AUD / f"{nm}.wav"); ctxs[nm] = c; wts[nm] = wt
        Z.append(sig(c))
    Z = np.vstack(Z); mean_z = Z.mean(0); Zc = Z - mean_z

    # ---- family-level symbol-coupling subnetworks (panel A chords) ----
    from collections import OrderedDict
    fam_members = OrderedDict()
    for nm in names:
        fam_members.setdefault(FAMILY[nm], []).append(nm)
    fam_names = list(fam_members)
    nFam = len(fam_names); nS = len(dr.syms)
    fam_fp = []
    for f in fam_names:
        c = np.mean([ctxs[s] for s in fam_members[f]], 0); c /= np.linalg.norm(c) + 1e-9
        space.set_context_vec(c); D1 = space.make_shifted_matrix(**dr.kw); space.set_context_vec(None)
        fam_fp.append((D1 @ D1.T - dr.C0)[dr.iu])
    fam_fp = np.vstack(fam_fp); fam_mean = fam_fp.mean(0)
    KE = 45  # top descriptor-pair edges kept per family
    fam_M = []
    for fi in range(nFam):
        res = fam_fp[fi] - fam_mean
        Msym = np.zeros((nS, nS))
        for p in np.argsort(np.abs(res))[::-1][:KE]:
            a, b = int(dr.oi[p]), int(dr.oj[p])
            if a != b:
                Msym[a, b] += abs(res[p]); Msym[b, a] += abs(res[p])
        fam_M.append(Msym)

    SYMC = plt.cm.tab10(np.linspace(0, 1, nS))
    fig = plt.figure(figsize=(14.0, 7.0))
    gs = fig.add_gridspec(2, nFam, height_ratios=[1.05, 0.82], hspace=0.40,
                          wspace=0.10, top=0.86, bottom=0.10, left=0.035, right=0.965)

    # (A) one chord/ribbon diagram per family: which symbol couplings it
    # distinctively activates (ribbon width = magnitude of the family-specific
    # coupling, aggregated over the top descriptor-pair edges).
    sang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, nS, endpoint=False)
    SX, SY = np.cos(sang), np.sin(sang)
    for fi, f in enumerate(fam_names):
        ax = fig.add_subplot(gs[0, fi]); ax.axis("off"); ax.set_aspect("equal")
        Msym = fam_M[fi]; mx = Msym.max() + 1e-9
        for a in range(nS):
            for b in range(a + 1, nS):
                if Msym[a, b] <= 0:
                    continue
                xs = np.linspace(0, 1, 24)
                bx = (1 - xs)**2 * SX[a] + 2 * (1 - xs) * xs * 0 + xs**2 * SX[b]
                by = (1 - xs)**2 * SY[a] + 2 * (1 - xs) * xs * 0 + xs**2 * SY[b]
                ax.plot(bx, by, color=FAM_COL[f], lw=0.4 + 3.4 * Msym[a, b] / mx,
                        alpha=0.62, zorder=2, solid_capstyle="round")
        for s in range(nS):
            ax.scatter(SX[s], SY[s], s=50, color=SYMC[s], edgecolors="k",
                       linewidths=0.4, zorder=3)
            ax.text(1.26 * SX[s], 1.26 * SY[s], short_sym(dr.syms[s]), fontsize=5.0,
                    ha="center", va="center", color="0.2")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_title(f, color=FAM_COL[f], fontsize=10.5, pad=0, fontweight="bold")
    fig.text(0.5, 0.905, "(A) Each sound family activates a different subnetwork of symbol "
             "couplings   (ribbon width = strength of the family-specific coupling)",
             ha="center", fontsize=9)

    # (B) morphing between two sounds = continuous navigation of the field
    a, b = ("thunder", "fire") if {"thunder", "fire"} <= set(names) else (names[0], names[-1])
    ka, kb = names.index(a), names.index(b)
    alphas = np.linspace(0, 1, 21)
    # same residual as panel A (vs the sound mean), so endpoints match the heatmap
    M = np.vstack([sig((1 - t) * ctxs[a] + t * ctxs[b]) for t in alphas]) - mean_z
    # show each endpoint's two dominant archetypes (from panel A's Zc); center
    # among just these so the storm->hearth crossover is visible (not a swell)
    show = sorted(set(list(np.argsort(Zc[ka])[::-1][:2]) + list(np.argsort(Zc[kb])[::-1][:2])))
    Ms = M[:, show] - M[:, show].mean(axis=1, keepdims=True)
    axB = fig.add_subplot(gs[1, 1:max(2, nFam - 1)])
    pal = plt.cm.tab10(np.linspace(0, 1, 10))
    for c, s in enumerate(show):
        axB.plot(alphas, Ms[:, c], "-", lw=1.8, color=pal[s % 10],
                 label=short_sym(dr.syms[s]))
    axB.axhline(0, color="0.6", lw=0.6, ls=":")
    axB.set_xticks([0, 0.5, 1])
    axB.set_xticklabels([NICE.get(a, a).replace("\n", " "), "morph", NICE.get(b, b).replace("\n", " ")],
                        fontsize=7.5)
    axB.set_xlabel("blend  α", fontsize=8)
    axB.set_ylabel("relative archetype emphasis", fontsize=8)
    axB.tick_params(labelsize=6.5)
    axB.legend(fontsize=6.5, loc="best", framealpha=0.9, ncol=2)
    axB.set_title(f"(B) Morphing {NICE.get(a, a).replace(chr(10), ' ')} → "
                  f"{NICE.get(b, b).replace(chr(10), ' ')}:\n"
                  "the field is traversed continuously, not switched",
                  fontsize=8.6, pad=6)

    fig.suptitle("A creative interface: navigating the archetypal space with sound "
                 "(CLAP audio → acoustic blend → Δ readout)", fontsize=11.5, y=0.965)
    save_fig(fig, "fig_delta_soundscape")
    plt.close(fig)


if __name__ == "__main__":
    main()
