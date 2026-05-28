# Paper figures

Scripts to generate every figure used in v2 of the paper.

## Quick start

```bash
# from the repo root
pip install -r requirements.txt
pip install ripser umap-learn matplotlib networkx   # if not already present

python paper/figures/make_all.py
```

Figures are written to `paper/v2/figures/` as PDF (paper-ready) + PNG (preview).

## Layout

| Script | Figure | Analysis | Notes |
|---|---|---|---|
| `fig_v1_umap.py`       | Symbolic field at rest — UMAP + ambiguity metrics | v1 Fig 1 (preserved) | Opening empirical figure. |
| `fig_v1_attention.py`  | Context-conditioned attention violins (EARTH × 4 themes) | v1 Fig 2 (preserved) | Pair with `fig01_delta_graph` in the paper. |
| `fig_v1_ppr.py`        | PPR contextual subgraph | v1 Fig 3 (preserved) | Different question than Δ-graph; both kept. |
| `fig01_delta_graph.py` | Context as relational rewiring — 3 contexts side-by-side | §6 Analysis 1 | Δ-graph centerpiece. |
| `fig02_topology.py`    | Topology intrinsic vs context-induced | §6 Analysis 2 | Needs `ripser`. |
| `fig03_phase_morphing.py` | Phase transitions in continuous morphing | §6 Analysis 3 | Sweeps α ∈ [0, 1]; slow (~21 PH computations). |
| `fig04_catalysts.py`   | Catalysts and structural integrity per archetype | §6 Analysis 4 | LOO-PH; produces a CSV companion. |
| `fig05_crossmodal.py`  | Cross-modal consistency (text/audio/image) | §6 Analysis 5 | Audio + image inputs need to be wired (placeholders in script). |

## Settings shared across scripts

- **Embedder backend.** Default `qwen3` (Qwen3-Embedding-0.6B) to match v1.
  Override via `--backend` (e.g. `cloudflare`, `sentence-transformer`).
- **Context-shift defaults.** Match the backend's standard `gate / relu /
  β=1.2 / τ=0.3 / within_symbol_softmax / γ=0.5 / pool_w=0.7`.  Change the
  `SHIFT_KW` dict at the top of each figure script if you want to ablate.
- **Anchor contexts.** Defined in `lakota_descriptors.py` (`C1 … C_scene`).
  Same set referenced in `paper/PLAN.md §6`.

## Caching

`_setup.build_space` pickles the constructed space (descriptor matrix + graph
+ centroids) under `paper/figures/.cache/space_<hash>.pkl`.  The hash includes
the descriptors dict and the embedder backend, so swapping descriptors or
embedders invalidates the cache automatically.

Delete `paper/figures/.cache/` to force a rebuild.

## Running individually

Each script is a single-purpose CLI:

```bash
python paper/figures/fig01_delta_graph.py
python paper/figures/fig02_topology.py --context C2
python paper/figures/fig04_catalysts.py --top 5
python paper/figures/fig05_crossmodal.py --audio recordings/scene.wav --image photos/scene.jpg
```

`-h` on any script prints its arguments.

## Wiring fig05 (cross-modal)

`fig05_crossmodal.py` ships with the text modality working end-to-end and
audio/image as `NotImplementedError` stubs.  To complete it:

1. Drop a WAV/MP3 recording matching `C_scene` (a thunderhead at dusk over a
   river) into a path you'll pass via `--audio`.
2. Drop a photograph matching the same scene into a path you'll pass via
   `--image`.
3. Implement `_embed_audio` and `_embed_image_via_caption` in the script —
   `web/backend/app/routes/context.py` (`/context/encode-audio`,
   `/context/encode-image`) is the production reference; either call the
   running backend over HTTP or replicate the CLAP / vision-LLM bridge inline.

The text-only baseline still produces a useful figure (just with a 1×1
agreement matrix).
