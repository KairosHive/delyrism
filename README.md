# DELYRISM

> Context-aware symbolic archetype explorer — watch meanings shift and relationships rewire when you give the engine a sentence, a sound, an image, or a blend of two contexts.

<p align="center">
  <img src="https://github.com/user-attachments/assets/76752b6c-6893-4eb8-b8c1-5d2670e7e5a0" width="720" alt="Meaning space with context shift arrows">
</p>

A FastAPI + Next.js stack. Slider tweaks are instant; backend memoises embeddings, UMAP layouts, and shift matrices so context changes don't trigger full recomputes. The previous Streamlit prototype lives on the [`old`](https://github.com/KairosHive/delyrism/tree/old) branch.

## Quickstart

```bash
conda create -n delyrism python=3.10 -y && conda activate delyrism
pip install -r requirements.txt -r web/backend/requirements.txt
cd web/frontend && npm install && cd ../..

# two terminals
uvicorn app.main:app --reload --port 8000 --app-dir web/backend
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm --prefix web/frontend run dev
# open http://localhost:3000
```

For Cloudflare-backed embedders + LLMs, set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. Or pick the local `sentence-transformer` backend in the sidebar for fully-offline use. Railway deployment: [`RAILWAY_SETUP.md`](RAILWAY_SETUP.md).

## What it does

Each Explorer panel is a different lens on the same shifted descriptor cloud:

| Panel | What it surfaces |
|---|---|
| **2D Meaning Space** | UMAP / t-SNE / PCA layout · symbol clusters · context-shift arrows · optional pull-intensity heatmap |
| **Rankings** | top archetypes for the current context (cosine coherence ⊕ personalized PageRank) |
| **Ambiguity** | dispersion / leakage / soft entropy per symbol — which archetypes blend, which stay sharp |
| **Descriptor attention** | softmax attention per archetype's descriptors under the current context |
| **Subgraph** | top-K archetypes + their top descriptors as a force-directed graph |
| **Within / between Δ heatmap** | descriptor-pair or symbol-pair cosine change (after − before) |
| **Δ-graph** | the headline relational view — strongest descriptor pairs whose similarity moved most |
| **Contextual transformations** | who migrated archetypes + per-archetype identity cards (before vs after top descriptors) |

## Context sources

Stack any combination. They all feed the same `v_ctx`:

- **Sentence** — free-text prompt in the Context Prompt card.
- **Symbol weights** — manually bias toward archetypes via sliders in the sidebar.
- **Image** — drop/paste an image; a Cloudflare vision LLM reads it as a short paragraph, then your text embedder encodes that paragraph (works with any text backend, no CLIP needed).
- **Audio** — upload or record (requires the CLAP embedder backend).
- **Alchemist mode** — two sentences A and B with a morph slider; the engine blends them server-side and every panel updates live as you drag.

## Shift strategies

The cloud rewriting is configurable. From the sidebar:

- **`gate`** — additive pull toward `v_ctx`, with per-descriptor gating (`relu` / `cos` / `softmax` / `uniform`). Fast, deterministic, default.
- **`pooling`** — convex blend (`avg`) or element-wise (`max` / `min`) toward `v_ctx`.
- **`reembed`** — re-encode each descriptor through the embedder with the context sentence prepended. Slowest, most semantically rich.
- **`hybrid`** — linear blend of `gate` and `reembed`.

Each strategy produces a different `D′` and every Δ panel updates accordingly. Rankings / Attention / Subgraph use the original `D` + `v_ctx` (they're a spotlight on a fixed landscape, not a landscape rewrite).

## Story generator

A separate tab. Weaves Δ-graph motifs into micro-fiction with:

- 15 tone presets (Pynchon, Borges, Calvino, Tarkovsky, García Márquez, Kafkaesque, etc.)
- 6 output forms (prose / short-story / poem / myth / incantation / vignette)
- Anchor archetype, motif source + density, language (EN / FR / ES), tense, POV
- Reads the same Δ-graph params you set in Explorer — motifs match what you see.

## Archetype Builder

A tab that talks to the Egregore companion service to compose new symbol sets from PDFs, images, or raw text.

## Programmatic API

The engine works standalone, without the web stack:

```python
from delyrism import SymbolSpace, TextEmbedder, context_delta_graph

embedder = TextEmbedder(backend="qwen3")
symbols = {"Fire": ["passion", "energy"], "Water": ["calm", "flow"]}
space = SymbolSpace(symbols, embedder, descriptor_threshold=0.2)

proposals = space.propose(sentence="intense transformation", topk=3, tau=0.3)
G = context_delta_graph(space, sentence="healing journey", top_abs_edges=20)
```

Full reference: [`FUNCTIONS_README.md`](FUNCTIONS_README.md).

## Symbol presets

Ten archetypal systems ship under [`delyrism/structures/`](delyrism/structures/) — Elements, Chakras, Jungian, Lakota, Mayan, Chinese Zodiac, Planets, Musical Modes, Sacred Architecture, Seasons of Life. Each is a `{symbol: [descriptors…]}` JSON, freely editable.

## Documentation

- [`web/README.md`](web/README.md) — frontend + backend architecture, dev setup
- [`RAILWAY_SETUP.md`](RAILWAY_SETUP.md) — single-service Docker deployment
- [`FUNCTIONS_README.md`](FUNCTIONS_README.md) — engine API reference
- [`delyrism/legacy/app.py`](delyrism/legacy/app.py) — original Streamlit prototype, kept for reference

Every sidebar control has a `?` tooltip — the parameter reference lives in the UI, not in this README.

## Use cases

- **Mythopoetic writing** — generate story seeds from Δ-graph motifs in 15 literary registers
- **Cultural / semiotic studies** — compare how cultural contexts shift symbolic meaning; build archetypes from historical or image corpora
- **Psychological exploration** — map Jungian-style archetypes; measure ambiguity and overlap; trace themes across audio transcripts
- **Music & audio analysis** — CLAP joint embeddings trace sound → archetype associations

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

```
@software{delyrism2026,
  title  = {Delyrism: Context-Aware Symbolic Archetype Explorer},
  author = {Bellemare, Antoine and contributors},
  year   = {2026},
  url    = {https://github.com/KairosHive/delyrism}
}
```

Built on Transformers, NetworkX, UMAP, CLAP, Plotly, Next.js. Inspired by archetypal psychology (Jung), distributional semantics, graph-based knowledge representation, and mythopoetic traditions.
