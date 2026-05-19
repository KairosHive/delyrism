# DELYRISM — Context-Aware Symbolic Archetype Explorer

A semantic engine for exploring symbolic relationships through **context-conditioned embeddings**, **graph diffusion**, and **attention mechanisms**. Watch how meanings shift and relationships reorganize when you provide textual, audio, or multimodal context.

> The app ships as a **FastAPI + Next.js** stack under [`web/`](web/). Slider tweaks are instant, plots are real interactive components, and the backend caches embeddings + UMAP layouts so context changes don't trigger full recomputes. See [`web/README.md`](web/README.md) and [`RAILWAY_SETUP.md`](RAILWAY_SETUP.md). The previous Streamlit app lives on the [`old`](https://github.com/KairosHive/delyrism/tree/old) branch.

<p align="center">
  <img src="https://github.com/user-attachments/assets/76752b6c-6893-4eb8-b8c1-5d2670e7e5a0" width="720" alt="Meaning space with context shift arrows">
</p>

## Quickstart

```bash
# 1. Create env
conda create -n delyrism python=3.10 -y && conda activate delyrism

# 2. Install
pip install -r requirements.txt -r web/backend/requirements.txt
cd web/frontend && npm install && cd ../..

# 3. Run — two terminals
uvicorn app.main:app --reload --port 8000 --app-dir web/backend
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm --prefix web/frontend run dev
# open http://localhost:3000
```

For Cloudflare-backed embeddings + story generation, set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. Or pick the local `sentence-transformer` backend in the sidebar for fully-offline use. For Railway deployment see [`RAILWAY_SETUP.md`](RAILWAY_SETUP.md).

## What it does

- **Symbol ranking** — context-aware ranking of archetypes combining cosine coherence with personalized PageRank diffusion through a descriptor graph.
- **Context shift** — four strategies (gate / reembed / pooling / hybrid) for moving descriptor embeddings under a given context. Visualize movement as 2D arrows, Δ-graph edges, and per-symbol Δ heatmaps.
- **Attention** — softmax over descriptor-context similarities, surfaced as per-symbol bar charts.
- **Ambiguity metrics** — dispersion / leakage / soft entropy per symbol so you can see which archetypes blend and which stay sharp.
- **Multi-modal context** — text, audio (CLAP), or manual symbol weights. Mix any combination.
- **Story generator** — weave delta-graph motifs into micro-fiction with 15 tone presets (Pynchon, Borges, Calvino, Tarkovsky, Garcia-Marquez, Kafkaesque, …), 6 output forms (prose / short-story / poem / myth / incantation / vignette), anchor archetype, motif source + density controls.
- **Archetype Builder** — compose new symbol sets from PDFs, images, or raw text via the Egregore companion service.

Every sidebar control has a `?` tooltip explaining what it does — the parameter reference lives there, not in this README.

## How it works

```
encode descriptors → build symbol↔descriptor graph (cosine threshold)
                  → encode context → softmax attention over descriptors
                  → shift descriptor embeddings (4 strategies)
                  → rank symbols: λ · cosine_coherence + (1−λ) · personalized_PageRank
                  → delta = S_after − S_before for the descriptor graph
```

The engine lives in [`delyrism/delyrism.py`](delyrism/delyrism.py) — the web layer (`web/backend/`) wraps it as HTTP routes with per-space caching. A parity test in [`web/tests/parity_test.py`](web/tests/parity_test.py) verifies every API endpoint returns the same numbers as direct engine calls.

## Use cases

- **Mythopoetic writing** — generate story seeds from delta-graph motifs in 15 literary registers.
- **Cultural / semiotic studies** — compare how cultural contexts shift symbolic meanings; compose archetypes from historical or image corpora.
- **Psychological exploration** — map Jungian-style archetypes; measure ambiguity and overlap; track therapeutic themes across audio transcripts.
- **Music & audio analysis** — CLAP joint embeddings let you trace sound → archetype associations.

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

Full function signatures in [`FUNCTIONS_README.md`](FUNCTIONS_README.md).

## Symbol presets

Ten archetypal systems ship under [`delyrism/structures/`](delyrism/structures/) — Elements, Chakras, Jungian, Lakota, Mayan, Chinese Zodiac, Planets, Musical Modes, Sacred Architecture, Seasons of Life. Each is a `{symbol: [descriptors...]}` JSON, freely editable. Drop your own JSON in there or compose new ones via the Archetype Builder tab.

## Documentation

- [`web/README.md`](web/README.md) — frontend + backend architecture, dev setup
- [`RAILWAY_SETUP.md`](RAILWAY_SETUP.md) — single-service Docker deployment
- [`FUNCTIONS_README.md`](FUNCTIONS_README.md) — engine API reference
- [`delyrism/structures/`](delyrism/structures/) — preset archetypal systems

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

Built on Transformers, NetworkX, UMAP, CLAP, Plotly, Next.js. Inspired by archetypal psychology (Jung), distributional semantics, graph-based knowledge representation, and mythopoetic traditions across cultures.
