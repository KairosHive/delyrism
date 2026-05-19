# Delyrism — new web stack (FastAPI + Next.js)

A reactive rewrite of the Streamlit app at the repo root. Same engine
(`delyrism/delyrism.py`), same capabilities, same default parameters — just
without the full-page reruns that Streamlit forces on every slider tick.

```
web/
  backend/       FastAPI server that wraps SymbolSpace / TextEmbedder
  frontend/      Next.js 14 (App Router) + TanStack Query + Zustand + Tailwind
  tests/         parity_test.py — direct-engine vs HTTP-API comparison
```

## Local development

### 1. Backend

```bash
# from repo root
pip install -r web/backend/requirements.txt
# the backend reuses everything else from the root requirements.txt
pip install -r requirements.txt        # engine deps (torch, transformers, …)

uvicorn app.main:app --reload --port 8000 \
  --app-dir web/backend
```

Routes are documented at `http://localhost:8000/docs`.

### 2. Frontend

```bash
cd web/frontend
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
# open http://localhost:3000
```

### 3. (Optional) Egregore builder

```bash
uvicorn miner_server:app --port 8765 --app-dir delyrism
# then the Archetype Builder tab will embed it via iframe
```

## Parity verification

```bash
python -m web.tests.parity_test
```

Builds a small SymbolSpace twice — once via the engine directly, once via the
HTTP API — and verifies that `/propose`, `/attention`, `/ambiguity`, `/shift`
and `/delta-graph` return identical numerical values within float tolerance.
All 47 checks pass on `sentence-transformer / all-MiniLM-L6-v2`.

## How reactivity works

The Streamlit app re-runs the entire 4000-line script on every widget change.
The new stack splits work into expensive and cheap operations:

| Operation                                  | Cost   | Triggered by                     | Cache    |
|--------------------------------------------|--------|----------------------------------|----------|
| `SymbolSpace` construction (embed all desc) | high   | "Build space" button             | server   |
| `propose`, `ambiguity`, `reduce-2d`        | medium | sentence / weight / param change | server (per-key) |
| Plot styling, top-K, color, hover, panning | zero   | client only                      | n/a      |

TanStack Query keys are derived from the exact subset of sidebar state each
endpoint cares about, so flipping "Draw convex hulls" doesn't re-query
`/propose`, and changing τ doesn't re-fetch the 2D projection.

## Endpoint surface

```
POST /spaces                  build (or reuse) a SymbolSpace
GET  /spaces/presets          list bundled JSON presets
GET  /spaces/presets/{name}   load a bundled preset
POST /propose                 ranked symbols
POST /attention               descriptor attention for a symbol
POST /ambiguity               dispersion / leakage / soft entropy
POST /reduce-2d               UMAP / t-SNE / PCA projection
POST /shift                   2D arrows from context-shift strategies
POST /delta-graph             top-|Δ| descriptor pair edges
POST /subgraph                top-K symbols + their top-M descriptors
POST /context/encode-text     embed a sentence
POST /context/encode-audio    embed an uploaded audio file (CLAP/AudioCLIP)
POST /context/set-override    push an external context vector into the space
POST /story/generate          Cloudflare Workers AI story generation
GET  /story/models            list supported provider models
GET  /builder                 Egregore service URL for the frontend iframe
GET  /miner                   (legacy alias for /builder)
GET  /backends                embedder catalog for the sidebar dropdown
GET  /healthz                 health probe
```
