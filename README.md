# DELYRISM — Context-Aware Symbolic Archetype Explorer

A semantic engine for exploring symbolic relationships through **context-conditioned embeddings**, **graph diffusion**, and **attention mechanisms**. Watch how meanings shift and relationships reorganize when you provide textual, audio, or multimodal context.

## Core Capabilities

### 🎯 Symbol Ranking & Proposals
- **Top Symbols**: Context-aware ranking combining semantic coherence with graph-based diffusion (Personalized PageRank)
- **Top Descriptors**: Extract the most semantically relevant descriptors for each symbol under a given context
- **Dual Scoring**: Blend coherence (direct similarity) and structural importance (graph connectivity)
- **Blind Spot Mode**: Discover symbols *least* aligned with your context (explore opposites)

### 🔍 Context Input Modalities
- **Text**: Natural language prompts and queries
- **Audio**: Upload files or record live (mic) using CLAP embeddings
- **Image**: Visual context via OpenCLIP (experimental)
- **Symbol Weights**: Manual weight sliders to bias exploration
- **Hybrid**: Combine multiple modalities simultaneously

### 📊 Visualization Suite
- **2D Meaning Space**: UMAP/t-SNE/PCA projections with context-shift vectors showing semantic movement
- **Descriptor Attention Heatmaps**: Per-symbol attention weights across multiple contexts
- **Contextual Subgraph**: Network view of top-k symbols and their strongest descriptor connections
- **Delta (Δ) Graph**: Visualize how descriptor-descriptor relationships strengthen or weaken under context
- **Symbol Similarity Heatmaps**: Before/after/delta matrices showing structural reorganization
- **Ambiguity Metrics**: Dispersion, leakage, and soft entropy for each symbol

### ⚡ Context Shift Strategies
- **Gate**: Fast additive shift weighted by descriptor-context similarity
- **Reembed**: Semantically rich re-encoding with context prepended
- **Pooling**: Geometric interpolation between descriptors and context
- **Hybrid**: Blend multiple strategies for optimal results

### 🎨 Generative Storytelling
- **Multi-Model Support**: Cloudflare Workers AI (Llama, Qwen, Mistral) and local Gemma models
- **Tone Presets**: Pynchon, Blake, Mystic-Baroque, Gnostic-Techno styles
- **Multi-Language**: Generate in English, French, or Spanish
- **Auto-Prompt**: Extract motifs from delta graph for narrative seeds

### 🏗️ Egregore - Archetype Miner
- **Unsupervised Discovery**: Mine archetypes from PDFs, images, or raw text
- **Semantic Clustering**: HDBSCAN + UMAP for pattern detection
- **LLM Refinement**: Optional GPT-based naming and descriptor extraction
- **Interactive Builder**: Visual space exploration with cluster selection

---

## Quickstart
```bash
# create env (recommended)
conda create -n delyrism python=3.10 -y
conda activate delyrism

# install
pip install -r requirements.txt

# run
streamlit run app.py
```
## How to Use

### 1️⃣ Define Your Symbol Space
- **Upload JSON**: Load a pre-defined `{symbol: [descriptors...]}` structure
- **Use Presets**: Start with included sets (Elements, Chakras, Jungian, etc.)
- **Mine New Archetypes**: Use Egregore to discover symbols from your corpus

### 2️⃣ Choose Embedding Backend
- **Qwen3** (recommended): Fast, high-quality embeddings with optional context prompting
- **Cloudflare**: Zero local GPU usage, API-based (requires credentials)
- **SentenceTransformers**: Classic approach, broad model support
- **CLAP**: Audio-text joint embeddings (required for audio context)
- **AudioCLIP**: Alternative audio backend via OpenCLIP

### 3️⃣ Provide Context
- **Text Prompt**: Natural language query (e.g., "transformation through struggle")
- **Audio Input**: Upload `.wav`/`.mp3` or record live with microphone
- **Symbol Weights**: Manually adjust sliders to emphasize specific archetypes
- **Mix All Three**: Combine text + audio + weights for rich contextual queries

### 4️⃣ Explore Outputs
Navigate through the tabbed interface:
- **Meaning Space**: 2D map with shift arrows
- **Rankings**: Top symbols and their scores breakdown
- **Attention**: Descriptor importance per symbol
- **Subgraph**: Network of relationships
- **Δ Graph**: Edge changes under context
- **Metrics**: Ambiguity and structural analysis

### 5️⃣ Fine-Tune Parameters
- **τ (tau)**: Attention temperature (lower = sharper focus)
- **β (beta)**: Shift strength for gate strategy
- **α (alpha)**: PageRank damping (higher = more local diffusion)
- **λ (lambda)**: Coherence/diffusion blend weight
- **Strategy**: Gate, reembed, pooling, or hybrid
- **Top-K**: Number of symbols/edges to display

## 📊 Visual Outputs - Complete Guide

### 1) 2D Meaning Space with Context Shifts
**Purpose**: Visualize the semantic landscape of all descriptors and observe how context pulls meanings in specific directions.

**What you see**:
- Colored scatter points (descriptors grouped by symbol)
- Stars (symbol centroids)
- Convex hulls around symbol clusters
- **Arrows** showing movement from base position to context-shifted position

**Controls**: Choose UMAP, t-SNE, or PCA; toggle hulls and centroids; adjust visualization style

<p align="center">
  <img src="https://github.com/user-attachments/assets/76752b6c-6893-4eb8-b8c1-5d2670e7e5a0" width="720" alt="Latent Map and Contextual Shifts">
</p>

---

### 2) Top Symbols for Context
**Purpose**: Rank symbols by relevance to your context using a hybrid scoring system.

**What you see**:
- Bar chart with composite scores (coherence + PageRank blend)
- Breakdown showing coherence vs. diffusion contributions
- Top descriptors for each ranked symbol with attention weights

**How it works**: 
- **Coherence**: Direct cosine similarity between context and symbol centroid
- **PageRank**: Graph-based diffusion measuring structural importance
- **λ parameter**: Controls the blend (λ=1 → pure coherence, λ=0 → pure diffusion)

---

### 3) Descriptor Attention Heatmap
**Purpose**: Show which descriptors within a symbol are most activated by different contexts.

**What you see**:
- Rows = contexts, Columns = descriptors of chosen symbol
- Color intensity = attention weight (softmax over descriptor-context similarity)
- Ordered by attention strength for easy interpretation

**Use case**: Compare how different prompts highlight different facets of the same archetype

<p align="center">
  <img src="https://github.com/user-attachments/assets/66b7f06f-947e-4dca-ba44-08a66d9b5ec7" width="720" alt="Descriptor Attention per Symbol">
</p>

---

### 4) Symbol Ambiguity Metrics
**Purpose**: Quantify structural properties of each symbol in the embedding space.

**Metrics explained**:
- **Dispersion**: Internal diversity (avg distance between symbol's own descriptors)
- **Leakage**: Semantic overlap with other symbols (fraction of k-NN outside symbol boundary)
- **Soft Entropy**: Ambiguity across symbols (entropy of softmax distribution over centroids)

**What to look for**: High leakage + high entropy = symbol blends with others; high dispersion = broad/multifaceted symbol

<p align="center">
  <img src="https://github.com/user-attachments/assets/2790e5b0-6cb5-4a7a-a45f-a84c229d6de7" width="720" alt="Symbol Ambiguity Metrics">
</p>

---

### 5) Contextual Subgraph (Network View)
**Purpose**: Extract and visualize a focused subgraph centered on the most context-relevant symbols.

**What you see**:
- Nodes = top-k symbols + their top-m descriptors
- Edges = symbol-descriptor memberships and descriptor-descriptor similarities
- Color-coded by symbol with layout options (spring, kamada-kawai, circular)

**Controls**: Adjust number of symbols and descriptors, choose layout algorithm, filter by symbol

**Use case**: Understand the local semantic neighborhood of your query

---

### 6) Within-Symbol Associative Increase (Δ Heatmap)
**Purpose**: Visualize how descriptor-descriptor similarity matrices change under context shift.

**What you see**:
- Three heatmaps: Before, After, and **Delta (change)**
- Rows/Columns = descriptors of selected symbol (ordered by attention)
- Warm colors (red) = increased similarity; cool colors (blue) = decreased similarity

**Interpretation**: Positive deltas show which descriptor pairs become more associated under your context

<p align="center">
  <img src="https://github.com/user-attachments/assets/bf07f76d-a69f-4d9a-b773-93dad2a59c1a" width="420" alt="Within-Symbol Associative Increase">
</p>

---

### 7) Context Δ (Delta) Graph
**Purpose**: Network visualization of the **strongest edge changes** across the entire descriptor space.

**What you see**:
- Nodes = descriptors (colored by parent symbol)
- Edges = top-N changes in cosine similarity (Δ = After - Before)
- Edge thickness/color = magnitude and direction of change

**Controls**:
- **top_abs_edges**: How many edges to show (ranked by |Δ|)
- **within_symbol**: Show only intra-symbol edges vs. all edges
- **symbol_filter**: Focus on specific symbols
- **connected_only**: Remove isolated nodes

**Use case**: Discover which semantic connections strengthen or weaken under your context; extract motifs for narrative generation

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ba762bc-c98d-4b39-bbba-2a2f4fedd5e6" width="720" alt="Context Delta Graph">
</p>

---

### 8) Symbol Similarity Heatmaps (Additional View)
**Purpose**: Compare pairwise symbol relationships before and after context application.

**What you see**:
- Matrix of symbol-symbol cosine similarities
- Side-by-side Before/After comparison
- Delta matrix highlighting relationship changes

**Use case**: Track how context reshapes the macro-structure of your symbol space


