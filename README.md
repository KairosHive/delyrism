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

---

## 🧠 Technical Deep Dive

### How Context Conditioning Works

#### The Core Algorithm
1. **Encode** all descriptors into a semantic embedding space (using transformer models)
2. **Build** a bipartite graph: Symbols ↔ Descriptors + Descriptor-Descriptor edges (by cosine similarity threshold)
3. **Compute context vector** from input (text, audio, or weights)
4. **Apply attention** using softmax over descriptor-context similarities (temperature τ)
5. **Shift embeddings** using one of four strategies:
   - **Gate**: `D' = norm(D + β * gate(D·ctx) * ctx)`
   - **Reembed**: Re-encode descriptors with context prepended
   - **Pooling**: `D' = norm((1-w)*D + w*ctx)`
   - **Hybrid**: Blend gate + reembed results
6. **Rank symbols** by combining coherence and graph diffusion (λ blend parameter)

#### Personalized PageRank
- Inject probability mass at descriptor nodes proportional to their context similarity
- Diffuse through the graph (α damping controls locality)
- Aggregate symbol scores from descriptor scores
- Combine with direct coherence for final ranking

#### Delta Graph Construction
1. Compute descriptor similarity matrices before and after shift: `S_before`, `S_after`
2. Calculate delta: `Δ = S_after - S_before`
3. Extract top-K edges by `|Δ|` (absolute change)
4. Build NetworkX graph with Δ-weighted edges
5. Optionally filter by symbol, connectivity, or within-symbol only

### Parameter Reference

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **τ (tau)** | 0.01–1.0 | 0.3 | Attention sharpness (lower = more focused) |
| **β (beta)** | 0.0–2.0 | 0.6 | Shift strength for gate strategy |
| **α (alpha)** | 0.0–1.0 | 0.85 | PageRank damping (higher = more diffusion) |
| **λ (lambda)** | 0.0–1.0 | 0.6 | Coherence vs. diffusion weight |
| **descriptor_threshold** | 0.0–1.0 | 0.2 | Min cosine similarity for descriptor edges |
| **membership_alpha** | 0.0–1.0 | 0.0 | Blend shift by symbol membership |
| **top_abs_edges** | 5–500 | 30 | Number of Δ edges to visualize |

### Embedding Backends Compared

| Backend | Speed | Quality | GPU | Audio | Notes |
|---------|-------|---------|-----|-------|-------|
| **Qwen3** | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Optional | ❌ | Best for CPU, int8 quantization |
| **Cloudflare** | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | N/A | ❌ | Zero local compute, requires API key |
| **SentenceTransformers** | ⚡⚡ | ⭐⭐⭐⭐ | Recommended | ❌ | Classic, broad model support |
| **CLAP** | ⚡⚡ | ⭐⭐⭐⭐ | Recommended | ✅ | Audio-text joint space |
| **AudioCLIP** | ⚡⚡⚡ | ⭐⭐⭐ | Optional | ✅ | Lightweight audio via OpenCLIP |

---

## 🚀 Advanced Features

### Membership Alpha Blending
Control how much context shifts respect symbol boundaries:
```python
space.make_shifted_matrix(
    sentence="your context",
    membership_alpha=0.5  # 0=full shift, 1=no shift
)
```
- **0.0**: Descriptors can drift arbitrarily far (explore novel connections)
- **1.0**: Descriptors stay within symbol clusters (preserve structure)
- **0.3–0.7**: Balanced (recommended range)

### Within-Symbol Softmax
Apply attention separately per symbol (useful when symbols have very different descriptor counts):
```python
space.propose(
    sentence="context",
    tau=0.3,
    within_symbol_softmax=True  # Each symbol's descriptors compete only with each other
)
```

### Dual Context Comparison
The app supports **Context A** and **Context B** modes for side-by-side analysis:
- Compare how different prompts affect the same symbol space
- Visualize trajectory between two semantic positions
- Track which descriptors shift most between contexts

### Audio Context Integration
1. Set embedder backend to `clap` or `audioclip`
2. Upload audio file or record live
3. System embeds audio into the same space as text
4. Override context vector with audio embedding
5. All visualizations now show audio-conditioned shifts

### JSON Import/Export
Save and load complete symbol spaces with metadata:
```json
{
  "symbols": {
    "Fire": ["passion", "energy", "transformation"],
    "Water": ["flow", "healing", "depth"]
  },
  "metadata": {
    "source": "custom",
    "created": "2026-01-15",
    "descriptor_threshold": 0.2
  }
}
```

### Programmatic API
Use the core engine without Streamlit:
```python
from delyrism import SymbolSpace, TextEmbedder

# Initialize
embedder = TextEmbedder(backend="qwen3", model="Qwen/Qwen3-Embedding-0.6B")
symbols = {"Fire": ["passion", "energy"], "Water": ["calm", "flow"]}
space = SymbolSpace(symbols, embedder, descriptor_threshold=0.2)

# Get proposals
proposals = space.propose(sentence="intense transformation", topk=3, tau=0.3)

# Get delta graph
from delyrism import context_delta_graph, plot_delta_graph
G = context_delta_graph(space, sentence="healing journey", top_abs_edges=20)
fig = plot_delta_graph(G, space, palette="Nord")
```

---

## 🎯 Use Cases

### 1. Mythopoetic Writing & Worldbuilding
- Explore symbolic associations for narrative themes
- Generate story seeds from delta graph motifs
- Track character archetypes across different emotional contexts
- Multi-language myth generation with tone presets

### 2. Psychological Analysis
- Map Jungian archetypes with contextual activation
- Visualize how life experiences shift symbolic importance
- Track therapeutic themes across session transcripts (audio)
- Measure archetype ambiguity and overlap


### 3. Cultural Studies & Semiotics
- Analyze how cultural contexts shift symbolic meanings
- Compare symbol systems across languages/cultures
- Mine archetypes from historical texts or image corpora
- Visualize semantic evolution over time

### 4. Music & Audio Analysis
- Use CLAP embeddings to explore audio-symbolic relationships
- Map musical qualities to archetypal descriptors
- Generate narratives from audio input
- Discover cross-modal metaphors (sound → symbol)

---

## 🛠️ Performance Optimization

### For Low-Resource Environments
```bash
# Use Cloudflare for embeddings (no GPU needed)
# In app: Select "Cloudflare" backend

# OR use quantized local model
# Qwen3 auto-applies int8 quantization on CPU
```

### For Speed
- Reduce `descriptor_threshold` to create sparser graphs (0.15–0.2 is good)
- Use PCA instead of UMAP for faster 2D projections
- Batch process embeddings (automatically handled)
- Cache symbol spaces (use Export/Save feature)

### For Quality
- Use `strategy="hybrid"` for richest semantic shifts
- Lower τ to 0.1–0.2 for sharper attention
- Increase `top_abs_edges` to 50–100 for comprehensive delta graphs
- Use Qwen3 or sentence-transformers for embeddings

### For Large Symbol Sets (100+ symbols)
- Increase `descriptor_threshold` to 0.25–0.3
- Use `connected_only=True` in delta graph
- Apply `symbol_filter` to focus on subsets
- Consider mining sub-spaces with Egregore

---

## 📦 Dependencies

### Core (required)
```
torch >= 2.0
transformers >= 4.35
sentence-transformers >= 2.2
numpy >= 1.23
networkx >= 3.0
scikit-learn >= 1.2
matplotlib >= 3.6
streamlit >= 1.28
```

### Optional (for full features)
```
umap-learn >= 0.5      # for UMAP projections
scipy >= 1.9           # for convex hulls
open-clip-torch >= 2.23  # for AudioCLIP/image
laion-clap >= 1.1      # for CLAP audio
librosa >= 0.10        # for audio processing
Pillow >= 9.0          # for image handling
streamlit-mic-recorder  # for live audio recording
```

### For Egregore (archetype mining)
```
hdbscan >= 0.8
PyPDF2 or pypdf        # for PDF parsing
openai                 # for LLM refinement (optional)
```

---

## 🔧 Configuration

### Cloudflare Workers AI Setup
Create `.streamlit/secrets.toml`:
```toml
[cloudflare]
account_id = "your_cloudflare_account_id"
api_token = "your_cloudflare_api_token"
```
Or set environment variables:
```bash
export CLOUDFLARE_ACCOUNT_ID="your_account_id"
export CLOUDFLARE_API_TOKEN="your_api_token"
```

### OpenAI API (for Egregore LLM refinement)
```toml
[openai]
api_key = "your_openai_api_key"
```

### Custom Symbol Sets
Place JSON files in `delyrism/structures/`:
```json
{
  "SymbolName": ["descriptor1", "descriptor2", "descriptor3"],
  "AnotherSymbol": ["desc_a", "desc_b"]
}
```

---

## 📚 Documentation

- **[FUNCTIONS_README.md](FUNCTIONS_README.md)**: Complete API reference with all function signatures
- **[Example Notebooks](notebooks/)**: Jupyter notebooks with detailed workflows
- **[Preset Symbol Sets](delyrism/structures/)**: Pre-built archetypal systems

---

## 🤝 Contributing

Delyrism is experimental research code. Contributions welcome:
- New embedding backends
- Additional visualization methods
- Symbol set templates
- Performance optimizations
- Documentation improvements

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

Built with:
- **Transformers** (Hugging Face) for semantic embeddings
- **NetworkX** for graph algorithms
- **Streamlit** for interactive interface
- **UMAP** for dimensionality reduction
- **CLAP/AudioCLIP** for multimodal embeddings

Inspired by concepts from:
- Symbolic cognition & archetypal psychology (Jung)
- Distributional semantics & word embeddings
- Graph-based knowledge representation
- Attention mechanisms in transformers
- Mythopoetic traditions across cultures

---

## 📬 Citation

If you use Delyrism in your research or creative work, please cite:
```
@software{delyrism2026,
  title={Delyrism: Context-Aware Symbolic Archetype Explorer},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/delyrism}
}
```

---

**Explore the liminal space where meaning shifts and archetypes dance.** 🌀
