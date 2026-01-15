# Delyrism: Symbol Space Explorer - Complete Function Reference

## Overview

Delyrism is a context-aware symbolic archetype exploration system that combines semantic embeddings, graph-based reasoning, and interactive visualization to explore symbolic relationships and generate mythopoetic narratives.

The system consists of two main components:
- **delyrism.py**: Core engine for semantic symbol space operations
- **app.py**: Streamlit-based interactive web interface

---

## Core Concepts

### Symbol Space
A **SymbolSpace** is a knowledge graph where:
- **Symbols** are high-level archetypes (e.g., Fire, Water, Jungian archetypes)
- **Descriptors** are semantic attributes that define each symbol
- **Context** can be provided via text or weighted symbols to shift the semantic space
- **Embeddings** capture semantic relationships using transformer models

### Context Conditioning
The system allows you to:
1. Provide a **text sentence** as context (e.g., "passion and transformation")
2. Assign **weights** to symbols (e.g., Fire=0.8, Water=0.2)
3. Combine both to create a rich contextual query
4. Use **audio** or **image** inputs (via CLIP/CLAP) as context

---

## delyrism.py - Core Engine

### 1. TextEmbedder Class

**Purpose**: Flexible text/audio/image embedding backend supporting multiple model architectures.

#### Backends Supported:
- **sentence-transformer**: Original SentenceTransformer models
- **qwen2/qwen3**: Qwen embedding models with EOS pooling
- **cloudflare**: Cloudflare Workers AI embedding API
- **audioclip**: AudioCLIP for text+audio joint embeddings
- **clap**: LAION CLAP for audio-text embeddings

#### Key Methods:

##### `__init__(backend, model, pooling, device, ...)`
Initialize embedder with specified backend.

**Parameters:**
- `backend`: Model backend type
- `model`: Specific model name/path
- `pooling`: Pooling strategy ("eos", "mean", "cls", "last")
- `device`: "cuda" or "cpu"
- `default_instruction`: Optional instruction prompt
- `default_context`: Optional default context

##### `encode(texts, instruction=None, context=None, batch_size=32)`
Encode text strings into semantic vectors.

**Parameters:**
- `texts`: List of strings to embed
- `instruction`: Optional instruction override
- `context`: Optional context override (can be per-text or global)
- `batch_size`: Batch size for processing

**Returns:** numpy array of shape `(n_texts, embedding_dim)`

##### `embed_audio_array(wave, sr)`
Embed audio waveform using CLAP or AudioCLIP backend.

**Parameters:**
- `wave`: Audio waveform as numpy array
- `sr`: Sample rate

**Returns:** Normalized embedding vector

---

### 2. SymbolSpace Class

**Purpose**: Core container for symbol-descriptor relationships with context-aware operations.

#### Initialization

##### `__init__(symbols_to_descriptors, embedder, descriptor_threshold, contextual_embeddings)`

**Parameters:**
- `symbols_to_descriptors`: Dict mapping symbol names to lists of descriptors
- `embedder`: TextEmbedder instance
- `descriptor_threshold`: Cosine threshold for descriptor-descriptor edges (default: 0.2)
- `contextual_embeddings`: If True, re-embed descriptors with context (default: False)

#### Context Management

##### `set_context_vec(vec)`
Manually override context vector (useful for audio/image inputs).

**Parameters:**
- `vec`: Numpy array or None to clear override

##### `ctx_vec(weights=None, sentence=None, ignore_override=False)`
Compute combined context vector from weights and/or sentence.

**Returns:** Normalized context vector

##### `conditioned_symbol(symbol, weights=None, sentence=None, tau=0.3)`
Get context-conditioned embedding and attention weights for a symbol.

**Parameters:**
- `symbol`: Symbol name
- `weights`: Optional dict of symbol weights
- `sentence`: Optional context sentence
- `tau`: Softmax temperature for attention

**Returns:** Tuple of (embedding_vector, attention_dict)

#### Symbol Proposal/Ranking

##### `propose(weights=None, sentence=None, topk=5, tau=0.3, lam=0.6, alpha=0.85, use_ppr=True, blind_spot=False)`
Propose top-k symbols given context, combining coherence and graph diffusion.

**Parameters:**
- `weights`: Symbol weights dict
- `sentence`: Context sentence
- `topk`: Number of symbols to return
- `tau`: Attention temperature
- `lam`: Blend factor (0=all diffusion, 1=all coherence)
- `alpha`: PageRank damping factor
- `use_ppr`: Enable Personalized PageRank
- `blind_spot`: If True, return lowest-scoring symbols

**Returns:** List of tuples `(symbol, combined_score, coherence, pagerank_score)`

##### `top_symbols_and_descriptors(sentence, method='ppr', topk_symbols=3, topk_desc=3, alpha=0.85, tau=0.02)`
Get top symbols and their most relevant descriptors.

**Parameters:**
- `sentence`: Context sentence
- `method`: "ppr" (PageRank) or "softmax" (attention)
- `topk_symbols`: Number of symbols
- `topk_desc`: Number of descriptors per symbol
- `alpha`: PageRank damping
- `tau`: Temperature

**Returns:** List of dicts with symbol, score, best_descriptors, descriptor_scores

#### Context Shifting

##### `make_shifted_matrix(weights=None, sentence=None, strategy='gate', beta=0.6, gate='relu', tau=0.5, ...)`
Create context-shifted descriptor embedding matrix.

**Parameters:**
- `weights`, `sentence`: Context specification
- `strategy`: "gate", "reembed", "pooling", or "hybrid"
  - **gate**: Additive shift weighted by similarity
  - **reembed**: Re-encode descriptors with context prepended
  - **pooling**: Weighted average with context vector
  - **hybrid**: Blend gate + reembed
- `beta`: Shift strength for gate/hybrid
- `gate`: Gate function ("relu", "cos", "softmax", "uniform")
- `tau`: Softmax temperature
- `within_symbol_softmax`: Apply softmax within each symbol separately
- `gamma`: Reembed blend weight for hybrid
- `pool_type`: "avg", "max", or "min" for pooling strategy
- `pool_w`: Context weight for avg pooling (0..1)
- `membership_alpha`: Blend original and shifted by symbol membership

**Returns:** Context-shifted descriptor matrix

##### `descriptor_similarity_matrices(weights=None, sentence=None, strategy='gate', order_by_attention=True, ...)`
Compute before/after/delta similarity matrices for each symbol.

**Returns:** Dict mapping symbol to:
- `descriptors`: Ordered descriptor names
- `S_before`: Similarity matrix before shift
- `S_after`: Similarity matrix after shift
- `S_delta`: Change in similarities

#### Metrics

##### `dispersion(symbol)`
Measure internal diversity of a symbol (avg cosine distance between its descriptors).

##### `leakage(symbol, k=10)`
Measure semantic overlap with other symbols (fraction of k-NN outside symbol).

##### `soft_entropy(symbol, tau=0.5)`
Measure ambiguity across symbols (softmax entropy over symbol centroids).

#### Visualization

##### `plot_map(method='umap', n_neighbors=15, with_hulls=True, include_centroids=True, figsize=(8,6), title='Descriptor map')`
2D projection of descriptor space with symbol-colored clusters.

**Parameters:**
- `method`: "umap", "tsne", or "pca"
- `n_neighbors`: UMAP neighbors parameter
- `with_hulls`: Draw convex hulls around symbol clusters
- `include_centroids`: Show symbol centroids as stars

**Returns:** Matplotlib figure

##### `plot_map_shift(weights=None, sentence=None, strategy='gate', method='umap', figsize=(14,5), ...)`
Side-by-side before/after comparison of context shift.

##### `plot_symbol_similarity_heatmaps(simdict, symbol, vmax=1.0, figsize=(13,4), only_delta=True)`
Heatmaps showing descriptor similarity changes.

##### `plot_attention(symbol, weights=None, sentence=None, tau=0.3, topk=10, figsize=(8,5), title=None)`
Bar chart of descriptor attention weights for a symbol under context.

##### `plot_symbol_predictions(weights=None, sentence=None, tau=0.5, lam=0.6, alpha=0.85, topk=None, figsize=(7,4), use_ppr=True)`
Bar chart of symbol proposal scores.

##### `plot_weight_sweep(symbol, weight_symbol, weight_range=(0,1,11), sentence=None, metric='coherence', tau=0.3, figsize=(8,5))`
Sweep one symbol's weight and plot how it affects another symbol's metric.

##### `plot_graph(node_size=600, font_size=10, context_symbols=None, figsize=(12,8))`
NetworkX visualization of the symbol-descriptor graph.

---

### 3. Standalone Plotting Functions

#### Context Trajectory Analysis

##### `plot_symbol_context_trajectory(space, symbol, sentences, tau=0.3, method='pca', figsize=(7,5), title=None)`
Show how a symbol's embedding moves through context space across different sentences.

##### `plot_symbol_centroid_shifts(space, symbol, sentences, tau=0.3, method='pca', figsize=(8,6), title=None, context_labels=None)`
Visualize symbol centroid shifts with arrows.

##### `plot_contexts_and_symbol_centroid(space, symbol, sentences, tau=0.3, method='pca', figsize=(7,5), context_labels=None)`
Plot context vectors and symbol centroid in reduced space.

##### `plot_centroid_context_plane(space, symbol, sentences, tau=0.3, figsize=(7,5), context_labels=None)`
Project onto the plane spanned by centroid and context vectors.

#### Meaning Landscapes

##### `plot_meaning_landscape(space, symbol, sentences, context_labels=None, method='pca', grid_res=110, sigma=0.19, figsize=(10,6))`
Heatmap showing "energy" landscape of meaning space.

##### `plot_meaning_energy_3d(space, method='pca', grid_res=220, sigma=0.14, figsize=(12,8))`
3D energy landscape across all symbols.

##### `plot_descriptor_attention_heatmap(space, symbol, contexts, context_labels=None, tau=0.1)`
Heatmap of descriptor attention across multiple contexts.

#### Delta Graph Analysis

##### `context_delta_graph(space, weights=None, sentence=None, strategy='gate', beta=0.6, gate='relu', tau=0.5, top_abs_edges=30, symbol_filter=None, within_symbol=True, connected_only=False, membership_alpha=0.0, ...)`
Build a graph of descriptor similarity changes (Δ edges).

**Parameters:**
- `top_abs_edges`: Keep only top N edges by |Δ|
- `symbol_filter`: List of symbols to include (None = all)
- `within_symbol`: If True, only edges within same symbol
- `connected_only`: If True, only keep largest connected component
- `membership_alpha`: Blend by symbol membership

**Returns:** NetworkX graph with Δ-weighted edges

##### `plot_delta_graph(G, space, layout='spring', node_size=500, edge_width_scale=5.0, edge_alpha_scale=0.85, figsize=(10,7), palette='Nord', title=None)`
Visualize the delta graph with colored nodes by symbol.

##### `plot_contextual_subgraph_colored(space, weights=None, sentence=None, topk=5, k_desc=3, strategy='gate', beta=0.6, gate='relu', tau=0.5, palette='Nord', layout='spring', figsize=(10,7), title=None, ...)`
Extract and visualize a focused subgraph around top-k symbols.

#### Ambiguity Analysis

##### `plot_ambiguity_metrics(space, symbols=None, figsize=(14,4))`
Show dispersion, leakage, and entropy for symbols.

---

## app.py - Streamlit Web Interface

### Main Interface Sections

#### 1. Symbol & Embedder Configuration
- Load/save symbol sets from JSON
- Choose embedding backend and model
- Configure descriptor threshold

#### 2. Context Input
- Text sentence input
- Symbol weight sliders
- Audio upload (mic recording or file)
- Image upload (with CLIP)
- Context vector visualization

#### 3. Archetype Proposals
- Top-k symbol proposals with scores
- Coherence vs. PageRank breakdown
- Descriptor attention heatmaps
- "Blind spot" mode (explore opposites)

#### 4. Shift Visualization
- Before/after descriptor maps
- Delta graph (similarity changes)
- Contextual subgraph
- Symbol trajectory plots

#### 5. Narrative Generation
- Multi-model support:
  - **Cloudflare Workers AI**: Llama, Qwen, Mistral, Gemma
  - **Local Gemma**: 2B/3B/3n models
- Tone presets: Pynchon, Blake, Mystic-Baroque, Gnostic-Techno
- Multi-language: English, French, Spanish
- Auto-prompt from top motifs

---

### Key Helper Functions in app.py

#### Narrative Generation

##### `build_gemma_prompt(context_sentence, motifs, tone='dreamy', pov='first', tense='present', target_words=(120,180), language='English')`
Build chat-style prompt for narrative generation.

**Parameters:**
- `context_sentence`: Context string
- `motifs`: List of symbols/descriptors to weave
- `tone`: Stylistic tone (simple or preset name)
- `pov`: "first" or "third" person
- `tense`: "present" or "past"
- `target_words`: Min/max word count
- `language`: "English", "French", or "Spanish"

**Returns:** List of chat messages

##### `generate_with_cloudflare(messages, model='@cf/meta/llama-3.1-8b-instruct', max_tokens=256, temperature=0.8, top_p=0.9)`
Call Cloudflare Workers AI for text generation.

##### `load_gemma(model_id, use_8bit=False, force_gpu=False)`
Load Gemma/Qwen/TinyLlama chat model.

##### `generate_with_gemma(tok_or_proc, mdl, messages, max_new_tokens=180, temperature=0.8, top_p=0.9, repetition_penalty=1.05)`
Generate text with local model.

#### Delta Graph Utilities

##### `top_motifs_from_delta_graph(G, k_nodes=10, positive_only=True)`
Extract top motifs from delta graph by Δ magnitude.

**Parameters:**
- `G`: Delta graph
- `k_nodes`: Number of nodes to return
- `positive_only`: Only include positive Δ edges

**Returns:** List of descriptor strings

#### Multimodal Adapters

##### `_ImageAdapter`, `_AudioAdapter`, `_TextAdapter`
Fragment classes that handle image/audio/text uploads and convert to embeddings for context override.

---

## Typical Workflow

### 1. Basic Symbol Exploration
```python
from delyrism import SymbolSpace, TextEmbedder

# Initialize
embedder = TextEmbedder(backend="qwen3", model="Qwen/Qwen3-Embedding-0.6B")
symbols = {
    "Fire": ["passion", "transformation", "energy", "destruction"],
    "Water": ["flow", "emotion", "depth", "healing"]
}
space = SymbolSpace(symbols, embedder)

# Query with context
proposals = space.propose(
    sentence="intense creative breakthrough",
    topk=3,
    tau=0.3
)
```

### 2. Context Shift Analysis
```python
# Get delta graph
G = context_delta_graph(
    space,
    sentence="gentle healing process",
    strategy="gate",
    beta=0.6,
    top_abs_edges=20
)

# Visualize
plot_delta_graph(G, space, palette="Nord")
```

### 3. Narrative Generation (via Streamlit)
1. Enter context: "A warrior finds peace"
2. Adjust symbol weights (Fire=0.7, Water=0.3)
3. Select tone preset: "Blake" (prophetic lyricism)
4. Click "Generate Story"
5. System automatically:
   - Proposes relevant symbols
   - Extracts top motifs from delta graph
   - Builds prompt with tone directives
   - Generates mythopoetic prose

---

## Advanced Features

### Membership Alpha Blending
Control how much the shift respects symbol boundaries:
- `membership_alpha=0.0`: Full shift (descriptors can drift far)
- `membership_alpha=1.0`: No shift (stay within symbol clusters)
- `membership_alpha=0.5`: Balanced (useful default)

### Within-Symbol Softmax
Apply attention separately per symbol:
- `within_symbol_softmax=True`: Each symbol's descriptors compete only with each other
- Useful when symbols have very different numbers of descriptors

### Audio/Image Context
```python
# Image (via OpenCLIP)
embedder = TextEmbedder(backend="audioclip")
# ... load image, embed it
img_vec = embedder.encode_image(image)
space.set_context_vec(img_vec)

# Audio (via CLAP)
embedder = TextEmbedder(backend="clap")
wave, sr = librosa.load("audio.wav")
audio_vec = embedder.embed_audio_array(wave, sr)
space.set_context_vec(audio_vec)
```

### Hybrid Strategies
- **gate**: Fast, additive shift
- **reembed**: Slow but semantically rich (re-encodes with context)
- **hybrid**: Best of both (blend gate + reembed)
- **pooling**: Geometric interpolation with context vector

---

## Performance Tips

### For Low-Resource Environments
1. Use Cloudflare backend for embeddings (no local GPU needed)
2. Enable `use_8bit=True` for local models
3. Set `torch.set_num_threads(4)` on CPU
4. Use smaller models (Qwen2.5-0.5B, SmolLM2)

### For Speed
1. Cache embeddings (automatically done in SymbolSpace)
2. Use `batch_size` parameter in `encode()`
3. Reduce `descriptor_threshold` to sparse graphs
4. Use PCA instead of UMAP for projection

### For Quality
1. Use Qwen3 or sentence-transformers for embeddings
2. Increase `top_abs_edges` in delta graphs
3. Lower `tau` for sharper attention
4. Use `strategy="hybrid"` for richer semantics

---

## Configuration Files

### JSON Symbol Set Format
```json
{
  "Fire": ["passion", "energy", "transformation"],
  "Water": ["flow", "healing", "emotion"],
  "Earth": ["grounding", "stability", "growth"]
}
```

### Streamlit Secrets (for Cloudflare)
Create `.streamlit/secrets.toml`:
```toml
[cloudflare]
account_id = "your_account_id"
api_token = "your_api_token"
```

---

## Dependencies

### Core
- `torch`, `transformers`, `sentence-transformers`
- `numpy`, `networkx`, `sklearn`
- `matplotlib`, `umap-learn`

### Optional
- `streamlit` (for web interface)
- `open-clip-torch` (for image/audio CLIP)
- `librosa` (for audio processing)
- `Pillow` (for image handling)
- `scipy` (for convex hulls)

---

## License & Citation

This is experimental research code. If you use it, please cite appropriately and respect the spirit of symbolic exploration.

**Key Innovation**: Context-aware symbolic reasoning combining:
- Neural semantic embeddings
- Graph-based diffusion (PageRank)
- Attention mechanisms
- Mythopoetic narrative generation
