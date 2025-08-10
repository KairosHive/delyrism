# DELYRISM — Archetype Explorer

Explore a **symbol ↔ descriptor** world and watch meanings shift with **text** or **audio** context.

## Features
- 2D **Meaning Space** (UMAP / t-SNE / PCA) with context-shift arrows  
- **Top symbols** ranked for your context (coherence + PageRank blend)  
- **Descriptor Attention** per symbol  
- **Contextual Subgraph** and **Δ (delta) graph**  
- **Audio context** from **upload** or **live mic recording**

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
## How to use

1. **Data** — Upload or paste your `{symbol: [descriptors...]}` JSON.  
2. **Embeddings** — Pick a backend: `qwen3`, `qwen2`, `original`, `audioclip`, or `clap`.  
3. **Context**
   - Type a prompt **or**
   - Open **“Or use an audio context”** → **Upload** a file or **Record** with the mic  
   - For audio, select the **`clap`** backend  
4. **Explore** — Tweak **β / τ / α / λ** and other sliders; inspect **Rankings**, **Attention**, **Subgraph**, and the **Δ Graph**.

## 📊 Results (Quick Tour)

### 1) Latent Map & Contextual Shifts
2D map of descriptors; arrows show how a context pulls meanings.
<p align="center">
  <img src="https://github.com/user-attachments/assets/76752b6c-6893-4eb8-b8c1-5d2670e7e5a0" width="720" alt="Latent Map and Contextual Shifts">
</p>

---

### 2) Descriptor Attention (per symbol)
Which descriptors the context highlights for a chosen symbol.
<p align="center">
  <img src="https://github.com/user-attachments/assets/66b7f06f-947e-4dca-ba44-08a66d9b5ec7" width="720" alt="Descriptor Attention per Symbol">
</p>

---

### 3) Symbol Ambiguity Metrics
Dispersion, leakage, and entropy—normalized for quick comparison.
<p align="center">
  <img src="https://github.com/user-attachments/assets/2790e5b0-6cb5-4a7a-a45f-a84c229d6de7" width="720" alt="Symbol Ambiguity Metrics">
</p>

---

### 4) Within-Symbol Associative Increase (Δ)
Heatmap of how descriptor–descriptor ties strengthen under context.
<p align="center">
  <img src="https://github.com/user-attachments/assets/bf07f76d-a69f-4d9a-b773-93dad2a59c1a" width="420" alt="Within-Symbol Associative Increase">
</p>

---

### 5) Context Δ Graph
Graph of strongest edge changes within network
<p align="center">
  <img src="https://github.com/user-attachments/assets/2ba762bc-c98d-4b39-bbba-2a2f4fedd5e6" width="720" alt="Context Delta Graph">
</p>


