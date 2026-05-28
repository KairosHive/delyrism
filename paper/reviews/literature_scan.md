# Literature scan — what's prior art, what's actually new

Quick survey done 2026-05-28 to check the chair's claim that *"embedding techniques have been widely explored"* against what's actually in the framework v2.

The framing in this note is technical / external-literature only — the **Indigenous-centered framing** in `PLAN.md §1` reframes which of these comparisons even matter. Read PLAN.md first.

---

## Prior art that is genuinely close

### Topological polysemy (the closest precedent)

- **Jakubowski, A. et al. (2020)** — *Topology of Word Embeddings: Singularities Reflect Polysemy* (ACL StarSem). [arXiv:2011.09413](https://arxiv.org/abs/2011.09413). Uses persistent homology on punctured neighborhoods of word embeddings; the H0 / H1 structure correlates with the number of senses. **This is the precedent for PH-on-embeddings.** The framework's PH layer is closer to this than to anything else in the literature.
- **Jakubowski et al. (2024)** — *Topological quantification of ambiguity in semantic search* ([arXiv:2406.07990](https://arxiv.org/abs/2406.07990)). Extends polysemy-PH to sentence-level ambiguity.
- **Uchendu & Le (2024)** — *Unveiling Topological Structures from Language: A Comprehensive Survey of TDA Applications in NLP* ([arXiv:2411.10298](https://arxiv.org/html/2411.10298v3)). Useful background; positions PH-in-NLP as a small but active subfield.
- **TopoBERT** ([repo](https://github.com/AdaUchendu/AwesomeTDA4NLP)) — interactive viz of transformer embedding topology across layers / fine-tuning. UX precedent for our topology tab.

### Visualizing context effects on embeddings

- **Coenen et al.** — *Language, Context, and Geometry in Neural Networks* (Google PAIR), [Context Atlas](https://pair-code.github.io/interpretability/context-atlas/blogpost/). Visualizes geometric reorganization of word representations as context changes. Visual precedent for our shift / Δ views; does not look at descriptor-pair couplings.
- *Visually Analyzing Contextualized Embeddings* ([arXiv:2009.02554](https://arxiv.org/pdf/2009.02554)) — UMAP + cluster-co-occurrence visualizations of contextualized embeddings.

### Pairwise / graph-similarity rewiring

Search for *"context-induced pairwise similarity rewiring"* turned up methodologically adjacent but conceptually distinct work:

- *Graph Embedding with Shifted Inner Product Similarity* ([arXiv:1810.03463](https://arxiv.org/pdf/1810.03463)) — a graph-embedding method using SIPS. Not about visualizing context effects.
- *Graph-induced pairwise constrained embedding* (JMLR 12) — must-link / cannot-link constraints. Not relevant.
- *Graph Embedding Using Constant Shift Embedding* — also unrelated.

**No close precedent found for the Δ-graph itself** (pairwise descriptor coupling matrix differenced before/after context, rendered as a signed graph of strengthened vs weakened relations). The closest analog is concept-drift / diachronic-embedding work, which differences over *time*, not over *thematic context applied to the same descriptors*.

### Sense disambiguation + symbolic integration

- *Contextualized Knowledge Base Aware Sense Embeddings* (Springer 2021) — WSD using KB structure and contextualized embeddings. Uses external lexical resources (WordNet); our framework uses a curated symbol→descriptor structure with no external KB.
- Standard WSD literature (Loureiro et al. 2021, Haber 2024 — both already cited in v1) treats the problem as token-level disambiguation in running text. Different problem.

---

## What is actually novel in the framework v2 (after this scan)

| Instrument | Novelty assessment |
|---|---|
| Symbol-as-descriptor-distribution + softmax attention | **Not novel.** Standard attention-over-descriptors. Setup, not contribution. |
| Dispersion / leakage / inter-symbolic entropy metrics | Mild combinations of existing notions. Useful framing but not the contribution. |
| PPR over symbol-descriptor bipartite graphs | Standard application of PPR; not novel. |
| **Δ-graph (pairwise coupling rewiring under context)** | **Appears genuinely novel.** No direct precedent surfaced. Closest neighbors are diachronic embedding analysis (different setting) and SIPS (different problem). |
| **PH applied to a fixed symbolic system under context shift** | **Extension of Jakubowski et al.** They do PH on individual word neighborhoods; we do PH on curated archetype clouds, on the *union* of archetypes, and crucially, *under context-shifted descriptor matrices*. The before/after-context PH comparison is new. |
| Set-level PH metrics (coverage, focus, balance, separation) | Adaptations / combinations of standard PH summaries — useful but mild. |
| Synergy bridges across symbols | Some Mapper-style precedent exists for cross-cluster structure; the cross-symbol-cycle framing on union vs separate PH is new in the polysemy / symbolic-system context. |
| Catalysts via leave-one-out PH | Standard LOO importance applied to PH summaries; mild novelty. |
| Identity cards + Migrations | Concrete UX novelty; the underlying signal is the shift, no separate technical contribution. |
| **Multi-modal context override (audio CLAP + image vision-LLM) into a fixed symbolic field** | Engineering-novel — not aware of work treating a curated symbolic system as a multi-modal hub. |
| **Morphing mode (continuous A↔B interpolation, with PH / Δ-graph live)** | UX novel; the underlying mechanism is just interpolation. |

---

## Implications for v2

1. **The chair's "widely explored" critique is partly right and partly wrong.** Symbol-as-distribution + softmax attention is not novel. The Δ-graph and the PH-of-symbolic-sets-under-context are.
2. **Cite Jakubowski et al. explicitly.** Position our PH work as extending topological polysemy from word-neighborhoods to *curated symbolic fields under contextual reshaping*.
3. **Lead with the Δ-graph** in §3 (the methods section). It has the cleanest novelty argument *and* the cleanest expression of the relational worldview (see PLAN.md §1).
4. **Set-quality metrics are mild novelty** — present them as instruments for community-grounded interpretation, not as standalone technical contributions.
5. **Multi-modal context is engineering novelty** — keep brief but include; it supports the "symbolic field is a hub of relations across modalities" framing.

---

## Sources

- [Topology of Word Embeddings: Singularities Reflect Polysemy (Jakubowski et al. 2020)](https://arxiv.org/abs/2011.09413)
- [Topological quantification of ambiguity in semantic search (2024)](https://arxiv.org/abs/2406.07990)
- [Unveiling Topological Structures from Language: A Survey of TDA Applications in NLP (Uchendu & Le 2024)](https://arxiv.org/html/2411.10298v3)
- [AwesomeTDA4NLP — repository of TDA-NLP work](https://github.com/AdaUchendu/AwesomeTDA4NLP)
- [Language, Context, and Geometry in Neural Networks (Coenen et al., Google PAIR)](https://pair-code.github.io/interpretability/context-atlas/blogpost/)
- [Visually Analyzing Contextualized Embeddings](https://arxiv.org/pdf/2009.02554)
- [Graph Embedding with Shifted Inner Product Similarity (1810.03463)](https://arxiv.org/pdf/1810.03463)
- [Contextual Embeddings — overview](https://www.emergentmind.com/topics/contextual-embeddings)
