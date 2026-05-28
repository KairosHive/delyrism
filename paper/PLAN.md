# Paper v2 — Improvement Plan

**Lead:** Antoine Bellemare-Pepin
**Co-author / knowledge holder:** Suzanne Kite (Oglala Lakota, Bard College, Wihanble Sʼa Center)
**Program:** Abundant Intelligences
**Working title:** TBD — see §13 (avoid using the system's project name as the paper title; the paper is about the framework, not the tool)

---

## 1 · The reframe — two simultaneous claims

The v1 paper was read by the NeurIPS Creative AI chair as *an embeddings paper that happens to use Lakota material*, and judged by embeddings-paper criteria. That reading missed the actual contribution. v2 needs to make **two simultaneous claims**, both first-class:

1. **Indigenous-centered / Abundant Intelligences scholarship.** The framework is built *from* Lakota relational epistemology in collaboration with a knowledge holder. The Lakota Shape Kit is the source of theory, not a case study. Meaning is constituted by **living relations** among descriptors, contexts, and interpreters — *Mitákuye Oyásʼiŋ*. This is the worldview; the instruments below are formal expressions of it.
2. **Technical NLP contribution.** A structural-probing framework for context-conditioned representations of curated lexicons: a pairwise-coupling decomposition of context effects (Δ-graph), a parametric family of context-shift operators for static embeddings, conditional persistent homology on symbolic fields, and structural prompting for narrative generation.

Sections 1–2 of the paper make claim 1 (relational worldview as foundation). Sections 3–6 make claim 2 (formal instruments + evaluation). The two are co-constitutive — the worldview is not decoration on the methods; the methods exist *because of* the worldview.

---

## 2 · What is already built that the paper should foreground

Already implemented and analytically usable:

| Instrument | What it measures / shows | Worldview reading |
|---|---|---|
| **Δ-graph** | Signed change in pairwise descriptor coupling under context | Context as relational rewiring |
| **Context-shift operator family** | `gate / reembed / pooling / hybrid` with β / γ / τ / pool_w / α_membership | Multiple modes of relational reconfiguration |
| **Per-symbol PH** | H0 cohesion, H1 loops, H2 voids for each archetype's descriptor cloud | Topological correlates of cyclical / non-linear meaning |
| **Set-level PH metrics** | Coverage_h1/h2, focus, separation, balance on the union | Shape of the symbolic field as a whole |
| **Conditional PH (overlay)** | PH recomputed on the context-shifted matrix `D'` | How context reshapes the topology of the field |
| **Synergy bridges** | H1 cycles spanning multiple archetypes | Cross-archetype relational loops |
| **Catalysts (LOO importance)** | Descriptors whose removal collapses persistent structure | Load-bearing words for the field's relational integrity |
| **Migrations & Identity Cards** | Descriptors whose nearest archetype flipped under context | Fluid membership — relations override fixed category |
| **Morphing mode** | Continuous interpolation between two contexts A↔B with live PH/Δ | Continuous relational reconfiguration |
| **Multi-modal context** | Audio (CLAP) / image (vision-LLM bridge) → single context-vector slot | Symbolic field as cross-modal interpretive hub |
| **Structural prompting** | LLM generation conditioned on Δ-graph / PH cycle / migration motifs | Analytic layer returned to the storytelling tradition |

v1 used only the first row (and only implicitly — the notebook had Δ-graph code but the paper didn't show it).

---

## 3 · Theoretical core — Section 2 of the paper

Title proposal: **"Meaning as Relational Rewiring."**

Outline:
1. **Lakota relational epistemology** — Kite's voice. Cite IPAI 2020 (Lewis et al.), Kite's `hel` paper, Kite's prior writing on Lakota-grounded AI design. Establish *Mitákuye Oyásʼiŋ* as the operative claim: nothing exists in isolation; meaning is the position of a thing in a living web.
2. **Why this is incompatible with the static-vector view in NLP**, and why context-sensitive vectors (BERT, Qwen, etc.) only partially address it — they contextualize the *vector*, not the *relational structure* of a curated symbolic field.
3. **Resonant Western literature** — Saussure (relational language), Peirce (triadic sign), distributional semantics, polysemy. These are *parallels*, not parents; we engage them where they converge with Lakota relational thought.
4. **Programmatic claim:** the instruments in Section 3 are not new embedding methods; they are **structural probes** of how a context reshapes the relational fabric of a curated symbolic field. Indigenous epistemology motivates *what to look at*; NLP gives us the tools to look.
5. **Position the work as Abundant Intelligences scholarship.** AI built *from* Indigenous knowledge systems in collaboration with knowledge holders. Cite the program; cite the Indigenous-AI commitments (IPAI, CARE principles, Te Hiku Media, Tahu Kukutai on Indigenous data sovereignty) — make the lineage visible.

This is where Kite's voice should be strongest. Decisions about which concepts to invoke, which protocols to honor in framing, which terms to leave un-translated — not technical choices.

---

## 4 · Technical contributions — Section 3 of the paper

Graded honestly. The paper should lead with A and C; B and D fill out the eval; E and F are supporting.

### A · Δ-graph — pairwise-coupling decomposition of context effects
**Strong novelty.** Given a descriptor matrix `D ∈ ℝⁿˣᵈ` (row-normalized) and a context-shift operator `S(c, θ)` producing `D' = S(D, c, θ)`, the signed difference

$$ \Delta = D'D'^{\top} - DD^{\top} $$

decomposes the context effect into per-pair contributions, rendered as a signed graph (edges colored by sign, weighted by magnitude). This localizes contextualization to *which lexical relations strengthen or weaken*, rather than to point displacement.

No direct precedent in the lit scan. Closest neighbors (diachronic embedding drift, attention-head rewiring probes) operate over different objects. Cite as new interpretability instrument.

### B · A parametric family of context-shift operators for static lexicons
**Moderate novelty + cleanest experimental backbone.** Four strategies, all already implemented:

- **`gate`** — gated additive: `D' = D + β · g_τ(⟨D, c⟩) · (c − ⟨D, c⟩)` with `g ∈ {relu, cos, softmax, uniform}`.
- **`reembed`** — re-encode each descriptor in a context-templated prompt (`"In this context: {sent}. Descriptor: {desc}"`).
- **`pooling`** — pooled per-archetype context blend with `pool_type ∈ {avg, max, min}` and weight `pool_w`.
- **`hybrid`** — convex combination of `gate` and `reembed` controlled by `γ`.

Plus `membership_α` for soft archetype membership and `within_symbol_softmax` for normalized attention.

No standard catalog of context-shift operators for *fixed* embeddings exists in the NLP literature — most "context-conditioned embedding" work just retrains contextualized models. We give a parameterized family and can ablate it as a unit.

### C · Conditional persistent homology of curated symbolic fields
**Strong novelty as an extension claim.** Jakubowski et al. (StarSem 2020) introduced PH on punctured neighborhoods of word embeddings as an unconditional polysemy detector. We extend in three directions:

1. **Curated archetype clouds** rather than unsupervised word neighborhoods — leveraging the lexicon designer's (or the community's) structure.
2. **Union PH with set-level descriptors** — `coverage_h1`, `coverage_h2`, `focus`, `separation_tightness`, `cohesion_balance`, `count_balance` as quantitative properties of an entire symbolic field, usable for comparing ontologies.
3. **Conditional PH** — recompute on `D'` and compare to the unconditional PH on `D`; ask which loops, voids, and bridges are *intrinsic* to the field vs *context-induced*.

The before/after-context PH comparison appears genuinely new. Synergy (cycles spanning multiple archetypes) and catalyst (LOO topological importance) decompositions are derived from this scaffold.

### D · Structural prompting from Δ-graph and PH motifs
**Engineering / framing novelty.** LLM generation conditioned on structural-analytic features (top Δ-edges, persistent H1 cycle members, migration patterns, transformation extrema) rather than retrieved text (standard RAG) or raw context (standard prompting). Distinct from RAG because the conditioning signal is *analytic-structural*, not lexical-retrieved.

### E · Symbolic field as cross-modal probe target
**Mild novelty / conceptually distinct from CLIP-style alignment.** Multi-modal inputs (audio via CLAP; image via vision-LLM bridge to text → embedder) addressed via the *same* context-vector slot probing a single text-symbolic field. The field stays fixed; only the probe changes. Different from multimodal embedding spaces that fuse modalities.

### F · Set-quality PH descriptors for ontology curation
**Mild novelty.** The set-level metrics from §C, packaged as practical tools for someone curating a symbolic field — diagnostic during construction, before / after / under-context comparison during use.

### G · Migrations & Identity Cards
**Not a separate contribution** — derived UX from §B + nearest-neighbor reassignment under perturbation. Stays as figures, not as a claim.

---

## 4.5 · What v1 already does well — keep this

The v1 paper is not a blank slate; several pieces work and should stay (lightly edited) in v2 rather than be rewritten. Treat this list as the conservative core.

| v1 element | Status in v2 | Notes |
|---|---|---|
| §1 Lakota Shape Kit introduction + Figure of the kit | **Keep**, expand with Abundant Intelligences framing | Already does the right cultural-introductory work. |
| §2 The Inherent Ambiguity of Meaning (Saussure, Peirce, Piantadosi, Solé, Cohn) | **Keep largely intact**, reframe as "resonant Western literature" rather than parent framework | The references and arguments are solid; only the positioning changes. |
| §3.1 Symbol as context-weighted distribution over descriptors | **Keep**, compress to setup for §4 | This is the substrate the new instruments operate on; can't remove it. |
| Ambiguity metrics: dispersion, leakage, inter-symbolic entropy | **Keep** | Useful as field descriptors alongside the new PH set-level metrics. Position them as the "first-order" descriptors and PH as "second-order" (shape, not just spread). |
| Figure 1 (UMAP + dispersion/leakage/entropy bars) | **Keep** | Strong first-look figure; stays as the opening empirical figure. |
| §3.2 Contextual Attention Mechanism + Figure 2 (Earth across emotion themes) | **Keep** | Earth-across-emotions is a clean demonstration of context-conditioned attention. Pair it with a Δ-graph view of the same shift so the reader sees attention-level *and* relational-coupling-level pictures together. |
| §3.3 Graph-Based Relational Modeling: PPR-based contextual subgraph + Figure 3 | **Keep**, position as complementary to Δ-graph | PPR shows *propagation* of relevance through the bipartite graph; Δ-graph shows *change* in pairwise coupling. They answer different questions. Keep both. |
| Figure 4 (star plots of symbol pairs) | **Reduce or drop** | The Δ-graph + identity cards subsume what the starplots showed at the pair level. Keep only if Kite specifically wants this view. |
| §4 Creative Potential | **Keep the framing**, point to structural prompting (§4D) as the concrete realization | The motivation section becomes the bridge between methods and applications, even if Analysis 6 is parked. |
| §5 Ethical Considerations | **Keep and expand** with CARE / OCAP / IPAI explicit references | Already in the right spirit; v2 should cite the protocols by name and discuss community review of the published materials. |

**What's new vs what's preserved, roughly:**
- §1–2 of v1 → §1–2 of v2, expanded with relational-epistemology and Abundant Intelligences framing.
- §3 of v1 → §4.1 and §4.6 of v2 (compressed setup + multi-modal).
- v1 had no analogue of §4.2–4.5 — these are the new technical contributions.
- v1 had no analogue of §6 (showcase analyses) — these are new demonstrations.
- v1 had no §7 (evaluation) at all — this is the largest addition.
- §5 of v1 → final ethics section, expanded.

---

## 5 · Methods — Section 4 of the paper

Re-presented around the instruments, with one Lakota-grounded worked example per instrument:

- **§4.1 The symbolic field** — descriptors per symbol, embedding setup. Compressed; this is setup.
- **§4.2 Context-shift operators** — formal definitions of the four strategies (B above); brief.
- **§4.3 Δ-graph** — definition, signed-graph rendering, top-edge selection, sign filter. Worked example: a sentence drawn from Kite's storytelling; show which descriptor relations strengthen vs weaken.
- **§4.4 Topology of the field** — per-symbol PH; set-level descriptors; conditional PH on `D'`; synergy and catalyst decompositions.
- **§4.5 Structural prompting** — motif extraction from Δ-graph / cycles / transformations; prompt assembly; LLM generation.
- **§4.6 Multi-modal probes** — same context-vector slot, three modalities.

---

## 6 · Showcase analyses — Section 5 of the paper

Five core analyses that the paper builds on, all runnable on existing endpoints. Each gets ~one figure + 1–3 paragraphs. Analyses 6 and 7 are set aside for now (see §6.6).

### Anchor contexts (working set)

To unblock concrete work I'm committing to a small set of context sentences. **These are placeholders Kite should swap.** They are deliberately nature- and cycle-themed, evocative without invoking ceremonial specifics, and chosen for empirical contrast — three for Analysis 1, two for Analysis 3, one for Analysis 5. If Kite has a better set, the analyses re-run on them in a few minutes.

| ID | Sentence | Register |
|---|---|---|
| `C₁` | *"A storm gathers over the hills, the air thick with waiting."* | atmospheric tension |
| `C₂` | *"Birds rise at first light, the river runs cold and steady."* | renewal / morning |
| `C₃` | *"Smoke curls upward toward stars; voices fall silent."* | stillness / ascent |
| `C_A` | *"An old wound surfaces in a new season."* | memory / cyclical return |
| `C_B` | *"Rain washes the trail clean; footprints disappear."* | renewal through dissolution |
| `C_scene` | *"A thunderhead gathers above the river at dusk."* | + matched audio + matched image |

Reasons for this set: `C₁ / C₂ / C₃` contrast on temporal mood (tension / morning / night) without overlapping; `C_A` and `C_B` are tonally distant enough to give Analysis 3 strong phase transitions; `C_scene` is concrete enough to record audio for and to photograph for the cross-modal analysis.

### Analysis 1 · Context as relational rewiring (Lakota Shape Kit, three contexts)
**Instruments:** Δ-graph + Identity Cards.
**Contexts:** `C₁`, `C₂`, `C₃`.

For each, compute the Δ-graph on the full Shape Kit and pick the top-N |Δ| edges. Render the three side by side with consistent edge-scale.

**Show:** the same lexicon yields radically different relational structures under different narrative framings — strengthened edges cluster intra-archetype under one context, bridge archetypes under another. Identity cards underneath show which descriptors *migrated* archetype membership under each context.

**Worldview reading:** context does not shift symbols, it activates particular *relations among descriptors*. The Shape Kit's interpretive openness becomes computationally visible.

### Analysis 2 · Topology of a symbolic field: intrinsic vs context-induced
**Instruments:** per-symbol PH + conditional PH (overlay) + set-quality metrics.
**Context:** `C₃` (the strongest single narrative framing in the set).

Compute PH on `D` (intrinsic) and on `D' = S(D, C₃)`. Compare:
- H1 cycles that persist in both (intrinsic relational features)
- H1 cycles that appear only under `C₃` (context-induced loops)
- H1 cycles that disappear under `C₃` (collapsed by the framing)

**Show:** side-by-side persistence diagrams + set-level metrics (coverage_h1, focus, separation) before/after. One worked example tracing the most-persistent H1 cycle descriptor-by-descriptor.

**Worldview reading:** some relational patterns are *of* the symbolic field (its intrinsic shape); others emerge through *living interpretation* (situated framing). The framework distinguishes these formally — which is something the Lakota tradition does naturally and Western NLP cannot.

### Analysis 3 · Phase transitions in continuous morphing
**Instruments:** morphing mode + Δ-graph + per-symbol PH (track over α).
**Contexts:** `C_A` ↔ `C_B`.

Sweep blend α ∈ {0, 0.1, …, 1.0}. At each α, track:
- per-symbol H1 count and total persistence,
- top Δ-graph edges and their signs,
- migration events (descriptors switching nearest archetype),
- separation_tightness on the union.

**Show:** meaning does not interpolate linearly. Specific α-values produce topology jumps, migration cascades, and Δ-edge sign flips — *phase transitions* in symbolic interpretation. Plot as α-trajectories of each tracked quantity, with phase-transition α-values highlighted.

**Worldview reading:** between two readings of a symbolic field there is not a smooth blend but a sequence of small re-organizations. The framework makes these legible.

Probably the most surprising figure in the paper.

### Analysis 4 · Catalysts and structural integrity of an archetype
**Instruments:** LOO catalysts + per-symbol PH.
**Context:** none (intrinsic analysis).

For each archetype, compute leave-one-out PH and rank descriptors by their impact on persistent H1+H2 sum. Compare with Kite's reading of which descriptors are *ceremonially / narratively central* vs *peripheral but structurally important*.

**Show:** table of top-3 catalysts per archetype; brief discussion of agreement / disagreement with the knowledge holder's reading.

**Worldview reading:** the framework identifies *load-bearing* descriptors — those holding a symbol's relational shape together. These are not necessarily the most-frequent or most-prototypical, which is consistent with how oral traditions emphasize specific words for ceremonial function.

### Analysis 5 · Cross-modal consistency of relational structure
**Instruments:** multi-modal context + Δ-graph.
**Context:** `C_scene` in three modalities.

Encode the same scene three ways:
- text — the sentence `C_scene`,
- audio — a brief recording matching the scene (thunder + river ambience), CLAP-encoded,
- image — a photograph of the scene, vision-LLM → text → embedder.

Compute Δ-graphs from each modality's context vector. Quantify cross-modal agreement on top-K Δ-edges (rank correlation, Jaccard@K).

**Show:** the symbolic field's relational rewiring is *stable across modalities* when input semantics match — sound, sight, and word probe the same field through different surfaces.

**Worldview reading:** symbol life is not text-bound. The same relational field receives input from many channels — what oral traditions have always understood about story, song, image, gesture.

### 6.6 · Set aside for now

**Analysis 6 (structural-prompting faithfulness eval against an LLM baseline)** and **Analysis 7 (Lakota Shape Kit vs Western-elements ontology comparison)** are parked. Both raise questions that aren't ours to answer alone:

- Analysis 6 reintroduces an LLM comparison the chair's critique would *expect* to see — but designing a faithfulness rubric that respects community standards is non-trivial, and the comparison may not be the right framing for the paper. Revisit after the community study design (§7a) is in place; the form of that study will tell us whether a structural-prompting demonstration belongs in the paper at all.
- Analysis 7 is politically delicate — a side-by-side of Lakota and Western symbolic systems risks reading as adversarial regardless of careful framing. Kite to decide if and how this comparison enters the paper.

---

## 7 · Evaluation — Section 6 of the paper

### 7a · Community-grounded validation (primary)
Co-designed with Kite. Possible forms (Kite to choose):
- **Semi-structured sessions** with Lakota knowledge holders. Show Δ-graphs, identity cards, migrations for 3–5 context sentences. Ask: does the relational pattern resonate? Where does it fail? Where does it surface something interesting?
- **Story co-creation comparison** with Analysis 6 outputs. Community readers + technical raters; report agreement themes, not Likert means.
- **Generative iteration** — knowledge holders propose contexts; framework returns Δ-graphs and cycles; iterate.

Participation, attribution, compensation, and writeup review follow community protocols. CARE / OCAP govern publication.

### 7b · Technical ablations and robustness (in service of 7a)

- **Shift-operator ablation (§4B).** Same context, all four strategies + parameter sweeps. Measure: shift magnitude `||D' − D||_F`, Δ-graph top-K edge agreement (Jaccard, Kendall-τ), per-symbol H1 count change. Show the strategies produce systematically different rewiring patterns — and identify which strategies preserve community-validated readings best.
- **Embedder invariance.** Re-run with at least three embedders (Qwen3-Embedding-0.6B, BGE-M3, OpenAI text-embedding-3 or open alternative). Report rank correlation of top Δ-edges and top-persistent cycles across embedders. Argument: the relational patterns are not artifacts of a single embedder.
- **Null model.** For each real context, generate a random shift of equal `||D' − D||_F`. Compute Δ-graph and PH metrics for both. Permutation test on (a) within-symbol edge concentration in Δ-graph, (b) Δ in coverage_h1 / focus. Establish that meaningful contexts produce structured rewiring, not just any rewiring of the right magnitude.
- **Descriptor stability.** Bootstrap-resample descriptors per archetype (drop 10–20% at random, repeat); measure stability of top Δ-edges and top-persistent cycles. Address R7Mo's "brittle to descriptor choice" critique directly.

### 7c · Why we do not compete with LLMs on creative output
The chair's *"ChatGPT can do poetry"* critique is not the relevant comparison. We are not making a generation-quality claim. The framework is a **structural probe of how a context reshapes a curated symbolic field**, with structural prompting as a downstream application. Where we do compare to LLMs (Analysis 6), it is on *structural faithfulness to the surfaced diagnostics*, not on generation quality.

---

## 8 · Reviewer concerns mapped to v2 sections

| # | v1 critique | v2 response | Section |
|---|---|---|---|
| 1 | Embedding techniques widely explored | True for the symbol-as-distribution baseline. The contributions are §4A (Δ-graph) and §4C (conditional PH on symbolic fields) — both novel; lit scan in `reviews/literature_scan.md` defends this. | §4 |
| 2 | No quantitative comparisons | §7b: shift-operator ablation, embedder invariance, null model, descriptor stability. | §7b |
| 3 | LLMs already do creative tasks | §7c: that's not the comparison. Where we do compare (Analysis 6), it's structural faithfulness, not generation quality. | §7c, §6.6 |
| 4 | No expert/community study | §7a — primary form of validation, co-designed with Kite. | §7a |
| 5 | Brittle to descriptor choice | §7b descriptor-stability bootstrap. | §7b |
| 6 | Brittle to embedder choice | §7b multi-embedder. | §7b |
| 7 | Author-chosen descriptors as limitation | Reframed: descriptor curation *by knowledge holders* is the method, not a flaw. Community-grounded AI requires Indigenous expertise to mean anything. | §2, §7 |

---

## 9 · Venue

Worth a conversation before final commit. Options:
- **NeurIPS Creative AI Track (resubmission)** — possible if v2 is unmistakably reframed; the prior reviewers' confusion was partly our framing.
- **NeurIPS Indigenous in AI workshop** if it runs again — strongest fit.
- **FAccT / AIES** — strong fit for the Abundant Intelligences framing.
- **EMNLP main / Findings or ACL** — viable for the technical contribution after §4 + §7 are firmed up.
- **AI & Society / Big Data & Society / Cultural Analytics** — journals that take this kind of work seriously.
- **NeurIPS TDA workshop** — for the §4C contribution if standalone.

Kite's perspective on venue probably matters more than mine.

---

## 10 · Open questions for Antoine + Kite

1. **Naming.** "Δ-graph", "synergy", "catalyst", "morphing" — current names are NLP-flavored. Is there language drawn from Lakota tradition (or co-designed with Kite) that carries the concepts more truthfully? Renaming should propagate through the codebase UI too.
2. **Working title for the paper** (avoid the system name; the paper is about the framework).
3. **Anchor contexts (§6).** I've committed to a placeholder set so work can start (`C₁ / C₂ / C₃ / C_A / C_B / C_scene`). Kite can swap any or all at any time — the analyses re-run in minutes. The question is just whether she wants to swap them now, later, or never.
4. **What to keep private.** Some descriptor sets, some context sentences, some interpretations are not for an open-access paper. The framework can be demonstrated without disclosing materials that belong to the community. Kite to decide.
5. **Community study design (§7a).** Form, participants, compensation, review of results — Kite leads.
6. **Authorship beyond AB & SK.** Community members who shape v2 — co-authors, acknowledgments, or both? Kite to decide.
7. **Venue.** See §9.

(The original "Analysis 7 — ontology comparison" question is parked in §6.6.)

---

## 11 · Folder layout

```
paper/
├── README.md
├── PLAN.md                              ← this document
├── original/                            # v1 submission
│   ├── neurips2025_submission.tex
│   └── symbol_ambiguity_clean.ipynb
├── reviews/
│   ├── neurips2025_reviews.md
│   └── literature_scan.md
└── v2/
    ├── (drafts go here)
    └── figures/                         # new figures for v2
```

---

## 12 · Next steps

In order; pause between for AB + Kite input.

1. ☐ Review this plan; flag anything wrong, missing, or not ours to decide.
2. ☐ Run **Analysis 1** with the placeholder contexts (`C₁ / C₂ / C₃`) — fast, demonstrates the Δ-graph workflow end-to-end, gives Kite something concrete to react to and swap contexts on.
3. ☐ Run **Analysis 4** (catalysts) — intrinsic, no contexts needed; surfaces a list Kite can react to immediately.
4. ☐ Decide venue (informs format, length, tone).
5. ☐ Run **Analyses 2, 3, 5** once the placeholder contexts are confirmed or swapped.
6. ☐ Run the §7b technical ablations (shift-operator, embedder invariance, null model, descriptor stability) — pure compute, parallel with anything else.
7. ☐ Kite drafts §2 (relational worldview / Abundant Intelligences) and shapes the framing throughout.
8. ☐ Design the §7a community study with Kite.
9. ☐ Full draft (preserving the v1 elements listed in §4.5).
10. ☐ Community review of draft before submission.

**Practical note on order:** steps 2 and 3 are independent of every open question; they can run today. Their outputs are the easiest material for Kite to react to and will sharpen every other decision.

---

## 13 · A note on naming

The system has a name; the paper need not use it. Treat the system's name as the implementation; the paper introduces *the framework* — a structural-probing approach to context-conditioned representations of curated symbolic fields, motivated by Lakota relational epistemology and developed within the Abundant Intelligences program. The implementation is one realization of the framework; the contribution is the framework itself.
