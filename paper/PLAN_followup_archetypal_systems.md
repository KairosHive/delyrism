# Follow-up paper — Comparative Archetypal Systems

**Working title (options):**
- *How Symbolic Systems Read a Text: Modes of Engagement and Cross-System Correspondence in Contextual Embeddings*
- *Many Lenses, One Text: A Comparative Instrument for Archetypal Systems*

**Status:** concept / plan. Depends on the v2 paper (the Δ readout, calibration, null model, modality-agnostic context slot) being published first — this paper *cites* it as the instrument and contributes the comparative layer on top.

---

## 1 · One-line thesis

Given a curated symbolic/archetypal system (Jungian archetypes, astrology, classical elements, chakras, Plutchik emotions, tarot, …), the same context-conditioned coupling readout characterizes **how that system engages a text** — not whether it "wins." Across many systems we measure (a) the *mode* of within-system engagement (monotonic / polar / distributed), and (b) the *cross-system bridges* a text induces between concepts of different systems — and we test whether those bridges recover the historical correspondence tables that occultists, astrologers, and comparative mythologists built by hand.

The contribution is a **comparative, non-hierarchical instrument over symbolic systems**, plus a falsifiable question: *do contextual embeddings encode the cross-ontology correspondences that human symbolic traditions converged on?*

---

## 2 · Relationship to the v2 paper

| v2 (foundation) | This paper (extension) |
|---|---|
| One curated field (Lakota), Δ readout, calibration, null, robustness | *Many* systems compared with the same instrument |
| Kernel figure: same text through Lakota vs an emotion kit | Generalized to N systems + an explicit engagement/bridge layer |
| §4.4 ambiguity = scalar margin/entropy | **Full profile shape** (mode of engagement), not a scalar |
| Within-symbol descriptor couplings | **Cross-system** descriptor couplings (bridges) |
| Modality-agnostic probe (audio) | Carries over; texts are the primary probe here |

Everything technical is reused: `build_space(descriptors=...)` accepts any symbol→descriptor dict; `_delta_common.DeltaReadout` gives the calibrated readout; the null model and multi-embedder robustness transfer directly.

---

## 3 · Core contributions (graded honestly)

- **(A) Modes of engagement — strong/novel.** A text engages different systems with qualitatively different *activation topologies* (monotonic, polar, distributed). This is a property of the system-geometry × text interaction, and it does **not** reduce to "which descriptors are closest." Primary contribution.
- **(B) Cross-system correspondence recovery — strong/novel and falsifiable.** Build one combined space over all systems; measure context-induced couplings *between* concepts of different systems; test recovery of known historical correspondence tables (element↔chakra, planet↔tarot, etc.). Either result is publishable: recovery ⇒ embeddings encode the traditions' structure; non-recovery ⇒ the traditions encode something embeddings don't.
- **(C) The comparative instrument — moderate/framing.** A reproducible pipeline turning "which interpretive lens fits, and how" into measurable, size-normalized, null-referenced quantities — useful for digital humanities, comparative mythology, cognitive science.
- **(D) Honest scope (carried from v2).** *Which* symbols/concepts activate is largely first-order similarity. The value-add is the *mode* (B-emergent shape) and the *cross-system bridge structure*, which are non-trivial. State this plainly, as in v2.

---

## 4 · The systems

Start **secular / public-domain** for the methodological core:

| System | ~#symbols | Notes |
|---|---|---|
| Jungian archetypes | 12 | Hero, Shadow, Anima/Animus, Wise Old Man, Trickster, … |
| Astrology — planets | 10 | Sun…Pluto; well-documented significations |
| Zodiac signs | 12 | for element/modality structure |
| Classical elements | 4–5 | Earth/Water/Fire/Air(/Aether) |
| Chakras | 7 | ordered axis (root→crown) |
| Plutchik emotions | 8 | wheel with opposition structure (built in v2 as a kit) |
| Tarot — major arcana | 22 | rich, has Golden Dawn correspondences |
| (optional) Big Five / MBTI | 5/16 | psychometric control system |

**Lakota Shape Kit:** *not* in the comparison set by default. Including sacred community material in a multi-system study is gated on explicit community consent and Kite's direction, and even then framed as a separate, non-benchmarked case — never inside a leaderboard. The mode-of-engagement framing (§5) is non-hierarchical, which helps, but the decision is the community's.

Descriptor authoring: each symbol gets 8–15 plain-English descriptors from standard reference sources (for the public systems), curated to avoid lexical leakage of the symbol's own name — same discipline as the v2 oblique probes. Provenance documented per system.

---

## 5 · Methods

### 5.0 Per-system calibration (recap)
For each system S: build its space, fit a **per-system reference battery** of oblique probes to estimate the context-generic mean fingerprint and per-symbol (mean, std). All readouts are de-biased against S's own battery — essential for fair cross-system comparison.

### 5.1 The within-system profile
For text `t` and system `S`, the readout gives a de-biased per-symbol response vector `a = wsz_S(t)` over S's symbols. Two derived objects:
- a **distribution** `p = relu(a) / Σ relu(a)` (symbols above baseline; preferred for shape metrics — interpretable as "above-resting activation"), and/or `p_τ = softmax(a/τ)` for a temperature-controlled view.
- the **structured-vs-null gate** (see 5.4).

### 5.2 Mode of engagement — how to compute distributed activation
Compute on `p` (after the null gate says the engagement is structured at all):

- **Effective support** `k_eff = exp(H(p))`, `H` = Shannon entropy. `k_eff ∈ [1, |S|]`. ≈1 monotonic, ≈2 polar, large ⇒ distributed.
- **Dispersion** `Disp = k_eff / |S| ∈ (0,1]` — size-normalized so it's comparable across systems of different cardinality. *This is the principal "distributed activation" statistic.* Also report the participation ratio `(Σ a_+)² / Σ a_+²` as a robustness check (a ≥ 0 part).
- **Peakedness** `Peak = p_(1)` and margin `m = a_(1) − a_(2)` (z-units) — monotonic signature.
- **Polarity / bimodality** — to separate "polar" (two real poles) from "monotonic" and "distributed":
  - `Pol = p_(2) / p_(1)` (≈1 ⇒ two co-equal) **and** top-2 mass `p_(1)+p_(2)` high while the rest are flat. A clean rule: *polar* if `p_(1)+p_(2) > 0.6` and `p_(2)/p_(1) > 0.5` and `k_eff < 3`.
  - Optionally a Hartigan dip test or bimodality coefficient on the sorted profile for a distribution-level statement.
- **Mode coordinates, not hard labels.** Place each (t, S) in a 2-D **mode space** with axes (Peak, Disp) — monotonic ↑Peak/↓Disp, distributed ↓Peak/↑Disp, polar in between with high `Pol`. Show the continuum; use the rule above only for summary counts. A ternary (monotonic / polar / distributed) is a clean alternative display.

**Normalization across systems (the whole ballgame).** Because `|S|`, descriptor counts, and anisotropy differ, raw `k_eff`/`Peak` aren't comparable. Two safeguards: (i) `Disp` is divided by `|S|`; (ii) standardize every mode coordinate against system S's **own null distribution** (5.4) → report `z`-scored coordinates so "monotonic for S" means "more peaked than S's chance baseline," not "more peaked than another system."

### 5.3 Cross-system bridges — concepts coupling across archetypes
Build **one combined space** `D_all = concat_S(D_S)`, row-normalized; tag each descriptor with `(system, symbol)`. Baseline Gram `C0 = D_all D_allᵀ` already contains within- and cross-system couplings.

Under a context: `D'_all = Shift(D_all, c, θ)`, `Δ = D'_all D'_allᵀ − C0`, diagonal zeroed.

- **Cross-system edges** = `Δ_ij` with `system(i) ≠ system(j)`. *Compute at descriptor granularity* and select top edges by `|Δ_res|` (residual vs a text battery mean) — symbol-level *signed* aggregation cancels (the lesson from v2's soundscape); aggregate to system↔symbol or symbol↔symbol only for display, using **magnitude**.
- **Two regimes per cross-pair**, using baseline `C0_ij`:
  - **Emergent bridge** — high `Δ_ij`, low `C0_ij`: the context *creates* a correspondence not present at rest. (Most interesting / discovery-oriented.)
  - **Reinforced correspondence** — high `Δ_ij`, high `C0_ij`: a stable link the context amplifies. (Most relevant to validating historical tables.)
- **Text-specificity filter.** Residualize bridges across a battery of texts; the constant component is anisotropy/generic-coactivation, the *text-specific residual* is the meaningful, navigable bridge. Report both, but headline the residual.
- **Honest caveat.** A cross-bridge under a dark text between Jung-Shadow and Pluto may be mere co-activation (both relevant to "dark"). The emergent-vs-baseline split and the text-specificity filter are what separate a *correspondence* from *co-relevance*; state this.

### 5.4 Null model (shared by 5.2 and 5.3)
For each system / the combined space, generate random context shifts of matched `‖D'−D‖_F` (as in v2 §4.6). Gives: (i) the "structured engagement" gate (is `a` above chance at all), and (ii) null distributions for every mode coordinate and bridge weight, so all reported quantities are chance-referenced.

---

## 6 · Experiments / figures

- **E1 — Mode-of-engagement map.** Text × system grid; each cell colored by mode (or a glyph showing the profile). Reveals system tendencies (columns), text fingerprints (rows), and the interesting **mode flips** (a text monotonic everywhere but polar in one system has found that system's dialectic). *Headline figure.*
- **E2 — System signatures.** Each system's *characteristic* mode tendency over a text corpus (some systems are constitutionally monotonic, others holistic). A point per system in mode space, with spread.
- **E3 — Cross-system bridge networks.** Per representative text, a multipartite graph: systems as groups, concepts as nodes, top text-specific cross-bridges as edges (emergent vs reinforced styled differently). Shows which Jung↔planet↔chakra correspondences a text lights up.
- **E4 — Correspondence recovery (validation).** Encode known tables (element↔chakra, planet↔tarot/Golden Dawn, sign↔element, …) as ground-truth graphs. Metric: do baseline `C0` and/or context-aggregated cross couplings rank true correspondences above non-correspondences (AUC, precision@k)? Two regimes (baseline geometry vs apt-text-modulated). *Falsifiable, novel either way.*
- **E5 — Grounding / construct validity.** Texts deliberately written in one system's register → does the readout localize to that system (and to the intended concept)? Cross-system analog of v2's oblique-probe grounding.
- **E6 — Robustness.** Multi-embedder agreement of modes and bridges; descriptor-stability bootstrap; everything null-referenced. Carried from v2 §4.6.

---

## 7 · Validation against correspondence tables (detail)

Candidate ground-truth sources (public, documented):
- **Element ↔ chakra** (earth–root, water–sacral, fire–solar-plexus, air–heart, …) — widely tabulated.
- **Zodiac sign ↔ element / modality** (fire: Aries/Leo/Sagittarius; cardinal/fixed/mutable).
- **Planet ↔ tarot major arcana** and **element ↔ suit** (Golden Dawn / Rider-Waite tradition).
- **Planet ↔ classical element / temperament** (traditional rulerships).

Metrics: rank-AUC and precision@k of embedding-derived cross couplings against each table; report per-table and pooled. Distinguish *baseline* recovery (geometry alone) from *context-modulated* recovery (does an apt text sharpen the right correspondences). A clean negative result is interesting too ("LLMs encode element↔chakra but not Golden Dawn planet↔tarot").

---

## 8 · Honest framing & limits (must stay in the paper)

- *Which* concepts activate ≈ first-order similarity; the **mode shape** and **cross-bridge structure** are the non-trivial parts. Same characterization discipline as v2 §4.2.
- "Distributed" must be split into *structured-holistic* (above null) vs *flat-no-fit* (at null) — never conflate.
- Cross-bridges: correspondence vs co-relevance separated by emergent-vs-baseline + text-specificity.
- Descriptor curation drives results → report provenance, run descriptor-stability bootstrap, and frame curation as a modeling choice, not ground truth.

---

## 9 · Ethics

- **Non-hierarchical by construction.** "Modes of engagement" and "correspondences" are descriptive, not a ranking — no system is declared "best." This is both better science and the safer framing.
- **Lakota material excluded by default** from the comparison; any inclusion is community-led, consented, framed separately, and never benchmarked against pop-spirituality systems. Decision belongs to Kite / the community (cf. the parked "Analysis 7" in the v2 PLAN).
- Astrology/tarot/chakras treated as *cultural-symbolic systems worth modeling*, neither endorsed nor debunked; the claim is about their representational structure in embeddings, not their validity.

---

## 10 · Corpus of texts

Diverse, oblique (not naming concepts): dream reports, myths/folktales, lyric poems, short news vignettes, emotional micro-narratives, philosophical fragments. Plus a constructed set with known intended system (for E5). Watch licensing; prefer public-domain (myth/folktale collections, PD poetry) and short author-written probes.

---

## 11 · Venue

- **Cultural Analytics / CHR (Computational Humanities Research)** — strongest fit for comparative symbolism.
- **Cognitive Science (CogSci)** — the "do embeddings recover human correspondence systems" question.
- **NLP (Findings / *SEM)** — framed as cross-ontology correspondence in contextual representations + the mode-of-engagement probe.

---

## 12 · Risks / open questions

1. Do modes genuinely differ across systems, or do all profiles look the same once null-normalized? (De-risk first — §13.)
2. Are cross-bridges text-specific, or dominated by a constant anisotropy mode (as the raw kernel was in v2)?
3. Does correspondence recovery beat chance, and is it in baseline geometry or only under apt context?
4. Descriptor-set sensitivity — how stable are modes/bridges to curation?
5. Temperature/τ and the relu-vs-softmax choice for `p` — pick by a principled rule, report sensitivity.

---

## 13 · Minimal de-risking prototype (run first)

Before committing, run on **4 secular systems** (Jung, astrology-planets, elements, Plutchik) over ~25 oblique texts:
1. Per-system calibration + null.
2. Mode coordinates (Peak, Disp, Pol), null-z-scored → does the mode space show separation and at least some **mode flips**?
3. Combined space → top text-specific cross-bridges; eyeball interpretability.
4. One correspondence table (element↔chakra is cleanest) → AUC vs chance.

Decision gate: proceed only if (2) shows real mode diversity **and** either (3) is interpretable or (4) beats chance. Report honestly if it collapses to similarity — same standard we've held throughout.
