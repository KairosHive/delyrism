# Plan — The Loom: Tradition-Anchored Generative Design through Symbolic-Field Distortion

*Third leg of the program. For later — drafted alongside `PLAN_followup_archetypal_systems.md`.*

---

## 0. Where this sits

The program is a trilogy, each leg evaluable on its own terms:

| Leg | Paper | Move | Genre / venue |
|---|---|---|---|
| **Instrument** | v2 (`neurips2025_v2.tex`) | *measure* contextual distortion in one curated field | ML / NeurIPS |
| **Atlas** | `PLAN_followup_archetypal_systems.md` | *map* how many systems engage a text (modes, bridges) | cultural analytics / CogSci |
| **Loom** | *this plan* | *weave* — use the readout generatively to inform design grounded in cultural narrative | design research / HCI |

The Loom is the most distinct in genre. Its evidence is **case studies, artifacts, and participatory critique**, not held-out accuracy. It is also the leg with the heaviest ethical load, because it *generates new things from cultural narratives* — which is exactly why it must be its own paper rather than a section bolted onto a methods paper. v2 already plants the bridge paragraph (§ Creative Applications) and a forward-pointer (§ Discussion); this plan is what that paragraph grows into.

---

## 1. One-line thesis

> The same instrument that reads how a story rewires a symbolic field can be run *inward* — surfacing the relational structure that one or many narratives activate, and offering that structure to designers and knowledge holders as a thinking surface for new, tradition-anchored design — provided the practice is participatory, consented, and co-authored rather than extractive.

## 2. Abstract (draft)

Design has always drawn on stories, but computational design tools flatten narrative into prompts or retrieved text, losing the *relational* structure that gives a tradition's stories their meaning. We present a generative design method built on the Δ symbolic-field instrument: given a narrative (or several), the readout surfaces which relations among a curated symbolic field's facets the story strengthens, weakens, or bridges — a legible map of the symbolic structure a story activates. We introduce a **weave** operation that combines the structures several narratives induce into a shared backbone, narrative-specific signatures, and *emergent bridges* that appear only when stories are read together, and we show how this structure feeds design: as extractable motifs, as a constraint surface a design is asked to honor, as a structural conditioning signal for generators, and as a tunable ambiguity dial. We develop and assess the method as participatory design research — co-design sessions, artifact analysis, and knowledge-holder critique — under community-governed protocols, and we foreground the ethics of working *with* living symbolic traditions rather than extracting from them. The contribution is a generative, culturally anchored complement to the analytic instruments of the program, and an honest account of what an embedding-derived structure can and cannot contribute to design that belongs to a community.

## 3. Contributions

1. **A generative inversion of the Δ readout** — from "how does this context rewire the field" to "what structure does this narrative offer a designer," with the human in the loop, not the autogenerator.
2. **The weave operation** — a principled way to combine the symbolic structures of multiple narratives (shared backbone / per-story signature / emergent cross-narrative bridges), at descriptor granularity with magnitude aggregation for display (the lesson carried from v2's soundscape: signed symbol-level aggregation cancels).
3. **Four design affordances grounded in field logic** — motif extraction, constraint surface, structural prompting, ambiguity dial — each tied to a concrete Δ readout rather than free-association.
4. **A participatory design-research evaluation** appropriate to the genre, plus an honest ethics framework for tradition-anchored generative work.

## 4. Method

### 4.1 Reading a narrative's symbolic structure (from v2)
Run the calibrated Δ readout on a story or its fragments → per-symbol response trajectory (§4.3 of v2), the Δ-graph of strengthened/weakened couplings (§4.1), and the de-biased signature. This is the "what does this story activate" map; the instrument already exists.

### 4.2 The weave — combining narratives
Given narratives `N_1 … N_k` (one community's corpus, or — only with consent — several traditions), compute each fingerprint, then:

- **Warp (shared backbone):** couplings strengthened across all/most narratives — the common relational core the stories agree on.
- **Weft (per-narrative signature):** couplings distinctive to a single narrative (its residual against the set mean) — what each story uniquely brings.
- **Emergent bridges:** couplings that appear only when narratives are read *together* (combined/sequential context) — high Δ but low in any single story, analogous to the cross-system bridges in the atlas plan; distinguish *emergent* (created by the combination) from *reinforced* (a resting coupling the stories amplify) via the resting Gram `C0`.
- **Operations:** union (associative breadth), intersection (shared core), difference (one narrative's signature vs the rest), and a **blend trajectory** (morph between narratives, as in the sound morph) — a continuous dial between two story-worlds.
- **Compute discipline:** descriptor-pair granularity; aggregate to symbol pairs by magnitude only for display; null-baseline the combination so "emergent" never silently means "noise."

### 4.3 From structure to design
The surfaced structure becomes design input, four ways:

- **Motif extraction:** top couplings and migrated descriptors as seed motifs for visual / object / spatial / experiential design.
- **Constraint surface:** the structure as a set of relations a design should *honor* (e.g., hold the coupling a community's stories foreground) — design-by-constraint rather than design-by-generation.
- **Structural prompting** (from v2): condition a generator on the structural motifs (top Δ-edges, descriptor migrations, trajectory turning points) so output stays in the field's logic.
- **Ambiguity dial** (from v2 §4.4): tune associative divergence (high-entropy context) vs tight grounding (low-entropy) — ambiguity as a design control surface.

### 4.4 The human loop
This is the load-bearing design move: the instrument is a **thinking surface in a co-design session**, not an autonomous generator. Outputs are co-authored with designers and knowledge holders; the readout's role is to make relational structure legible and navigable, surfacing directions a practitioner might not otherwise consider, while authorship and judgment stay human and community-held.

## 5. Evaluation (design-research standards, not accuracy)

- **Co-design / workshop studies:** does the surfaced structure help practitioners ideate? Think-aloud, artifact analysis, reflective critique. Primary evidence.
- **Knowledge-holder critique:** are the surfaced structures faithful and respectful to the tradition? The primary *correctness* check (mirrors v2's stance that community review is the real validation).
- **Contrastive probe (carefully, not a horse-race):** design-with-instrument vs design-without on matched briefs, to surface *what the instrument adds* — read qualitatively, not scored.
- **Artifact analysis:** outputs judged by design-research criteria (novelty, narrative groundedness, cultural specificity), via expert panels.
- **Honest negative space:** report where the structure adds nothing beyond reading the stories directly — the go/no-go question of §9.

## 6. Ethics (load-bearing — this is why it's its own paper)

- **Whose stories, under what consent.** Community-authored corpora only; public-domain/secular material for general demonstration. No combining of sacred materials across traditions without explicit, specific consent.
- **Authorship & benefit.** Outputs co-authored; benefit-sharing; the community holds authority over what is made from its narratives and how it circulates.
- **Refusal & sovereignty.** CARE, IPAI, Indigenous data sovereignty; right to withhold, withdraw, and decline circulation at any stage.
- **Lakota material gated by default.** Consistent with the rest of the program: `lakota_descriptors.py` is community material, not to be modified, benchmarked, or freely combined; sacred-text fragments require knowledge-holder review; no invented paraphrases of sacred passages. The Lakota Shape Kit appears in the Loop only with the knowledge holder's direction, never as a default demonstration set.
- **Anti-appropriation as a design stance.** The paper's spine is *working with* vs *extracting from*. The instrument never licenses harvesting a tradition's structure for design that the community neither authors nor benefits from. Make this a stated design principle, not a disclaimer.
- **Lineage:** Indigenous design and Two-Eyed Seeing (Etuaptmumk), decolonial and more-than-human design, participatory / co-design, Abundant Intelligences.

## 7. Related work to situate against

Participatory & co-design; Indigenous and decolonial design; creativity-support tools and computational creativity; narrative-based and story-driven design; design thinking / generative design methods; cultural analytics. Position the contribution as *relational-structure surfacing for design* — distinct from RAG (retrieves text), from style transfer (mimics surface), and from prompt engineering (passes raw context).

## 8. Honest caveats (carry the program's discipline)

- The Δ readout is, to high accuracy, a first-order re-expression of the encoder's contextualization (v2 §4.2). Its value here is **legibility and navigability for design**, not access to structure the model lacks.
- The instrument reads how a *model* represents a field; it does not validate cultural meaning — only knowledge holders do.
- Appropriation risk is real and central; the method is only as ethical as the consent and co-authorship around it.
- "Emergent bridges" between narratives must be null-tested; otherwise the weave invents connections.

## 9. Minimal de-risking prototype (go/no-go before committing)

A small **secular** case study: combine 3–4 public-domain narratives (e.g., Aesop fables, or a set of public folk tales) over a generic symbolic field; run the weave (warp / weft / emergent bridges); hand the surfaced structure to a designer (or run a structured self-study) on a concrete brief. **Gate:** does the structure produce usable, narrative-grounded design directions that a practitioner judges as adding something *beyond reading the stories directly*? If yes → scope the participatory study with community partners. If no → the weave collapses to "read the stories," and the Loom stays an affordance note in v2 rather than a paper.

## 10. Venue options

Primary: **DIS**, **Creativity & Cognition (C&C)**, **CHI** (design/HCI). Also: **TEI** (if embodied/installation), **Design Issues** / **Digital Creativity** (journal), or a cultural-analytics track. Frame under Abundant Intelligences throughout.

## 11. Relationship to the other two legs

- Uses the **instrument** (v2) to read each narrative.
- Borrows the **atlas's** cross-system bridge math for the weave's emergent bridges (`C0`-conditioned emergent-vs-reinforced split, descriptor-level compute, null baselines).
- Is the only leg whose primary output is *artifacts and practice*, and whose primary evaluator is the community — which is why it stands alone.
