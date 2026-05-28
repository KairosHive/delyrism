# NeurIPS 2025 Creative AI Track — Submission #151

**Title:** Decoding Ambiguous Signs: Context-Sensitive Alignment of Visual Symbolic Systems and Language Embedding Models

**Authors:** Antoine Bellemare-Pepin, Suzanne Kite

**Submitted:** 10 Aug 2025
**Decision:** Reject (27 Sept 2025)

---

## Program Chair Decision

> Reject. While this paper tackles visual symbolic systems with embedding models, the embedding techniques have been widely explored. However, the authors do not perform quantitative comparisons — it remains unclear whether existing embedding models can solve this problem. Although the authors state relations to creative applications, existing models may be able to solve those problems, e.g., we can merely use ChatGPT to achieve poetry composition, without the need of external embedding models.

**Three concrete deficiencies:**

1. Embedding techniques are well-known; novelty unclear.
2. No quantitative comparisons against baseline embedding models.
3. Creative applications (poetry, story) already solvable by general-purpose LLMs — what does our framework add?

---

## Official Review — Reviewer R7Mo

**Rating:** 3 (Marginally around acceptance threshold, borderline)
**Confidence:** 3 (Reviewer fairly confident)
**Date:** 1 Sept 2025

> This paper models the ambiguity of visual symbols — using the Lakota Shape Kit as its case — by treating a symbol not as one fixed meaning but as a distribution that shifts with context. The paper provides visualization of semantic neighborhoods and quantitative ambiguity metrics, show how salient facets change, pairwise co-activations.
>
> It's a thoughtful, well-framed analysis and visualization of context-sensitive meaning for Creative AI track, and the figures make the ideas legible. However, there's also a lack of evidence beyond the diagrams: there's no user study with community experts, no task-based evaluation, and the approach depends on author-chosen descriptors and a particular embedding model.

**Four concrete deficiencies:**

1. Evidence is qualitative (diagrams) only.
2. No user study with community experts.
3. No task-based evaluation.
4. Brittle dependence on author-chosen descriptors and a single embedding model — robustness unestablished.

---

## Summary of attack surfaces

| # | Concern                                          | Source        | Severity |
|---|--------------------------------------------------|---------------|----------|
| 1 | Embedding technique novelty                      | Chair         | High     |
| 2 | No quantitative comparison vs baselines          | Chair, R7Mo   | High     |
| 3 | No task-based evaluation                         | R7Mo          | High     |
| 4 | No expert/community user study                   | R7Mo          | Med–High |
| 5 | Brittle to descriptor choice                     | R7Mo          | Medium   |
| 6 | Brittle to embedder choice                       | R7Mo          | Medium   |
| 7 | LLMs already do creative tasks                   | Chair         | Medium   |

All seven must be addressed (directly or by reframing) in v2.
