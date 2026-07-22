# AAAI title + abstract (draft, 2026-07-22)

> **READ FIRST — integrity note.** Everything below is written from results that are IN HAND, except one
> sentence marked `[RESULT — fill from the campaign]`. That sentence is the graph-vs-no-graph headline and
> it is being computed right now by the 5-seed confirmatory val campaign. Do NOT submit with a fabricated
> number there. Two honest phrasings are supplied — use whichever the campaign actually yields.
>
> **Scope caveat you must keep:** all reported numbers are on the DEVELOPMENT (val) fold. The sealed
> challenge split is sequestered. If AAAI reviewers will expect a held-out test number, either run the
> steward-only sealed evaluation once, or state the development-fold scope explicitly in the paper.

---

## Title — three options

**A. (balanced — recommended while the result is pending; survives either outcome)**
Does the Graph Help? Diagnosing a Silent Regularizer Confound in Protein–Program Graphs for T-Cell
Perturbation Prediction

**B. (finding-forward; best if the corrected comparison stays null/at-parity)**
When the Graph Switches Itself Off: A Regularizer Confound in Evidence-Gated Perturbation Graphs

**C. (model-forward; best if the corrected graph reliably wins)**
Evidence-Gated Intervention-Informed Protein–Program Graphs for T-Cell Perturbation Response Prediction

---

## Abstract (~190 words; one sentence pending)

Predicting how primary human T cells respond to genetic perturbations is central to immunotherapy design,
yet whether protein-interaction priors actually help over expression-only baselines remains unsettled. We
present EG-IPG, an evidence-gated, intervention-informed graph model that predicts perturbation-induced
transcriptional program responses, emits calibrated uncertainty, and returns a faithfulness-audited
minimal predictive subgraph as a rationale. Evaluating the graph prior rigorously proved harder than the
modeling: we show that a standard edge-sparsity regularizer—an unnormalized sum over edges—dominates the
gradient direction of the model's learnable edge gates and drives them to ~1e-7 within the first epoch in
every seed, silently disabling message passing so that "graph" and "no-graph" models become identical up
to spare parameters. A prior multi-seed comparison was therefore confounded by construction rather than
informative. **[RESULT — fill from the campaign: see the two phrasings below.]** Under a pre-registered
protocol—leakage-safe target-grouped splits, an architecture search confined to an inner training fold,
five paired seeds, and family-wise error control (both Bonferroni and Holm)—we re-measure the graph's
contribution honestly. Our diagnosis, correction, and evaluation protocol form a reusable template for
claiming, or refuting, graph benefit in biological prediction.

---

## The pending sentence — two honest versions

**If the campaign CONFIRMS a graph benefit** (condition_gated − expression_only > 0, CI excludes zero,
survives BOTH corrections):
> Repairing the objective so the gates survive, the evidence-gated graph reliably outperforms an
> expression-only baseline (Δsystema = [X], 95% CI [[lo], [hi]], p = [p] after Holm and Bonferroni) across
> five paired seeds, while restoring the interpretable, faithful rationales the collapsed model could not
> produce.

**If the campaign is NULL / at parity** (CI crosses zero after correction):
> Repairing the objective so the gates demonstrably function, the graph still does not reliably beat an
> expression-only baseline (Δsystema = [X], 95% CI [[lo], [hi]], crosses zero after correction)—so the
> earlier apparent negative was untested-by-construction, and the corrected comparison is an honest
> parity; the model's value then rests on calibrated, faithful rationales rather than raw accuracy.

---

## Where each number comes from (so nothing is invented)

- Regularizer magnitude / gate collapse (~103×, ~1e-7, all seeds, epoch 0): `docs/h1-optimization-notes.md`,
  `next_goal_after_gate_collapse.txt`. SOLID.
- "Prior comparison confounded by construction": the 5-seed campaign's graph arms had dead gates.
  `feature_list.json` feat-011 evidence. SOLID.
- Δsystema, CI, p, corrections: `data/results/screening_lambda0/robustness_5seed.json` — PRODUCED BY THE
  RUNNING CAMPAIGN. Do not fill until it exists; get them with `multiseed.py --seeds 0,1,2,3,4`.
- Architecture search (design choices ablated): `docs/feat011-arch-search-notes.md`. SOLID (inner-holdout
  selection, n=1 — describe as selection, not as the headline).
- Rationale faithfulness audit: feat-012, runs on the campaign's live-gate checkpoints — PENDING.

## What still has to be true before this is submittable

1. The campaign lands and `robustness_5seed.json` gives real contrasts → fill the pending sentence.
2. Pick the title matching the outcome (A now; B or C once known).
3. Decide dev-fold vs sealed-test framing (your governance call).
4. The faithful-rationale claim needs the feat-012 audit to have RUN on a live-gate checkpoint (pending).
