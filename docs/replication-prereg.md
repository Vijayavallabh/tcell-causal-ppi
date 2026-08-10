# Pre-registration — multi-dataset replication of the EG-IPG null

**Frozen 2026-08-03, before any replication lane was launched. This file is not edited after the
first lane lands.** Corrections, if a genuine error is found, are appended as dated
`AMENDMENT (YYYY-MM-DD)` blocks at the bottom, never by editing the text above them. The dataset
evidence this rests on is `docs/replication-dataset-survey.md`, written the same day.

The point of pre-registering a *negative* result is narrow and specific: a null is cheap to
manufacture. Every knob left open until after the numbers are visible (which DE test, which seeds
count, which contrast is "the" contrast, whether a collapsed-gate run is a result) is a knob that can
be turned toward the answer already written in the paper. This document closes them.

---

## 1. Hypotheses and the contrast family

The family is the four pre-registered contrasts already implemented in
`screening/multiseed.py:CONTRASTS`, on the primary endpoint `systema`, paired by seed:

| Key | Contrast | Question |
|---|---|---|
| `h1_vs_no_graph` | `condition_gated` - `expression_only` | does the evidence-gated graph beat no graph? |
| `h2a` | `typed_static` - `expression_only` | does a typed static graph beat no graph? |
| `h2b` | `condition_gated` - `typed_static` | does condition gating beat static? |
| `promotion_margin` | `untyped_gnn` - `expression_only` | does *any* graph beat no graph? |

Statistic: one-sample two-sided t on the per-seed differences, df = n-1, 95% CI = mean +/- t*se.
Multiplicity: **both Bonferroni and Holm over the family of four, both always reported**, and
`survives_family_wise` requires BOTH. Choosing whichever correction rescues a claim after seeing the
numbers is the look-elsewhere effect; recording both makes it unavailable.

**Primary contrast is fixed per dataset by design, not by outcome:**

- Datasets with **>= 2 experimental conditions** (`perturbation_2` present): primary is
  `h1_vs_no_graph`. Currently Frangieh (3), Shifrut (2), Datlinger (2).
- Datasets with **1 condition**: `condition_gated` degenerates to `typed_static`, because its
  condition embedding is indexed by a constant. h1 is **not defined** there and will not be reported
  as if it were. Primary is **`h2a`**. Currently Replogle, Norman, Papalexi.

This assignment is made now, from the survey table, before any model has been trained.

## 2. Design held fixed across every dataset

1. **Program basis refit INSIDE the train fold.** No basis fitted on data that includes val or any
   held-out target. The basis is a response-derived transformation and is fenced accordingly.
2. **Blocked target-OOD split**, sequence-similarity family blocking, same generator as the
   reference (`tcell_pipeline.splits`) into a fresh `SPLITS_ROOT` per dataset.
3. **Same four arms**, same nested family, same primary endpoint `systema`.
4. **`lambda_graph = 0`.** The unnormalised edge penalty at 0.01 annihilates the gates in epoch 0;
   running the replication at the confounded setting would test nothing.
5. **>= 4 seeds per arm**, seeds drawn in order 0,1,2,3,4.
6. **Epoch cap 20 with `EARLY_STOP_PATIENCE = 10`**, batch size 8, matching the reference campaign.
7. **PINNACLE context matched to the dataset's cell type** where one exists (survey section 3);
   where none exists, ESM-2-only node features, recorded in the run config as an explicit ablation.

## 3. Decisions that must be fixed before data are seen, and are

- **DE method.** Pseudobulk by (target, condition, replicate) using summed raw counts, then a
  moderated t-test against the matched control pseudobulks within the same condition, producing
  `log_fc`, `zscore`, `p_value`, `adj_p_value` (Benjamini-Hochberg), `baseMean`, `lfcSE`. Chosen
  because `pydeseq2`/`decoupler` are not installed and adding a dependency mid-campaign is itself a
  degree of freedom. If a dataset cannot yield >= 2 replicate pseudobulks per (target, condition),
  it is **dropped**, not switched to a different test.
- **Minimum cells per pseudobulk: 25.** A (target, condition, replicate) cell with fewer is dropped
  and the drop is counted in the dataset's report.
- **Gene space.** Symbol join to the reference's measured-gene space; a dataset's gene overlap is
  reported (survey section 2) and no gene is imputed.
- **Target space.** Single-gene targets only. Norman's double perturbations are excluded from the
  split and from every contrast.

## 4. Kill criteria — what makes a run not a result

- **Gate collapse.** Mean edge gate is logged **every epoch on every graph arm**. A final gate mean
  `<= 1e-3` means the graph was switched off, so the arm measures nothing about the graph. Such a
  lane is reported as **UNDECIDABLE**, is excluded from every contrast, and is never averaged in as
  evidence for the null. Undecidable is not the same as decided-against.
- **Underpowered target axis.** A dataset whose blocked split yields fewer than **50 held-out
  family groups** cannot support a family-blocked OOD claim; its result is reported as
  *preliminary/qualitative* and is barred from any headline or pooled estimate. On the survey
  numbers this is expected to bind on Shifrut (21 targets) and Datlinger (32).
- **Seed attrition.** A contrast's `n` is the number of seeds where BOTH arms completed. Any dropped
  seed is named in the report with its reason. `n` never shrinks silently.
- **Zero variance across seeds** is the signature of seeds that did not propagate, not evidence of
  an effect, and is reported as degenerate.

## 5. Integration rules

A per-dataset result may be promoted into the paper as a headline **only if all of**:

1. `>= 4` seeds completed for **both** arms of that dataset's primary contrast;
2. the dataset passed the target-axis criterion in section 4;
3. every graph arm in the contrast had live gates (`> 1e-3`);
4. the result is a **null (parity) or graph-worse**.

Anything failing 1-3 goes into the report only. Anything failing 4 triggers section 6.

**Pooling.** Once `>= 4` datasets have landed, pool per contrast with **both** a fixed-effect and a
random-effects (DerSimonian-Laird) estimate, and report a heterogeneity statistic (Q, I^2). h1 and
h2a are pooled **separately**, over their own eligible datasets, never merged into one "the graph
does not help" number. Weights carry the per-dataset target counts so the reader can see that one
dataset dominates. A pooled bounded null is the strongest available claim; substantial heterogeneity
is an equally publishable and more interesting one, and will be reported as such rather than
smoothed away.

## 6. Contradiction stop

If any dataset shows the graph **helping** — the primary contrast positive and surviving **both**
Bonferroni and Holm — then: do not rewrite the null, do not integrate, do not re-run the dataset
hoping it reverses. Snapshot the artifacts, flag it at the top of `RESULTS_SUMMARY.md` as
`UNEXPECTED — NEEDS HUMAN REVIEW` with the numbers, and continue the remaining datasets. A positive
replication is a finding, and adjudicating it is a human decision.

## 7. What would falsify the paper's claim

Stated now so it cannot be redefined later. The paper claims the typed protein-program graph does not
help, and that a prior apparent version of this result was confounded by a silent regulariser. That
claim is falsified if, on a dataset with live gates, a correctly context-matched graph, `>= 4` seeds
and an adequately powered blocked target-OOD split, the primary contrast is positive and survives
both corrections. It is *weakened* (not falsified) if the pooled random-effects estimate crosses zero
with heterogeneity so large that the datasets are evidently not measuring one quantity — in which
case the honest report is that the question is dataset-dependent, and the single-dataset null does
not generalise.

## 8. Reporting

Every result carries: `n`, the per-seed values, both corrected p-values, gate health per arm, epochs
run, the PINNACLE context used (or the ESM-2-only flag), the number of held-out family groups, and
the name and reason of any dropped seed.

---

## Amendments

### AMENDMENT (2026-08-03), before any replication lane launched

Section 3 fixed the DE unit as "(target, condition, replicate)" without saying what *replicate*
means, and the harmonised files do not agree on one. Inspecting the six candidates shows each
carries a different replicate column, so leaving the word undefined would have let it be chosen
after seeing results. Pinned now, per dataset, with the count actually present in the file:

| Dataset | Replicate unit | Column | Levels | Groups (target x condition) | Median cells/group | % groups >= 25 cells |
|---|---|---|---|---|---|---|
| ShifrutMarson2018 | donor | `replicate` | 2 (D1, D2) | 42 | 502 | 100% |
| DatlingerBock2017 | experimental replicate | `replicate` | 6 | 64 | 74 | 89% |
| FrangiehIzar2021 | sgRNA targeting the same gene | `sgRNA` | 819 (~3/target) | 747 | 217 | 94% |
| NormanWeissman2019 | 10x lane | `gemgroup` | 8 | 237 | 354 | 100% |
| ReplogleWeissman2022 (RPE1) | batch | `batch` | 56 | 2,394 | 72 | 89% |

Rationale for the one non-obvious choice: Frangieh has no replicate/donor/batch column, so the
replicate unit is the **sgRNA**, which is the standard within-target biological replicate in pooled
Perturb-seq. This is recorded because it is a weaker replicate than a donor or a batch: guides
targeting one gene share the biological sample and differ only in cut site, so their variance
understates true biological variance and the resulting p-values are anti-conservative. Frangieh's
contrasts are therefore interpreted on effect size and CI, not on the p-value alone.

Shifrut's 2 donors are the minimum the section-3 rule allows (`>= 2` replicate pseudobulks). It
passes, but with df = 1 on the treatment side its CIs will be very wide, which compounds the
target-axis limitation already recorded for it in section 4.

No other section is changed.
