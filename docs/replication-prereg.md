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

### AMENDMENT 2 (2026-08-10): the replicate-pseudobulk rule was stricter than the reference itself

**This amendment RELAXES a rule, which is the direction that deserves scrutiny, so the justification
and the timing are both stated explicitly.**

**The error.** Section 3 required "two or more replicate pseudobulks per (target, condition)" with a
25-cell floor, and section 4 dropped any dataset that could not meet it. That rule was never applied
to the reference dataset. The reference's supervised target is
`data/raw/GWCD4i.DE_stats.h5ad`, a 16.8 GB artifact **downloaded from the source publication** and
produced by the authors' own differential-expression pipeline over all cells per (perturbation,
condition) against matched controls. So the design compared replication datasets, held to a
per-replicate pseudobulk standard, against a reference held to no such standard. That is not a
conservative choice; it is an inconsistent one, and it discards datasets for failing a test the
reference never took.

**The correction.** The DE unit becomes: **all cells for a (target, condition) against the pooled
control cells of the same condition**, which is the standard Perturb-seq contrast and what the
reference uses. The 25-cell floor is retained, now applied per (target, condition) rather than per
(target, condition, replicate). Uncertainty comes from the cell-level fit rather than from replicate
pseudobulks. Where a dataset does carry a genuine replicate axis (donor for Shifrut, gemgroup for
Norman, batch for Replogle), it is recorded in the provenance and used as a covariate if the DE
method supports one, but its absence no longer excludes a dataset.

**Why this is not result-shopping.** No replication model has been trained. Zero replication arms have
run, so no replication result exists that this amendment could have been chosen to favour. The change
was prompted by an audit of how the reference target was constructed, not by any outcome. It is also
outcome-symmetric: it admits datasets that could support a positive replication just as readily as a
null. The pre-registered contrast family, the n>=4 bar, both corrections, the gate-collapse kill
criterion, the blocked target-OOD split, and the fold-local basis refit are all UNCHANGED.

**What it changes in practice.** Under the corrected rule, targets clearing the floor:
Replogle RPE1 **2,122** (was 12), Frangieh **246** (was 216), Norman **105**, Papalexi 25,
Datlinger 31, Shifrut 21. Replogle RPE1 moves from "dropped" to the best-powered candidate available,
within an order of magnitude of the reference's 11,526 targets.

**What does NOT change.** The target-axis floor of section 4 still applies: a dataset yielding fewer
than 50 held-out family groups is reported as preliminary and barred from any headline or pooled
estimate. On these numbers that still binds on Shifrut, Datlinger and Papalexi. Single-condition
datasets still make h2a the primary contrast, because `condition_gated` degenerates there.

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

---

## Amendment 3 — 2026-08-10 (BEFORE the first replication lane is trained)

Written before any replication arm has been trained, so no result can have motivated it. Two
decisions were forced by measurement during the stage 4-6 wire-up.

### 3.1 Program dimension K is not portable across datasets

The reference screen fits a K=128 fold-local program basis on 33,983 DE rows. A replication dataset
has one DE row per (target x condition), so its row count is its target count times its condition
count, and the train fold is 60% of that. K cannot exceed the number of train rows: the basis is
rank-limited by its own input.

Measured train-fold rows, and the K each dataset can carry:

| Dataset | DE rows | train rows | K |
|---|---|---|---|
| ReplogleWeissman2022_rpe1 | 2,122 | ~1,273 | **128 (reference value)** |
| ReplogleWeissman2022_K562_essential | 2,003 | ~1,201 | **128 (reference value)** |
| FrangiehIzar2021_RNA | 702 | ~421 | **128 (reference value)** |
| TianKampmann2021_CRISPRi | 184 | ~110 | 32 (deviation) |
| TianKampmann2021_CRISPRa | 100 | ~60 | 16 (deviation) |
| NormanWeissman2019_filtered | 105 | ~63 | 16 (deviation) |
| PapalexiSatija2021_eccite_RNA | 25 | ~15 | 8 (chain smoke only, never headlined) |

RULE: K = 128 wherever train rows >= 256; otherwise the largest power of two <= train_rows/2.
Any dataset run at K != 128 is a **deviation from the reference architecture** and must be labelled
as such wherever it appears. The pooled estimate is reported TWICE - over all datasets, and over the
K=128 subset alone - and if those two disagree, the K=128 subset is the one that speaks to the
reference architecture. Rationale for pre-registering the rule rather than dropping small datasets:
silently shrinking capacity on some datasets and not others is exactly the kind of unlogged
weakening that manufactures a null, and dropping them instead would leave the design at three.

### 3.2 Primary contrast per dataset is fixed by its condition count, not chosen later

The condition gate needs >= 2 contexts. On a single-condition dataset `condition_gated` is
arithmetically identical to `typed_static` - it would report a number, and the number would be
uninformative about gating. Fixed in advance:

- **FrangiehIzar2021_RNA** (3 conditions) - PRIMARY h1: condition_gated vs expression_only.
  This is the ONLY qualified dataset that can test h1 at all.
- **every other dataset** (1 condition) - PRIMARY h2a: typed_static vs expression_only.
  `condition_gated` is NOT run there; its absence is by design, not attrition.

h1 and h2a are pooled SEPARATELY and never merged. h1 therefore pools over n=1 dataset, which is a
stated limit of this replication, not a result about gating.

### 3.3 PINNACLE context assignment (fixed now, logged per lane)

| Dataset | Cell type | PINNACLE context used |
|---|---|---|
| FrangiehIzar2021_RNA | melanoma | `melanocyte` |
| ReplogleWeissman2022_rpe1 | RPE1 | `retinal_pigment_epithelial_cell` |
| ReplogleWeissman2022_K562_essential | K562 | **none** - ESM-2 features only |
| TianKampmann2021_CRISPRi/a | iPSC neuron | matching context if present, else **none** |

Where the context is `none` the graph arm carries ESM-2 node features and no PINNACLE channel. That
is a weaker graph arm by construction and is reported as such; it is recorded here so it cannot
later be mistaken for evidence about the graph.

### 3.4 Kill criteria (unchanged in substance, restated for the replication lanes)

A lane whose mean edge gate falls to <= 1e-3 is an UNDECIDABLE experiment and is reported as such -
never as evidence the graph does not help. Gate mean is logged every epoch on every graph arm.

---

## Amendment 4 — 2026-08-16 (BEFORE any A1 lane is trained)

Registers a MECHANISTIC DIAGNOSTIC, not a confirmatory hypothesis. Nothing here changes the
confirmatory family, its `family_size`, or any landed aggregation.

### 4.1 What question this answers, and why it needs pre-registering

On the frozen fold at n=7, edge typing costs h2a = -0.0120 systema (7/7 seeds, survives Bonferroni
AND Holm) while the untyped GCN is the best graph arm at +0.0043. That says WHICH component costs
the graph its benefit and not WHY. Two explanations are confounded inside the existing contrast:

  P  the relation PARTITION is the wrong inductive bias for this task;
  C  typed message passing carries 4x the message parameters over the same edges (2,396,160 against
     599,040 on the synthetic fixture), so the damage is CAPACITY and not about evidence types.

Both predict the same sign on h2a, so the landed family cannot separate them. Two new arms can.

### 4.2 The arms, fixed now

**`typed_shared`** — `SharedWeightTypedGraphEncoder`, `typed_static` with ONE `_RelMessage` tied
across all four relations instead of one each. Signed messages, edge features, complex nodes and the
gate pinned to 1.0 are all unchanged. Implemented in `src/tcell_pipeline/baselines/graph_baselines.py`
and covered by two tests in `src/tests/test_graph_baselines.py` (module identity, quarter parameter
count, live intervention against `typed_static`).

**`typed_permuted`** — per-relation parameters retained, but each PP edge's RELATION LABEL is
randomly reassigned among the three PP relations, preserving each relation's edge count exactly.
Edge attributes travel with the edge, so only which weight matrix processes it changes.
`complex_membership` edges are not permuted (they join different node types). The permutation is
drawn from the TRAINING SEED, so the five lanes average over five partitions rather than reporting
one lucky one.

### 4.3 The identifiability statement, recorded before the numbers exist

Under `norm='add'` (what `typed_static` runs) a layer computes `sum_r sum_{u in N_r(v)} f_r(u)`.
Tying `f_r = f` makes that identically `sum_{u in N(v)} f(u)`: the partition stops affecting the
aggregate at the same moment the parameters drop. **`typed_shared` alone therefore cannot attribute
a difference to parameter count.** This corrects the decision rule drafted in `NEXT_ACTIONS.txt`,
which read a positive `typed_shared - typed_static` as evidence for C on its own. It is not.
`typed_permuted` breaks the tie because it holds the parameter count at typed_static's while
destroying the typing's information content.

### 4.4 Decision rule, fixed before running

Primary endpoint `systema_pert_specific_delta`, paired per seed on the frozen `blocked_target_ood`
fold, seeds 0-4 (n=5, matching the landed reference family rather than a new n). Contrasts:

  D1  `typed_shared   - typed_static`
  D2  `typed_permuted - typed_static`

Read as a 2x2 on which of D1/D2 clear correction:

| | D2 null (typing carries no information) | D2 positive |
|---|---|---|
| **D1 null** | the typed STRUCTURE hurts, and neither its parameters nor its labels are the route | the true partition is worse than a random one at equal capacity: the typing is actively misleading |
| **D1 positive** | capacity and partition are jointly the route; with D2 null, the labels contribute nothing that the shared arm loses | both routes live; report both effect sizes and claim neither exclusively |

A D2 that is significantly NEGATIVE (permuting HELPS) is itself the finding that the annotation is
worse than noise at equal capacity, and is reported as such rather than folded into the table.

### 4.5 Multiplicity, and what this may not be used for

D1 and D2 form their own diagnostic family of size 2; Bonferroni and Holm are both reported over
that family and `survives_family_wise` requires BOTH, exactly as for the confirmatory family.
Diagnostic arms are NOT added to the confirmatory family and do not change its `family_size` of 4.

Neither arm may be used to promote a graph claim. `typed_shared` and `typed_permuted` beating
`typed_static` says something about the typed encoder, not about whether a graph prior helps: the
relevant graph claim remains `untyped_gnn - expression_only`, already on the record. If a diagnostic
arm beats `expression_only` and survives both corrections, the contradiction stop in section 6
applies unchanged — snapshot, flag, continue, do not rewrite the null around it.

### 4.6 Lane validity

The gate-collapse kill criterion does not bind here: both arms pin the gate to 1.0 by construction,
so a gate mean of 1.0 is correct rather than evidence of collapse. A lane is valid if it completes
at least as many epochs as its paired `typed_static` lane did before early stopping and returns a
finite primary metric; a lane that fails is reported as a dropped seed, by name and reason, and
shrinks n rather than being silently replaced.

---

## Amendment 4a — 2026-08-16 (correction to 4.2, BEFORE any `typed_permuted` lane is trained)

Amendment 4 is unchanged except where this says otherwise. `typed_shared` was already running when
this was written; `typed_permuted` had not started, so this is a pre-registration and not a
post-hoc note.

### 4a.1 The claim in 4.2 that was wrong

4.2 said of `typed_permuted` that "edge attributes travel with the edge, so only which weight matrix
processes it changes". That is false for the obvious implementation. The neighbourhood sampler ranks
candidate neighbours BY RELATION — `_PRIORITY_BONUS` in
`src/tcell_pipeline/graph/neighborhood_sampler.py` gives `physical_ppi` and `co_complex` a 1e6 bonus
over `functional_assoc` — so a target whose neighbourhood exceeds `NEIGHBORHOOD_CAP` = 512 keeps its
physical and co-complex neighbours and drops functional ones. Permuting the labels in the stored
graph therefore changes WHICH NEIGHBOURS ARE IN THE SUBGRAPH, and `typed_permuted` would have
differed from `typed_static` in two ways at once: the routing AND the sampled neighbourhood. D2
would not have been interpretable, and nothing in the result would have shown it.

### 4a.2 The design that replaces it

The relabelling happens AFTER sampling, through a `_sample` hook on `TypedGraphEncoder` whose default
is the plain sampler call (so every existing arm is bit-identical). `PermutedTypedGraphEncoder`
samples under the TRUE relations and then moves each protein-protein edge, with its attributes, into
its permuted relation store. The node set and the pooled edge multiset are therefore identical to
`typed_static`'s, edge for edge, and only which weight matrix processes an edge changes. This is
pinned by `test_permuted_relations_leave_the_sampled_neighbourhood_untouched`, which asserts it on a
fixture whose cap actually binds.

Two further properties are fixed here rather than left to the implementation:

- **Globally consistent.** An edge's permuted label is a pure function of its GLOBAL endpoints and
  the seed, so the same edge is relabelled the same way in every subgraph it appears in. The permuted
  partition is therefore one fixed alternative partition of the same edge set. A per-subgraph
  reshuffle would instead make the four modules a random router, which changes the architecture and
  reintroduces exactly the confound this arm exists to remove.
- **Exact global counts.** The two hash thresholds are read off the sorted hashes of every PP edge,
  so each relation keeps its original edge count exactly. Under `norm='add'` a relation's
  contribution to a node update scales with its degree, so a count that drifted would be a second
  intervention riding along with the relabelling. A first implementation put the threshold
  inclusivity the wrong way round and moved one edge; the count test caught it before any lane ran.

### 4a.3 What does not change

D1, D2, the 2x2 reading, the diagnostic family of size 2 with both corrections required, the ban on
using either arm to promote a graph claim, and the lane-validity rules all stand as written in
Amendment 4. Seeds are 0-4 on the frozen `blocked_target_ood` fold, paired per seed against the
landed `typed_static` lanes, primary endpoint `systema_pert_specific_delta`. The permutation is drawn
from the training seed, so the five lanes average over five partitions.

---

## Amendment 5 — 2026-08-16 (BEFORE the full-fold A3 re-scoring is read)

Registers a RE-SCORING of predictions that already exist. No training, no new fold, no new seed. What
is being fixed in advance is the endpoint list, the orientation rule, the multiplicity and the decision
rule — everything that could otherwise be chosen after seeing which metric was kind.

DISCLOSURE, so the record is complete: a 200-row plumbing smoke of the driver was run before this was
written and its numbers were seen. They are not used, quoted or carried forward, and no endpoint or
rule below was chosen because of them. The analysis this amendment governs is the full 4,400-row val
fold at the campaign's five seeds.

### 5.1 The endpoints, verified rather than assumed (checked 2026-08-16)

| endpoint | who reports it | what it is |
|---|---|---|
| `pearson_delta` | TxPert | Pearson between predicted and observed response over all genes, per perturbation, macro-averaged |
| `pearson_delta_top20` | GEARS | the same restricted to each perturbation's top-20 observed DE genes |
| `mse_top20` | GEARS | mean squared error over those top-20 genes; GEARS' headline |
| `edistance_scperturb` | scPerturb | E-distance as scPerturb computes it, on SQUARED euclidean distances |
| `energy_distance` | Szekely | energy distance on plain euclidean distances |

The DE subset is taken from the OBSERVED response, never the prediction.

`edistance_scperturb` is reported for commensurability and NOT as distributional evidence: with squared
distances the statistic collapses algebraically to `2*||mean(X) - mean(Y)||^2`, a difference of means
that cannot distinguish two populations with the same mean and different spread. `energy_distance` is
the distributional endpoint. Both are computed over the distribution of RESPONSES across held-out
perturbations, not over cell populations — this pipeline predicts one pseudobulk response per (target,
condition) and has no per-cell predictions to compare. Any claim about single-cell distributions
remains out of reach and stays hedged.

### 5.2 Orientation

Correlations count upward, errors and distances downward. Every metric is signed to larger-is-better
before any contrast is formed, so a positive delta always favours the first-named arm. Without this a
graph arm could be made to look good by an endpoint that runs backwards.

### 5.3 Multiplicity — the part that could be gamed, fixed now

Five endpoints times the four pre-registered contrasts is TWENTY simultaneous tests. Correcting only
within each endpoint's family of four and then reporting whichever endpoint was kind is the
look-elsewhere effect this project's `fallacy_scan.py` exists to catch.

Both bars are computed and both are reported:

- **within-metric, m = 4** — comparable to every other number in the paper, and the bar under which
  the campaign's own results were judged;
- **across-metric, m = 20** — the honest bar for the question "did anything survive anywhere once we
  looked under five endpoints".

A claim that a contrast SURVIVES the re-scoring requires the across-metric bar, under Bonferroni and
Holm both. The within-metric numbers are context, not the claim.

### 5.4 Decision rule

- **Closed** when the null holds under at least one endpoint an outside positive was reported in, and
  under the distributional endpoint. That is the paper's commensurability hedge discharged on its
  metric half.
- **A POSITIVE that clears the across-metric bar fires the contradiction stop** (section 6, unchanged):
  snapshot, flag at the top of `RESULTS_SUMMARY.md`, continue. The null is not rewritten around it.
- **Endpoints that DISAGREE IN SIGN on the same contrast are a result, not a nuisance**, and are
  reported as one. A sign that depends on which reported metric is chosen bounds what any single-metric
  claim in this literature can mean — including ours.
- Everything computed is reported. There is no endpoint here that can be dropped after the fact: the
  five are named above and the artifact carries all twenty cells.

### 5.5 What this cannot settle

The commensurability hedge has two halves. This closes the metric half only. Outside results are also
obtained on different splits, and re-scoring our predictions cannot speak to that; the split half stays
hedged, and no sentence anywhere may use this re-scoring to adjudicate another paper's claim.
