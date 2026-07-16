# Module 6 — Evaluation Metrics + Simple Baselines (feat-009 + feat-006) (design + as-built)

Date: 2026-07-16 · Depends on: feat-003 (blocked split), feat-008 (`EGIPGModel.forward` → `delta_z`,
`delta_x`, `sigma`), feat-005 (frozen fold-local basis `B`). Consumers: feat-007 (graph baselines, shares
the baseline protocol + output schema), feat-010 (external comparators), feat-011 (screening / the
near-null-signal freeze gate runs these metrics), feat-013 (reproducibility references).

Design source: `EG_IPG_architecture_walkthrough.md` §10 (Evaluation: The Metric-Qualified Endpoint) and
`perturbation_informed_causal_protein_program_graphs_report.md` §Evaluation Metrics / §Baselines. Where
they disagree the walkthrough wins.

## Purpose

Make model output **scorable** and give every headline table a set of mandatory simple references. Module 6
supplies (a) the prediction metrics + the G2-MQ metric-qualification gate + the control-reference
safeguards, delivered as **two independent implementations that must agree on a fixed fixture**, and (b)
six simple baselines behind one fit/predict contract writing a **common prediction schema**.

## Scope

- **In (feat-009):** `metrics.py` (8 metric groups), `metrics_ref.py` (independent 2nd impl of the subtle
  ones), `metric_qualification.py` (G2-MQ gate + control constructors), `control_reference.py` (§10.5).
- **In (feat-006):** `baselines/simple_baselines.py` (Zero / PerturbedMean / ConditionMean / Ridge /
  NearestNeighbor / LowRank) + `evaluation/output_schema.py` (the common prediction parquet).
- **Out:** graph baselines (feat-007), external comparators (feat-010), and — deferred within feat-006 —
  the **elastic-net** and **CatBoost/gradient-boosting** baselines named in the feature description.

## Metrics (walkthrough §10.4)

Every metric is computed **per row, then macro-averaged**. A row is one perturbation-target-by-condition
response (the report's observational unit), so per-row *is* per-perturbation; this avoids micro-averaging
(pooling all genes across all rows into a single correlation), the leakage-prone shortcut the G2-MQ gate
is designed to reject.

| Function | Definition |
|---|---|
| `mae`, `rmse` | per-row error, macro-averaged |
| `pearson_corr`, `spearman_corr` | per-row (rank) correlation |
| `systema_pert_specific_delta(pred,true,train_mean)` | **primary H1 endpoint**: `corr(pred−train_mean, true−train_mean)` — removes the average treatment effect so the model isn't credited for predicting the generic response |
| `centroid_accuracy(pred,true,all_true)` | fraction of rows whose prediction is cosine-closest to its own true centroid vs the bank |
| `topk_recall(pred,true,k)` | per-row recall of the strongest-\|magnitude\| genes |
| `sign_accuracy(pred,true,top_n)` | sign correctness among the strongest true effects |
| `program_cosine(pred_z,true_z)` | per-row cosine of program-delta vectors |
| `signed_de_metrics(probs,labels)` | macro-F1, per-class precision/recall, AUPRC for up/down DE calls (AUROC omitted — report only reports it alongside prevalence-aware metrics) |

**Zero / constant / non-finite convention.** A row whose predicted or true vector carries no signal
(zero variance, or non-finite) contributes a correlation/cosine of **0.0**. This makes a zero predictor
score worst — exactly what G2-MQ requires — and, crucially, is implemented **identically** in both
`metrics.py` and `metrics_ref.py` so the two agree by construction rather than only on a lucky fixture.
`centroid_accuracy` treats a degenerate (zero-norm) prediction as a **miss**, not a tied-with-everything
hit.

## Two independent implementations (report: "two independent implementations on a fixed fixture")

`metrics.py` is hand-vectorised numpy; `metrics_ref.py` loops per row and leans on `scipy.stats` /
`scipy.spatial` / `sklearn`. They re-implement mae, rmse, pearson, spearman, the Systema
perturbation-specific delta, centroid accuracy, and program cosine — the metrics whose algebra is subtle
enough to warrant a cross-check. The agreement test exercises a non-degenerate fixture **and** zero /
constant / non-finite rows.

## The G2-MQ gate (walkthrough §10.1)

`qualify_metric(fn, neg_controls, pos_refs)` returns `{passed, ordering_correct, dynamic_range,
neg_scores, pos_scores}`. A metric qualifies iff **every negative scores strictly below every positive**
(`max(neg) < min(pos)`). Negative-control constructors: `zero_prediction`, `perturbed_mean_prediction`,
`label_permutation` (N1 — a **derangement**, so no row keeps its own target identity and the negative
reliably reaches the null floor), `row_shuffle` (N2). Positive references: `oracle_prediction`,
`guide_split_half` (a stand-in for guide-level split-half agreement; real runs replace it with guide-level
MuData). Controls are built with a supplied RNG (never a global seed), so the seed is preserved per §10.5.

## Control-reference safeguards (walkthrough §10.5)

`independent_control_metric(pred, true, ctrl_a, ctrl_b)` subtracts **independent** control estimates
(`ctrl_a` off the prediction, `ctrl_b` off the truth) — the corrected estimator. `shared_control_diagnostic`
subtracts one shared control and is a **bias diagnostic only**. `null_control_predictor(ctrl)` re-emits
its control, so under the independent estimator its delta is zero and it scores ~0 — the intentionally
non-informative predictor §10.5 mandates.

## Baselines (report §Baselines)

Common `BaseBaseline`: `fit(X, z, conditions=None)` → `predict(X, conditions=None)` → `(delta_z (M,K),
delta_x (M,G))`. Baselines predict in **program space**; gene space is decoded through the frozen fold-local
basis, `delta_x = delta_z @ B.T` (the decoder's `B @ delta_z` pathway). `basis=None` → an empty `(M,0)` gene
block so program-only evaluation still works. `X` is treated as an opaque feature matrix — the harness
decides whether it holds q_pre context (ridge) or a target profile (kNN).

- **ZeroBaseline** — no-effect (zeros).
- **PerturbedMeanBaseline** — Systema non-control mean = the average training perturbation effect.
- **ConditionMeanBaseline** — per-condition mean; unseen condition → global fallback.
- **RidgeBaseline** — multi-output Ridge, `X → z`.
- **NearestNeighborBaseline** — kNN mean of neighbours' `delta_z` by profile.
- **LowRankBaseline** — truncated-SVD program subspace + ridge map into it + reconstruct.

sklearn returns a 1-D vector for a single target; `_as_columns` keeps it a **column** `(M,1)`, never
`atleast_2d`'s transposed `(1,M)` — the K==1 / rank==1 bug the review caught.

## Output schema (report §Baselines: common output schema)

`predictions/<model>/<split>/<seed>.parquet` with `row_index`, `delta_z_0..K-1`, `delta_x_0..G-1`,
`sigma_0..K-1`. Written atomically; read back with columns sorted by integer suffix. Baselines with no
calibrated uncertainty write `sigma = 0`. One schema lets the test steward score any model identically and
keeps challenge-split scoring model-agnostic. `config.PREDICTIONS_ROOT`.

## Review history

**Round 1** — adversarial workflow review (2026-07-16): 6 finder dimensions → per-finding adversarial
verify. **8/8 confirmed, all fixed**. Headline: the degenerate-predictor guard in `centroid_accuracy`;
non-finite agreement between the two implementations; the N1 derangement; the single-program `(M,1)`
baseline shape; three too-weak tests upgraded.

**Round 2** — xhigh workflow-backed `/code-review` of the committed diff (2026-07-16): **12 findings**
(7 CONFIRMED correctness, 2 PLAUSIBLE, 3 cleanup), all fixed — see
`docs/reviews/2026-07-16-code-review-module6.md`. These caught two-impl divergences round 1 missed (it only
corrupted `pred`, never `true`): non-finite `true` collapsing `centroid_accuracy` to 0.0, the `1e-12`
norm-floor inflating tiny-norm wrong-direction predictions, the FP-fragile `std==0` constant guard (both
impls now gate on `max==min`), product-form underflow (separate roots), the missing `topk`/`sign`
degeneracy guard, the baseline `X=None`/`conditions=None` contract, the `**kwargs` control hook, and three
cleanups (vectorised `topk`/`sign`, `rng.permuted` row-shuffle, a shared `_arrays.to_numpy`).

## Verification

`./init.sh` green at **145 tests** (92 prior + 53 Module 6: 30 in `test_metrics.py`, 23 in
`test_baselines.py` — 14 functions, three parametrized). Fully synthetic — no marts required. Both metric
implementations verified to agree on non-degenerate, zero/constant, non-finite (`pred` and `true`),
high-dimensional constant, tiny-norm, and extreme-scale rows under `-W error`.
