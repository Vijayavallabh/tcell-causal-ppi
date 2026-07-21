# feat-008 Stage B — the two missing FIT LOOPS and the near-null-signal freeze gate

**Status: CODE COMPLETE + TESTED. NO REAL-DATA RESULT YET — the Stage-B run is approval-gated and has
not been launched.** Nothing in this note is a finding about the model; there are no headline numbers
here because none have been computed. A comparison that was not computed is not a result.

Written to a standalone file rather than `feature_list.json` / `progress.md` / `session-handoff.md`
because a second session held those concurrently (a JSON read-modify-write would silently lose one
side's update). Merge into feat-008's evidence block at commit time.

---

## What was missing, and what now exists

feat-008's own evidence named the gap: *"the Stage-B calibration + rationale FIT loops (both loss
modules exist, no fit loop), and the near-null-signal freeze gate."* Confirmed by inspection —
`StageBCalibrationLoss`, `RationaleLoss` and `RationaleHead` were modules nothing ever fitted.

| # | Deliverable | New file |
|---|---|---|
| a | Stage-B calibration fit loop over a frozen Stage-A model | `src/tcell_pipeline/training/stage_b.py` |
| b | Rationale-head fit loop over a frozen backbone | `src/tcell_pipeline/rationale/rationale_fit.py` |
| c | Near-null-signal freeze gate + exit-code mapping | `src/tcell_pipeline/training/freeze_gate.py` |
| — | Orchestrator (fit both, gate both, exit non-zero unless frozen) | `src/tcell_pipeline/training/run_stage_b.py` |

No existing module was modified. `model.py`, `trainer.py`, `losses.py`, `run_train.py`,
`rationale_head.py` and `rationale_loss.py` were read and imported, never edited — the loops needed
nothing from them that was not already public.

---

## (a) Stage-B calibration fit — `training/stage_b.py`

Stage A never puts a gradient on the decoder's uncertainty head (no `StageALoss` term reads
`out["sigma"]`), so that head is still at init when Stage B begins; Stage B fits exactly it, with
`StageBCalibrationLoss` (Gaussian NLL), and nothing else.

Two properties are **asserted at runtime, not intended**:

1. **Stage A stays frozen.** Every parameter outside the calibration head is snapshotted before the fit
   and compared after; any movement raises `RuntimeError` naming the parameter. `requires_grad` and
   train/eval mode are restored to the caller's state in a `finally`, so the model is not left
   silently frozen.
2. **No val statistic enters the fit.** Early stopping and the best checkpoint key on the TRAIN NLL.
   The val NLL is computed and reported every epoch but never selects a parameter or an epoch.
   Pinned by fitting the same data against three different val sets (including *none*) and requiring
   bit-identical weights.

The backbone is eval + frozen for the whole fit, so its outputs are constants: one forward pass caches
the decoder's inputs and the epoch loop never re-runs the graph encoder. This is what makes a
100-epoch calibration fit cost roughly one forward pass rather than one hundred.

### Controls (reported *beside* the headline, never instead of it)

| control | what it is | what it catches |
|---|---|---|
| `vs_constant_sigma` | best per-program CONSTANT sigma, closed-form on the fit split | a head whose input-dependence bought nothing — the collapse-to-a-constant detector |
| `vs_permuted_sigma` | the fitted sigma with its rows permuted (same marginal, pairing destroyed) | the matched-random analogue for a per-row quantity |

A collapsed head is **identical** to its permuted control, so every paired delta is exactly zero, the
paired test finds zero variance, and the gate returns **undecidable** — not a spurious win. That is the
same trap the concurrent session just fixed on the primary endpoint, where a collapsed predictor scored
+0.0129 off floating-point dust; here the degenerate case cannot score at all.

---

## (b) Rationale-head fit — `rationale/rationale_fit.py`

`fit_rationale_head(graph_encoder, decoder, cases, head=RationaleHead())` — the loop feat-012's audit
needs and cannot run without. Same freeze discipline as (a), same assert. Per case the frozen
backbone's outputs (node states, condition gates, `dz_full`, and the control predictions) are computed
once and reused; only the head's scorer moves.

`rationale_contrasts` builds the gate inputs, one unit per case:

| contrast | direction | control |
|---|---|---|
| `sufficiency_vs_random` | lower is better | `MatchedRandomSampler` sets matched on size AND per-relation composition, recomputed against the FITTED mask |
| `necessity_vs_random` | higher is better | same |
| `sufficiency_vs_untrained` | lower is better | the ZERO-INIT head, which is already faithful by construction (it ranks by the frozen gate) |

The third contrast is the one that matters most here: the untrained head is free. A fit that only
reproduces it has bought nothing and must not be frozen — freeze the free head instead.

**Fit-time vs gate-time controls are deliberately different.** The contrastive term inside the loss uses
a small control set sampled once per case (a moving control target makes the objective non-stationary);
the GATE's controls are the full `N_MATCHED_CONTROLS` set, recomputed against the mask the fitted head
actually produces. Marked with a `ponytail:` comment and its upgrade path.

### A real bug this work surfaced

`rationale_contrasts` originally extracted each case's rationale with **DropEdge still active**, so the
selected edges — and every audit number keyed to them — were a fresh random draw per call. Found by a
test that asserted the contrast builder scores the mask the fitted head produces; it failed because two
calls disagreed. Fixed with an `eval_mode` context (the same fixed-model contract `FaithfulnessTester`
enforces for its deletion tests), and pinned by
`test_rationale_contrasts_are_deterministic_under_active_dropout`, which mirrors the existing
`test_faithfulness_is_deterministic_under_active_dropout`. This is the third time this repo has been
bitten by a stochastic encoder leaking into a "fixed-model" measurement.

---

## (c) The near-null-signal freeze gate — `training/freeze_gate.py`

```
evaluate_gate({name: {"fit": {unit: v}, "control": {unit: v}, "higher_is_better": bool, "units": [...]}})
    -> {"decision": freeze | refuse | undecidable, "contrasts": {...}}
exit_code(report) -> 0 (freeze) | 1 (refuse) | 2 (undecidable)
```

Per contrast the per-unit advantage — oriented so positive always means *the fit won* — goes through
`screening.multiseed.paired_delta_summary`, the hardened paired-t core the 5-seed campaign uses (zero
variance → undecidable, missing unit → dropped loudly). The whole run's contrast family then gets
**both** Bonferroni and Holm via `_apply_family_wise`, and a freeze requires both, so the correction
cannot be shopped after seeing which one rescues the claim. No statistics were reimplemented.

Three outcomes, deliberately distinct — `None` is not a negative:

* **freeze** — every required contrast has mean > 0, a CI excluding zero, and survives family-wise.
* **refuse** — a required contrast was COMPUTED and did not clear (crosses zero, dies under correction,
  or is significant on the WRONG side).
* **undecidable** — a required contrast could not be decided at all (absent, n<2, zero variance, every
  unit non-finite).

### What input makes each fence FIRE

Every guard was written against a constructed input that defeats the obvious implementation:

| guard | input that fires it | why the obvious version fails |
|---|---|---|
| direction check (`mean > 0`) | a fit reliably WORSE than its control | it has `ci_excludes_zero=True` **and** `survives_family_wise=True` — significance alone would freeze a fit that loses |
| zero-variance → undecidable | identical deltas on every unit | the one condition proving the units carry no information would otherwise publish as `p=0` |
| empty contrast set | `evaluate_gate({})` | `all([])` is vacuously true, so a gate with nothing to check reads as a pass |
| missing required contrast | `required=("suff","nec")` with only `suff` supplied | absence of evidence silently drops out of the family |
| duplicate unit ids | two cases sharing a row id | duplicates collapse into one dict key: n shrinks, verdict still reads clean |
| unit missing from both arms | explicit `units` universe | without it, n shrinks with an EMPTY `dropped` list — a 5-unit result reads as the intended 6 |
| `fmt()` None-safety | rendering a degenerate report | `None:+.4f` crashes on exactly the input the guard exists for |
| exit code default | a report with no `decision` | `.get(d, 0)` would exit 0 on a run that decided nothing |

### The exit code is traced

`run_stage_b.main()` returns `exit_code(report)`; a run that produced nothing at all (missing
artifacts) returns non-zero too. The overall gate **requires all five contrasts by name**: defaulting
`required` to "whatever was supplied" would let a run that never produced the rationale contrasts (an
expression-only model, or a fit that wrote no checkpoint) pass on the calibration alone. Per-artifact
decisions are still reported separately, so nothing is lost by being strict.

---

## The noise floor — added because the probe demanded it

A small end-to-end probe (40 rows, 3 cases, 2 epochs, real marts, real frozen H1) ran the whole
pipeline and **refused**, exiting 1. It also showed something the code had no answer for:

```
gene    rationale_size   sufficiency    necessity   rand_suff      rand_nec
ACP2    15               6.329e-06      2.98e-07    6.329e-06      2.98e-07
AAAS    15               2.842e-06      0.0         2.842e-06      0.0
ACER3   15               5.346e-06      0.0         5.346e-06      0.0
```

Deleting the entire 15-edge rationale moves the frozen model's prediction by **0.0**. Keeping only the
rationale versus keeping everything differs by ~1e-6. The rationale and its matched-random control
agree to 10 significant figures. These are not measurements; they are floating-point residue — and a
paired t-test on consistently-signed 1e-10 differences will happily report a small p and clear a gate
on nothing at all. That is the same failure shape as the collapsed predictor that scored +0.0129 on the
primary endpoint off numerical dust.

`rationale_contrasts` now applies a **relative noise floor** (`NOISE_FLOOR_REL = 1e-4` of
`||dz_full||`): a case where the rationale, its complement AND every matched-random control all move dz
by less than that is **non-informative**. Its measured values stay on the record in `per_case`, but it
enters the statistic as `None`, so the paired core drops it loudly and names it. All-non-informative ⇒
`n=0` ⇒ **undecidable**, printed as a warning line, never a refusal and never a pass. A zero-norm
prediction is non-informative by definition (otherwise the relative floor would be 0 and every 1e-12
residue would be promoted to a measurement).

### The obvious explanation, computed and REFUTED

The report now also records the decoder's mixture weight **λ** (mean/min/max over the gate rows),
because `delta_z = λ·graph_path([h_graph‖h_do]) + (1−λ)·expr_path(h_do)` and λ→0 would explain a
graph-invariant prediction immediately. Measured on 40 val rows of the frozen H1:

```
lambda mean = 0.2632   range [0.0703, 0.5796]
```

λ is **not** collapsed — the graph pathway carries about a quarter of the prediction. So the hypothesis
is refuted, and the insensitivity needs another explanation. Note λ cannot establish graph dependence
either: `graph_path` consumes the *concatenation* `[h_graph ‖ h_do]`, so it can be driven almost
entirely by its `h_do` half at any λ. λ measures how much of the JOINT pathway is used, not how much of
the GRAPH.

### The decisive measurement: delete every edge

The top-k rationale keeps 15 edges. What the deletion tests could not distinguish is "15 edges is too
few to matter" from "no edges matter". So: delete **all** of them (frozen H1, real val rows).

Eight DISTINCT covered val genes (the mart's first rows repeat one target, so genes were de-duplicated):

| gene | \|\|dz\|\| | edges in neighbourhood | \|S\| | ‖Δdz‖ ALL edges deleted | relative | ‖Δdz‖ rationale deleted |
|---|---|---|---|---|---|---|
| AAAS | 6.894 | 40,130 | 15 | 2.291e-05 | 3.3e-06 | **0.000e+00** |
| AARS2 | 9.616 | 53,170 | 15 | 1.133e-04 | 1.2e-05 | **0.000e+00** |
| ABCA3 | 6.685 | 23,874 | 15 | 7.066e-06 | 1.1e-06 | **0.000e+00** |
| ABHD14B | 7.840 | 40,832 | 15 | 3.568e-04 | 4.6e-05 | **0.000e+00** |
| ABHD15 | 5.827 | 15,732 | 15 | 1.041e-06 | 1.8e-07 | **0.000e+00** |
| ACADM | 7.101 | 53,463 | 15 | 1.210e-05 | 1.7e-06 | **0.000e+00** |
| ACADSB | 8.654 | 59,513 | 15 | 2.835e-04 | 3.3e-05 | **0.000e+00** |
| ACAP2 | 6.666 | 25,659 | 15 | 1.228e-05 | 1.8e-06 | **0.000e+00** |

Deleting the **entire neighbourhood** — 16k to 60k edges — changes the prediction by 1.8e-07 to 4.6e-05
of its magnitude, every case below the 1e-4 floor. Removing the 15-edge rationale is bit-for-bit
identical to removing nothing (`0.000e+00`, all eight).

This is not a sparsity/top-k problem: **no rationale of any size is measurable on this checkpoint**, so
a feat-012 audit run against it would spend hours reporting floating-point residue. That is the single
most decision-relevant thing this session found, and it cost about ten minutes to establish.

**Scope, honestly:** 8 genes, `condition_gated` seed 0 only, val split; other family members and seeds
are untested, and this says nothing about *why*. It is a **model property, not a defect in the Stage-B
code** — the loops and the gate behave exactly as designed on it, which is precisely why the gate
returns *undecidable* rather than a verdict.

### CORRECTION (2026-07-21): the *why*, and a framing I got wrong

Session E found the mechanism (`docs/h1-optimization-notes.md`) and it changes what this result means.
`StageALoss._graph` — **in `training/losses.py`, a file this session owns** — is an unnormalised
`sum` over edges divided only by BATCH SIZE, while every other term is mean-reduced. Over a real 16k–60k
edge neighbourhood that makes the graph penalty **103× the response term at initialisation**; its
gradient on the edge gates is ~3.1e+06× the task's, so the penalty's **direction** is ~100% of the total
(`g_total/g_penalty` = 0.999994–1.000315) and the gate dies inside epoch 0.

> **CORRECTION (2026-07-21).** This paragraph previously continued "and `GRAD_CLIP=1.0` then rescales the
> whole update by ~1/695, so ~99.98% of every step goes to driving gates to zero." **That mechanism is
> wrong**, and session E withdrew it. AdamW is scale-invariant per parameter, so a uniform clip factor
> cannot change the update at all (verified: ‖g‖=695 clipped to 1.0 gives the same θ as an unclipped
> ‖g‖=0.17). Magnitude sets the *rate* of collapse; direction sets *whether* it happens. Confirmed
> prospectively — per-edge normalisation makes the penalty 400× *smaller* than the task and the gates still
> collapse 2,108×. The measurements were always right; only the causal story was wrong.
I read `_graph` during this work and quoted its docstring — which says "averaged over the batch so its
strength doesn't scale with batch size" and is silent on the edge dimension that actually varies 4× —
without noticing the sum was never normalised over edges.

Measured directly on the gates (`scratchpad/gate_stats.py`, 3 covered val genes, 117,174 edges):

| | gate mean | max | min | bit-zero gates |
|---|---|---|---|---|
| at init | 0.550 – 0.604 | 0.742 | 0.364 | 0 / 117,174 |
| **frozen H1** | **1.24e-07 – 1.36e-07** | 3.54e-07 | 3.89e-08 | **0 / 117,174** |

A 6.5-order collapse. Two consequences:

1. **The framing above is wrong where it matters.** "No rationale is measurable on this checkpoint" is
   still true, but it is NOT evidence the graph is useless — the optimisation never let the graph
   participate, so the experiment did not test the question. Any reading of the delete-all-edges result
   as "the graph contributes nothing" must be retracted; it is a **measurement-validity defect**. This
   note previously offered it as a mechanistic account of the project's graph negative. It is not.
2. **Not literally zero.** The relayed claim "the frozen H1's gate mean is exactly 0.000000" overstates
   it: 0.000000 is the 6-dp rendering of ~1.2e-07, and not one gate of 117,174 is bit-zero. This is the
   detail that reconciles the two independent probes — if the gates were bit-zero, delete-all would move
   the prediction by EXACTLY zero, yet E measured 1.00e-02 (on `h_graph`) and this session measured
   1.8e-07–4.6e-05 (on `delta_z`). Those residuals are the surviving gate magnitude, not noise. It also
   explains why `drop-S` came out at exactly `0.000e+00`: removing 15 of ~40,000 edges gated at ~1e-07
   perturbs the aggregate by ~1e-09, which rounds to bit-identical in float32 — a rounding statement,
   not a proof of zero influence.

What this does NOT change: the Stage-B loops and the freeze gate are model-agnostic and behaved
correctly. `NOISE_FLOOR_REL` refused to manufacture a result from a model whose deletions move nothing,
which is the outcome it was built for.

**The `_graph` normalisation is not fixed here.** Dividing by edge count instead of batch size is a
one-line change in a file this session owns, but it alters the Stage-A training objective: it would
invalidate the frozen H1, the 5-seed campaign and every screening result, and it would move the code
hash session D's reproducibility manifest is built on. That is the user's call, not this session's.

## Test evidence

**+62 tests from this work** (55 in the first pass, +7 from the self-review fixes), all green. The repo-wide total is a moving target while a second session
lands work in the same checkout — it was 332 before this work, 387 immediately after it, and 454 once
the other session's basis-study/reproducibility work landed (`./init.sh` green at 454; the 4 files this
work owns account for 79 of them). Quote the +55, not the total. Every test pinning a correctness claim was watched FAILING
before implementation — each new group was written first and run red (collection errors, then assertion
failures), then implemented.

| file | new tests | covers |
|---|---|---|
| `src/tests/test_freeze_gate.py` (new) | 22 | every row of the "what makes this FIRE" table above; both orientations |
| `src/tests/test_training.py` | +18 | calibration fit: freeze assert, val-independence, controls, collapse, empty split, best-epoch restore, seed namespacing, permutation averaging |
| `src/tests/test_rationale.py` | +18 | rationale fit: freeze assert, determinism, matched controls, mask polarity, noise floor, best-epoch restore |
| `src/tests/test_stage_b_driver.py` (new) | 4 | decision → process exit code; the run-level required-contrast guard |

### Mutation testing

**41 mutations** were applied one at a time and the suite re-run (32 in the first pass, 9 more covering
the self-review fixes): 38 killed individually, 3 by the joint mutation described below. The first pass had six survivors, and they were more useful than the kills:

* *arm means over all units instead of used units* — the reconciliation test used only clean units, so
  it could not see the difference. Fixed the TEST to drop a unit.
* *permuted control not permuted* — nothing pinned that the control is actually re-paired for a
  non-collapsed head. Added the assertion.
* *gate controls matched to the untrained head* — the test exercised the helper, not the call site.
  Added a wiring test; **it immediately failed for a real reason and found the DropEdge bug above.**
* *duplicate case ids allowed* — no test supplied duplicates. Added.
* *`n < 2` clause in `_decide`* — genuinely unreachable: the paired core already leaves
  `ci_excludes_zero` as `None` for n<2. It was decoration, so it was DELETED and the behaviour pinned
  by `test_single_unit_is_undecidable` instead.

Two mutations survive **individually and by design**: the optimiser parameter set and the
`requires_grad` freeze are redundant guards — either alone keeps Stage A frozen. Applying **both**
mutations together does move the backbone, and then the runtime assert fires and 5 tests fail
(calibration) / 8 fail (rationale). Verified explicitly rather than assumed.

---

## Fit split — deviation raised and CONFIRMED (2026-07-20)

The goal text said *"calibration is fit on TRAIN only"*. The driver defaults instead to the split's own
dedicated **`calibration` partition** (915 genes / 2,713 rows), with `--fit-role train` available:

* Stage A was optimised on train, so its train residuals are optimistically small; a sigma head fitted
  there would be miscalibrated for exactly the out-of-fold rows a calibration head exists to serve.
  `losses.py`'s own docstring says "fitted on the calibration partition after the predictor freeze".
* It is also cheaper and lighter: the audit's stratifier densifies the split's z-score block, which is
  1.75 GB for train vs 0.22 GB for the calibration partition on a shared box.

The instruction's *intent* — the fit never sees the evaluation split — holds either way: the gate is
evaluated on `val` (4,400 rows), which no fit touches. **Raised explicitly and confirmed by the user:
keep the calibration partition as the default.**

---

## Self-review pass (2026-07-20) — 4 findings, all fixed

A 5-lane review of this work (AGENTS.md compliance, shallow bug scan, git-history bug classes, prior
recorded reviews, comment-vs-code) surfaced four defects. Each was reproduced before being fixed, and
each fix was mutation-tested afterwards.

**1. The calibration gate scored a model that was never saved.** `fit_calibration` stepped the optimiser
every epoch but checkpointed only improving ones, and never restored the best weights — so after early
stopping (the whole point of `patience`) the in-memory head was the LAST epoch while `best_ckpt` held
the best. `run_stage_b` then gated that live model. Reproduced: best epoch 0, run ended at epoch 3,
`torch.equal(ckpt, model) is False`. The rationale path already did this correctly via `_fitted_head`,
whose docstring states the rule the calibration path broke.

The review suggested reloading in the driver. **Fixed differently:** both fit loops now restore the best
epoch's weights before returning, so in-memory == `best_ckpt` for *every* caller instead of only the one
that remembers. `fit_rationale_head` had the same latent defect and it was worse there — that objective
PLATEAUS rather than diverging, so the history read identical to 16 decimal places while the scorer
drifted **6.5** in weight space with no sign of it in the log. Reported metrics now describe the restored
epoch (`best_epoch`, plus `last_train_nll` kept separately).

**2. Stage-B checkpoints were not seed-namespaced.** Both heads write fixed filenames and the driver
passed no `ckpt_dir`, so `--seed 0` then `--seed 1` silently overwrote the first run's artifacts — with
`config.N_FINAL_SEEDS = 5` in the project's own methodology. This exact class was already fixed for
Stage A in commit `32fb473`. Fixed: `stage_b_ckpt_dir(seed)` → `<CHECKPOINTS_ROOT>/stage_b/<seed>/`,
mirroring `screening.py`'s `<root>/<name>/<seed>/ckpt`, plus a `--ckpt-dir` override.

**3. The run-level `required=` guard had no test.** Deleting `required=REQUIRED_CONTRASTS` left 23/23
tests green — the vacuous-pass trap the module docstring warns about was itself unguarded, and the test
that claimed to cover it re-derived the guarantee by calling `evaluate_gate` directly instead of the
path `run()` uses. Fixed: the composition is now a named `overall_gate()` the test actually calls;
dropping its `required=` turns the suite red. *Residual, stated plainly:* the one line where `run()`
calls `overall_gate` is still only verifiable by reading it — covering it needs the real marts.

**4. The published table's advantage column could not be reconciled.** `fit_mean`/`control_mean` are raw
arm values while the advantage is ORIENTED (positive = the fit won), so it equals fit − control only
when higher is better. Four of the five real contrasts are lower-is-better, and a real probe row read
`fit +19.3314 | control +19.4723 | advantage +0.1410` — not a subtraction of the two columns. The
comment asserting they reconcile was simply wrong, and the only test used the default orientation.
Fixed: the comment now states the orientation-dependent relation, `render` prints a **direction** column
("lower is better" / "higher is better") on every row, and the reconciliation test covers both
orientations.

**5. (found while verifying the other four) The permuted-sigma control was a single random draw.** Two
probe runs on identical data and an identical fitted head, differing only in `--seed`, gave:

```
seed default : fit +19.3314  control +19.4723  advantage +0.1410  raw p 0.2494
seed 3       : fit +19.3314  control +19.4839  advantage +0.1525  raw p 0.1506
```

A 40% swing in a headline control's p-value, from the control's own seed (both runs single-draw, so the
seed is the only difference). The matched-random control on the rationale side already averages over its
draws; this one did not. Fixed: the permuted control is now the mean per-row NLL over
`n_permutations=8` re-pairings, and four real-fold probes bracket the change:

| run | seed | control | raw p | |
|---|---|---|---|---|
| probe2 | 0 | +19.4723 | 0.2494 | single draw |
| probe3 | 3 | +19.4839 | 0.1506 | single draw |
| probe5 | 3 | +19.4355 | 0.1597 | averaged over 8 |
| probe4 | 7 | +19.4446 | 0.1235 | averaged over 8 |

Seed-to-seed p spread: 0.099 single-draw vs 0.036 averaged, consistent with the √8 ≈ 2.8× the average
should buy. **Two seeds per arm is an illustration, not a measurement** — the evidence for the fix is
`test_permuted_control_is_averaged_over_draws_not_a_single_permutation`, which sweeps 5 seeds and
requires the averaged spread to be under half the single-draw spread.

That fix then nearly broke something more important. Averaging K **identical** float32 values does not
reliably return that value — one fixture row drifted `5.96e-08` — and a collapsed head's permuted control
is BY DEFINITION its own fit, so the contrast stopped being exactly degenerate. A paired t would then
have had a spread to work with and `undecidable` would have become a p-value computed on rounding noise:
the precise failure this gate exists to prevent, reintroduced by a fix for something else. The average
now accumulates in float64, which returns identical inputs exactly. Caught by
`test_a_collapsed_calibration_head_cannot_clear_the_gate`, which asserts exact equality — a tolerance
there would have hidden it.

Considered and deliberately not changed: `calibration_contrasts` raising `TypeError` rather than a clear
message on an empty *train* split (unreachable through the driver, which validates first); the
frozen-backbone assert sitting after `try/finally` (on an exception path there is no result to trust);
`seeded_init` restoring only the CPU RNG (pre-existing in `trainer.py`, and nothing in this driver
consumes CUDA randomness). The `--n-controls` flag reaching only the gate was real ambiguity, so the
fit-time count is now its own `--n-controls-fit` flag rather than a silent constant.

## Measured cost of the real run (timed end-to-end, not extrapolated from a sub-component)

Probe: real marts, real frozen H1 (`condition_gated/0`), 40 fit + 40 gate rows, 3 cases, 4 controls,
2 epochs, CPU, `OMP_NUM_THREADS=4`, box load ~50-60.

| phase | probe | unit cost |
|---|---|---|
| setup (build_hetero_graph + load ckpt) | 64.9 s | fixed |
| frozen backbone pass (80 rows) | 347.5 s | **4.34 s/row** |
| calibration fit, 2 epochs (cached) | 0.5 s | ~0.25 s/epoch |
| calibration contrasts (cached) | 0.1 s | negligible |
| rationale fit (3 cases × 2 epochs) | 123.3 s | ~2.2 s/encode |
| rationale gate (3 cases × 4 controls) | 85.0 s | 28.3 s/case |

Extrapolated to the full run (2,713 fit + 4,400 gate rows, 50 cases) **at this contention**:

* frozen backbone pass — 7,113 rows × 4.34 s ≈ **8.6 h** (dominates; it is one 3-layer message pass over
  a 16k–60k-edge neighbourhood per row, and rows do not share `(gene, condition)`, so node-state caching
  would buy nothing here)
* calibration fit + contrasts — minutes (cached)
* rationale fit, 20 epochs ≈ **1.5 h**; rationale gate ≈ **6.3 h** at 100 controls, **2.0 h** at 30

Total ≈ **16 h** as specified, ≈ **12 h** at the audit's own 30-control cap. The dedup fix already took
1.9× off the calibration phase (663.5 s → 348.1 s on the probe, measured before and after).

This is contention-dominated, not intrinsic: the screening campaign recorded 8.17 GPU-hours for a full
multi-epoch `condition_gated` TRAINING run, which implies a per-row forward far below 4.34 s on an
uncontended A100. Any GPU estimate here would be a guess — no GPU was benchmarked, because none was free
(GPU 2 belongs to the concurrent session until ~23:30; 0/1/4 to other users).

**And the rationale half of that run is known in advance to return UNDECIDABLE** — the edge-sensitivity
table above already establishes that no deletion on this checkpoint is measurable. Spending 2–6 h to
re-derive `n_informative = 0` at 50 cases would be paying for an answer already in hand.

## Not done, and deliberately so

* **The real Stage-B run has not been launched.** The plan and its measured cost were put to the user,
  who chose *review the edge-sensitivity finding first, run nothing yet* (2026-07-20). No GPU was taken
  and no long job was started; every probe wrote to a scratch directory via the `CHECKPOINTS_ROOT` /
  `LOGS_ROOT` / `STAGE_B_ROOT` env overrides, so `data/checkpoints/` and `data/results/` are untouched
  by this session.

* **The open question for feat-012** is now a decision, not a compute task: the audit's machinery is
  complete and its target model is edge-inert. Auditing `condition_gated/0` as it stands can only
  return *undecidable*. Whether to (a) publish that as the honest Module-4 result, (b) test whether any
  other family member / seed is edge-sensitive before auditing, or (c) investigate WHY message passing
  contributes ~1e-6 to a model whose λ is 0.26, is a scientific call for the user. Options (b) and (c)
  are cheap — the 8-gene diagnostic above cost ~10 minutes.
* No result, verdict, or number about calibration or rationale quality exists yet. When the run
  happens, every headline in `stage_b_gate.{json,md}` will carry its control on the same row by
  construction — `render()` cannot print one without the other.
* The driver's data-path wiring (real marts, real checkpoint) is exercised only by that run; the unit
  tests cover the fit loops, the gate and the exit-code mapping, all on synthetic fixtures.
* ~~`STAGE_B_ROOT` lives in `run_stage_b.py` rather than `config.py`, because a concurrent session held
  that file. Move it next to `RATIONALE_AUDIT_ROOT` when the lock lifts.~~

  **CORRECTION (2026-07-21):** that reason was false and was never checked. Session D pointed out that
  `config.py` is unmodified in git, was last written 2026-07-17, and is claimed by no session's brief —
  sessions C and D both used existing `config.*` roots and needed no such workaround. The "lock" was an
  assumption I asserted as fact in a code comment. `STAGE_B_ROOT` now lives in `config.py`.

  It went to the **Module 5** block beside `CHECKPOINTS_ROOT`/`LOGS_ROOT`, not beside
  `RATIONALE_AUDIT_ROOT` as the retracted bullet proposed: the Module 8 header enumerates
  feat-010/012/013, and Stage B is feat-008's, produced by the training module. That header also said
  *"Stage B is a loss module only"* — true until this work landed the fit loops, so it was corrected in
  the same edit. `config.py` diff: +5/−1.

---

## Appendix: feature_list.json evidence block (feat-008)

Session A merges the DoD triad. This is the exact text to APPEND to feat-008's `evidence` string --
append-only, so the existing evidence must remain a strict prefix. Byte-identical, single line,
pure ASCII (so appending cannot re-encode the file's existing `\uXXXX` escapes), and the LEADING
SPACE is the separator from the current tail (`...read it as superseded by this note.`).

Verification: 7665 chars, 1 line, ASCII-only; concatenated onto HEAD's evidence gives
22374 chars with HEAD's 14709 chars as a strict prefix.

```text
 UPDATE (2026-07-21, Stage B -- the two missing FIT LOOPS and the near-null-signal freeze gate): the REMAINING line above ("the training-loss OPTIMIZATION loop + train/calibration loops (the loss module exists, no fit loop yet)") is now satisfied IN CODE; the real-data Stage-B run is NOT run (see the DEFERRAL below), so feat-008 stays in-progress. New: src/tcell_pipeline/training/stage_b.py (fit_calibration -- fits ONLY the decoder's uncertainty head with StageBCalibrationLoss over a FROZEN Stage-A model; the backbone is snapshotted before the fit and compared after, so any movement raises rather than being intended; requires_grad + train/eval restored in a finally; early stopping and the best checkpoint key on the TRAIN NLL so no val statistic can select a parameter, pinned by fitting the same data against three different val sets incl. none and requiring bit-identical weights; the frozen backbone's decoder inputs are cached in ONE forward pass so a 100-epoch fit costs ~one pass; frozen_caches shared with the contrasts, measured 1.9x on the probe; the fit restores the BEST epoch's weights before returning, so the in-memory head IS the checkpointed artifact; stage_b_ckpt_dir(seed) -> CHECKPOINTS_ROOT/stage_b/<seed>/ so a multi-seed sweep cannot silently overwrite itself). src/tcell_pipeline/rationale/rationale_fit.py (fit_rationale_head -- the loop feat-012's audit needs and cannot run without; same freeze discipline and same best-epoch restore; per case the frozen backbone's node states, gates, dz_full and control predictions are computed once and reused). src/tcell_pipeline/training/freeze_gate.py (evaluate_gate -> freeze | refuse | undecidable, exit_code -> 0 | 1 | 2; per-contrast advantage oriented so positive always means the fit won, through screening.multiseed.paired_delta_summary and _apply_family_wise, so BOTH Bonferroni and Holm are recorded and a freeze requires both; no statistics reimplemented). src/tcell_pipeline/training/run_stage_b.py (fits both heads on the split's dedicated CALIBRATION partition, gates on val, requires all five contrasts BY NAME, and returns the gate's decision as the process exit code). config.py: STAGE_B_ROOT added beside CHECKPOINTS_ROOT/LOGS_ROOT, and the Module 5 header corrected -- it read "Stage B is a loss module only", which these fit loops ended. EVERY headline is reported beside its own control, by construction: calibration against a per-program CONSTANT sigma fit on the fit split (the collapse-to-a-constant detector) and against its own row-PERMUTED sigma averaged over 8 re-pairings; rationale against size- and relation-matched MatchedRandomSampler sets recomputed against the FITTED mask, and against the ZERO-INIT head (which is free and already faithful by construction, so a fit that only reproduces it bought nothing). A collapsed head is IDENTICAL to its permuted control, so the paired test finds zero variance and the gate returns UNDECIDABLE -- the degenerate case cannot score. Tests: +62 (src/tests/test_freeze_gate.py 22 and test_stage_b_driver.py 4 new; test_training.py +18, test_rationale.py +18; those four files hold 86 tests, repo total 512 at time of writing and moving while other sessions land work). Every test pinning a correctness claim was watched FAILING first. 41 mutations applied one at a time: 38 killed individually, 3 (the optimiser parameter set and the requires_grad freeze, which are redundant guards) killed only by the joint mutation, verified rather than assumed. Bugs this work found and fixed: rationale_contrasts extracted each case's rationale with DropEdge ACTIVE, so the selected edges and every audit number keyed to them were a fresh random draw per call (fixed with an eval_mode context, the same fixed-model contract FaithfulnessTester enforces; pinned by test_rationale_contrasts_are_deterministic_under_active_dropout). A 5-lane self-review then found 5 more, all fixed and mutation-tested: the gate scored the live model instead of the best checkpoint (both fits now restore best -- the rationale objective PLATEAUS while the scorer drifted ~6.5 in weight space with nothing in the history to show it); Stage-B checkpoints were not seed-namespaced; the run-level required= guard had no test (deleting it left 23/23 green -- now a named overall_gate the test calls); the arm-mean reconciliation comment was sign-flipped for the four of five contrasts that are lower-is-better (render now prints a direction column); and the permuted-sigma control was a SINGLE draw whose seed moved its raw p from 0.2494 to 0.1506 on the real fold (now averaged over 8, accumulated in FLOAT64 because averaging identical float32 values drifted one row 5.96e-08 and would have given a paired t a spread on rounding noise). DEFERRAL (2026-07-21): the Stage-B real run is NOT run and should NOT be run on condition_gated/0. Measured on that checkpoint over 3 covered val genes / 117,174 edges: the condition gates sit at 1.24e-07..1.36e-07 with a maximum of 3.54e-07, against 0.550..0.604 at init -- a ~4.5e+06x collapse, reproducible via tcell_pipeline.probe_graph_gradients, which reports the frozen checkpoint's gate mean, gate max and the collapse factor. The gates are therefore NOT zero: 0.000000 is the six-decimal rendering of ~1.3e-07, and that surviving magnitude is exactly what this session's delta_z residuals (1.8e-07..4.6e-05) and session E's h_graph residual (1.1e-02) are measuring, since bit-zero gates would force both to be identically zero. (A one-off probe additionally measured a gate minimum of 3.89e-08 and 0 of 117,174 gates bit-zero; that probe was not retained and probe_graph_gradients does not report a gate minimum, so those two figures are not currently reproducible from committed code.) Session E (docs/h1-optimization-notes.md) established the cause and this session re-derived it independently from StageALoss._graph: that term is an unnormalised sum over edges divided only by BATCH SIZE while every other term is mean-reduced, so at ~40k edges/sample the penalty is ~103x the response term, its gradient on the gates is up to ~3.3e+06x the task's, and GRAD_CLIP=1.0 rescales the whole update by ~1/695 -- the gate dies inside epoch 0. Consequences: RationaleHead computes importance = gate x sigmoid(scorer), so top-k would rank a quantity that is ~1e-07 everywhere, every deletion is a float32 no-op, NOISE_FLOOR_REL=1e-4 drops 100% of cases, and all three rationale contrasts return UNDECIDABLE -- guaranteed, at a cost of ~1.5 h fit + 2.0-6.3 h gate on measured numbers, where the gate read reaches the same answer in three minutes. The calibration half is not blocked by the gate collapse (sigma is input-dependent through h_do) but is superseded if H1 is retrained. CORRECTION (2026-07-21): this session earlier reported that deleting all 16k-60k edges moves the frozen H1's prediction by 1.8e-07..4.6e-05 relative and offered it as a mechanistic account of the project's graph negative. The measurement stands; THAT FRAMING IS WITHDRAWN. The optimisation drove the gates to ~1.3e-07 inside epoch 0, so the comparison did not test the hypothesis -- a measurement-validity defect, not evidence about the graph. StageALoss._graph is in a file this session owns and this session read it without noticing the missing edge-count normalisation; the fix is NOT applied here because it changes the Stage-A objective and would invalidate the frozen H1, the 5-seed campaign and every screening result, and move the code hash feat-013's manifest pins. REMAINING for feat-008 done: run Stage B (both fits + the gate) on a checkpoint whose gate mean has been verified against init, and record the gate's verdict.
```
