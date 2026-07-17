# Code review — Module 8 (External Comparators + Rationale Audit + Sealed Eval + Reproducibility)

Date: 2026-07-17 · Scope: the feat-010 + feat-012 + feat-013 diff (comparators/, rationale/rationale_audit.py,
evaluation/sealed_eval.py, reproducibility/, config.py) + the four new test files.

## Method

Adversarial workflow review (`module8-review`): 5 finder dimensions run in parallel — sealed-math,
comparators (leakage), rationale-audit, reproducibility, contract+tests — each producing structured findings
with a concrete failure scenario, then **per-finding skeptical verification** (an independent agent that tries
to REFUTE each claim by tracing the real control/data flow, defaulting to REFUTED unless it can name inputs
that produce a wrong result). 13 candidates → 6 CONFIRMED + 3 PLAUSIBLE + 4 refuted.

## Confirmed + fixed

| # | Sev | File | Defect | Fix |
|---|-----|------|--------|-----|
| 1 | high | `reproducibility/fallacy_scan.py` | A detector that **raised** was recorded `evaluated: True, flagged: None`, so `complete` stayed True and the errored detector counted as a clean part of an 11/11 scan → `verify` returned `REPRODUCIBLE` with a fallacy never examined. | Errored detectors are `evaluated: False` and listed in a new `errored` field; `complete` requires all 11 to run cleanly → `incomplete` → `PARTIALLY_REPRODUCIBLE`. |
| 2 | medium | `rationale/rationale_audit.py` | CORUM source-ablation iterated only `_PP_RELATIONS`, leaving every (100%-CORUM) `complex_membership` edge active, so `source_ablation['corum']` was systematically understated + mislabeled. | `_source_keep_mask` ablates over PP **and** `complex_membership` relations (guarded on column count). |
| 3 | medium | `rationale/rationale_audit.py` | `covered` was a stratum key AND a post-selection skip, so uncovered picks consumed and discarded audit-case slots → `n_audited` < `n_cases` even when covered targets remained. | Uncovered targets are filtered out **before** `_select_cases`; `n_uncovered_in_dataset` reported separately. |
| 4 | medium | `reproducibility/fallacy_scan.py` | `ecological` computed a group-mean correlation that is degenerately ±1 with ≤2 groups → spurious flag → false `NOT_REPRODUCIBLE`. | Requires ≥3 groups; with fewer it returns `flagged: False` with `aggregate_corr: None`. |
| 5 | low | `evaluation/sealed_eval.py` | No minimum-row guard: a 0/1-row challenge fold seals a NaN / zero-width bootstrap CI — and the split is write-once. | `min_rows` (default 2) guard raises before anything is written. |
| 6 | low | `comparators/compatibility_report.py` | `public_only = "public" in EXPOSURE_CLASS.lower()` is True for `"non-public (proprietary…)"`. | Explicit `PUBLIC_ONLY` class flag on each adapter; report reads the flag. |

(Findings 1 and "a crashed detector scored as a clean 11/11 pass" were the same root cause from two finders.)

## Regression tests added

`test_every_detector_has_a_working_flag_path` (all 11 flag paths), `test_errored_detector_is_not_counted_as_clean_coverage`,
`test_verify_partial_when_a_fallacy_detector_errors`, `test_ecological_needs_three_groups`,
`test_corum_ablation_reaches_complex_membership_edges`, `test_evaluate_refuses_degenerate_fold`,
`test_bootstrap_lcb_on_nonconstant_diffs`, `test_public_only_is_explicit_flag_not_substring`.

## Confirmed correct (finder attempted, refuted / no defect)

- Sealed eval: per-row `_rowwise_systema` matches `systema_pert_specific_delta`; the paired-row bootstrap LCB
  (2.5th percentile, best baseline by point estimate) and the two-clause H1 rule are correct; challenge-fold /
  baseline-prediction row alignment is enforced; the write-once seal holds.
- Comparators: both fit using **only** training responses (no val/challenge leakage); STRING source filtering,
  the Stable-Shift low-rank + presence-weighted graph-conv, and the TxPert softmax attention are correct.

## Verification (pass 1)

`./init.sh` green at **200 tests** (171 prior + 29 Module 8). Fully synthetic.

---

# Pass 2 — xhigh workflow `/code-review` of the committed module (5ea8a4b)

63 agents (one finder per correctness angle + a cleanup finder, an independent verifier per distinct
(file, line), then a ranked report) over `1f1f52e..5ea8a4b`, explicitly told not to re-report pass 1's
findings. **15 CONFIRMED, all fixed.** Most were verified by the reviewers actually executing the code.

The unifying theme: **each subsystem failed toward its own headline claim** — the verifier certified
REPRODUCIBLE on checks it never performed, and the sealed evaluator's seal could be walked around.

## Tier 1 — the verifier certified what it never checked

| # | File | Defect | Fix |
|---|---|---|---|
| 1 | `verify.py:40` | `_resolve` returned **absolute** manifest paths unchanged. config.py's roots are absolute by default, so a manifest built from them sent every hash/schema check at the **original run's files** — verifying that run against itself. Reviewers proved an **EMPTY checkout returns REPRODUCIBLE** with every hash 'pass'. | `_resolve` rejects absolute paths and any `..` escape → the check reports `missing` → CANNOT_VERIFY. |
| 2 | `verify.py:120` | An absent (or misspelled) `hashes` block emitted **zero** critical checks, so a manifest that hashed nothing sailed through as REPRODUCIBLE. `_DETERMINISTIC` was only used to *label* checks that existed, never to require them. | Every `_DETERMINISTIC` name absent from the manifest now emits a `missing` critical check. |
| 3 | `verify.py:94` | `_check_decision` passed **vacuously**: `bool(None) == bool(None)` when both records omit `h1_confirmed`, and the tolerance loop skipped every numeric key missing from `observed` — the critical confirmatory check could pass having compared nothing. | Requires `h1_confirmed` in both records **and** ≥1 numeric field in common; else `missing`. |
| 4 | `verify.py:82` | `_check_config` returned `"skip"` whenever `config_snapshot` was None — **the default argument** — and `_verdict` treats `skip` as clean. A reproduction that changed `DELTA_PRED` 0.05→0.01 (which alone flips the H1 call) certified as fully REPRODUCIBLE. | Config is now **critical**; an unverifiable config is `missing` → CANNOT_VERIFY. |
| 5 | `verify.py:93` | `tolerance` defaulted to **0.0** while `sealed_eval` emits no `tolerance` key → bit-exact float equality demanded → a rerun differing by 6e-16 returned NOT_REPRODUCIBLE. | `DEFAULT_DECISION_TOLERANCE = 1e-6`. |

## Tier 2 — the sealed seal was walkable

| # | File | Defect | Fix |
|---|---|---|---|
| 6 | `sealed_eval.py:77` | **The write-once seal was keyed on `seed`** — a free parameter that only drives the bootstrap RNG. A steward who got `lcb_95=0.048` (just under the 0.05 margin) could re-run at `seed=1`, pass the existence check, re-open the sequestered fold and seal a confirming decision — with no `force` and no error. This is **the garden-of-forks fallacy this very module ships a detector for**. | The seal is now **per split** (any existing sealed result for the split blocks), since the fold is opened once. |
| 7 | `sealed_eval.py:111` | The H1 rule's second clause is a **tautology**: the perturbed-mean baseline predicts `train_mean`, systema subtracts `train_mean` → a constant row → ρ ≡ **exactly 0.0** for any data, so `beats_perturbed` reduces to `ρ_EGIPG > 0`. The sealed record read as though a treatment-mean hurdle had been cleared. | Kept (spec-mandated) but **documented, not silently patched** — the docstring and a new `perturbed_mean_reference_note` field in the sealed JSON state the structural zero. |

## Tier 3 — fallacy detectors firing on clean data / passing on undefined input

| # | File | Defect | Fix |
|---|---|---|---|
| 8 | `fallacy_scan.py:95` | `regression_to_mean` measured both deviations against the **pooled** grand mean of baseline+followup, so any uniform shift was misread as regression: `regression_to_mean(b, b - 10)` (correlation exactly 1.0, zero regression) **flagged** → false NOT_REPRODUCIBLE. Also asymmetric (`b + 10` never flagged). | Each series is measured against **its own** mean (shift-invariant, symmetric). |
| 9 | `fallacy_scan.py:62` | `berkson` had no min-size guard; `_corr` returns 0.0 for <2 rows, which the comparison read as "selection destroyed the association" → a 1-row or 0-row selection **flagged** with no collider present. | `min_selected=3`, else `Unevaluable`. |
| 10 | `fallacy_scan.py:148` | `reverse_causation(0.0, 0.0)` **flagged** — no association in either direction means no causal claim to invalidate. `margin` polarity was also inverted vs `berkson`/`collider`. | Requires `min_forward=0.1` before flagging; `margin` now raises the bar, matching its siblings. |
| 11 | `fallacy_scan.py:23` | `_corr`'s `np.ptp(x) == 0` guard never catches NaN (`nan == 0` is False), so a single NaN reached `np.corrcoef` and `_sign(nan)` raised `ValueError: cannot convert float NaN to integer` → the detector errored → verdict silently downgraded. | `_corr` drops non-finite pairs; `_sign` is non-finite-safe. |
| 12 | `fallacy_scan.py:109` | `survivorship` fell back to `surv = full` with **zero survivors**, reporting a clean unflagged pass that counted toward 11/11 — the same "a check that didn't run must not certify" hazard pass 1 fixed for errored detectors, reached via an in-detector fallback. | Raises `Unevaluable`. |

A new `Unevaluable(ValueError)` makes this principle explicit: degenerate input **raises**, so the existing
errored machinery records it, coverage drops below 11/11, and the verdict is PARTIALLY — never a silent pass.
`ecological`'s ≤2-group case was converted from pass-1's "don't flag" to `Unevaluable` for the same reason.

## Tier 4 — audit + provenance

| # | File | Defect | Fix |
|---|---|---|---|
| 13 | `rationale_audit.py:273` | `audit_rationale` moved `model` to `device` but **never moved `head`**, and `_audit_one`'s `device` param was dead → a cuda input into a CPU `nn.Linear` killed the whole audit on the first case. | `head = head.to(device)`; the dead param is now the `seed`. |
| 14 | `rationale_audit.py:187` | `_stability` enables DropEdge, which draws from the **global** torch RNG that the audit's `seed` never touches → `mean_stability` in `audit_report.json` was not reproducible from `(model, head, dataset, seed)` (reviewers measured 1.0 / 0.778 / 0.852 at one audit seed). | The global RNG is seeded from the audit seed and its prior state restored. |
| 15 | `txpert_public.py:42` | `wrapped = _TXPERT_AVAILABLE` asserted `wrapped_upstream: true` from **mere importability** — any module named `txpert` on `sys.path`, including a name-squat — in the one artifact whose purpose is provenance audit, while `predict` never touches upstream code on any path. | `wrapped = False` always (it is the public reimpl); importability is recorded separately as `upstream_importable`. |

## Bonus: a latent Module 4 bug the CUDA fix surfaced

Fixing #13 let the audit reach the GPU for the first time, which immediately failed in **pre-existing Module 4
code**: `RationaleHead._select` builds `rel_id`/`local` on CPU but indexes them with `idx` from a CUDA `topk`
→ `RuntimeError: indices should be either on cpu or on the same device`. `RationaleHead` had never run on GPU
(its tests are CPU-only). Fixed with `idx = idx.cpu()` — the root cause of "the audit dies on CUDA" spanned
both modules.

## Regression tests (+15, all red-green verified)

`test_verify_cannot_verify_absolute_manifest_path`, `test_verify_empty_checkout_is_never_reproducible`
(**proved**: reverting the fix returns REPRODUCIBLE on an empty checkout), `test_verify_cannot_verify_missing_hashes_block`,
`test_verify_cannot_verify_vacuous_decision`, `test_verify_cannot_verify_without_config_snapshot`,
`test_verify_not_reproducible_on_config_change`, `test_verify_decision_tolerance_defaults_to_float_noise`,
`test_seal_is_per_split_not_per_seed`, `test_perturbed_mean_reference_is_structurally_zero`,
`test_regression_to_mean_ignores_a_uniform_shift`, `test_berkson_needs_enough_selected_rows`,
`test_reverse_causation_needs_a_forward_association`, `test_survivorship_with_zero_survivors_is_unevaluable`,
`test_nan_input_does_not_crash_detectors`, `test_stability_is_reproducible_from_the_audit_seed`,
`test_audit_runs_on_cuda` (a real GPU run, not a mock).

One test written in this pass was itself caught by red-green: `test_verify_empty_checkout_is_never_reproducible`
initially passed even with the bug reverted (a relative predictions path incidentally dragged the verdict to
PARTIALLY), so it was strengthened to make every manifest path absolute — it now discriminates.

## Verification (pass 2)

`./init.sh` green at **215 tests** (200 + 15), exit 0, CUDA test executed (not skipped).
