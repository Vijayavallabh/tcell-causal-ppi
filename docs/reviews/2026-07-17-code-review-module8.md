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

## Verification

`./init.sh` green at **200 tests** (171 prior + 29 Module 8). Fully synthetic.
