# Code review — Module 6 (Evaluation Metrics + Simple Baselines)

Date: 2026-07-16 · Scope: the Module 6 diff (feat-009 + feat-006). Method: a dynamic multi-agent
adversarial-review workflow — 6 finder dimensions over the 8 new files, each finding then handed to an
independent verifier prompted to **refute** it (default REFUTED unless a concrete failing input is
constructed). 8 findings surfaced; **8 confirmed, 0 refuted, 0 uncertain**. The verifiers downgraded every
finding to minor/nit (all latent edge cases, none exercised by the current finite fixtures/baselines);
all 8 were nonetheless fixed.

## Findings and resolutions

1. **`centroid_accuracy` credited a zero predictor 1.0** (metrics.py, metric-correctness).
   A zero-norm prediction row normalises to cosine 0 against *every* centroid, so `own >= rowmax - eps`
   became `0 >= -eps` → True for all rows → 1.0, inverting the "zero predictor scores worst" intent (and
   `metrics_ref` agreed at 1.0, so the cross-check was blind to it). **Fixed:** a degenerate (zero-norm /
   non-finite) prediction row is a **miss** (`pred_ok = isfinite(p).all(1) & norm(p) > 0`), in both
   implementations. New test `test_centroid_accuracy_penalizes_degenerate_predictor`.

2. **Non-finite rows diverged between the two implementations** (metrics.py:34, metric-agreement).
   `metrics.py`'s `np.where(den > 0, …)` treated `nan > 0` as False → silent 0.0, while `metrics_ref`
   fell through its `std == 0` guard into scipy → NaN (and sklearn's finite-check *crashed*
   `centroid_accuracy`). **Fixed:** both implementations contribute **0.0** on non-finite rows by
   construction — finite-guarded row reducers in `metrics.py`, a `_finite()` guard in `metrics_ref`,
   `spearman_corr` masks non-finite **before** ranking (else the row ranks into a finite vector and scores
   spuriously), and `centroid_accuracy` sanitises the sklearn input. New test
   `test_two_implementations_agree_on_non_finite_rows` (asserts finite + equal), verified under
   warnings-as-errors.

3. **`label_permutation` was not a derangement** (metric_qualification.py, gate-and-control, nit).
   A plain permutation leaves ~1 fixed point on average (and is the identity ~50% of the time at n=2), so
   the N1 negative retained perfectly-scored rows and could tie the oracle → spuriously fail the gate.
   **Fixed:** draw a derangement (resample up to 16×, else `np.roll` by 1 for n≥2). New test
   `test_label_permutation_is_a_derangement`.

4. **`RidgeBaseline` transposed single-program output** (simple_baselines.py, baseline-correctness).
   For K==1, `np.atleast_2d` promotes sklearn's 1-D `(M,)` prediction to `(1, M)` instead of `(M, 1)`,
   breaking `delta_x = dz @ B.T`. **Fixed:** `_as_columns` reshapes 1-D → `(M, 1)`.

5. **`LowRankBaseline` hit the same bug at rank==1** (simple_baselines.py, baseline-correctness).
   Same `atleast_2d` transpose in the reduced-coordinate reconstruction. **Fixed** by the same
   `_as_columns` helper. Findings 4–5 covered by `test_ridge_and_low_rank_handle_single_program`.

6. **`topk_recall` / `sign_accuracy` never exercised gene selection** (test_metrics.py, test-quality).
   Only tested with `pred == ±true`, where the selected index set is invariant — a bug selecting the
   *weakest* genes would still pass. **Fixed:** added partial-overlap cases with hand-built magnitudes
   asserting intermediate values (2/3) only correct magnitude-selection produces.

7. **The agreement test never exercised the zero/constant convention** (test_metrics.py, test-quality).
   The two implementations guard degeneracy with independent code, yet the agreement fixture had only
   non-degenerate rows. **Fixed:** `test_two_implementations_agree_on_degenerate_rows` (zero + constant
   rows) plus the non-finite agreement test from finding 2.

8. **`signed_de_metrics` only tested with perfect probs** (test_metrics.py, test-quality).
   `probs == labels` plus a vacuous `0 ≤ x ≤ 1` assertion — discrimination, precision≠recall, and
   imperfect-case AUPRC unverified. **Fixed:** `test_signed_de_metrics_imperfect_probs_discriminate`
   (P=R=0.5 on up, P=0.75/R=1.0 on down, AUPRC<1, macro-F1∈(0,1)).

## Outcome

`./init.sh` green at **131 tests** (92 prior + 39 Module 6), zero warnings; both metric implementations
verified to agree on non-degenerate, zero/constant, and non-finite rows under warnings-as-errors.
