# feat-005 — method × K basis study (characterisation only)

**Session C, 2026-07-20.** This is a CHARACTERISATION STUDY. It changes nothing: the production
basis stays frozen, every candidate is fit in memory, and all output lands under
`data/results/basis_study/`. If some other method/K scores better that is a FINDING to report,
not a licence to swap the basis.

Evidence block for `feature_list.json` — session A merges this at commit time. Do not edit the DoD
triad from this session.

---

## 1. Frozen-basis integrity

Verified against the fingerprint recorded independently at 2026-07-20 19:00, *before* this session
started (so a match is a real check, not a file compared to itself).

| file | recorded sha256 (first 32) | observed at session START | mtime |
|---|---|---|---|
| `data/intermediate/gene_program_loadings.parquet` | `d5a8883da20dd8ef8824f565188c5f45` | `d5a8883da20dd8ef8824f565188c5f45` ✓ | 2026-07-16 11:07:17 ✓ |
| `data/intermediate/program_response.parquet` | `2c29ba9535e293dd003836f3aad50d42` | `2c29ba9535e293dd003836f3aad50d42` ✓ | 2026-07-16 11:07:17 ✓ |

END-OF-SESSION re-check (2026-07-21 04:0x, after the full sweep): **both unchanged** —
`d5a8883da20dd8ef8824f565188c5f45…` and `2c29ba9535e293dd003836f3aad50d42…`, mtimes still
2026-07-16 11:07:17. `find data/intermediate -newermt "2026-07-20 19:00"` returns nothing, so no file
in that directory was written at all this session. `run_program_basis` was never invoked.

`run_program_basis` was **not run at all** this session, and neither `basis_study.py` nor
`run_basis_study.py` may reference `save_program_basis` / `save_program_response` /
`PROGRAM_LOADINGS_PATH` / `PROGRAM_RESPONSE_PATH` / `write_parquet_atomic`. That is enforced by an
AST-level test (`test_study_never_references_the_frozen_basis_write_path`), not by text matching —
the modules' own docstrings name those functions to explain why they are never called, and a guard
that fires on prose is one you learn to silence. Verified to fire: injecting a real
`save_program_basis` import into the driver turns the test red; removing it turns it green.

## 2. Metric definitions — pinned against the recorded numbers BEFORE any new fit

The recorded feat-005 evidence quotes *"centered reconstruction MAE 0.687 vs 0.817 predict-zero
baseline"* and *"22.7% exact-zero loadings"* for the frozen sparse_pca K=128 fit, but not the
formulas. Rather than guess, the definitions were recovered by scoring the **frozen B and A already
on disk** (read-only — no fitting involved) under three candidate readings:

| candidate definition | recon MAE | zero-baseline |
|---|---|---|
| **centred target, centred recon — `MAE(Z−μ, A·Bᵀ)`** | **0.6865** | **0.8174** |
| raw target, mean-added recon — `MAE(Z, A·Bᵀ+μ)` | 0.6865 | 0.8190 |
| raw target, raw recon — `MAE(Z, A·Bᵀ)` | 0.6887 | 0.8190 |
| *recorded* | *0.687* | *0.817* |

Only the first reproduces **both** recorded numbers, so the study uses it:

- **Reconstruction** — `recon_mae = mean|Zc − A·Bᵀ|` where `Zc = Z − μ`, `μ` = gene-wise mean over
  the train rows; baseline `= mean|Zc|`; `explained_frac = 1 − recon/baseline`
  (frozen cell: 1 − 0.6865/0.8174 = **16.0%**, matching the recorded "16% explained").
  `explained_frac` is `None` when the baseline is 0 — nothing to explain is UNDECIDABLE, not 0%.
- **Sparsity** — `zero_frac = mean(B == 0)`, **exact** zeros, no tolerance
  (frozen cell: **22.69%**, matching the recorded 22.7%; a `|B| < 1e-9` tolerance would inflate it).
  Plus `n_dead` = count of all-zero columns (frozen cell: **0**).
- **Stability** — §3 below.

### Reconstruction is reported on two targets, deliberately

The methods do not all model the same matrix. NMF is fit on `max(Z, 0)` (z-scores are signed; NMF
needs non-negativity), and TruncatedSVD does not centre. Scoring NMF against the signed centred
target and reading a ranking off it would report "NMF reconstructs worse" when the true statement is
"NMF answers a different question". So every cell reports:

- `recon_mae` — the **common** centred target `Zc`, comparable across all cells, and the one that
  matters downstream (the decoder reconstructs *signed* `delta_x` through B);
- `native_recon_mae` + `native_target` — each method's own target (`centred` / `raw_uncentred` /
  `positive_part`), which is the fair measure of how well the method did its own job.

## 3. Stability — resampling protocol, stated before the sweep was run

A basis is identified only up to **permutation and sign**, so an elementwise correlation between two
bases is meaningless. Protocol, fixed in `run_basis_study.py` as module constants:

| parameter | value |
|---|---|
| resamples per cell | `N_RESAMPLE = 3` → 3 pairwise comparisons (0,1), (0,2), (1,2) |
| subsample fraction | `RESAMPLE_FRAC = 0.8` of the TRAIN rows, **without replacement** |
| row seed | `RESAMPLE_SEED = 20260720`, resample *r* uses `default_rng(20260720 + r)` |
| factorisation seed | **held FIXED** at `config.SPLIT_SEED = 0` across resamples |
| matching | Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on `|cosine|` between B columns |
| statistic | mean `|cosine|` over the K matched pairs, then averaged over the 3 pairs |

The factorisation seed is held fixed **on purpose**: varying it too would conflate data-sensitivity
with initialisation noise, and the question being asked is "would a different sample of the training
data recover the same programs?"

Degenerate handling — a dead (all-zero) component has undefined cosine (0/0):

- a dead component scores 0 and is counted in `n_dead_across_resamples`; it drags stability down,
  which is correct (a dead program is not a stable program) but is reported separately so the number
  stays interpretable;
- if **either** basis in a pair is entirely dead there is nothing to match and that pair is
  `None` — **UNDECIDABLE**, never `0.0`, which would read as a real low-stability result. Cells
  carry `n_undecidable_pairs`.

Fold-locality: resamples are drawn from the TRAIN rows only. The fence in `fit_program_basis` /
`train_row_indices` is not widened; no val/cal/challenge row enters any factorisation.

### Held-out reconstruction (why the K axis is otherwise uninterpretable)

In-sample reconstruction is **monotone in K by construction** — K=512 wins every time, and a table
of in-sample numbers cannot answer "is K=128 the right choice". Each stability resample therefore
does double duty: fit on the 80%, then score the held-out 20% (still TRAIN rows) by least-squares
projection onto span(B), centred by the **fit subset's** mean so the held-out rows contribute nothing
to the transform. LS projection rather than each method's own `transform()` keeps this
method-agnostic and needs no fitted encoder. Cost: zero extra fits.

## 4. Multiplicity — stated before looking at any result

17 cells (4 methods × 4 K, plus the VAE at K=128) is a lot of chances to find a winner by luck.

- **Reconstruction** is the only axis with a real sampling distribution available: per-row MAE is
  persisted for every cell (`data/results/basis_study/row_mae/*.npy`), so each candidate is compared
  to the frozen cell (sparse_pca K=128) as a **paired** contrast over the 21,262 train rows. Family
  = the 16 candidate-vs-frozen contrasts; **both Bonferroni and Holm** are reported, and neither is
  chosen after seeing which rescues a claim. Persisting per-row residuals up front is what makes it
  possible to settle the statistic *after* the sweep without re-fitting anything.
  Caveat recorded with the result: this is a **within-sample** comparison over correlated rows — it
  says a basis reconstructs *these* rows better, not that it generalises.
- **Sparsity** is a deterministic property of a single fit. There is no sampling distribution, so it
  is reported descriptively with **no p-value**.
- **Stability** has only 3 pairs per cell. That is not enough to power a per-cell test, so it too is
  reported descriptively (mean + the individual pair values), explicitly *not* as a significance
  claim.

## 5. Convergence evidence

`_factor` silences `ConvergenceWarning` by design, so a capped fit was previously invisible to any
caller — an outer `warnings.catch_warnings()` sees nothing. `fit_program_basis` now takes an optional
`info` dict (purely additive; the production path passes nothing and is numerically unchanged)
recording `n_iter` / `max_iter` / `converged`:

- iterative methods (sparse_pca, nmf, fastica): `converged = n_iter < max_iter`;
- **svd**: `converged = None`. Randomized SVD has no iteration cap to hit, and claiming `True` would
  invent evidence it never produced;
- **vae**: `converged = None`. A fixed epoch budget is not a convergence criterion.

A fit that hit its cap is **UNDECIDABLE** — not evidence that the method reconstructs worse. Any such
cell is reported as capped rather than ranked.

## 6. Sanity cell — the harness reproduces the frozen fit

Run before trusting any other cell, via `run_basis_study --only sparse_pca:128`:

| metric | this harness | recorded |
|---|---|---|
| recon MAE (centred) | **0.686475** | 0.687 ✓ |
| zero-baseline MAE | **0.817409** | 0.817 ✓ |
| explained fraction | **16.02%** | 16% ✓ |
| exact-zero loadings | **22.694%** | 22.7% ✓ |
| dead programs | **0** | 0 ✓ |
| convergence | `converged=True`, `n_iter=2` of 100 | not previously recorded |

Fit time **361 s** (vs 289 s recorded for the production run) — the box is shared, carrying load
average ~57/64 from two other live sessions, with this study pinned to `OMP_NUM_THREADS=4`.

### FINDING — a recorded number that does not reproduce

The feat-005 evidence block contains the parenthetical *"sparse_pca trades reconstruction for
sparsity vs svd ~0.61"*. That does not reproduce. Measured at K=128:

| cell | recon MAE (common centred target) | recon MAE (own native target) |
|---|---|---|
| sparse_pca K=128 (frozen) | 0.6865 | 0.6865 (centred) |
| svd K=128 | **0.6876** | **0.6851** (raw uncentred) |

svd is within ±0.003 of sparse_pca on either target, not 0.08 better. Since the same harness
reproduces the sparse_pca cell to four decimals, the harness is not the suspect. The "~0.61" appears
to be an uncomputed comparison. Flagged for a dated `CORRECTION:` append to feat-005's evidence —
**not edited from this session**, since the DoD triad is shared state with two other live sessions.

## 7. Timing probe — the basis for the sweep estimate

Real cells, full 21,262 × 10,282 train matrix, `--no-stability` (one fit + scoring each), measured
under load ~57 at `OMP_NUM_THREADS=4`. These are whole-pipeline times, not sub-component benchmarks.

| cell | fit s | recon MAE | explained | zero_frac | converged |
|---|---|---|---|---|---|
| svd K=64 | 8.7 | 0.7051 | 13.7% | 0.0% | n/a (no cap) |
| svd K=128 | 10.2 | 0.6876 | 15.9% | 0.0% | n/a |
| svd K=256 | 13.3 | 0.6694 | 18.1% | 0.0% | n/a |
| svd K=512 | 22.0 | 0.6449 | 21.1% | 0.0% | n/a |
| sparse_pca K=64 | 171.0 | 0.7030 | 14.0% | 14.9% | True |
| sparse_pca K=128 | 361.0 | 0.6865 | 16.0% | 22.7% | **True** (2 of 100) |
| nmf K=64 | 85.4 | 0.8358 | **−2.2%** | 47.9% | **False** (100 of 100) |
| fastica K=64 | 564.2 | 0.7023 | 14.1% | 0.0% | True (94 of 100) |

**sparse_pca is linear in K — measured, not assumed.** K=64 → K=128 costs 171 s → 361 s, a 2.11×
factor for 2× K. Two points, so the extrapolation to K=256/512 rests on measurement.

**Full-cell path validated end-to-end** on the cheapest cell (svd K=64, all four fits + Hungarian +
held-out): stability **0.887** (pairs 0.874 / 0.897 / 0.891, 0 undecidable, 0 dead), held-out
explained **13.73%** against in-sample 13.75%. Held-out ≈ in-sample at K=64 is the expected control
when N ≫ K; the gap is what makes the K axis readable at K=512. Cell total 57.4 s on an 8.7 s fit,
which fixes the scoring overhead at ~60 s/cell (used by the budget model, not by any result).

### NMF K=64 is UNDECIDABLE, not a loss

It hit the iteration cap *and* explains a **negative** fraction — worse than predicting zero — on the
common target (−2.2%) and on its own positive-part target (−4.5%). A capped fit is not evidence the
method reconstructs worse, so this cell is reported as capped and is **not ranked**. Whether to raise
`MAX_ITER` for NMF is a compute-budget decision, recorded below rather than decided unilaterally.

Note on objective mismatch, applying uniformly to every cell: all four methods minimise a squared
(L2) error, while the reported metric is MAE — the definition the recorded evidence uses and that the
sanity cell reproduces. No method is scored on the loss it optimises, so the comparison is even, but
no cell's MAE should be read as "this method failed at its own objective".

## 8. Approved sweep plan (2026-07-20)

Two decisions were escalated rather than taken unilaterally, because both trade honesty against
compute. Both were approved as recommended:

1. **NMF iteration budget** — `max_iter=500` for **nmf at K=128 only** (the K matching the frozen
   basis); 100 elsewhere. At 100 everywhere every NMF cell caps out and NMF is undecidable
   everywhere; at 500 everywhere NMF alone costs ~6 h. This buys one decidable NMF point where the
   comparison matters. The other three NMF cells are expected to report `converged=False` and are
   **not ranked**. Encoded in `cell_max_iter()`.
2. **Bounded sweep** — cheapest cell first, **90 min hard cap per cell**, **10 h total budget**.
   Anything not completed is written as an explicit `not_measured` cell (`timeout`,
   `budget_exhausted`, or `failed`), never left absent and never left blank.

Mechanism: each cell runs as a **subprocess** so the cap is enforced by the OS. An in-process alarm
would sit unserviced inside a long BLAS call — precisely where a runaway cell spends its time.

Cells whose four fits cannot fit inside the cap **degrade to fit+score only** rather than burn the
whole cap and write nothing (`wants_stability()`); they report real reconstruction and sparsity with
`stability_computed=False`. On the measured cost model that is fastica K=256 and K=512.

Projected **8.7 h against the 10 h budget** (assuming fastica linear in K; its K=128 point was being
measured when the plan was fixed). Planned order and per-cell decisions:

| order | cell | est. fit | est. cell | stability | max_iter |
|---|---|---|---|---|---|
| 1–4 | svd K=64/128/256/512 | 0.1–1.2 m | 1.5–4.9 m | yes | 100 |
| 5 | nmf K=64 | 1.4 m | 5.8 m | yes | 100 |
| 6 | nmf K=128 | 2.8 m | 10.7 m | yes | **500** |
| 7 | sparse_pca K=64 | 2.9 m | 10.7 m | yes | 100 |
| 8–9 | nmf K=256, sparse_pca K=128 | 5.7 m | 20.4 m | yes | 100 |
| 10 | fastica K=64 | 9.4 m | 33.0 m | yes | 100 |
| 11–12 | nmf K=512, sparse_pca K=256 | 11.4 m | ~39.8 m | yes | 100 |
| 13 | fastica K=128 | 18.8 m | 64.9 m | yes | 100 |
| 14 | vae K=128 | ~20 m (never timed) | 69.0 m | yes | 20 epochs |
| 15 | sparse_pca K=512 | 22.8 m | 78.5 m | yes | 100 |
| 16 | fastica K=256 | 37.6 m | 128.8 m → capped | **no** | 100 |
| 17 | fastica K=512 | 75.2 m | 256.7 m → capped | **no** | 100 |

The vae row is the one cell with **no measured cost at all** — its 20 m is a guess, and it is
subject to the same cap as everything else.

## 9. Results

Sweep ran 2026-07-20 21:40 → 2026-07-21 04:00 (6.3 h), **16 of 17 cells completed**, all under the
approved bounds. Full table `data/results/basis_study/method_k_table.csv`, contrasts
`contrasts_vs_frozen.csv`, per-cell JSON under `cells/`, per-row residuals under `row_mae/`.

> `fit_seconds` in the table is **not** comparable to the §7 probe timings: the probes ran under
> load ~57 from two other live sessions, the sweep largely after they finished (fastica K=64:
> 564 s probe vs 226 s in-sweep). Treat those numbers as provenance, not as a benchmark.

### 9.1 The headline: the frozen choice is unremarkable, and that is the successful outcome

K=128 is the only **fully decidable** column — sparse_pca converges in 2 iterations, svd has no
iteration cap, and nmf/fastica were given the 500-iteration budget and both converged (257 and 115).
So this is the one place where methods are compared on equal footing:

| method | in-sample | **held-out** | overfit gap | stability | exact-zero | converged |
|---|---|---|---|---|---|---|
| fastica | 16.22% | **15.59%** | 0.63% | 0.729 | 0.0% | True (115/500) |
| svd | 15.88% | **15.55%** | 0.33% | 0.760 | 0.0% | n/a |
| **sparse_pca (FROZEN)** | 16.02% | **15.41%** | 0.61% | 0.612 | **22.7%** | True (2/100) |
| vae | 14.39% | **14.11%** | 0.28% | **0.258** | 0.0% | n/a (20 epochs) |
| nmf | −1.43% | 13.11% | — | 0.869 | 55.7% | True (257/500) |

The frozen basis sits **0.18 percentage points** of held-out explained fraction behind the best cell
at its own K. That gap is *consistent* — paired by resample (each resample scores both cells on the
same held-out rows) the differences are 0.00145 / 0.00147 / 0.00151 MAE, same sign, tight — so it is
a real ordering, not noise. It is also **practically negligible**, and it is bought by giving up the
sparsity: sparse_pca is the only method combining that accuracy with sparsity — 22.7% exact zeros
against 0.0% for svd, fastica and the VAE, which are dense to the last coefficient. NMF yields more
zeros still (55.7%) but pays 2.3 pp of held-out (13.11%) and is decidably unsuited to a signed
target (§9.3).

> **CORRECTION (2026-07-21, caught by session A at merge review):** this paragraph previously read
> "sparse_pca is the only method here producing any exact zeros at all", which is **false** — NMF at
> K=128 has 55.7% exact zeros, more than double, and it is not an excluded cell (it converged
> 257/500 and appears in the table directly above). The claim is exactly the class of error this
> study was correcting in the `svd ~0.61` figure: a ranking asserted past what was computed, with the
> refuting number already sitting in my own table. The corrected sentence above restricts the claim
> to what the data supports — sparse_pca is unique in *combining* competitive accuracy with sparsity,
> not in producing sparsity.

**Nothing found here justifies changing the frozen basis.** Reported as a finding, per the brief;
the decision is not this session's to take.

### 9.2 The K axis: reconstruction buys accuracy and pays in stability

held-out explained fraction / Hungarian-matched stability:

| method | K=64 | K=128 | K=256 | K=512 |
|---|---|---|---|---|
| sparse_pca | 13.69% / 0.841 | 15.41% / 0.612 | 16.82% / 0.350 | 18.33% / **0.225** |
| svd | 13.73% / 0.887 | 15.55% / 0.760 | 17.07% / 0.570 | 18.60% / 0.381 |
| fastica | 13.77% / 0.792 | 15.59% / 0.729 | 17.17% / 0.628 | 18.70% / 0.457 |
| nmf | 11.50% / 0.843 | 13.11% / 0.869 | 14.70% / 0.778 | 16.57% / 0.693 |

Every doubling of K buys roughly **+1.5 to +1.8 pp** of held-out reconstruction and costs roughly
**0.15–0.25** of matched stability. sparse_pca degrades fastest and monotonically:
**0.841 → 0.612 → 0.350 → 0.225**. At K=512 its programs barely survive an 80% resample at all,
which for a basis whose whole purpose is *interpretable programs* is a strong argument against
pushing K up. K=128 is a defensible middle, which is what the frozen configuration already uses.

The K=512 stability was **predicted at ~0.2 from the first three points before it was run** and
measured at 0.225 (pairs 0.2239 / 0.2276 / 0.2235, 0 undecidable, 0 dead) — recorded here because a
prediction that survives contact with the measurement is worth more than one made afterwards.

### 9.2a The K=512 column, now that it is complete

| method | held-out | stability | exact-zero | converged |
|---|---|---|---|---|
| fastica | **18.70%** | 0.457 | 0.0% | False (100/100) |
| svd | 18.60% | 0.381 | 0.0% | n/a |
| sparse_pca | 18.33% | **0.225** | **51.1%** | True (6/100) |
| nmf | 16.57% | 0.693 | 69.2% | False (100/100) |

sparse_pca at K=512 reaches held-out reconstruction within 0.37 pp of the best cell in the entire
study **and** the highest sparsity of any non-NMF cell (51.1%) — but it is the least stable basis
measured apart from the VAE. Two of the four cells in this column are capped and therefore not
ranked. This column is *not* a fair method comparison for that reason; §9.1 (K=128) is.

The in-sample/held-out gap widens with K exactly as expected (svd: 0.02 pp at K=64 → 2.45 pp at
K=512), confirming the held-out column is doing the job it was added for.

### 9.3 NMF is decidably unsuited here — and it is the ENCODING, not the subspace

NMF is the one method whose two reconstruction columns disagree wildly, and the reason is
methodological rather than numerical:

- **in-sample** scores the method's own non-negative A → **−1.43%**, worse than predicting zero;
- **held-out** scores an LS projection onto span(B), which may use negative coefficients → **+13.11%**.

So NMF's *subspace* is serviceable; its *non-negative encoding* cannot represent a signed centred
target. This is decidable at K=128 because that cell **converged** (257/500) — it is not a capped-fit
artifact — and it still reports **−2.91% on its own positive-part target**. NMF at K=64/256/512 hit
the 100-iteration cap (`converged=False`) and are **not ranked**.

This also means the two reconstruction columns are not interchangeable across methods: in-sample
asks "how good is this method's encoder?", held-out asks "how good is the subspace it found?".
Held-out is the fairer cross-method number precisely because it is encoder-agnostic.

### 9.4 Shallow VAE (deliverable 2)

Built as a single-linear-layer VAE (encoder `Linear(G, 2K)`, decoder `Linear(K, G)` without bias, so
B is literally the decoder weight and the result is a *basis* comparable to the factorisations), 20
epochs, Adam, β=1. At K=128 it reconstructs **14.11%** held-out — below every matrix factorisation
except NMF — and is by a wide margin the **least stable basis in the study (0.258)**. It produces no
sparsity. **The shallow VAE is not competitive here.** Its `converged` is reported as `None`: a fixed
epoch budget is not a convergence criterion, and no claim is made that it had finished training.

### 9.5 Multiplicity — and why it carries no information at this n

Family = the **15** candidate-vs-frozen paired contrasts actually run (16 candidates minus
sparse_pca K=512, which has no residuals to test; the frozen cell is excluded from its own family).

**Every one of the 15 contrasts has `p_raw` underflowing to exactly 0.0**, so Bonferroni and Holm
both return 0.0 and every contrast "survives" both corrections. That is not evidence of importance —
it is what a paired test over 21,262 rows does. The svd K=128 contrast is significant under both
corrections at an effect size of **−0.14 pp**. Accordingly:

- the **multiplicity correction is reported as required and is not load-bearing**;
- conclusions rest on `delta_explained_frac`, the effect-size column;
- the paired test is on **in-sample** reconstruction — the axis most confounded by capacity, since it
  is monotone in K by construction. The informative axis (held-out) has only 3 resamples per cell, so
  it is reported descriptively with its per-resample values, never with a p-value. **This is the
  study's main statistical limitation**: the axis with a real test is the least meaningful one.
- cross-K contrasts compare bases with different parameter counts; only the **within-K** comparison
  in §9.1 is a clean method comparison.

### 9.6 The cell that defeated the cap, and a cost model that was wrong

`sparse_pca K=512` first came back `not_measured: "timeout"` — it exceeded the 90 min cap even after
degrading to fit+score only, and was recorded with explicit nulls rather than left absent or blank.

**Resolved 2026-07-21.** Re-run with a 4 h cap it completed in **96.3 min** — it had needed six more
minutes than the cap allowed. Stability and held-out were then backfilled separately (3 resample
fits, 4 h 31 m at `OMP_NUM_THREADS=16`), reusing the fit already on disk rather than repeating the
96 min: the fit is deterministic under a fixed seed, so re-running it would only have reproduced
identical numbers. The backfill path refuses to write into a `not_measured` stub — a stub has no fit
at all, and attaching stability to one would fabricate a half-real cell that reads as measured.

**All 17 cells are now complete on all three axes.**

Two operational notes from that run:

- `OMP_NUM_THREADS=16` bought far less than hoped: 9,861 s of CPU over 6,315 s wall is only **1.56×
  effective parallelism**, because MiniBatchSparsePCA's LARS path is largely serial. Threads are not
  the lever for this estimator.
- The job had no per-resample progress output, so its remaining time could only be *inferred* from
  CPU-time rather than read. A `log()` per resample would have made that exact. Recorded as a real
  observability gap, not smoothed over.

The cost model underestimated the cell badly, which is worth recording as a measurement lesson:

| method | measured fit seconds, K=64 → 512 | implied scaling |
|---|---|---|
| sparse_pca | 184 → 354 → 1212 → **>5400** | **super-linear and accelerating** (1.9×, 3.4×, >4.5× per doubling) |
| fastica | 226 → 166 → 169 → 180 | **flat in K** (~90% is a K-independent whitening SVD) |
| svd | 8.7 → 10.1 → 13.4 → 21.4 | sub-linear |
| nmf | 86 → 363* → 450 → 887 | ~linear (*K=128 ran 500 iters) |

The two-point extrapolation for sparse_pca (exponent 1.08 from K=64→128) held at K=256 and broke at
K=512.

**Cost constants refitted 2026-07-21** from the completed sweep (all one run, so like-for-like;
the §7 probe numbers ran under load ~57 and are 2–3× slower for fastica):

| method | base @ K=64 | exponent | note |
|---|---|---|---|
| svd | 8.7 s | 0.43 | sub-linear |
| nmf | 86.4 s | 1.12 | ~linear, O(N·G·K)/iteration |
| sparse_pca | 183.7 s | **1.36** | still a **lower bound** — K=512 exceeded 5400 s for one fit |
| fastica | 225.7 s | **0.00** | measured −0.11, clamped to 0 (cost cannot fall with K) |
| vae | 32.5 s | 1.0 | assumed; one data point |

The sparse_pca row is the one that matters: even the refitted 1.36 exponent predicts 52 min for a
single K=512 fit when the true value is >90 min. The retry below therefore does **not** trust the
model — it measures the single fit first and only then decides whether the resamples are affordable.
That is the same discipline the §7 probe used, applied to the one cell that defeated it.

## 10. What a follow-up would need

Both `sparse_pca K=512` items are **done** (§9.6) — the table has no gaps. What remains open:

- **A held-out contrast with enough resamples to test rather than describe.** `N_RESAMPLE=3` was
  fixed before the sweep and is honoured; raising it is a new pre-registration, not a post-hoc tweak.
  This is the study's real statistical limitation (§9.5): the axis carrying a genuine paired test
  (in-sample) is the one most confounded by capacity, while the informative axis (held-out) has only
  3 values per cell.
- **A decidable NMF/fastica row away from K=128.** Five of the 16 method×K cells hit `max_iter=100`
  and are reported capped and unranked. The convergence budget was spent at K=128 by design, so the
  K=64/256/512 comparisons for those two methods remain undecidable rather than decided.
- **A dated `CORRECTION:` append to feat-005's evidence** for the `svd ~0.61` figure that does not
  reproduce (§6). Not done here: the DoD triad is shared state with two other live sessions.

---

## Test evidence

`src/tests/test_basis_study.py` — 45 tests, all watched failing before implementation.
Mutation testing of the load-bearing lines: **9/10 caught**.

The one survivor is an **equivalent mutant**, not a coverage gap: removing `.manual_seed(seed)` from
the VAE's `torch.Generator()` changes nothing observable, because `torch.Generator()` has a fixed
default seed (67280421310721) *and* `torch.manual_seed(seed)` still drives layer init — so both
claims under test (same seed → identical basis, different seed → different basis) genuinely still
hold. The explicit generator seed is kept anyway so the dependence survives a future refactor.

Constructed inputs that defeat a plausible-but-wrong implementation:

| claim | input constructed to break it |
|---|---|
| matching is a true assignment | B1 has two near-identical components, B2 has one of them plus an unrelated one. Greedy nearest-neighbour lets both claim the same column and reports **1.0000** (columns used `[0,0,2,3]`); Hungarian is forced onto the unrelated column and reports **0.7761**. |
| stability is sign-invariant | permuted + sign-flipped copy of a basis must score exactly 1.0 |
| dead components are undecidable | all-zero second basis → `None`, not `0.0` |
| exact-zero sparsity | one exact zero + one `1e-12` value → 0.5, not 1.0 (a tolerance would say 1.0) |
| VAE survives its target | `Zc × 1e4` at `lr=1e-1`: unclamped `logvar` overflows and B comes back **non-finite**; clamped it stays finite |
| capped fits are visible | `max_iter=1` on each iterative method → `converged is False` |
| the frozen basis cannot be written | injecting a real `save_program_basis` import into the driver turns the AST guard red |
