# Progress archive — per-module build logs, 2026-07-16/17

Moved out of `progress.md` to keep the live log readable. These are completion records for
Modules 6, 7 and 8 and the graph-throughput refactor — history, not current state.
Earlier material: `docs/history/progress-archive-2026-07.md`.

---

## Graph throughput refactor (2026-07-17) — the ceiling was the SAMPLER, not the message passing

The deferred task was written as "mini-batch the message passing with PyG `Batch`". Profiling the real
graph first said that was the wrong target, so both were fixed, sampler first:

| on an A100, per row | before | after |
| --- | --- | --- |
| `sample_subgraph` | 581 ms (**95%**) | 22 ms |
| message passing | 34 ms (5%) | batched |
| **forward+backward, bs=8** | **667 ms** | **61 ms** |
| GPU utilisation (median) | **1%** | **46%** (p90 94%) |
| projected 21,262-row epoch | **3.94 h** | **0.36 h** |

- **Root cause: `sample_subgraph` scanned the whole edge table per row.** `torch.isin(ei[a], frontier)`
  over 6.9M functional_assoc edges × 2 directions × 3 relations × 2 hops, plus `_induce`'s full-graph
  remap gather — `torch.isin` alone was 59% of sampler time, `_induce` 28%, `argsort` 0.03%. ~8M edges
  swept to find a few thousand. Fixed with a CSR neighbour index (`_NeighborIndex`) built **once** per
  graph (0.85 s, ~130 MB): incident-edge lookup is now O(sum of the node set's degree). 581 → 22 ms/row.
- **Then** message passing (now the majority) was mini-batched via `Batch.from_data_list`: one set of
  relational kernels per batch instead of a per-row Python loop. The condition gate is scattered per
  edge (each edge gated by ITS OWN sample's condition) and the readout attends per sample via
  `to_dense_batch` + a key-padding mask, so no attention leaks across targets.
- **Equivalence is the correctness gate, and both are exact**: the sampler is bit-identical to the
  full-scan reference (subgraph shapes on the real graph are unchanged — min 465 / max 512 nodes, mean
  30,751 PP edges), and batched forward matches the per-sample loop edge-for-edge. Both oracles are
  self-contained in the tests (they must NOT import the sampler's policy constants — a shared constant
  moves both sides and the test silently loses its teeth; that was caught during this work). Teeth
  verified by injecting 6 defects (selection priority, traversal direction, cap off-by-one, edge order,
  condition broadcast, attention leak, reversed gate split) — each failed the suite.
- DropEdge is train-only and random, so equivalence is asserted in `eval()`; that is the honest limit.
- **bs=8 is the sweet spot**: bs=32 gives 15.2 vs 16.3 rows/s for 3× the memory (31.5 GB vs 10.0 GB).
- **Saturation is NOT reached and the reason is known**: the batch is still sampled row-by-row on CPU
  (~22 ms/row, ~36% of the step), so the GPU idles between batches. Next ceiling = batch-aware sampling,
  a per-target subgraph cache (11,526 unique targets over 33,983 rows ≈ 2.9 rows/target), or sampling in
  DataLoader workers. Not done — YAGNI until 0.36 h/epoch actually hurts.
- Real-data smoke, all three §10.6 nested members, 600 rows / 1 epoch (machinery only, NOT science):
  untyped_gnn systema=0.0534, typed_static systema=0.0531, condition_gated systema=0.0432.
- **xhigh /code-review (37 agents) → 15 defects, all fixed** (`docs/specs/2026-07-17-graph-throughput-minibatch.md`
  carries the full record). One real regression: `h_graph` preallocated float32 broke dtype under autocast.
  The important one was **test blindness** — `_grow`'s order-restoring sort could be deleted with all 237
  tests green while the sampler diverged on 35 of 60 fixture genes, because the 4 hand-picked probe genes
  were exactly the ones that cannot fail; the module docstring asserted the opposite of the real contract.
  Also: a vacuous all-ones assertion (`allclose` is True on a zero-length gate), CSR cache staleness (an
  appended edge was silently never seen), `torch.split` views, and 3 hot-path wastes worth 9.4× → **10.9×**.
  **Lesson: hand-picked probe cases are a way of choosing what the test cannot see; mutate the one line the
  central claim rests on, not just the 6 you thought of.**
- **All 15 findings are now fixed, including the 2 first waved through.** (a) `incident()`'s duplicate-free
  precondition was docstring-only -> enforced (free next to the gather it guards; a duplicate silently
  corrupts the sampled neighbourhood). (b) **Eval peak memory was a REAL regression, not inherent**: the
  per-row loop bounded eval peak to ONE subgraph regardless of batch, and `collect_predictions` defaults to
  `batch_size=BATCH_SIZE=64` under `no_grad`. Both encoders now message-pass at most
  `config.GRAPH_ENCODE_CHUNK` (default 8) subgraphs at once and stitch — **batch 64 / no_grad: 12.53 GB ->
  2.01 GB (6.2x) and slightly faster (2.9s -> 2.5s)**, results bit-identical at every chunk size. The
  original goal had asked for "a working batch size / chunking path"; it had not been built.
  **Honest limit: chunking does NOT cut TRAINING memory** (autograd retains every chunk until backward —
  batch-64 training still peaks at 55.9 GB); batch size is the training knob, chunking bounds evaluation.

## Module 8 (External Comparators + Rationale Audit + Sealed Eval + Reproducibility) — this session (2026-07-17)

New `src/tcell_pipeline/comparators/` (feat-010), `rationale/rationale_audit.py` (feat-012),
`evaluation/sealed_eval.py` + `reproducibility/` (feat-013). Everything downstream of screening: the external
comparators the H1 model must beat, the rationale stress-test, the write-once sealed challenge eval + H1 rule,
and the clean-checkout reproducibility verifier + 11/11 fallacy scan. Like Module 7 this is verification
machinery, not a compute campaign. **Not yet committed.**

- **Comparators** (`comparators/`) — `StableShiftAdapter` (REIMPLEMENTED: fold-local low-rank SVD from train
  only + one STRING graph-conv onto held-out targets), `TxPertPublicAdapter` (PUBLIC-ONLY STRING
  score-attention aggregator; records whether upstream TxPert is importable), `source_adjacency` (shared
  STRING-channel builder), `compatibility_report.py` (license / exposure / checkpoint / explicit PUBLIC_ONLY).
  Two distinct comparator families under the ≤2-family / 16-trial caps.
- **Rationale audit** (`rationale/rationale_audit.py`) — `audit_rationale(model, head, dataset, n_cases=50)`
  on the frozen H1 + head (no training); stratified (degree×effect×condition×coverage, uncovered filtered
  before selection); reuses `FaithfulnessTester`/`MatchedRandomSampler`/`RationaleHead` + adds minimality
  curve, source ablation (BioPlex/HuRI/STRING/CORUM incl the 100%-CORUM membership edges), GInX-by-sparsity,
  stability → `audit_report.json`.
- **Sealed eval** (`evaluation/sealed_eval.py`) — `SealedEvaluator.evaluate` forwards the frozen model over
  the sequestered challenge fold and applies **LCB₉₅(ρ_EGIPG−ρ_best) > 0.05 AND ρ_EGIPG > ρ_perturbed_mean**
  (10 000 paired-row bootstrap on per-row systema); write-once (refuses overwrite; `min_rows` guard).
- **Reproducibility** (`reproducibility/`) — `verify_reproducibility(checkout, manifest)` → REPRODUCIBLE /
  PARTIALLY / NOT / CANNOT_VERIFY (deterministic hashes + schema + config + checkpoint + decision + the
  11-detector fallacy scan); an errored detector can't silently certify a clean 11/11.
- **config:** COMPARATORS_ROOT, RATIONALE_AUDIT_ROOT, SEALED_ROOT, REPRODUCIBILITY_ROOT, DELTA_PRED=0.05,
  N_BOOTSTRAP=10000, N_RATIONALE_AUDIT_CASES=50.
- **Adversarial review** (`docs/reviews/2026-07-17-code-review-module8.md`) — 5 finder dimensions → 13
  candidates → skeptical verify → **6 confirmed + 3 plausible, all fixed** with regression tests: [high] a
  crashed fallacy detector was counted as clean 11/11 (→ false REPRODUCIBLE); [med] CORUM ablation ignored
  membership edges; [med] uncovered targets burned audit slots; [med] ecological flagged spuriously with ≤2
  groups; [low] no min-row guard before sealing; [low] `public_only` substring matched "non-public". The
  sealed-math + comparator-leakage finders confirmed those paths correct.
- **xhigh `/code-review` pass 2 (2026-07-17, 63 agents, over the committed `5ea8a4b`)** — **15 confirmed,
  all fixed.** Theme: *each subsystem failed toward its own headline claim.* The verifier certified
  REPRODUCIBLE on an **empty checkout** (absolute manifest paths hashed the original run — proved by the
  reviewers), on a manifest with **no hashes block**, on a decision record pinning **nothing**
  (`bool(None)==bool(None)`), and with the config check **skipped by default** (so a changed `DELTA_PRED` —
  the knob that flips H1 — certified clean); `tolerance` defaulted to 0.0 demanding bit-exact floats. The
  sealed **write-once seal was keyed on the bootstrap `seed`**, so bumping it re-opened the sequestered fold
  and resealed a possibly-confirming decision — **the garden-of-forks this module ships a detector for**.
  Four fallacy detectors fired on clean data or passed on undefined input (`regression_to_mean(b, b−10)`
  flagged despite correlation 1.0; `reverse_causation(0,0)` flagged; NaN crashed `_sign`; zero-survivor
  `survivorship` silently passed). The audit crashed on any non-CPU device and its `stability` wasn't
  reproducible from the seed. The TxPert report asserted `wrapped_upstream` from mere importability.
  **Bonus:** the CUDA fix surfaced a **latent Module-4 bug** — `RationaleHead._select` indexes CPU tensors
  with a CUDA `topk` index; the head had never run on GPU. All fixed; new `Unevaluable` makes "a check that
  didn't run never certifies" explicit. The H1 second clause is a proven tautology (ρ_perturbed_mean ≡ 0
  under systema) — **documented in the sealed JSON, not silently patched**.
- **Pass 3 — adversarial verification OF the pass-2 fixes (17 agents)** — re-attacking each fix found **2
  STILL EXPLOITABLE + 9 PARTIAL**. Verdict on pass 2, recorded plainly: *the fixes were point patches that
  satisfied their own regression tests.* (a) The **seal** had only moved from `seed` to `split` — an equally
  caller-supplied label never bound to the fold, so `"Challenge"` / `"challenge_rerun"` / `"calib/../challenge"`
  / a swapped root / a TOCTOU race all re-sealed the same fold. Now keyed on a **`fold_fingerprint`** with a
  sanitized label and an **atomic `O_EXCL` claim**. (b) The **decision check** added presence guards but never
  touched the **`bool()` coercion that was the defect** (`bool("false") is True`). Now strict bools. (c) The
  **root cause pass 2 missed**: `_corr` returned a `0.0` **sentinel** for three degeneracies, indistinguishable
  from a real zero — it now **raises `Unevaluable`**, closing the false-flag path in berkson/collider/
  ecological/simpson at once. Plus: scale-invariant `regression_to_mean`; the reverse-causation floor moved to
  the **stronger** direction (the old one silently unflagged the archetypal trap); input validation across
  **all 11** detectors; a **whitelist** verdict; a **tolerance cap**; malformed manifests → verdict not
  traceback; lexical containment (symlinked stores work); the CUDA RNG leak; a **conditional** H1 note (it was
  an unconditional literal that contradicted the value beside it).
- **Two bugs in my own work caught by red-green:** `_safe_split_label` rejected *every* label
  (`(os.altsep or "") in s` is `"" in s` on POSIX), and a test passed with the bool coercion reverted.
- **Verification:** `./init.sh` green at **224 tests** (171 prior + 53 Module 8), exit 0, incl. a real CUDA
  audit run. Every original reviewer attack replayed against the fixed code and confirmed closed; all fixes
  red-green verified. Spec `docs/specs/2026-07-17-module8-comparators-audit-sealed-repro.md`; review record
  `docs/reviews/2026-07-17-code-review-module8.md` (all three passes).
- **Standing lesson for this module:** a fix that only satisfies its own regression test is not a fix — ask
  what *class* the defect belongs to and where else that class lives.

## Module 7 (Graph Baselines + Screening Harness) — this session (2026-07-16)

New `src/tcell_pipeline/baselines/graph_baselines.py` (feat-007) + `src/tcell_pipeline/screening/` package
(feat-011). Gives the H1 predictor its graph references and the machinery to screen the §10.6 nested family
under the report's frozen trial budget. **Not yet committed.**

- **Graph baselines** (`baselines/graph_baselines.py`) — `NetworkPropagationBaseline` (non-neural symmetric-
  normalised PPI diffusion; predict = proximity-weighted mean of training responses; isolated/absent → zero),
  `UntypedGraphEncoder` (homogeneous GCNConv, all edges one type, no gates → `(h_graph,None,None)`),
  `StaticTypedGraphEncoder` (`TypedGraphEncoder` with the condition gate pinned to 1.0 — overrides only
  `_gate`, §10.6 member #2). The two neural encoders drop into `EGIPGModel(graph_encoder=…)` and train
  through the existing Stage-A `Trainer`.
- **Screening** (`screening/screening.py`) — `screen_config` (train → reload best ckpt → score val in
  dataset order → write predictions [output schema] + metrics row → return the suite; primary =
  `systema_pert_specific_delta`), `run_screening` (H2a typed-static>expr-only, H2b condition-gated>typed-
  static; **failure-isolating**), nested-family factories (fresh model per call — the weight-sharing fix).
- **Experiment registry** (`screening/experiment_registry.py`) — immutable `run-NNNN` ids; enforces the
  **32 EG-IPG / 16-per-comparator** trial caps; logs every run incl **failed**; null/empty-manifest tolerant.
- **config:** SCREENING_ROOT, REGISTRY_PATH, MAX_EGIPG_TRIALS=32, MAX_COMPARATOR_TRIALS=16,
  N_SCREENING_SEEDS=1, N_FINAL_SEEDS=5.
- **Review** (`docs/reviews/2026-07-16-code-review-module7.md`) — adversarial workflow, 11 agents, 3 findings
  confirmed+fixed (tautological H2a test, `load_registry` null-`runs` crash, driver `ID_MAPPING_PATH` guard);
  a pre-review fix caught a shared perturbation-encoder that would co-train configs' weights.
- **Real-data smoke** (A100, blocked-target-OOD, 40-row/1-epoch/batch-4) — all 4 wave members
  trained+scored+registered `completed`. **Honest negative:** graph variants don't beat expression-only
  (systema 0.377 / 0.362 / 0.348; H2a Δ=−0.015, H2b Δ=−0.015, neither supported) — the near-null-signal
  regime on untrained-to-convergence models. **Memory ceiling:** the typed encoder OOMs 80 GB on real dense
  subgraphs at batch 32 (first real training of the graph model — Module 5's real run was expr-only); fits at
  batch 4 / `expandable_segments`; failure isolation keeps one OOM lane from aborting the wave.
- **Verification:** `./init.sh` green at **171 tests** (145 prior + 26 Module 7, after the xhigh review's
  15 fixes). Spec `docs/specs/2026-07-16-module7-screening.md`; review record
  `docs/reviews/2026-07-16-code-review-module7.md` (both review passes + the 4-tier resolution).
- **xhigh /code-review resolved (2026-07-16):** 15 findings across 4 tiers — Tier 1 (registry distinct-config
  cap, valid summary.json, exit codes, **network-prop scoring path**), Tier 2 (seed-namespaced ckpts,
  gpu_hours, comparator-family cap), Tier 3 (best-vs-last + completed-log tests, `seeded_init` weight-init
  reproducibility), Tier 4 (one-pass val scoring, CSR train_mean, shared `response_metric_suite`, dup-name
  guard). All four tiers re-validated on the real blocked-target-OOD fold.
- **Full real-data pipeline run (2026-07-17):** M1-M6 all green on full data (M5 best_val 3.4690; M6 egipg
  systema 0.0810 edges ridge 0.0806, G2-MQ PASSED). **M7 graph screening is compute-bound on full data** —
  single-threaded per-subgraph sampling, GPU ~0%; untyped_gnn didn't finish 1 epoch over 21,262 rows in
  ~11h. Workaround: 4 configs + network-prop on a **1,000-row fold, one A100 each in parallel** (~55 min) →
  H2a +0.0010 (nominally supported), H2b −0.0062 (not) — noise at 1 epoch. **RESOLVED 2026-07-17** by the
  graph throughput refactor above (3.94 h → 0.36 h/epoch): the diagnosis "single-threaded per-subgraph
  sampling" was right and was the 95%; "mini-batch the encoders" alone would have addressed only the 5%.
  The full-fold screening campaign is now tractable and is the next compute task.
  `run_full_pipeline.sh` runs M1-7 unattended under nohup. See session-handoff for detail.

## Module 6 (Evaluation Metrics + Simple Baselines) — this session (2026-07-16)

New packages `src/tcell_pipeline/evaluation/` (feat-009) + `src/tcell_pipeline/baselines/` (feat-006).
Makes model output **scorable** and gives every headline table its mandatory simple references. Fully
synthetic tests — no marts required. **Round 1 committed as `9f4f9d6`; the round-2 xhigh-review fixes are
not yet committed.**

- **Metrics** (`evaluation/metrics.py`) — 10 fns / 8 groups, per-row then macro-averaged (a row = one
  perturbation-target×condition response, so per-row *is* per-perturbation): mae, rmse, pearson_corr,
  spearman_corr, **systema_pert_specific_delta** (primary H1 endpoint — `corr(pred−train_mean,
  true−train_mean)`), centroid_accuracy, topk_recall, sign_accuracy, program_cosine, signed_de_metrics
  (macro-F1 + per-class P/R + AUPRC; AUROC omitted). Zero/constant/**non-finite** rows → 0.0 (a zero
  predictor scores worst).
- **Second independent impl** (`evaluation/metrics_ref.py`) — loops + scipy/sklearn re-implementation of
  mae/rmse/pearson/spearman/systema/centroid/program_cosine; agrees with `metrics.py` on a fixed fixture
  **and** on zero/constant/non-finite rows.
- **G2-MQ gate** (`evaluation/metric_qualification.py`) — `qualify_metric` (all-negatives < all-positives)
  + control constructors zero / perturbed-mean / label-perm-N1 (a **derangement**) / row-shuffle-N2 +
  oracle / guide-split-half.
- **Control-reference safeguards** (`evaluation/control_reference.py`, §10.5) — independent vs shared
  control estimators + `null_control_predictor` (~0 under the corrected estimator).
- **Common output schema** (`evaluation/output_schema.py`) — `predictions/<model>/<split>/<seed>.parquet`
  (row_index + delta_z_0..K-1 + delta_x_0..G-1 + sigma_0..K-1; atomic; baselines write sigma=0).
- **Six baselines** (`baselines/simple_baselines.py`) — common `BaseBaseline`
  (fit(X,z,conditions)→predict→(Δz (M,K), Δx (M,G)); Δx = Δz @ B.T via the frozen basis, basis=None → empty
  gene block): Zero / PerturbedMean / ConditionMean / Ridge / NearestNeighbor / LowRank. **Deferred within
  feat-006:** elastic-net + CatBoost.
- **config:** METRICS_TOP_K=20, METRICS_SIGN_TOP_N=50, PREDICTIONS_ROOT.
- **Verified:** `./init.sh` **145 passed** (92 prior + 53: 30 in `test_metrics.py`, 23 in
  `test_baselines.py`), zero warnings; both metric impls agree on non-degenerate + zero/constant +
  non-finite (`pred` AND `true`) + high-dim-constant + tiny-norm + extreme-scale rows under `-W error`.
- **Review round 1** (dynamic workflow, 6 dimensions × per-finding verify — 8/8 confirmed then fixed):
  centroid degenerate-predictor guard; non-finite agreement; N1 derangement; single-program `(M,1)` shape;
  3 too-weak tests upgraded.
- **Review round 2** (xhigh workflow-backed `/code-review` of `9f4f9d6` — 12 findings, all fixed;
  `docs/reviews/2026-07-16-code-review-module6.md`): non-finite-`true` centroid collapse; the `1e-12`
  norm-floor inflating tiny-norm wrong-direction preds; the FP-fragile `std==0` constant guard (both impls
  now gate on `max==min`); product-form underflow (separate roots); `topk`/`sign` degeneracy guard; the
  baseline `X=None`/`conditions=None` contract; the `**kwargs` control hook; + 3 cleanups (vectorised
  `topk`/`sign`, `rng.permuted` shuffle, shared `_arrays.to_numpy`). +14 regression tests.
- Design+as-built: `docs/specs/2026-07-16-module6-evaluation.md`.

## Full real-data run incl. Module 6 (2026-07-16)

Re-ran every non-destructive real-data entrypoint end-to-end, GPU where it helps (M0 excluded — destructive).
- **M1** encoder: 33,983 rows on **A100** in 2.15 s (15.8k rows/s); all `h_do` finite; q_post fence held.
- **M2** typed graph on **A100**: `h_graph` finite; condition-varying gates; attention sums to 1.
- **M3** M1→M2→M3 on **A100**: finite; λ∈[0.46,0.55]; σ>0; expr-only λ=0.
- **M4** rationale (CPU, thread-pinned): sufficiency < matched-random; necessity > matched-random; `predictive_rationale`.
- **M5** Stage-A on **A100**: full 21,262-row train fold, expr-only, 2 epochs, best_val 3.4711 → checkpoint.
- **M6** (NEW `run_module6_smoke.py`): scored the trained model + all 6 baselines on the real 4,400-row val fold
  (model forward on **A100**). **Ridge is the strongest baseline** (systema 0.081) and edges the 2-epoch model
  (0.078) — the baseline-vs-model comparison Module 6 exists for, matching the near-null-signal caveat.
  **G2-MQ systema gate PASSED** (negs ≤0.026 < guide-split-half 0.938 < oracle 1.0, range 0.911); §10.5
  null-control predictor → 0.0 under independent controls; output-schema roundtrip on 4,400 preds.
- **Real-data gap caught + fixed:** `control_baseline_expr` is NaN for ~1,558 rows (the encoder imputes; the
  sklearn baselines don't) → `nan_to_num` on the baseline feature matrix in the smoke.
- **GPU relevance (honest):** M1/M2/M3 encoders are genuinely GPU-accelerated; Stage-A training is
  data-loading-bound; M4/M6 baselines+metrics are numpy/sklearn (CPU) — GPU doesn't apply there.

> **Archived** — the Module 3/4/5 sections and their real-data runs are in [`docs/history/progress-archive-2026-07.md`](docs/history/progress-archive-2026-07.md).

