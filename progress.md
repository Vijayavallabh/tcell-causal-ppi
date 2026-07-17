# Session Progress Log

## Current State

**Last Updated:** 2026-07-17 (**graph throughput refactor: 10.9× end-to-end, `./init.sh` green at 242**, xhigh `/code-review` 15 defects fixed. Prior: Module 8 `5ea8a4b` → xhigh `/code-review` 15 defects fixed `2edb44f` → pass-3 adversarial verification OF those fixes found 2 still exploitable + 9 partial, root causes fixed `6a68882` → real-data drivers `97f8451`.)
**Active Feature:** the graph mini-batch refactor is **done** — the compute ceiling that blocked feat-010/011/012/013 is lifted (3.94 h → 0.36 h per 21,262-row epoch; GPU util median 1% → 46%). The four campaigns are now **unblocked but not yet run**: feat-011 (32-trial screening + 5-seed promotion), feat-010 (16-trial comparators), feat-012 (50-case audit on the frozen H1), feat-013 (sealed opening + clean-checkout reproduction). Also open: feat-006 (elastic-net + CatBoost), feat-008 (Stage-B calibration + rationale fit loops + freeze gate), feat-005. Next: the screening campaign → a converged/promoted model → the deferred campaigns.

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

## Status

### What's Done

- [x] feat-001 Environment & Data Download — **DONE**
  - Env imports OK (anndata 0.13.1, mudata 0.3.10, h5py 3.16.0)
  - Aggregate layer downloaded to `data/raw/` (~101 GB): 4 HDF5 + 15 suppl tables + 12 jsonld
  - Cell-level files intentionally excluded (storage-blocked)
- [x] feat-002 Data Inspection & ID Mapping — **DONE**
  - `examples/` inspectors; `src/tcell_pipeline/id_mapping.py`
  - Ran online on real DE: 12311 unique Ensembl (11526 targets / 10282 measured / 9497 both), all HGNC
    resolved, UniProt/Entrez filled via mygene.info
  - One-to-many UniProt DISAMBIGUATED (reviewed-canonical strategy, see Decisions): 33 multi-accession
    genes -> 23 resolved to a confident canonical, 10 genuine multi-product loci flagged
    `uniprot_ambiguous` with all candidates kept in `uniprot_alternatives`
- [x] feat-004 PPI Graph Construction — **DONE**
  - `ppi_graph.py` typed-edge harmonizer + `complex_membership.py` CORUM bipartite membership
  - All 5 sources fetched + merged on real data -> `data/graphs/protein_edges.parquet`:
    **7,980,907 edges** (bioplex 118162, huri 52256, biogrid 1218142, string 13715404, corum 77696)
  - `complex_membership.parquet`: 18,932 memberships / 5,628 complexes (CORUM 5.3)
  - `ppi_degree_*` computed from the graph into perturbation_condition

- [x] **Module 0 data pipeline** (`src/tcell_pipeline/`, 9 modules + `run_module0.py`) ran end-to-end on
  real data. Derived marts written under `data/intermediate/`, `data/graphs/`, `data/manifests/`
  (all gitignored): id_mapping, DE layers (zscore/log_fc sparse NPZ; neglog10_p_value/neglog10_adj_p_value/
  baseMean/lfcSE dense NPY), de_obs/de_var, protein_edges, complex_membership, perturbation_condition
  (33983 rows; 187 without UniProt; 32425/33983 with a control baseline), control baseline + donor
  profiles, feature_availability.yaml (**q_pre=43 / q_post=13 / metadata=2, leakage fence disjoint**).
- [x] **This session's fixes** (commits e453964..eab027e):
  - UniProt disambiguation via reviewed-canonical strategy (`choose_uniprot`); `uniprot_ambiguous` flags
    only equal-evidence ties (10 loci: CDKN2A p16/p14ARF, GNAS, MOCS2, TMPO...)
  - HuRI download: apex host `interactome-atlas.org` (cert invalid for the `www.` subdomain)
  - CORUM download: migrated to CORUM 5.3 fastapi endpoint (old `coreComplexes.txt.zip` path gone);
    handles new `subunits_gene_name` schema; per-source TLS-verify skip for the broken helmholtz cert chain
  - feature_availability: `KNOWN_METADATA_COLS` allowlist so the leakage-fence REVIEW warning fires only
    on genuinely-unexpected metadata (row_index/mapping_status no longer cry wolf)
  - `mygene` added to requirements.txt
- [x] 23 pytest tests in `src/tests/` (synthetic fixtures; added test_control_profiles.py,
  test_complex_membership.py); `init.sh` green (compileall + pytest)

- [x] **Module 1 Perturbation & Context Encoder** (feat-014, `src/tcell_pipeline/encoders/`) — **DONE**
  - `PluggableEmbeddingStore` (frozen PLM 1280 / PINNACLE 512 by UniProt; zero-fallback when the
    parquet is absent, in-memory cache, NOT an nn.Module), `TargetEncoder` (no trainable gene-ID
    embedding; h_target R^1796), `ContextEncoder` (trainable condition Embedding(3,64) + donor PCs
    through Linear(32,32), no free donor-ID embedding), `QualityEncoder` (n_guides +
    single_guide_estimate + zeros(64) guide-seq placeholder; h_quality R^66), `PerturbationEncoder`
    (fusion Linear(1958->256)+LayerNorm -> h_do R^256; rejects q_post cols at the boundary).
  - 503,264 trainable params, CPU-only, batch-first. Config: PLM_EMBED_DIM/PINNACLE_EMBED_DIM/
    GUIDE_SEQ_EMBED_DIM/H_DO_DIM/CONDITIONS/PLM_EMBEDDINGS_PATH/PINNACLE_EMBEDDINGS_PATH.
  - NaN guard (`as_float_vector` nan_to_num): missing control_baseline_expr (1558/33983) and
    n_guides can't poison the LayerNorm'd h_do. Upgrade path = fold-fit imputation in Module 3 loader.
  - 10 tests (test_encoders.py) + real-data smoke on perturbation_condition/de_obs -> finite (4,256).

- [x] **Module 1 real embedding ingestion** (feat-015, `embeddings_plm.py` + `embeddings_pinnacle.py`) — **DONE**
  - `embeddings_plm.py`: real **ESM-2 650M** (1280-d, mean-pooled), UniProt-REST sequences, resumable,
    **device-aware (GPU)**. Ran on an A100 -> **11419/11419 mart proteins embedded** (100% PLM coverage), finite.
  - `embeddings_pinnacle.py`: real **PINNACLE** (Figshare 22708126) `cd4-positive helper t cell` context.
    Real dim is **128** (config placeholder was 512 -> **corrected to 128**). Gene-symbol->UniProt via id_mapping;
    **1119 embeddings, 1070/11419 mart proteins covered** (contextual — rest keep zero fallback).
  - Live encoder dims now: target.out_dim **1412** (1280+128+4), fusion Linear(1574->256), **404,960** params.
  - Tests rewritten to **real data/embeddings — no synthetic parquets** (still 10 in test_encoders.py).
  - **GPU enabled**: swapped torch cu130->**cu126** (host driver is CUDA 12.2; cu13x can't see the 5x A100s).
    requirements.txt: +fair-esm, +cu126 install note.

- [x] **Module 2 Typed Graph Encoder** (feat-016, `src/tcell_pipeline/graph/`) — **DONE**
  - `graph_builder.build_hetero_graph` -> PyG HeteroData + gene_to_idx: 25440 protein nodes
    (frozen 1412-d TargetEncoder descriptor, graph-derived degrees, zero-fallback) + 5628 complex
    nodes (index-only, learned embedding in the encoder); 4 relations (physical_ppi 1123205 /
    co_complex 48389 / functional_assoc 6857702 / complex_membership 18932) with 8-d edge features.
  - `neighborhood_sampler.sample_subgraph`: physical/co-complex-first then score-fill, cap 512
    proteins + member complexes, induced HeteroData preserving orig_idx.
  - `typed_graph_encoder.TypedGraphEncoder`: 3-layer per-relation MessagePassing with signed
    message `tanh(W_sign h_u)*relu(W_mag h_u)`, condition gate `sigmoid(w_gate[h_cond||f_e])`
    computed once/relation and returned as `edge_gates` for Module 4, residual FFN+LayerNorm,
    DropEdge 0.1. `graph_readout.GraphReadout`: 4-head cross-attention (q=h_do) -> h_graph R^256.
  - CPU **and** CUDA (device-aware). 8 synthetic tests (`test_graph.py`) + real-data smoke
    (`graph/run_module2_smoke.py`): full graph in ~18s, CD3E neighbourhood, Module 1 h_do ->
    Module 2 h_graph (4,256) finite on GPU, gates differ by condition, attention sums to 1.

- [x] **feat-003 Leakage-Safe Splits** (`src/tcell_pipeline/splits.py`) — **DONE**
  - Design in `docs/specs/2026-07-15-feat-003-leakage-safe-splits.md` (from the experiment-plan
    report). **Empirical measurement forced the algorithm**: naive connected-components collapses on
    every axis (physical 1-hop → 95% giant component, complex → 23%, raw ESM cos≥0.95 → 92%,
    Louvain → 42%). Hard block = sequence/paralog family via **representative (non-chaining)
    clustering on centered ESM-2 embeddings** (cos≥0.85 → 3.1% largest family) + CORUM co-membership
    under a 5% size cap (capped union-find). Physical-PPI neighbourhood is **audit-only** (95%
    hairball; per report G1 + Phase-1 step 6/9 "publish the similarity distribution").
  - **4-role** partition (train/val/calibration/challenge, ~60/15/10/15; realized 62.5/13/7.9/16.6)
    + random diagnostic split. Frozen + hashed to `data/splits/` (git-tracked): blocked/random CSVs,
    manifest.json, leakage_report.json. **Effectiveness validated** (corrected post-review): challenge
    genes with a ≥0.85 train paralog cut 53.8% (random) → 26.4% (blocked) = 51% reduction.
  - 8 synthetic tests (`test_splits.py`); `./init.sh` green (54 pytest).
- [x] **Post code-review fixes** (feat-016 + feat-003; committed 7760624) — **DONE**
  - Applied the verified `/code-review` findings (`docs/reviews/2026-07-15-code-review-feat-016-feat-003.md`).
  - **feat-003 leakage-safety (split CSVs byte-identical, sha256 unchanged):** audit now publishes
    cap-induced family splits via an uncapped pre-cap component pass (post-cap "no split" assertion was
    blind to families the 5% cap must break: `cap_induced_family_splits=1`); sequence residual centered
    in one global frame (was mismatched per-subset means, understating leakage → 53.8/26.4/51% vs old
    53.5/28.1/47%); `run()` fails closed when PLM embeddings absent (was silent fail-open).
  - **feat-016 bugs:** graph degree columns reordered to match Module 1 `[physical, functional, complex]`;
    `encode_one` moves `h_do` to device; flaky signed-message test seeded + false `<1.0` bound dropped.
  - **Tier 3 (all addressed):** cheap defenses (edge-feature `nan_to_num`, gene-symbol `dropna`,
    unknown-source fail-fast); dead config constants removed; OOV culture_condition raises a legible
    `ValueError` (`_condition_index`); diagnostic random split uses cumulative-boundary allocation (no
    truncated tail — `random.csv` regenerated, blocked split + effectiveness numbers unchanged);
    `edge_gates[rel]` now length E (one per original edge) for all relations, not 2E-doubled for PP
    (full gate→edge identity API still deferred to Module 4).
  - Regenerated `data/splits/`; **57 pytest** green (+3 regression checks).

### What's In Progress

- **feat-005 Latent Program Extraction** — fold-local basis machinery + frozen sparse_pca production
  loadings done; the 4-method × 4-K comparison (reconstruction / sparsity / stability) + shallow-VAE basis remain.
- **feat-008 EG-IPG Model** — M1+M2+M3 decoder/EGIPGModel + Module 4 rationale head / loss / faithfulness
  eval built; the training-loss OPTIMIZATION loop + train/calibration loops remain (and feat-007 is not-started).

### What's Next

1. feat-008 training loop: wire RationaleLoss + the decoder losses into a Stage-A (predictor) then
   Stage-B (rationale) fit; then feat-011 screening consumes it.
2. feat-006 Simple Baselines / feat-007 Graph Baselines — consume the frozen splits, unblock the feat-008 comparison.
3. feat-005 method×K comparison + shallow VAE (extraction machinery done; only the study remains).
4. Optional: near-null-signal check on development data before freezing H1 (2026-07-14 finding).

## Blockers / Risks

- [ ] `data/raw` ~101 GB near the 105 GiB soft cap; derived marts now also on disk (protein_edges ~35 MB,
  DE layers, control profiles) — watch disk before feat-005 program bases land
- [ ] **Near-null-signal regime (2026-07-14 finding):** this CD4+ screen may be near-null-signal (models
  barely beat the mean). Confirm a detectable above-mean signal before freezing H1; a rigorous negative
  benchmark is a valid outcome.
- [x] RESOLVED: HuRI + CORUM downloads (both source URLs migrated this session)
- [x] RESOLVED: id_mapping UniProt/Entrez (online mygene pass done; only 6 no-hit genes remain, HGNC-resolved)

## Decisions Made

- **UniProt disambiguation**: pick the gene's reviewed human canonical (UniProt REST gene_exact+reviewed)
  by annotation-score then lexical; flag only equal-evidence ties; keep the gene as the perturbation unit
  (CRISPRi knocks down the whole locus) — no forced single-protein pick, alternatives preserved
- **CORUM host**: broken TLS chain (certifi also fails) -> per-source verify skip for `corum` only
- Data scope: aggregate layer only; cell-level (~1.6 TiB) excluded
- Donor key = physical CE codes; independent NTC controls come from pseudobulk (DE has none)
- Distributional metrics: do not use Wasserstein/Energy distance as a sole headline metric
- Stable-Shift (feat-010): first-party code unconfirmed; plan a row-compatible reimplementation
- **Embeddings (feat-015)**: PLM = real ESM-2 650M (1280-d, mean-pooled); PINNACLE = real published
  128-d contextual vectors (NOT the 512 placeholder), `cd4-positive helper t cell` context to match the
  CD4+ screen (configurable via config.PINNACLE_CONTEXT). Frozen features; artifacts gitignored + regenerable.
- **GPU**: host has 5x A100 80GB but the CUDA-12.2 driver can't run the default cu13x torch; use the
  cu126 build (`torch==2.13.0+cu126`, minor-version compat). Embedding generation runs on GPU.
- **Module 2 (feat-016)**: protein node features reuse Module 1's frozen 1412-d TargetEncoder
  descriptor (degrees recomputed from the graph so all 25440 nodes have them); complex nodes are
  index-only (learned embedding lives in the encoder). Custom PyG `MessagePassing` per relation —
  RGCNConv/GATConv can't express the signed `tanh*relu` message or the condition gate. Condition
  gate `alpha` depends only on `h_cond` + edge features (not `h_u`), so it's computed once and
  reused across layers and returned as `edge_gates`. ponytail: per-target subgraph loop in forward
  (upgrade to PyG mini-batching if Module 3 graph-encode throughput demands it).

> **Archived** — the per-feature *Files Added* lists (feat-003/014/015/016) are in [`docs/history/progress-archive-2026-07.md`](docs/history/progress-archive-2026-07.md).

## Notes for Next Session

- `examples/` scripts double as data-understanding docs
- Read the experiment plan report for detailed feature specs (2026-07-14 literature refresh)
- Before feat-011 screening / freezing H1, run the near-null-signal check
- Module 0 marts are on disk but gitignored; rerun `python src/tcell_pipeline/run_module0.py` to regenerate
