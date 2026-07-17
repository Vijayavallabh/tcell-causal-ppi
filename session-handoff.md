# Session Handoff

## Current Objective

- Goal: Build the EG-IPG model for T cell perturbation response prediction
- Current status: **Module 0 + Module 1 encoder (feat-014) + real PLM/PINNACLE embeddings (feat-015) +
  Module 2 typed graph encoder (feat-016) + leakage-safe splits (feat-003) done. Module 3 Program Decoder +
  Module 4 Sparse Predictive-Rationale Head + Module 5 Loss + Training (Stage A loop + Stage B calibration
  loss) built** — the four model modules are now **trainable**. feat-005 (latent program extraction)
  **in-progress** (fold-local basis + frozen sparse_pca production loadings done; method×K comparison + VAE
  remain), feat-008 (EG-IPG model) **in-progress** (M1+M2+M3 decoder/EGIPGModel + Module 4 rationale head /
  faithfulness + **Module 5 Stage A training loop + Stage B calibration loss** built; the **Stage-B
  calibration + rationale FIT loops + the near-null-signal freeze gate remain**, and feat-007 is
  not-started). Done: feat-001/002/003/004/014/015/016. **NEW — Module 6 (Evaluation Metrics + Simple
  Baselines) built earlier:** **feat-009 (metrics) done**, **feat-006 (simple baselines) in-progress**
  (6 of 8 baselines — elastic-net + CatBoost deferred). **NEWEST — Module 7 (Graph Baselines + Screening
  Harness) built this session:** **feat-007 (3 graph baselines) done**, **feat-011 (screening harness)
  in-progress** (harness + registry done; the 32-trial campaign + 5-seed promotion remain). Next: the full
  screening campaign with convergent training, feat-006 remainder (elastic-net + CatBoost), feat-010
  external comparators, or feat-008 Stage-B fit loops. **Module 7 committed (`6b6021f`) + its xhigh
  `/code-review` fully resolved (15 findings, Tiers 1-4: `9db57ae`→`32fb473`→`4e25f4b`→`04e6148`).**
- Branch / commit: main. **Module 5 (Loss + Training) committed this session** — all code + docs +
  state-file syncs in a single commit: the new `training/` package (`losses.py`, `dataset.py`, `trainer.py`,
  `run_train.py`, `__init__.py`), `config.py` (Module 5 constants), `src/tests/test_training.py`,
  `docs/specs/2026-07-16-module5-training.md`, README (train section), a Module 4-spec cross-ref, and
  feature_list/progress/handoff. Prior landmarks: Module 4 (`rationale/` package + `encode_subgraph`
  enabler + docs) `953bd3f`/`b094b5e`, full real-data run + warnings cleanup `2bf1653`/`6dcf196`, Module 3
  (Program Decoder) + feat-005 sparse_pca basis `172a506`/`fc385ef`, feat-016 (Module 2) `100a505`,
  feat-003 `35e3999`. The two planning docs (report + walkthrough) carry as-built notes but are gitignored
  (local-only). **Latest committed is always `git log -1` on main.**

## Completed This Session (Module 7 — Graph Baselines + Screening Harness; feat-007 + feat-011)

- **feat-007 (Graph Baselines) done** — `src/tcell_pipeline/baselines/graph_baselines.py`: three PPI-graph
  references. `NetworkPropagationBaseline` (non-neural symmetric-normalised diffusion; predict =
  proximity-weighted mean of training responses; isolated/absent → zero). `UntypedGraphEncoder` (homogeneous
  GCNConv, all edges one type, no gates). `StaticTypedGraphEncoder` (`TypedGraphEncoder` + condition gate
  pinned to 1.0 — overrides only `_gate`; §10.6 nested member #2). The two neural encoders honour the
  `graph_encoder` forward contract, so they train through the existing Stage-A `Trainer`.
- **feat-011 (Screening Harness) in-progress** — `src/tcell_pipeline/screening/`: `screen_config` (train →
  reload best ckpt → score val → write predictions [output schema] + metrics row; primary = `systema`),
  `run_screening` (H2a/H2b on `systema`, **failure-isolating**), `experiment_registry` (immutable ids, the
  32 EG-IPG / 16-comparator trial caps, all runs logged incl failed), `run_screening.py` driver. Harness +
  registry done; **the 32-trial campaign + 5-seed promotion is the remaining compute work.**
- **config:** SCREENING_ROOT, REGISTRY_PATH, MAX_EGIPG_TRIALS=32, MAX_COMPARATOR_TRIALS=16,
  N_SCREENING_SEEDS=1, N_FINAL_SEEDS=5.
- **Adversarial review** (`docs/reviews/2026-07-16-code-review-module7.md`) — 11 agents, 3 findings
  confirmed+fixed; correctness-critical dims (diffusion math, encoder wiring, eval alignment) clean. Plus a
  pre-review fix: a shared perturbation-encoder that would have co-trained two configs' weights.
- **Real-data smoke** (A100, blocked-target-OOD, bounded 40-row/1-epoch/batch-4) — all 4 wave members
  trained+scored+registered `completed`. **Honest negative:** graph variants don't beat expression-only
  (systema 0.377 expr-only / 0.362 typed-static / 0.348 condition-gated; H2a Δ=−0.015, H2b Δ=−0.015, neither
  supported). **Memory ceiling found:** the typed encoder OOMs 80 GB on real dense subgraphs at batch 32
  (first real training of the graph model); fits at batch 4 / `expandable_segments` — CPU is the report's
  home for graph message passing. `./init.sh` green.
- **xhigh `/code-review` fully resolved (2026-07-16)** — a second, deeper pass (6 finders → 24 candidates →
  21 verifiers) found **15 verified findings** (4 refuted; correctness-critical maths again held), fixed
  across four tiers (`9db57ae`→`32fb473`→`4e25f4b`→`04e6148`): registry distinct-config cap, valid
  summary.json, exit codes, **the network-propagation baseline's scoring path** (`score_network_propagation`
  + `run_screening(extra_scorers=…)`), seed-namespaced ckpts + `gpu_hours`, `MAX_COMPARATOR_FAMILIES=2`,
  `seeded_init(seed)` weight-init reproducibility (the Trainer's seeded gens cover only data shuffling — a
  real Module-5 reproducibility gap now closed), one-pass val scoring, CSR `train_mean`, and one shared
  `response_metric_suite`. `./init.sh` green at **171 tests**; all four tiers re-validated on real data.
  Review record `docs/reviews/2026-07-16-code-review-module7.md`. **All committed on main.**

## Full real-data pipeline run (2026-07-17)

- **Modules 1-6 validated on full real data** (all smokes green): M1 encoder + leakage fence; M2
  condition-varying gates; M3 basis on 21,262 train rows; M4 rationale on real OOD; **M5** Stage-A expr-only
  5-epoch training (21,262 train / 4,400 val, **best_val 3.4690**, checkpoint written); **M6** — trained
  egipg **systema 0.0810** just edges ridge 0.0806, G2-MQ gate PASSED (range 0.911), null-control ≈0.
- **M7 graph screening is compute-bound on full data.** The per-target subgraph sampling + per-row message
  passing is single-threaded CPU (`torch.set_num_threads(1)`), GPU ~0% util: `untyped_gnn` (the *fastest*
  graph config) did not finish ONE epoch over 21,262 rows in ~11 h. **Full-data graph screening is not
  practical as-is.**
- **Workaround used:** ran the 4 nested configs + network_propagation on a **1,000-row fold in parallel,
  one A100 each** (`scratchpad/screen_one.py` + `parallel_screen.sh`) → all 5 done in ~55 min (vs ~2.5 h
  sequential). Same-fold H2a/H2b: systema expr-only 0.0402 / untyped 0.0404 / typed-static 0.0412 /
  condition-gated 0.0350 / network-prop 0.0237. **H2a +0.0010 (nominally supported), H2b −0.0062 (not)** —
  noise-dominated at 1 epoch / 1k rows; the near-null-signal regime. Results in `data/results/screening/`.
- **DEFERRED — the real perf fix:** mini-batch the graph encoders (PyG `Batch` over sampled subgraphs) so
  message passing runs on many at once → true single-GPU saturation + tractable full-data runs. The
  `ponytail:` upgrade in `TypedGraphEncoder`/`UntypedGraphEncoder`; touches Module 2/7, must preserve the
  Module-4 edge_gates contract + keep 171 tests green. **This is the top graph-throughput task.**
- `run_full_pipeline.sh` (repo root) runs Modules 1-7 unattended under nohup (M1-M4 fanned across 4 GPUs,
  M5→M6 in dep order, M7 last); Module 0 excluded (DESTRUCTIVE).

## Completed Earlier (Module 6 — Evaluation Metrics + Simple Baselines; feat-009 + feat-006)

**Round 1 committed as `9f4f9d6`; the round-2 xhigh-review fixes are NOT yet committed** — awaiting the
commit go-ahead. All fully synthetic (no marts). `./init.sh` green at **145 tests** (92 prior + 53).

- **feat-009 (metrics) — done.** New package `src/tcell_pipeline/evaluation/`: `metrics.py` (10 fns / 8
  groups, per-row → macro; primary H1 endpoint `systema_pert_specific_delta`; zero/constant/non-finite →
  0.0), `metrics_ref.py` (independent 2nd impl agreeing on a fixture + degenerate + non-finite rows),
  `metric_qualification.py` (G2-MQ `qualify_metric` + control constructors incl. N1 **derangement**),
  `control_reference.py` (§10.5 independent vs shared control + `null_control_predictor`),
  `output_schema.py` (`predictions/<model>/<split>/<seed>.parquet`, atomic, sigma=0 default).
- **feat-006 (simple baselines) — in-progress.** New package `src/tcell_pipeline/baselines/`:
  `simple_baselines.py` — common `BaseBaseline` (fit(X,z,conditions)→predict→(Δz (M,K), Δx (M,G)); Δx = Δz
  @ B.T; basis=None → empty gene block); Zero / PerturbedMean / ConditionMean / Ridge / NearestNeighbor /
  LowRank. **Deferred:** elastic-net + CatBoost (named in the feature description, out of this goal's scope).
- **config:** METRICS_TOP_K, METRICS_SIGN_TOP_N, PREDICTIONS_ROOT.
- **Tests:** `src/tests/test_metrics.py` (30), `src/tests/test_baselines.py` (23 cases / 14 functions).
- **Review round 1** (committed in `9f4f9d6`): dynamic adversarial workflow → 8/8 fixed (centroid
  degenerate-predictor guard, non-finite agreement, N1 derangement, single-program `(M,1)` shape, 3 tests
  upgraded). **Review round 2** (xhigh workflow-backed `/code-review` of `9f4f9d6` — 12 findings all fixed;
  `docs/reviews/2026-07-16-code-review-module6.md` round-2 section): two-impl divergences round 1 missed
  (non-finite `true` collapse, `1e-12` norm-floor, FP-fragile `std==0` → both gate on `max==min`,
  product-form underflow), the `topk`/`sign` degeneracy guard, the baseline `X=None`/`conditions=None`
  contract, the `**kwargs` control hook, + 3 cleanups. New file `evaluation/_arrays.py`.
- **Round-2 fixes committed** as `fe3a724` (`evaluation/*` incl. new `_arrays.py`, `baselines`, tests, docs,
  state triad).
- **Full real-data run incl. Module 6** (2026-07-16, see progress.md): M1/M2/M3 encoders + M5 Stage-A train
  + M6 model-forward on **A100**, M4 rationale on CPU. New `src/tcell_pipeline/run_module6_smoke.py` scores
  the trained model + 6 baselines on the real val fold (ridge is the strongest baseline, edging the 2-epoch
  model — near-null-signal regime), G2-MQ systema gate PASSED, §10.5 null-control → 0, schema roundtrip.
  Real-data gap fixed: `nan_to_num` the baseline feature matrix (`control_baseline_expr` NaN for ~1.5k rows).
  A fully-trained model (more epochs; graph variant) is the real H1-vs-baseline test.

> **Archived** — the Module 3/4/5 + feat-003/015/016 completed sections are in [`docs/history/session-handoff-archive-2026-07.md`](docs/history/session-handoff-archive-2026-07.md).

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Compile + tests | `./init.sh` | Pass | **92 passed** on torch cu126 (79 prior + 13 Module-5 `test_training.py`); compileall clean; zero warnings |
| Module 5 unit tests | `pytest src/tests/test_training.py` | Pass | 13 passed (synthetic, tiny fixture marts incl. donor profiles): Stage A shapes+gradient flow, graph-gate penalty + **batch-normalization**, Stage B Gaussian NLL+grad, DE probs∈[0,1], learnable λ mixture, dataset keys+q_post fence+dz source, **donor pool+resampler (distinct real donors)**, 2-epoch checkpointed run, param-update, **real donor-invariance signal (train non-zero / val 0 / off 0)**, **empty-split guard** |
| Module 5 real-data Stage A smoke | `python -m tcell_pipeline.training.run_train --expr-only --n-max 256 --epochs 3` | Pass | trains + back-props + atomic checkpoint; **real 4-donor invariance term falls 2.15→0.19 over 3 epochs via the encoder; val invariance 0 (deterministic)**. Full-graph M1→M2→M3 (donor invariance active) + `--no-donor-invariance` (L_graph on real edge_gates) both PASSED |
| Module 5 FULL train fold on A100 | `run_train --expr-only --epochs 3 --device cuda` | Pass | **21,262 train / 4,400 val** on A100; best_val 3.468; train response 3.333→3.324, de 0.173→0.120, donor-invariance 0.113→0.0016; val invariance 0.0; atomic best+last checkpoints. Full-graph capped run on A100 exercised wired `edge_confidences` (source-aware L_graph). **GPU note:** Stage-A training is data-loading/CPU-bound (A100 ~1-3% util); M1/M2/M3 encode smokes are the GPU-accelerated part (24.7k rows/s) |
| Module 4 unit tests | `pytest src/tests/test_rationale.py` | Pass | 10 passed (synthetic): imp∈[0,1], top-k sorted, sufficiency<matched-random, necessity>matched-random, matched-random size+relation match, structural_ood (deleted-fraction vs independent count + component monotonicity), loss components+gradients, expr-only→empty rationale, label predictive_rationale not causal, faithfulness determinism under active DropEdge |
| Module 4 real-data smoke | `python src/tcell_pipeline/rationale/run_module4_smoke.py` | Pass | real PPI graph, A1BG neighbourhood (33,754 edges, |S|=15): sufficiency<matched-random, necessity>matched-random, structural-OOD audit, labelled `predictive_rationale` (not causal) |
| Module 3 unit tests | `pytest src/tests/test_programs.py` | Pass | 12 passed (synthetic): basis shapes ×4 methods, fold-local rows, decoder shapes, λ∈[0,1], σ>0, B-is-buffer, Δx=B·Δzᵀ+r, expr-only variant, full EGIPGModel forward |
| Module 3 real-data smoke | `python src/tcell_pipeline/run_module3_smoke.py` | Pass | fold-local SVD basis on 21,262 real train rows (18s) → B(10282,128); M1→M2→M3 on 4 real perturbations finite, λ∈[0.46,0.55], σ>0; expr-only λ==0 |
| Module 3 basis orchestrator | `python -m tcell_pipeline.programs.run_program_basis --method svd` | Pass | 6.2s → gene_program_loadings.parquet (B 10282×128) + program_response.parquet (A 21262×128), gitignored; challenge-overlap assert held |
| feat-005 production basis (sparse_pca) | `run_program_basis --method sparse_pca` | Pass | 289s on the real train fold → re-froze B(10282,128)/A(21262,128); all finite, fold-locality exact (saved rows==21,262 train), 22.7% zero loadings, recon MAE 0.687 vs 0.817 zero-baseline |
| Full real-data run (all built features) | init.sh + splits + M1/M2/M3 smokes + sparse_pca | Pass | 69 tests; splits byte-identical; M1 33,983 rows finite; M2 25,440-node graph gates differ; M3 M1→M2→M3 finite λ∈[0.38,0.62] σ>0; sparse_pca basis frozen |
| feat-003 split tests | `pytest src/tests/test_splits.py` | Pass | 8 passed; grouping/cap/no-split/determinism/audit-fail-closed/fractions |
| feat-003 real split | `python -m tcell_pipeline.splits` | Pass | 11525 genes → 5141 family groups (largest 5%); wrote data/splits/*; sequence leakage **26.4% (blocked) vs 53.8% (random) = 51% cut** (corrected global-frame residual); split CSVs byte-identical to the 35e3999 freeze |
| Module 2 graph tests | `pytest src/tests/test_graph.py` | Pass | 8 passed; synthetic graph (structure, 2-hop cap, condition gate differs, signed msg, forward finite, edge_gates, zero/absent target, attn sums to 1) |
| Module 2 real-data smoke | `python src/tcell_pipeline/graph/run_module2_smoke.py` | Pass | full 25440-node graph ~18s; CD3E nbhd 512 proteins; Module 1 h_do -> h_graph (4,256) finite on GPU; gates differ by condition; attn sums to 1 |
| Encoder tests (real data) | `pytest src/tests/test_encoders.py` | Pass | 10 passed; real PLM+PINNACLE parquets, real marts — no synthetic parquets |
| PLM generation (GPU) | `python -m tcell_pipeline.embeddings_plm` | Pass | 11419/11419 proteins, 1280-d, finite; A100, 100% util |
| PINNACLE ingestion | `python -m tcell_pipeline.embeddings_pinnacle` | Pass | 1119 embeddings (128-d), 1070/11419 mart coverage (CD4 helper context) |
| Encoder real-data e2e | head of perturbation_condition/de_obs -> PerturbationEncoder | Pass | h_do (8,256) finite; real PLM+PINNACLE vectors flow through |
| Module 1 full-mart smoke | `python src/tcell_pipeline/run_module1_smoke.py` | Pass | on GPU (cuda), 33,983 rows in ~2s; all finite; PLM 33796, PINNACLE 3135 coverage; q_post rejected |
| Module 0 full run (prior) | `python src/tcell_pipeline/run_module0.py` | Pass | all 7 steps on real data; 7.98M edges; leakage fence disjoint |

> **Archived** — the per-module *Files Added* lists are in [`docs/history/session-handoff-archive-2026-07.md`](docs/history/session-handoff-archive-2026-07.md).

## Decisions Made

- Module 1 batch contract: `PerturbationEncoder.forward` takes a dict with keys `uniprot_id` (list),
  `ppi_degree_physical/functional/complex`, `control_baseline_expr`, `culture_condition` (str names or
  long indices), `donor_pc` (a single (B,32) tensor — loader stacks donor_pc_00..31), `n_guides`,
  `single_guide_estimate`. Any q_post key raises. The Module 3 data loader builds this dict.
- **Embeddings are real (feat-015)**: PLM = ESM-2 650M (1280-d, mean-pooled), 100% mart coverage;
  PINNACLE = real published 128-d contextual vectors (`cd4-positive helper t cell`, config.PINNACLE_CONTEXT),
  1070/11419 coverage. Frozen + pluggable; artifacts gitignored, regenerate via the two embeddings_* modules.
- **GPU**: use `torch==2.13.0+cu126` on this host (CUDA-12.2 driver can't run the default cu13x wheel); the
  5x A100s are otherwise invisible to torch. Embedding generation AND the encoder run on GPU — the encoder
  is device-aware (`PerturbationEncoder().to('cuda')`); TargetEncoder/QualityEncoder build CPU tensors that
  forward moves to the fusion's device. Tests default to CPU (portable); the GPU test runs only when CUDA is present.
- UniProt: reviewed-canonical pick; flag only equal-evidence ties; gene is the perturbation unit
- CORUM host has a broken TLS chain -> per-source verify skip for `corum` only
- Data scope: aggregate layer only; donor key = physical CE codes; controls from pseudobulk
- Near-null-signal regime (2026-07-14): confirm above-mean signal before freezing H1; negative result is valid
- Stable-Shift (feat-010): first-party code unconfirmed; plan a row-compatible reimplementation

## Blockers / Risks

- `data/raw` ~101 GB near the 105 GiB soft cap; feat-015 added PINNACLE raw (~1.3 GB) + PLM embeddings
  parquet (~58 MB) + uniprot sequence cache — watch disk before feat-005
- Near-null-signal regime: H1 superiority not guaranteed on this CD4+ screen
- (Resolved this session: HuRI + CORUM downloads; id_mapping UniProt/Entrez online pass)

## Next Session Startup

1. Read `AGENTS.md`.
2. Read `feature_list.json` and `progress.md`.
3. Review this handoff.
4. Run `./init.sh` before editing.

## Recommended Next Step

- **Module 5 (Loss + Training) is committed on main** (`git log -1`); working tree clean, `./init.sh` green
  (**92 tests**, zero warnings). The `training/` package, config, tests, the Module 5 spec, README, and
  state-file syncs all landed in one commit. Stage A trains M1+M2+M3 (with **real donor-invariance**); Stage
  B calibration is a loss module.
- To **advance feat-008 (the last pieces)**:
  1. **Stage A production run** — fit `EGIPGModel` on the full **train** fold with `run_train.py` (fold-local:
     `PerturbationDataset("train")`), select on **val**, then **freeze** the H1 checkpoint. Before the freeze,
     run the **near-null-signal check** on development data (H1 superiority is not guaranteed on this CD4+
     screen — a negative result is valid). The graph path is CPU-bound per subgraph AND donor resampling
     ~triples it, so a full multi-A100 run wants **PyG mini-batching** of the subgraphs + the donor
     node-state cache (the two `ponytail:` notes in `typed_graph_encoder.forward` / `trainer._donor_variants`),
     or `--no-donor-invariance` for a fast pass.
  2. **Stage B fit loops** — on the frozen H1: fit `StageBCalibrationLoss` on the **calibration** partition,
     and fit `RationaleHead` with Module 4's `RationaleLoss` (both loss modules exist; the fit loops don't).
- **Donor invariance is now a real trained signal** (donor resampling over the 4 real `control_donor_profiles`
  donors; penalty on `Var(Δz)` directly — see the Module 5 section). Efficiency upgrade for graph runs: cache
  the donor-independent graph node states so donor variants re-run only readout+decoder (`ponytail:` in
  `trainer._donor_variants`).
- **The 3 items the xhigh review flagged are now implemented (round 3):** (1) `Δz_true` mismatch fixed —
  **`z@B` for every row** (one consistent fold-local target; `program_response` dropped as a training
  dependency, and from run_train's required-gate). (2) `edge_confidences` **wired** — per-edge source
  confidence (edge-feature score column, [0,1]) threaded `TypedGraphEncoder.forward` → `EGIPGModel.forward`
  (`out["edge_confidences"]`) → Trainer → `L_graph`, so its unsourced term down-weights well-sourced edges
  (real data ~21k → ~18k). (3) `Subset` silent-disable fixed — `Trainer._resolve_donor_pool` unwraps
  wrapper `.dataset` chains. The paralogue HGNC collision (GPR89A/GPR89B→`GPHRA`) surfaced earlier is an
  upstream **feat-002** id_mapping item. Still genuinely open: a shared/nuisance decomposition for donor
  invariance (needs a nuisance head), and PyG mini-batching + a donor node-state cache for graph throughput.
- To **finish feat-005**: add the 4-method × 4-K (64/128/256/512) comparison on reconstruction / sparsity /
  stability + a shallow-VAE basis (the extraction machinery is done and the sparse_pca production loadings
  are frozen — only the study remains).
- The **FaithfulnessTester + MatchedRandomSampler** built this session are also the machinery **feat-012**
  (predictive-rationale audit: necessity / sufficiency / minimality / stability vs matched random + structural-OOD)
  will run once the model is trained.
- Alternatively start **feat-006 (simple baselines)** / **feat-007 (graph baselines)** — both consume
  the frozen `data/splits/` and unblock the feat-008 comparison.
- feat-003 calibration knob left open (`docs/specs` + leakage_report.json): the centered-cosine threshold
  (0.85) and 5% cap can be tuned on the published paralog-similarity distribution; the sequence residual
  is 26.4% (vs 53.8% random) — tighten via pairwise must-links or curated families if a downstream result
  needs it. The report also now surfaces `cap_induced_family_splits` (the 5%-cap must break any family
  bigger than one role's budget) — tightening the cap trades partition balance against that residual.
