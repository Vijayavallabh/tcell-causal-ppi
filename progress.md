# Session Progress Log

## Current State

**Last Updated:** 2026-07-16 (Module 6 committed as `9f4f9d6`; xhigh workflow-backed /code-review round 2 → 12 findings all fixed [**round-2 fixes NOT yet committed**])
**Active Feature:** Module 6 (Evaluation Metrics + Simple Baselines) — **feat-009 done**, **feat-006 in-progress** (6 of 8 simple baselines; elastic-net + CatBoost deferred). Also still open: feat-008 **in-progress** (Stage-B calibration + rationale fit loops + near-null-signal freeze gate remain), feat-005 in-progress, feat-007 not-started. Next: feat-006 remainder (elastic-net + CatBoost), feat-007 (graph baselines — now unblocked by the shared baseline protocol + output schema), or feat-008 Stage-B fit loops.

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

## Full real-data run on GPU incl. Module 5 (2026-07-16)

Ran every non-destructive real-data entrypoint end-to-end (M0 excluded — destructive), GPU where it helps.
- `./init.sh` — **92 passed**, zero warnings.
- **Module 1** (encoder): 33,983 rows on **A100** in **1.38 s** (24.7k rows/s); all `h_do` finite; q_post fence held.
- **Module 2** (typed graph): 25,440-node graph on **A100**; `h_graph` finite; per-condition gates differ; attn sums to 1.
- **Module 3** (M1→M2→M3): on **A100**; finite; λ∈[0.31,0.67]; σ>0; expr-only λ=0 (fold-local SVD basis on 21,262 rows in 17.8 s).
- **Module 4** (rationale/faithfulness): real PPI graph (A1BG, 33,754 edges, |S|=15); sufficiency<matched-random; necessity>matched-random; structural-OOD audit; labelled `predictive_rationale` (CPU, thread-pinned).
- **feat-003 splits**: re-run **byte-identical** (sha256 unchanged, git clean) → deterministic; 51% leakage reduction.
- **Module 5 Stage A** (NEW; `run_train.py` gained `--device cpu|cuda`):
  - **expr-only, FULL train fold** (21,262 train / 4,400 val), 3 epochs, **A100**: best_val 3.468; train response 3.333→3.324, de 0.173→0.120, **donor-invariance 0.113→0.0016** (encoder learning donor-invariance); **val invariance 0.0** (deterministic, the train-only fix); atomic best+last checkpoints (~80 MB) written.
  - **full-graph M1→M2→M3**, capped, **A100**: all components incl. **donor-invariance 1.55** + **wired `edge_confidences`** (source-aware `L_graph` ≈16.6k); trains + checkpoints; back-props through the confidence-weighted graph term.
- **GPU relevance (honest):** the Module 1/2/3 encode smokes are genuinely GPU-accelerated (24.7k rows/s). **Stage-A training is data-loading-bound** (per-row pandas + sparse slice in the DataLoader) and the graph path is **CPU-bound** in the per-subgraph loop, so the A100 sat ~1-3% utilized during training — compute isn't the bottleneck. PyG mini-batching + a batched loader (+ the donor node-state cache) are the documented upgrades to make training GPU-bound.

## Module 5 (Loss + Training) — this session (2026-07-16)

New package `src/tcell_pipeline/training/` implementing walkthrough §8 — makes the four model modules
**trainable**. Two frozen stages (§8.1): **Stage A** fits the H1 predictor (Module 1+2+3); **Stage B**
loss modules (calibration + Module 4 rationale) are fitted *after* the H1 freeze — no fit loop here by design.

- **StageALoss** (`losses.py`): `L_pred = L_response(Huber program, δ=1) + λ_gene·L_gene(Huber gene) +
  λ_DE·L_DE(focal BCE up/down, γ=2) + λ_inv·L_invariance + λ_graph·L_graph`. **DEHead** = `Linear(256,2G)`
  over `h_do`; DE up/down labels derived from `|Δx_true(zscore)| ≥ 1.645` (two-sided 10% tail — the
  `adj_p<0.1` proxy the dataset contract carries). `f_shared = Linear(K,K)` donor-invariance; `L_graph` =
  `Σ|ᾱ| + Σ(1−conf)·ᾱ²` from `edge_gates` (conf defaults 0 → L2 on gates; optional `edge_confidences`).
- **StageBCalibrationLoss** (`losses.py`): Gaussian NLL `0.5·Σ[log σ² + (Δz−Δẑ)²/σ²]` — a **loss module
  only** (fitted on the calibration partition after the H1 freeze; no loop).
- **PerturbationDataset** (`dataset.py`): split-aware (`blocked_target_ood.csv`); `__getitem__ →
  (batch_dict, target, condition, Δz_true, Δx_true, row_index)`; **q_pre-only** (leakage fence held
  downstream — `q_post` never in features); `Δz_true` = `program_response` A for train rows else `z@B`
  projection out-of-fold; `Δx_true` = zscore row; `+ collate`. All paths injectable (tiny-fixture tests).
- **Trainer** (`trainer.py`): AdamW(1e-3/1e-5) over the model **and** loss params (DE head + `f_shared`);
  the frozen basis B (`persistent=False` buffer) is **neither optimised nor checkpointed**; grad-clip 1.0,
  patience-10 early stop, atomic best+last checkpoints to `data/checkpoints/`, per-epoch logs to `data/logs/`.
- **run_train.py**: Stage A orchestrator on real marts (`--lr/--epochs/--batch-size/--seed/--n-max/--expr-only`);
  pins `torch.set_num_threads(1)`. `RationaleLoss` (Module 4) **not** reimplemented.
- **config:** `LR/WEIGHT_DECAY/MAX_EPOCHS/EARLY_STOP_PATIENCE/BATCH_SIZE/GRAD_CLIP/HUBER_DELTA/FOCAL_GAMMA/
  LAMBDA_DE/LAMBDA_INV/LAMBDA_GRAPH/LAMBDA_GENE/DE_CALL_ZSCORE/DONOR_INVARIANCE/DONOR_INVARIANCE_SAMPLES/
  CHECKPOINTS_ROOT/LOGS_ROOT`.
- **Donor invariance is REAL, not inert (2026-07-16 intelligent fix + post-review hardening).** An
  adversarial review first found the naive `_invariance` (group mart rows by `(target,condition)`) inert —
  the mart's `donor_pc` is the condition-level *mean* of the donors, so no donor pair exists to group. The
  individual donors survive in `control_donor_profiles.parquet` (**4 real donors** × 3 conditions), so the
  term was reformulated to **donor resampling**: the Trainer re-runs the encoder under distinct real donor
  PC vectors (in eval so DropEdge doesn't contaminate the signal) and `L_invariance` penalises the
  **variance of `Δz` across donors, directly**. (A *second* xhigh review caught that penalising
  `Var(f_shared(Δz))` with a free `f_shared` is degenerate — collapses to `W=0` under weight decay,
  re-inert — so `f_shared` was dropped and we penalise raw `Δz`, which has no trivial solution and forces
  the encoder itself.) Verified on real data: the term **optimises via the encoder, 2.15 → 0.19 over 3
  epochs**, and is computed in **train only** so the val metric stays deterministic.
  `config.DONOR_INVARIANCE`(=True) / `DONOR_INVARIANCE_SAMPLES`(=2); `--no-donor-invariance` opts out.
  `ponytail:` donor resampling ~triples the CPU-bound graph Stage-A cost — cache node states to remove it.
- **Verification:** `./init.sh` green — **92 tests** (79 prior + 13 new `test_training.py`, all synthetic:
  Stage A shapes+gradient flow, graph-gate penalty + **batch-normalization**, Stage B NLL+grad, DE
  probs∈[0,1], learnable λ mixture, dataset keys+q_post fence+dz source, **donor pool+resampler**, 2-epoch
  checkpointed run, param-update, **real donor-invariance signal (train nonzero / val 0 / off 0)**,
  **empty-split guard**), **zero warnings**. Real-data Stage A smoke PASSED: expr-only (256×3, invariance
  2.15→0.19, val invariance 0) and full-graph M1→M2→M3 — all train, back-prop, checkpoint.
- **Post-review round 2 (xhigh `/code-review`, 15 verified defects; correctness ones fixed):** the
  degenerate `f_shared` (above); stochastic donor term leaking into **val** → now train-only; silent no-op
  when donor profiles absent → fail-fast + honest log; `L_graph` batch-size-dependent → mean-reduced;
  `torch.manual_seed` global reseed → dedicated `Generator`s; empty-split crash → clear `ValueError`;
  DEHead sized from `model.decoder.h_do_dim`; de_obs↔pc row-count guard; `DONOR_COLS` reused.
- **Round 3 — the 3 items round 2 flagged are now implemented:** (a) **`Δz_true` mismatch fixed** — `z@B`
  for **every** row (one consistent fold-local target; `program_response` no longer a training dependency,
  dropped from the dataset + the run_train required-gate). (b) **`edge_confidences` wired** — per-edge
  source confidence (edge-feature score column, [0,1]) threaded `TypedGraphEncoder.forward` →
  `EGIPGModel.forward` (`out["edge_confidences"]`) → Trainer → `L_graph`, so its unsourced term now
  down-weights well-sourced edges (real data: L_graph ~21k → ~18k). (c) **`Subset` silent-disable fixed** —
  `Trainer._resolve_donor_pool` walks wrapper `.dataset` chains. Verified: full-graph training back-props
  through the confidence-weighted `L_graph` (exit 0); `test_graph` checks confidences aligned per-edge to
  gates + clipped to [0,1]; `./init.sh` green at 92 tests. Refuted (round 2): a false device mismatch.
- **Remaining for feat-008 done:** the Stage-B calibration + rationale **fit loops** (both loss modules
  exist, no fit loop), the near-null-signal freeze gate, and feat-007 (graph baselines) still not-started.
- Design + as-built: `docs/specs/2026-07-16-module5-training.md`.

## Module 4 (Sparse Predictive-Rationale Head) — this session (2026-07-16)

New package `src/tcell_pipeline/rationale/` implementing walkthrough §7 / report §Module 4. **Stage B**:
fitted AFTER the H1 predictor freeze — a **predictive rationale, NOT a causal mechanism** (deletion
scores are fixed-model perturbation tests, report line 499/718). No training loops (out of scope by
design — modules + loss + faithfulness eval only).

- **RationaleHead** (`rationale_head.py`): per edge `imp = ᾱ · sigmoid(Linear([h_u‖h_v‖f_e]))` (gate ×
  learned relevance, both in [0,1]); the scorer is **zero-initialised** so an untrained head ranks by the
  frozen condition gate (faithful by construction — training refines it). Top-k selection across all 4
  relations → `selection_mask` + `selected` (sorted). Output labelled `predictive_rationale`, never
  `causal`; `edge_gates=None` (expression-only member) → empty rationale.
- **RationaleLoss** (`rationale_loss.py`): `λ_sp·|S| + λ_suff·‖dz_S−dz_full‖² + λ_nec·relu(δ−‖dz_\S−dz_full‖)²
  + λ_ct·contrastive`. Pure function of pre-computed deltas + importance; differentiable to the head when
  the caller passes soft-mask deltas.
- **FaithfulnessTester** (`faithfulness.py`): fixed-model deletion tests — `sufficiency`/`necessity` re-run
  the FROZEN encoder with the rationale kept / removed (their gate zeroed) and measure how `Δz` moves;
  `structural_ood_audit` reports degree / component-count / sparsity / hop-distance before-vs-after.
- **MatchedRandomSampler** (`matched_random.py`): negative controls matched on per-relation edge count
  (→ size + relation composition + sparsity). ponytail: richer degree/connectivity/hop matching deferred.
- **Module-2 enabler:** `TypedGraphEncoder.encode_subgraph(...)` added to expose final node states +
  accept a per-edge gate mask; `encode_one` unchanged (delegates to it, 3-tuple contract preserved).
- **config:** `RATIONALE_TOP_K=15`, `RATIONALE_TAU=0.5`, `LAMBDA_SPARSE/SUFF/NEC/CONTRAST`, `N_MATCHED_CONTROLS=100`.
- **Verification:** `./init.sh` green — **78 tests** (69 prior + 9 new `test_rationale.py`, all synthetic:
  imp∈[0,1], top-k sorted, sufficiency<matched-random, necessity>matched-random, matched-random size+relation
  match, structural_ood dict, loss components+gradients, expr-only→empty rationale, label predictive_rationale
  not causal). Real-data `run_module4_smoke.py` **PASSED** on the real PPI graph (A1BG neighbourhood:
  sufficiency<matched-random, necessity>matched-random, labelled `predictive_rationale`).
- **Perf note:** this 64-core box (shared with CVAT workers) thrashes torch's thread pool on the tiny
  per-subgraph GNN ops (2.5s→20ms per encode); the CPU-only Module-4 tests + smoke pin `torch.set_num_threads(1)`.
- **Post-review fixes (xhigh `/code-review` — 13 verified findings, 3 refuted; all confirmed resolved):**
  (correctness) `FaithfulnessTester` now forces the encoder+decoder into **eval** on every deletion re-run —
  `@torch.no_grad` suppresses gradients but NOT DropEdge, so on a train-mode encoder the "fixed-model"
  sufficiency/necessity scores were **stochastic** (+ a determinism regression test; caller's train/eval
  state restored); `structural_ood_audit` sparsity is now **PP-scoped** to match its degree/component/hop
  metrics; the tautological audit test now checks the deleted-fraction against an **independent count** +
  component monotonicity. (cleanup) optional cached `dz_full` via a public `delta_z()`, `torch.topk` in
  `_select` (was a per-edge `float()` sync + full sort on ~33k edges), DRY `_PP_RELATIONS`, vectorised
  `_pp_edges`, smoke on the public API. Kept as spec-mandated: `edge_attrs` param + `subgraph_edges` output.
  `./init.sh` green at **79 tests** (+1 determinism); Module 4 real-data smoke re-run **PASSED**.
- **Remaining for feat-008 done:** the training-loss OPTIMIZATION loop + train/calibration loops (the loss
  module exists, no fit loop yet); feat-007 graph baselines still not-started. The FaithfulnessTester +
  MatchedRandomSampler are also the machinery feat-012 (predictive-rationale audit) will run on the trained model.

## Full real-data run + warnings cleanup (2026-07-16)

Re-ran every non-destructive real-data entrypoint end-to-end (M0 excluded — it re-downloads multi-GB PPI
DBs and overwrites the frozen marts), GPU where the code is device-aware. All green:
- `./init.sh` — **79 passed** (compileall clean).
- Module 1 smoke (feat-014/015): 33,983 rows on **A100 (cuda)** @24.7k/s; all `h_do` finite; q_post fence held.
- Module 2 smoke (feat-016): 25,440-node graph on **cuda**; CD3E nbhd 512 proteins; per-condition gates differ; attn sums to 1.
- Module 3 smoke (feat-008 slice): M1→M2→M3 on **cuda**; finite; λ∈[0.42,0.65]; σ>0; expr-only λ==0.
- Module 4 smoke (feat-008): real PPI graph; sufficiency<matched-random; necessity>matched-random; labelled `predictive_rationale`.
- feat-005 production basis: `run_program_basis --method sparse_pca` re-fit the real train fold in **304s**
  → B(10282,128)/A(21262,128); all finite; fold-locality exact; 22.7% zero loadings / **0 dead programs**;
  recon MAE 0.686 vs 0.817 baseline. Parquets gitignored.
- feat-003 splits: re-run **byte-identical** (4/4 sha256), 26.4% blocked vs 53.8% random = 0.509 cut.
- **GPU:** 5× A100 80GB, torch 2.13.0+cu126 (CUDA-12.2 driver). M1–M3 ran on cuda; the basis fit is
  sklearn/CPU and Module 4 is CPU-only by design.

**Warnings cleanup (commit `2bf1653`):** silenced the expected warnings the run surfaced — `torch.jit.script`
DeprecationWarning (torch_geometric 2.8 on torch 2.13; filtered in `tcell_pipeline/__init__.py` before PyG
import) and sklearn `ConvergenceWarning` (LARS early-stop / capped-iter NMF/FastICA; scope-silenced in
`program_basis._factor`). `./init.sh` now **79 passed with a clean warnings summary** (was "79 passed, 4
warnings"); verified under `-W error::DeprecationWarning` + a ConvergenceWarning-leak probe. Module 4
docs sync (README + `docs/specs/2026-07-16-module4-rationale-head.md`) committed at `b094b5e`.

## Module 3 (Program Decoder) — prior session

New package `src/tcell_pipeline/programs/` + `src/tcell_pipeline/model.py` implementing walkthrough §6.
Scope was Module 3 only — Module 4, losses, and training loops deliberately excluded.

- **feat-005 (Latent Program Extraction) — in-progress.** `program_basis.py`: `fit_program_basis`
  (Z_train ≈ A·Bᵀ) with method dispatch — `sparse_pca` (MiniBatchSparsePCA, the scalable sparse
  variant; the paper default, ~15 min on full train), plus `nmf` / `fastica` (ICA) / `svd`; K from
  `config.PROGRAM_DIM=128`. `train_row_indices` is the fold-locality gate (train-role genes → row
  indices; challenge-overlap `assert` in the orchestrator). `save/load_program_basis` +
  `save_program_response` (atomic parquet, gene axis = full de_var order). `run_program_basis.py`
  orchestrator (`--method`, `--K`). Ran `--method svd` on **21,262 real train rows × 10,282 genes**
  in 6.2 s → `gene_program_loadings.parquet` (B 10282×128) + `program_response.parquet` (A 21262×128),
  both gitignored under `data/intermediate/`. **Remaining for done:** 4-method × 4-K comparison
  (reconstruction / sparsity / stability) + shallow VAE.
- **feat-008 (EG-IPG Model) — in-progress (Module-3 slice).** `program_decoder.py` `ProgramDecoder`:
  graph path `Linear(512,K)` + expr-only path `Linear(256,K)`, sigmoid mixture gate `λ∈[0,1]`, softplus
  uncertainty `σ`, gene decode `Δx = B·Δzᵀ + r` with **B a frozen `register_buffer` (not a Parameter)**.
  `model.py` `EGIPGModel` wraps M1+M2+M3; `graph_encoder=None` → expression-only nested variant (λ pinned
  to 0, no edge gates). **Remaining:** Module 4 sparse rationale head, losses, train/calibration loops.
- **Verification:** `./init.sh` green — **69 tests** (57 prior + 12 new `test_programs.py`, all synthetic:
  basis shapes across all 4 methods, fold-local row selection, decoder shapes, λ∈[0,1], σ>0, B-is-buffer,
  `Δx=B·Δzᵀ+r`, expr-only variant, full EGIPGModel forward). Real-data `run_module3_smoke.py` **PASSED**
  end-to-end (M1→M2→M3 on 4 real perturbations: finite, λ∈[0.46,0.55], σ>0; expr-only λ==0).
- **config additions:** `PROGRAM_DIM=128`, `PROGRAM_METHOD="sparse_pca"`,
  `PROGRAM_LOADINGS_PATH`, `PROGRAM_RESPONSE_PATH`, `PROGRAM_COL_PREFIX`.
- **Post-review fixes (xhigh `/code-review`, all 13 findings resolved):** FastICA basis returns
  `mixing_` (loadings) not `components_`; `program_basis` buffer `persistent=False`; independent
  `raise`-based fold-leak guard; σ floored at `1e-12`; expr-only variant drops the graph residual bias;
  `load_program_basis` errors clearly on duplicate gene symbols; complete mart guards in the M3 smoke +
  orchestrator; overridable decoder dims on `EGIPGModel`; shared `build_encoder_batch` (encoders/batch.py)
  + `load_zscore_rows`; hoisted decoder `joint`; removed the dead `GENE_LEVEL_DIM` alias. `./init.sh`
  69 green; M1/M2/M3 smokes + orchestrator all re-run clean.

## Full real-data run — all modules/features (2026-07-15)

Ran every non-destructive real-data entrypoint end-to-end (M0 excluded — it re-downloads multi-GB PPI
DBs and overwrites the frozen marts). All green:
- `./init.sh` — **69 passed** (compileall clean); `splits.py` (feat-003) — **byte-identical** to the
  frozen freeze (all 4 sha256 match), effectiveness 26.4% vs 53.8% random (0.51 cut).
- Module 1 smoke (feat-014/015): all **33,983** rows finite @24.7k/s; real PLM 33,796 / PINNACLE 3,135
  coverage; NaN guard held; q_post leakage fence rejected the injected column.
- Module 2 smoke (feat-016 + feat-004 graph): 25,440-node graph / ~8M typed edges; CD3E nbhd @cap 512;
  per-condition gates differ; readout attention sums to 1.
- Module 3 smoke (feat-008 + feat-005 slice): fold-local basis on 21,262 train rows; M1→M2→M3 finite;
  λ∈[0.38,0.62]; σ>0; expr-only variant λ=0.
- **feat-005 production basis:** `run_program_basis --method sparse_pca` fit the real train fold in
  **289s** → froze `gene_program_loadings.parquet` (B 10282×128) + `program_response.parquet`
  (A 21262×128), replacing the earlier svd smoke output. Validated: all finite; fold-locality exact
  (saved rows == 21,262 train); **22.7% zero loadings**, no dead programs; centered recon MAE **0.687**
  vs 0.817 zero-baseline (sparse_pca trades reconstruction for sparsity vs svd ~0.61). feat-005 stays
  in-progress — method×K comparison + shallow VAE remain. Parquets gitignored (regenerable).

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

## Files Added This Session (feat-003 — leakage-safe splits)

- `src/tcell_pipeline/splits.py` (NEW): family grouping (capped union-find), 4-role partition, random
  split, leakage audit, `run()` → frozen `data/splits/` artifacts
- `src/tests/test_splits.py` (NEW): 8 synthetic tests
- `docs/specs/2026-07-15-feat-003-leakage-safe-splits.md` (NEW): design doc (report-derived + empirical)
- `data/splits/` (NEW, git-tracked): `blocked_target_ood.csv`, `random.csv`, `manifest.json`, `leakage_report.json`
- `src/tcell_pipeline/config.py` — feat-003 constants (SPLITS_ROOT, SPLIT_ROLES/FRACTIONS/SEED, SEQ_SIM_COSINE_THRESHOLD, GROUP_SIZE_CAP, artifact paths)
- `feature_list.json` (feat-003 → done), `progress.md`, `session-handoff.md`

## Files Added Prior This Session (feat-016 — Module 2 typed graph encoder)

- `src/tcell_pipeline/graph/__init__.py` (NEW)
- `src/tcell_pipeline/graph/graph_builder.py` (NEW): `build_hetero_graph()` -> HeteroData + gene_to_idx
- `src/tcell_pipeline/graph/neighborhood_sampler.py` (NEW): `sample_subgraph()`
- `src/tcell_pipeline/graph/typed_graph_encoder.py` (NEW): `TypedGraphEncoder` + signed message + condition gate
- `src/tcell_pipeline/graph/graph_readout.py` (NEW): `GraphReadout` cross-attention
- `src/tcell_pipeline/graph/run_module2_smoke.py` (NEW): real-data smoke (build graph, CD3E, encode)
- `src/tests/test_graph.py` (NEW): 8 synthetic Module 2 tests
- `src/tcell_pipeline/config.py` — Module 2 constants (GRAPH_*, EDGE_*, N_RELATION_TYPES, PROTEIN_FEATURE_DIM, ...)
- `feature_list.json` — feat-016 added, status done; `progress.md`, `session-handoff.md` — state sync

## Files Modified Prior Session (feat-015 — real embeddings + GPU)

- `src/tcell_pipeline/embeddings_plm.py` (NEW): ESM-2 650M generator (resumable, GPU-aware)
- `src/tcell_pipeline/embeddings_pinnacle.py` (NEW): PINNACLE CD4-context -> UniProt mapper (Figshare download)
- `src/tcell_pipeline/config.py` — PINNACLE_EMBED_DIM 512->128; +PINNACLE_RAW_DIR/FIGSHARE_URL/CONTEXT
- `src/tcell_pipeline/run_module1_smoke.py` (NEW): full-mart real-data smoke (33,983 rows, GPU-native, Module 1 analogue of run_module0.py)
- `src/tcell_pipeline/encoders/{context,perturbation}_encoder.py` — device-aware forward (encoder runs on GPU via .to('cuda'))
- `src/tests/test_encoders.py` — rewritten to real PLM+PINNACLE data (no synthetic parquets); dim literals 1796->1412
- `requirements.txt` — +fair-esm, +cu126 torch install note
- `feature_list.json` — feat-015 added, status done
- `progress.md`, `session-handoff.md` — state sync

### Prior session (feat-014 — Module 1 encoder)

- `src/tcell_pipeline/encoders/` (NEW package): `_tensor.py`, `embedding_store.py`, `target_encoder.py`,
  `context_encoder.py`, `quality_encoder.py`, `perturbation_encoder.py`, `__init__.py`
- `src/tcell_pipeline/config.py` — Module 1 constants; `src/tests/test_encoders.py` (10 tests); feat-014 done
- Post-review leakage-fence hardening (xhigh /code-review, 3 CONFIRMED/PLAUSIBLE latent findings):
  `feature_availability.py` — `_is_donor_pc` tightens the bare `donor_pc_` prefix to digits-only, and
  `_assert_disjoint_fence()` makes `classify_columns` REFUSE at runtime when a name is in both
  Q_PRE_COLS and Q_POST_COLS (previously the output-level disjointness test was a tautology that
  couldn't catch it). `test_feature_availability.py` — config disjointness pin, behavioral raise-on-
  overlap test, donor-prefix test, committed-manifest drift guard. pytest now **37 total**.

Prior session (Module 0 fixes, commits e453964..1732def): id_mapping.py reviewed-canonical UniProt
disambiguation; ppi_graph.py HuRI apex URL + CORUM 5.3 fastapi + `_corum_gene_col` + TLS skip;
complex_membership.py CORUM schema; feature_availability.py + config.py KNOWN_METADATA_COLS allowlist;
requirements.txt `mygene`; AGENTS.md session-handoff.md first-class artifact.

## Notes for Next Session

- `examples/` scripts double as data-understanding docs
- Read the experiment plan report for detailed feature specs (2026-07-14 literature refresh)
- Before feat-011 screening / freezing H1, run the near-null-signal check
- Module 0 marts are on disk but gitignored; rerun `python src/tcell_pipeline/run_module0.py` to regenerate
