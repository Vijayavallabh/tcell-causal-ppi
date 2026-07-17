# session-handoff.md — archived history (Modules 3-5, feat-003/015/016, per-module file lists)

> Archived from `session-handoff.md` (2026-07-17) to keep the live file's restart-read short. Nothing changed — verbatim historical detail.

## Completed Prior Session (Module 5 — Loss + Training; feat-008)

Built Module 5 (walkthrough §8) as a new package `src/tcell_pipeline/training/`, making the four model
modules **trainable**. Two frozen stages (§8.1): **Stage A** fits the H1 predictor (Module 1+2+3); the
**Stage B** loss modules (calibration + Module 4 rationale) are fitted after the H1 freeze — no fit loop
here by design. **Committed on main this session** (`git log -1`).

- **StageALoss** (`losses.py`) — `L_pred = Huber response (program + gene, δ=1) + λ_DE·focal-BCE DE
  up/down (DEHead=Linear(256,2G), γ=2, labels from |zscore|≥1.645 as the `adj_p<0.1` proxy the dataset
  carries) + λ_inv·donor-invariance (f_shared=Linear(K,K)) + λ_graph·L_graph (Σ|ᾱ| + Σ(1−conf)ᾱ² from
  `edge_gates`; conf defaults 0)`. **StageBCalibrationLoss** — Gaussian NLL, a **loss module only**.
  **DEHead** — `Linear(256,2G)` → up/down logits, `.probs()` in [0,1].
- **PerturbationDataset** (`dataset.py`) — split-aware (`blocked_target_ood.csv`); `__getitem__ →
  (batch_dict, target, condition, Δz_true, Δx_true, row_index)`; **q_pre-only** (fence held downstream);
  `Δz_true` = `program_response` A for train rows else `z@B` projection out-of-fold; `Δx_true` = zscore
  row; `+ collate`. Paths injectable → tiny-fixture tests.
- **Trainer** (`trainer.py`) — AdamW(1e-3/1e-5) over model **and** loss params; frozen B
  (`persistent=False` buffer) **neither optimised nor checkpointed**; grad-clip 1.0, patience-10 early
  stop, atomic best+last checkpoints (`data/checkpoints/`), per-epoch logs (`data/logs/`).
- **run_train.py** — Stage A orchestrator on real marts (`--lr/--epochs/--batch-size/--seed/--n-max/
  --expr-only`); pins `set_num_threads(1)`. `RationaleLoss` (Module 4) **not** reimplemented.
- **config** — `LR/WEIGHT_DECAY/MAX_EPOCHS/EARLY_STOP_PATIENCE/BATCH_SIZE/GRAD_CLIP/HUBER_DELTA/
  FOCAL_GAMMA/LAMBDA_DE/LAMBDA_INV/LAMBDA_GRAPH/LAMBDA_GENE/DE_CALL_ZSCORE/CHECKPOINTS_ROOT/LOGS_ROOT`.
- **Donor invariance — REAL (intelligent fix + post-review hardening), not inert.** A review first found
  the naive `_invariance` (group mart rows by `(target,condition)`) inert: the mart's `donor_pc` is the
  condition-level *mean* of the donors. The real donors survive in `control_donor_profiles.parquet` (**4
  real donors** × 3 conditions), so it was reformulated to **donor resampling** — the Trainer re-runs the
  encoder under distinct real donor PC vectors and `L_invariance` penalises the **variance of `Δz` across
  donors, directly**. (A *second* xhigh review caught that penalising `Var(f_shared(Δz))` with a free
  `f_shared` is degenerate — it collapses to `W=0` under weight decay, re-inert — so `f_shared` was dropped;
  raw-`Δz` variance has no trivial solution and forces the encoder itself.) Verified: **optimises via the
  encoder, 2.15 → 0.19 over 3 epochs**, computed **train-only** so the val metric stays deterministic.
  `config.DONOR_INVARIANCE` / `DONOR_INVARIANCE_SAMPLES`; `--no-donor-invariance` opts out. `ponytail:`
  donor resampling ~triples the CPU-bound graph Stage-A cost (cache node states to remove it).
- **Post-review round 2 (xhigh `/code-review`, 15 verified defects; correctness fixed):** the degenerate
  `f_shared`; the stochastic donor term leaking into **val** → train-only; silent no-op when donor profiles
  absent → fail-fast + honest log; `L_graph` batch-size-dependent → mean-reduced; `manual_seed` global
  reseed → dedicated `Generator`s; empty-split crash → clear `ValueError`; DEHead sized from the decoder;
  de_obs↔pc guard; `DONOR_COLS` reused.
- **Round 3 — the 3 round-2-flagged items implemented:** `Δz_true = z@B` for every row (consistent target,
  `program_response` dropped as a dependency); `edge_confidences` **wired** through the graph → model →
  `L_graph` (unsourced term now source-aware, ~21k → ~18k); `Subset` silent-disable fixed via
  `_resolve_donor_pool`.
- **Verification** — `./init.sh` green, **92 tests** (79 prior + 13 new `test_training.py`, synthetic),
  zero warnings. Real-data Stage A smoke PASSED: expr-only (256×3, invariance 2.15→0.19, val invariance 0)
  and full-graph M1→M2→M3 — all train, back-prop, write atomic checkpoints.
- **Remaining (feat-008):** the Stage-B calibration + rationale **fit loops** (both loss modules exist,
  no fit loop), the near-null-signal freeze gate, and feat-007 (graph baselines, still not-started).
- Design + as-built: `docs/specs/2026-07-16-module5-training.md`.

## Completed This Session (Module 4 — Sparse Predictive-Rationale Head; feat-008)

Built Module 4 (walkthrough §7 / report §Module 4) as a new package `src/tcell_pipeline/rationale/`.
**Stage B** — fitted AFTER the H1 predictor freeze; a **predictive rationale, NOT a causal mechanism**
(deletion scores are fixed-model perturbation tests, report line 499/718). No training loops (out of scope
by design — modules + loss + faithfulness eval only). **Committed on main this session** (`git log -1`).

- **RationaleHead** — per-edge importance `ᾱ · sigmoid(Linear([h_u‖h_v‖f_e]))` (both factors in [0,1]);
  scorer **zero-initialised** so an untrained head ranks by the frozen condition gate (faithful by
  construction). Top-k over all 4 relations. Output labelled `predictive_rationale`, never `causal`;
  no graph (`edge_gates=None`) → empty rationale.
- **RationaleLoss** — `λ_sp·|S| + λ_suff·‖dz_S−dz_full‖² + λ_nec·relu(δ−‖dz_\S−dz_full‖)² + λ_ct·contrastive`;
  differentiable to the head via soft gate weights.
- **FaithfulnessTester** — fixed-model deletion tests (`sufficiency`, `necessity`) re-run the frozen encoder
  with the rationale kept / removed; `structural_ood_audit` reports degree / components / sparsity /
  hop-distance before-vs-after.
- **MatchedRandomSampler** — negative controls matched on per-relation edge count (size + relation composition).
- **Module-2 enabler** — `TypedGraphEncoder.encode_subgraph(...)` exposes final node states + accepts a
  per-edge gate mask; `encode_one` unchanged (delegates; 3-tuple contract preserved; all its callers untouched).
- **config** — `RATIONALE_TOP_K=15`, `RATIONALE_TAU=0.5`, `LAMBDA_SPARSE/SUFF/NEC/CONTRAST`, `N_MATCHED_CONTROLS=100`.
- **Verification** — `./init.sh` green, **78 tests** (69 prior + 9 new `test_rationale.py`, synthetic).
  Real-data `run_module4_smoke.py` **PASSED** on the real PPI graph (A1BG neighbourhood:
  sufficiency<matched-random, necessity>matched-random, labelled `predictive_rationale`).
- **Perf** — this 64-core box (shared with CVAT workers) thrashes torch's thread pool on tiny per-subgraph
  GNN ops (2.5s→20ms/encode), so the CPU-only Module-4 tests + smoke pin `torch.set_num_threads(1)`.
- **Post-review fixes (xhigh `/code-review` — 13 verified findings, 3 refuted; all confirmed resolved):**
  (correctness) `FaithfulnessTester` forces encoder+decoder **eval** on every deletion re-run — `no_grad`
  suppresses gradients but not DropEdge, so the fixed-model scores were stochastic on a train-mode encoder
  (+ determinism regression test, caller state restored); `structural_ood_audit` sparsity **PP-scoped** to
  match its connectivity metrics; tautological audit test replaced with an independent-count + component-
  monotonicity check. (cleanup) optional cached `dz_full` via public `delta_z()`, `torch.topk` selection,
  DRY `_PP_RELATIONS`, vectorised `_pp_edges`, smoke on the public API. Kept as spec-mandated: `edge_attrs`
  param + `subgraph_edges` output. `./init.sh` **79 tests**; smoke re-run PASSED.
- **Remaining (feat-008):** the training-loss OPTIMIZATION loop + train/calibration loops (loss module exists,
  no fit loop); feat-007 not-started. FaithfulnessTester + MatchedRandomSampler are also the machinery
  **feat-012** (predictive-rationale audit) will run on the trained model.

## Also This Session (2026-07-16) — full real-data verification + warnings cleanup

- **Full real-data run** (GPU where device-aware; M0 excluded — destructive): `./init.sh` **79 passed**;
  M1 (33,983 rows, **A100**), M2 (25,440-node graph, A100), M3 (M1→M2→M3, A100), M4 (real PPI graph, CPU)
  all PASSED; `sparse_pca` production basis re-fit + validated (**304 s**, B 10282×128 / A 21262×128,
  fold-locality exact, 22.7% zero loadings / 0 dead programs, recon MAE 0.686 vs 0.817); `splits.py`
  **byte-identical** (4/4 sha256). GPU: 5× A100 80GB (torch 2.13.0+cu126).
- **Warnings cleanup** (commit `2bf1653`): silenced the expected third-party `torch.jit.script` deprecation
  (torch_geometric 2.8 on torch 2.13) + sklearn `ConvergenceWarning` (LARS / NMF / FastICA) at their
  sources; `./init.sh` now 79 passed with a **clean warnings summary** (was 4 warnings). Module 4 docs sync
  (README + `docs/specs/2026-07-16-module4-rationale-head.md`) committed at `b094b5e`.

## Completed Prior Session (Module 3 — Program Decoder; feat-005 + feat-008 scaffold)

Built Module 3 (walkthrough §6) as a new package `src/tcell_pipeline/programs/` + top-level
`src/tcell_pipeline/model.py`. Scope was Module 3 only; Module 4, losses, and training were
deliberately excluded (per the goal). **Committed on main this session** (code + docs + state-file
syncs in one commit; `git log -1` for the hash).

- **feat-005 (Latent Program Extraction) — in-progress.** `programs/program_basis.py`: fold-local
  `fit_program_basis` (Z_train ≈ A·Bᵀ) with method dispatch `sparse_pca` (MiniBatchSparsePCA — scalable
  sparse variant, paper default, ~15 min on full train) / `nmf` / `fastica` (ICA) / `svd`; K from
  `config.PROGRAM_DIM=128`. `train_row_indices` = fold-locality gate; `save/load_program_basis` +
  `save_program_response` (atomic parquet). `programs/run_program_basis.py` orchestrator (`--method/--K`,
  challenge-overlap assert). Ran `--method svd` on **21,262 real train rows × 10,282 genes** in 6.2 s →
  `gene_program_loadings.parquet` (B 10282×128) + `program_response.parquet` (A 21262×128), gitignored.
  **Production `sparse_pca` basis since fit** on the real train fold (`--method sparse_pca`, **289 s**) →
  re-froze B 10282×128 / A 21262×128 over the svd smoke output; all finite, fold-locality exact
  (saved rows == 21,262 train), 22.7% zero loadings, recon MAE 0.687 vs 0.817 zero-baseline.
  **Remaining:** 4-method × 4-K comparison (reconstruction / sparsity / stability) + shallow VAE.
- **feat-008 (EG-IPG Model) — in-progress (Module-3 slice).** `programs/program_decoder.py`
  `ProgramDecoder`: graph path `Linear(512,K)` + expr-only `Linear(256,K)`, sigmoid mixture gate
  `λ∈[0,1]`, softplus uncertainty `σ`, gene decode `Δx = B·Δzᵀ + r` with **B a frozen `register_buffer`,
  not a Parameter**. `model.py` `EGIPGModel` wraps M1+M2+M3; `graph_encoder=None` → expression-only
  nested variant (λ pinned to 0). **Remaining:** Module 4 sparse rationale head, losses, train/cal loops.
- **Verification:** `./init.sh` green — **69 tests** (57 prior + 12 new `test_programs.py`, all synthetic).
  Real-data `run_module3_smoke.py` **PASSED** end-to-end (M1→M2→M3 on 4 real perturbations: finite,
  λ∈[0.46,0.55], σ>0; expr-only λ==0). `config.py` +`PROGRAM_DIM/PROGRAM_METHOD/`
  `PROGRAM_LOADINGS_PATH/PROGRAM_RESPONSE_PATH/PROGRAM_COL_PREFIX`.

## Completed This Session (post code-review fixes — feat-016 + feat-003)

Applied the verified findings from the xhigh `/code-review`
(`docs/reviews/2026-07-15-code-review-feat-016-feat-003.md`). **Committed at `7760624`.**

- **Tier 1 (feat-003 leakage-safety) — split CSVs byte-identical (sha256 unchanged), audit corrected:**
  - `splits.py` audit now publishes **cap-induced family splits** via an uncapped pre-cap component
    pass (`_precap_labels`). The old post-cap "no family group spans >1 role" assertion was blind to
    families the 5% cap *must* break: real data has `cap_induced_family_splits=1` (one giant
    single-linkage family), `family_challenge_sharing_train_frac=0.41` (an upper bound inflated by
    single-linkage chaining — the pairwise residual, 26.4%, is the true leakage).
  - `_sequence_residual` now centers train+challenge in **one global frame** (was per-subset means →
    mismatched frames understating similarity). Corrected effectiveness **53.8%→26.4% = 51% reduction**
    (was 53.5→28.1=47). `manifest.json` gains `sequence_block_active`, `n_genes_with_embedding`.
  - `run()` **fails closed** when PLM embeddings are absent (was silent fail-open publishing a
    sequence-leaky split as safe); override with `SPLITS_ALLOW_NO_SEQUENCE=1`.
- **Tier 2 (feat-016 active bugs):** `graph_builder` degree columns reordered to
  `[physical, functional, complex]` to match Module 1's `TARGET_SCALAR_KEYS`; `typed_graph_encoder`
  `encode_one` now moves `h_do` to the module device (was a device-mismatch crash on the public entry
  point); `test_graph.py` signed-message test seeded + the false `|out| < 1.0` bound dropped (relu is
  unbounded → ~13% flake).
- **Tier 3 (all addressed):**
  - Cheap defenses: `graph_builder` `nan_to_num` on edge features (symmetric with node features),
    `.dropna()` on gene symbols, fail-fast on unknown PPI source; dead config constants
    `N_RELATION_TYPES` / `RELATION_TYPES` / `SPLIT_AUDIT_HOPS` removed.
  - `#10` OOV `culture_condition` now raises a legible `ValueError` via `_condition_index` (still
    fail-fast — closed 3-value vocab, invalid input); the "never crash a batch" docstring narrowed to
    unknown *genes* only.
  - `#11` diagnostic random split uses cumulative-boundary allocation (last boundary == n, no truncated
    tail). `random.csv` regenerated; **`blocked_target_ood.csv` still byte-identical** to 35e3999 and the
    effectiveness numbers are unchanged (gene-level baseline doesn't shift at N=11525).
  - `#12` returned `edge_gates[rel]` is now length **E** (one per original edge) for *all* relations,
    aligned to the sub-graph `edge_index` (was 2E-doubled for PP — the mirror carried an identical gate).
    Full gate→(u,v) identity-forwarding API still **deferred to Module 4** (its consumer isn't built).
- **Regenerated** `data/splits/`; **57 pytest green** (+3 regression checks: OOV raises, edge_gates
  length == E, random split covers all items at small N).

## Completed This Session (feat-003 — leakage-safe splits)

Design brainstormed against the experiment-plan report; spec in
`docs/specs/2026-07-15-feat-003-leakage-safe-splits.md`. **The approved CC-over-3-axes design was
revised after empirical measurement proved it collapses** (naive connected-components → giant
components on every axis: physical 95%, complex 23%, ESM cos≥0.95 92%, Louvain 42%).

- [x] `src/tcell_pipeline/splits.py`: hard block = sequence/paralog family via **representative
  (non-chaining, CD-HIT-style) clustering on centered ESM-2 embeddings** (cos≥0.85 → 3.1% largest
  family) + CORUM co-membership, under a 5%-of-genes **capped union-find** (3986 giant merges refused).
  Physical-PPI neighbourhood is **audit-only** (95% one component — can't be a hard block; report G1 +
  Phase-1 6/9 want its distribution *published*, not zeroed).
- [x] **4-role** partition (train/val/calibration/challenge ~60/15/10/15; realized 62.5/13/7.9/16.6),
  assigned by whole family group, seeded, deficit-greedy. Random diagnostic split (row-level).
- [x] Frozen + hashed to **`data/splits/`** (git-tracked): `blocked_target_ood.csv`, `random.csv`,
  `manifest.json`, `leakage_report.json` (machine-readable: hard-asserts no family group split across
  roles; publishes per-axis train→challenge residual + fail-closed audit).
- [x] **Effectiveness validated** (numbers corrected in the post-review pass below): challenge genes
  with a ≥0.85 train paralog cut **53.8% (random) → 26.4% (blocked) = 51% reduction** (the ~26% floor is
  irreducible given dense ESM geometry). 8 synthetic tests (`test_splits.py`). `./init.sh`: **54 passed**.

## Completed This Session (feat-016 — Module 2 typed graph encoder)

New package `src/tcell_pipeline/graph/` (PyG torch_geometric 2.8), a component of feat-008 built ahead
(depends only on Module 0 outputs + Module 1's h_do):

- [x] `graph_builder.build_hetero_graph()` -> `(HeteroData, gene_to_idx)`: **25440** protein nodes keyed
  by upper-case HGNC, each carrying the **same frozen 1412-d descriptor as Module 1's TargetEncoder**
  (PLM 1280 + PINNACLE 128 + 3 graph-derived degrees + control_baseline_expr, zero-fallback);
  **5628** complex nodes (index-only, the learned `nn.Embedding` lives in the encoder). 4 relations
  split by the `is_*` flags + bipartite membership, each with an 8-d edge feature
  (source one-hot(5)|score|is_direct_binary|n_supporting). Real edge counts: physical_ppi 1123205,
  co_complex 48389, functional_assoc 6857702, complex_membership 18932.
- [x] `neighborhood_sampler.sample_subgraph()`: grows physical/co-complex first then score-fills, caps
  at **512** proteins, pulls in member complexes, returns an induced HeteroData preserving `orig_idx`.
- [x] `typed_graph_encoder.TypedGraphEncoder(nn.Module)`: 3-layer per-relation custom PyG
  `MessagePassing` (RGCNConv/GATConv can't express this) with **signed message**
  `tanh(W_sign h_u)*relu(W_mag h_u)` and **condition gate** `sigmoid(w_gate[h_cond(64)||f_e(8)])`
  computed once per relation (layer-independent) and returned as `edge_gates` for Module 4; residual
  FFN+LayerNorm per node type; DropEdge 0.1. `graph_readout.GraphReadout`: 4-head cross-attention
  (q=h_do, K=V=node states) -> **h_graph R^256**, attention sums to 1. `forward(target_genes,
  conditions, h_do)` loops per-target subgraphs; targets absent from the PPI graph -> zero h_graph.
- [x] **CPU and CUDA** (device-aware; sampled subgraphs moved to the module device). config: GRAPH_HOPS,
  NEIGHBORHOOD_CAP, GRAPH_HIDDEN_DIM, GRAPH_LAYERS, GRAPH_N_HEADS, EDGE_DROPOUT, EDGE_FEATURE_DIM,
  N_RELATION_TYPES, COMPLEX_EMBED_DIM, CONDITION_EMBED_DIM, RELATION_TYPES, PROTEIN_FEATURE_DIM.
- [x] Verified: **8** synthetic tests (`test_graph.py`) + `graph/run_module2_smoke.py` real-data smoke
  (full graph in ~18s, CD3E neighbourhood 512 proteins/740 complexes, real Module 1 h_do -> Module 2
  h_graph (4,256) finite on GPU, gates differ by condition, attention sums to 1). `./init.sh`: **46** passed.

## Completed Prior Session (feat-015 embeddings + GPU-native Module 1 encoder + real-data smoke)

The feat-014 encoder left both target-embedding stores at zero-fallback (no parquet on disk). This
session generated the real embeddings so the PerturbationEncoder runs on real target vectors:

- [x] `embeddings_plm.py`: real **ESM-2 650M** (1280-d, mean-pooled over final-layer residues, BOS/EOS/pad
  excluded). Sequences from the UniProt REST accessions endpoint (cached to `uniprot_sequences.parquet`);
  resumable (skip embedded, atomic checkpoint). **Device-aware** — ran on an A100 -> **11419/11419 mart
  proteins embedded** (100% PLM coverage), all finite.
- [x] `embeddings_pinnacle.py`: real **PINNACLE** (Li et al. 2024, Figshare article 22708126) contextual
  embeddings. Real dim is **128** — config's 512 was a placeholder, **corrected to 128**. Took the
  `cd4-positive helper t cell` context (the CD4+ screen's cell type; `config.PINNACLE_CONTEXT`); gene-symbol
  -> UniProt via id_mapping -> **1119 embeddings, 1070/11419 mart proteins covered** (contextual embeddings
  only span in-network proteins; the rest keep the zero fallback).
- [x] Live encoder dims now derive to target.out_dim **1412** (1280+128+4), fusion `Linear(1574->256)`,
  **404,960** trainable params (was 1796 / 503,264 under the 512 placeholder).
- [x] Tests rewritten to **real data/embeddings — no synthetic parquets** (10 tests in `test_encoders.py`):
  real PLM present-loaded + absent-id zero-fallback + dim-mismatch guard, real PINNACLE CD4-context load,
  forward/NaN tests on the real perturbation_condition + de_obs marts.
- [x] **GPU enabled**: host has 5x A100 80GB but the CUDA-12.2 driver can't run the default cu13x torch;
  swapped to `torch==2.13.0+cu126` (minor-version compat). requirements.txt documents the cu126 install.
- [x] Embedding artifacts are gitignored under `data/intermediate/`; regenerate via
  `python -m tcell_pipeline.embeddings_{plm,pinnacle}`.
- [x] **Module 1 encoder made device-aware (GPU-native)**: `ContextEncoder`/`PerturbationEncoder` forward
  move constructed tensors to the module's device, so `PerturbationEncoder().to('cuda')` runs the whole
  forward on GPU (TargetEncoder/QualityEncoder build CPU tensors that forward relocates). Tests default to
  CPU (portable); `test_encoder_runs_on_gpu_when_available` runs only when CUDA is present. Suite now **38**.
- [x] **`run_module1_smoke.py`** (NEW): full-mart real-data verification — drives all 33,983 rows through
  the encoder on GPU (~2s), asserts every h_do finite, checks the leakage fence rejects the mart's real
  q_post columns. Exits non-zero on any NaN/fence breach. The Module 1 analogue of `run_module0.py`.

Prior session (feat-014): `src/tcell_pipeline/encoders/` package — five nn.Modules fused into `h_do` R^256,
q_pre inputs only, no trainable gene-ID embedding, no free donor-ID embedding, leakage fence at the boundary,
NaN guard. Earlier: ~100 GB download, `examples/`, README, Module 0 + code-review fixes, UniProt/HuRI/CORUM.



---

## Files Added (this session, Module 5 — Loss + Training)

- `src/tcell_pipeline/training/__init__.py`, `losses.py`, `dataset.py`, `trainer.py`, `run_train.py` (NEW package)
- `src/tests/test_training.py` (NEW, 13 synthetic tests incl. donor pool + real donor-invariance + post-review guards)
- `docs/specs/2026-07-16-module5-training.md` (NEW — design + as-built + the real donor-invariance fix)
- `src/tcell_pipeline/config.py` — Module 5 constants (`LR`, `WEIGHT_DECAY`, `MAX_EPOCHS`,
  `EARLY_STOP_PATIENCE`, `BATCH_SIZE`, `GRAD_CLIP`, `HUBER_DELTA`, `FOCAL_GAMMA`, `LAMBDA_DE/INV/GRAPH/GENE`,
  `DE_CALL_ZSCORE`, `DONOR_INVARIANCE`, `DONOR_INVARIANCE_SAMPLES`, `CHECKPOINTS_ROOT`, `LOGS_ROOT`)
- `src/tcell_pipeline/training/dataset.py` — `load_donor_pool` + `sample_donor_variants` (real per-donor
  `control_donor_profiles` resampling for the donor-invariance term)
- Round-3 cross-module edits (edge_confidences wiring): `graph/typed_graph_encoder.py`
  (`_edges_with_gates`/`encode_subgraph`/`forward` now also return per-edge source confidence),
  `model.py` (`out["edge_confidences"]`), `graph/run_module2_smoke.py` + `src/tests/test_graph.py` (3-tuple
  `forward` unpacking + a confidence-alignment assertion)
- `README.md` (Train the H1 predictor / Stage A section), `docs/specs/2026-07-16-module4-rationale-head.md`
  (Stage A cross-ref), the two gitignored planning docs (§8 / §Loss as-built notes)
- `feature_list.json` (feat-008 → Module-5 addendum, stays in-progress), `progress.md`, `session-handoff.md`

## Files Added (this session, Module 4 — Sparse Predictive-Rationale Head)

- `src/tcell_pipeline/rationale/__init__.py`, `rationale_head.py`, `rationale_loss.py`, `faithfulness.py`,
  `matched_random.py`, `run_module4_smoke.py` (NEW package)
- `src/tests/test_rationale.py` (NEW, 9 synthetic tests)
- `src/tcell_pipeline/graph/typed_graph_encoder.py` — added `encode_subgraph(...)` (exposes final node
  states + accepts a per-edge gate mask); `encode_one` delegates to it (3-tuple contract preserved)
- `src/tcell_pipeline/config.py` — Module 4 constants (`RATIONALE_TOP_K`, `RATIONALE_TAU`, `LAMBDA_SPARSE`,
  `LAMBDA_SUFF`, `LAMBDA_NEC`, `LAMBDA_CONTRAST`, `N_MATCHED_CONTROLS`)
- `feature_list.json` (feat-008 → Module-4 addendum, stays in-progress), `progress.md`, `session-handoff.md`

## Files Added (prior session, Module 3 — Program Decoder)

- `src/tcell_pipeline/programs/__init__.py`, `program_basis.py`, `program_decoder.py`,
  `run_program_basis.py` (NEW package)
- `src/tcell_pipeline/model.py` (NEW — EGIPGModel M1+M2+M3)
- `src/tcell_pipeline/run_module3_smoke.py` (NEW — real-data e2e smoke)
- `src/tcell_pipeline/encoders/batch.py` (NEW — shared `build_encoder_batch` for the M1/M2/M3 smokes)
- `src/tests/test_programs.py` (NEW, 12 synthetic tests)
- `src/tcell_pipeline/config.py` — Module 3 constants (PROGRAM_DIM, PROGRAM_METHOD,
  PROGRAM_LOADINGS_PATH, PROGRAM_RESPONSE_PATH, PROGRAM_COL_PREFIX)
- `feature_list.json` (feat-005 + feat-008 → in-progress), `progress.md`, `session-handoff.md`
- `data/intermediate/{gene_program_loadings,program_response}.parquet` (gitignored; now the sparse_pca production fit)

## Post-review fixes (Module 3 — xhigh `/code-review`, all 13 findings resolved)

An xhigh workflow review of the Module 3 diff surfaced 13 verified defects; all resolved:
- **Correctness:** FastICA basis now returns `mixing_` (loadings), not `components_` (unmixing);
  `program_basis` buffer is `persistent=False` (a stale checkpoint can't clobber the gene-aligned B);
  `run_program_basis` fold-leak guard is now an independent `raise` (not a tautological `assert`);
  σ has a `1e-12` floor (no float32 underflow to 0); the expression-only variant drops the graph
  residual bias (clean §10.6 ablation); `load_program_basis` raises a clear error on duplicate gene
  symbols; the M3 smoke + orchestrator guards now cover every mart they read.
- **Cleanup:** `EGIPGModel` takes overridable `h_graph_dim/h_do_dim`; shared `build_encoder_batch`
  (encoders/batch.py) replaces the batch dict duplicated across the 3 smokes; shared `load_zscore_rows`
  helper; hoisted the decoder's `joint` concat; removed the dead `GENE_LEVEL_DIM` config alias.
- **Verified:** `./init.sh` 69 tests green; all three real-data smokes (M1/M2/M3) + the basis
  orchestrator re-run clean.

## Files Added (this session, feat-003 — leakage-safe splits)

- `src/tcell_pipeline/splits.py` (NEW), `src/tests/test_splits.py` (NEW, 8 tests)
- `docs/specs/2026-07-15-feat-003-leakage-safe-splits.md` (NEW): design doc (report-derived + empirical)
- `data/splits/{blocked_target_ood,random}.csv`, `{manifest,leakage_report}.json` (NEW, git-tracked frozen artifacts)
- `src/tcell_pipeline/config.py` — feat-003 constants (SPLITS_ROOT, SPLIT_ROLES/FRACTIONS/SEED, SEQ_SIM_COSINE_THRESHOLD, GROUP_SIZE_CAP, artifact paths)
- `feature_list.json` (feat-003 → done), `progress.md`, `session-handoff.md`

## Files Added (this session, feat-016 — Module 2)

- `src/tcell_pipeline/graph/{__init__,graph_builder,neighborhood_sampler,typed_graph_encoder,graph_readout,run_module2_smoke}.py` (NEW)
- `src/tests/test_graph.py` (NEW): 8 synthetic Module 2 tests
- `src/tcell_pipeline/config.py` — Module 2 constants (GRAPH_*, EDGE_*, N_RELATION_TYPES, COMPLEX/CONDITION_EMBED_DIM, RELATION_TYPES, PROTEIN_FEATURE_DIM)
- `feature_list.json` (feat-016 added, done), `progress.md`, `session-handoff.md`

## Files Changed (prior session, feat-015)

- `src/tcell_pipeline/embeddings_plm.py` (NEW): ESM-2 650M generator (resumable, GPU-aware)
- `src/tcell_pipeline/embeddings_pinnacle.py` (NEW): PINNACLE CD4-context -> UniProt mapper (Figshare download)
- `src/tcell_pipeline/config.py` — PINNACLE_EMBED_DIM 512->128; +PINNACLE_RAW_DIR/FIGSHARE_URL/CONTEXT
- `src/tests/test_encoders.py` — rewritten to real PLM+PINNACLE data (no synthetic parquets); 1796->1412
- `requirements.txt` — +fair-esm, +pyyaml (was undeclared), +cu126 torch install note
- `README.md` — GPU/cu126 setup note + "Precompute target embeddings" step; PINNACLE 128-d detail
- `src/tcell_pipeline/encoders/embedding_store.py` — docstring refresh (embeddings now generated)
- `src/tcell_pipeline/encoders/{context,perturbation}_encoder.py` — device-aware forward (runs on GPU when .to('cuda'))
- `src/tests/test_encoders.py` — +test_encoder_runs_on_gpu_when_available (skips without CUDA)
- `src/tcell_pipeline/run_module1_smoke.py` (NEW): full-mart real-data smoke, GPU-native (Module 1 analogue of run_module0.py)
- `feature_list.json` (feat-015 added, done), `progress.md`, `session-handoff.md`
- Prior session (feat-014): `src/tcell_pipeline/encoders/` package + config Module 1 constants + test_encoders.py

