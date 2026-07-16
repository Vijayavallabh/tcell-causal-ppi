# Module 5 — Loss + Training (feat-008 remainder) (design + as-built)

Date: 2026-07-16 · Depends on: feat-014 (Module 1 → `h_do`), feat-016 (Module 2 → `h_graph`,
`edge_gates`), feat-008 (Module 3 → `EGIPGModel`/`ProgramDecoder` → `Δz`, `Δx`, `σ`, `λ`), feat-005
(frozen fold-local basis `B` + `program_response` `A`), feat-003 (blocked split). Consumers: feat-011
(screening), feat-012 (predictive-rationale audit), feat-013 (reproducibility).

Design source: `EG_IPG_architecture_walkthrough.md` §8 (Full Loss Function) and
`perturbation_informed_causal_protein_program_graphs_report.md` §Loss Function. Where they disagree the
walkthrough wins (the more-recent authoritative plan).

## Purpose

Make the four model modules **trainable**. Module 5 supplies the Stage A optimisation objective + loop
for the H1 predictor (Module 1 + 2 + 3), the split-aware supervised dataset, and the Stage B calibration
loss (a loss module only — fitted after the H1 freeze). Module 4's `RationaleLoss` is the Stage B
rationale objective and is **not** reimplemented here.

## Two frozen stages (walkthrough §8.1)

- **Stage A — H1 predictor (frozen first).** `Trainer` fits Module 1 + 2 + 3 with `StageALoss`.
- **Stage B — secondary heads (after the freeze).** `StageBCalibrationLoss` (Gaussian NLL) + Module 4's
  `RationaleLoss`. Both are **loss modules only** — no fit loop here; the report freezes H1 before
  fitting calibration/rationale so joint fine-tuning can't change H1 predictions. The Stage-B *fit
  loops* remain feat-008's last piece.

## Scope

- **In:** `StageALoss` (response + gene Huber, focal-BCE DE head, donor-invariance, edge-gate graph
  regulariser), `StageBCalibrationLoss` (Gaussian NLL), `DEHead`, `PerturbationDataset` (split-aware,
  q_post-fenced), `Trainer` (AdamW, grad-clip, early stop, atomic checkpoints, per-epoch logs),
  `run_train.py` (Stage A orchestrator on real marts).
- **Out (deferred to feat-008 / feat-012):** the Stage-B calibration + rationale *fit loops*, the
  quality-weighted `L_repro` paired sensitivity analysis (H1 is unweighted by default, §8.4), top-k
  ranking loss, conformal calibration, and the near-null-signal freeze gate.

## Objective — `StageALoss` (`losses.py`, `nn.Module`, walkthrough §8.2)

    L_pred = L_response + lambda_gene * L_gene
           + lambda_DE * L_DE + lambda_inv * L_invariance + lambda_graph * L_graph

1. **Response reconstruction.** `L_response = Huber(Δẑ, Δz_true, δ=1)` (program level) and
   `L_gene = Huber(Δx̂, Δx_true, δ=1)` (gene level), both mean-reduced. Unweighted (`w_i = 1`) — the
   confirmatory H1 default (§8.4). `lambda_gene = 0.5`.
2. **DE classification.** `DEHead = Linear(256, 2·G)` over `h_do` → up / down logits. Targets are the
   per-gene DE up/down calls, derived from `Δx_true` (the DE z-score): `|z| ≥ DE_CALL_ZSCORE (1.645)` is
   the two-sided 10% tail — the proxy this dataset carries for `adj_p < 0.1`, since the raw `adj_p`
   layer is not part of the `__getitem__` contract. **Focal** BCE (`γ = 2`) so the abundant non-DE genes
   don't drown the rare affected ones. Numerically stable (`binary_cross_entropy_with_logits`,
   both-class modulation `(1 − p_t)^γ`). `lambda_DE = 0.1`.
3. **Donor invariance (real per-donor resampling).** The model's program prediction `Δz` must not
   depend on WHICH donor conditions the encoder. The training mart's `donor_pc` is only the *mean* of
   the real per-donor control profiles, but the individual donor vectors survive in
   `control_donor_profiles.parquet` (**4 real donors** × 3 conditions, verified). So each **train** step
   the Trainer re-runs the encoder under `DONOR_INVARIANCE_SAMPLES` distinct real donor PC vectors per
   row (donor resampling), and `L_invariance` penalises the **per-example variance of `Δz` across those
   donors, directly**. Penalising `Δz` itself — not a learnable projection `f_shared(Δz)` — is
   deliberate: a free `f_shared` is trivially minimised by collapsing its weights to 0 (weight decay +
   the variance objective both drive `W→0`), so it would decay back to inert without ever pressuring the
   encoder. Raw-`Δz` variance has no such degenerate solution — driving it down forces the encoder to
   emit donor-invariant predictions, consistent with the donor-averaged response target. **Verified on
   real data:** train invariance falls **2.15 → 0.19** over 3 epochs (via the encoder, since there is no
   free projection to collapse). Donor forwards run with the encoder in eval so DropEdge doesn't
   contaminate the signal (gradients still flow), and only in **train** (a stochastic resample kept out
   of the validation metric so early-stopping stays deterministic). `lambda_inv = 0.1`; off / no donor
   pool → a clean no-op. (A shared/nuisance split keeping legitimate donor-specific signal would need a
   paired nuisance head — deferred.)
4. **Graph regularisation.** From `edge_gates`: `graph_λ_sparse · Σ|ᾱ| + graph_λ_unsrc · Σ(1 − conf)·ᾱ²`.
   The sparsity term drives sparse condition gates; the second penalises reliance on low-confidence
   edges. The model output carries only the gates, so `conf` defaults to 0 (every edge treated as
   unsourced → a plain L2 on the gates); an optional `edge_confidences` argument down-weights the
   penalty for well-supported edges when a caller wires them in. `None` gates (expression-only) → 0.
   `lambda_graph = 0.01`.

## Stage B calibration — `StageBCalibrationLoss` (`losses.py`, §8.3)

Gaussian negative log-likelihood `0.5·Σ[log σ² + (Δz − Δẑ)²/σ²]` over the frozen H1 program deltas
(`F.gaussian_nll_loss`, `var = σ²`, eps-floored). **Loss module only** — fitted on the calibration
partition after the predictor freeze; no training loop here by design.

## Dataset — `PerturbationDataset` (`dataset.py`, `torch.utils.data.Dataset`)

Split-aware (`blocked_target_ood.csv` role filter). `__getitem__ → (batch_dict, target_gene, condition,
Δz_true, Δx_true, row_index)`.

- **Features are q_pre only.** `build_encoder_batch` assembles the encoder contract from
  `perturbation_condition` + `de_obs`; the leakage fence is enforced downstream (`PerturbationEncoder`
  raises on any `q_post` column). `q_post` never enters features.
- **`Δz_true`** — the precomputed program score `A` from `program_response` for train-fold rows (the
  exact target `B` was fit to); for out-of-fold rows (val / calibration / challenge, which have no `A`)
  it is the z-score projected onto the frozen loadings, `z @ B`. Keyed on availability in
  `program_response`, so it is correct even though `program_response` holds only train rows.
- **`Δx_true`** — the per-gene z-score row from the sparse DE layer (`zscore.npz`), sliced by
  `row_index`.
- `collate` merges per-sample encoder batches into one batched dict + parallel lists/tensors. All paths
  are constructor kwargs (default to `config`) so tests point at tiny fixtures.

## Trainer — `trainer.py`

AdamW(`lr = 1e-3`, `wd = 1e-5`) over the model **and** the loss's own parameters (the DE head +
`f_shared` live on `StageALoss`). The frozen basis `B` is a `persistent=False` decoder buffer, so it is
**neither optimised nor written to the checkpoint** (a stale checkpoint can't clobber the gene-aligned
basis). Gradient clip `max_norm = 1`, early stopping on validation total (`patience = 10`), atomic
best + last checkpoints to `data/checkpoints/`, per-epoch loss components to `data/logs/`. `run()`
returns `{best_ckpt, best_val, epochs_run, history}`. CPU-only default (`device="cpu"`); the encoders
self-place their sub-outputs so a GPU run also works.

## Orchestrator — `run_train.py`

`PYTHONPATH=src python -m tcell_pipeline.training.run_train --epochs N --batch-size B [--expr-only]
[--n-max K]`. Loads the split, builds train/val `PerturbationDataset`s, assembles `EGIPGModel` on the
real PPI graph with the frozen basis, trains Stage A, checkpoints. `--expr-only` fits the
expression-only nested variant (`graph_encoder=None`, `λ` pinned to 0). Pins `torch.set_num_threads(1)`
(the 64-core box thrashes torch's pool on tiny per-subgraph GNN ops).

## Config additions (`config.py`)

`LR = 1e-3`, `WEIGHT_DECAY = 1e-5`, `MAX_EPOCHS = 100`, `EARLY_STOP_PATIENCE = 10`, `BATCH_SIZE = 64`,
`GRAD_CLIP = 1.0`, `HUBER_DELTA = 1.0`, `FOCAL_GAMMA = 2.0`, `LAMBDA_DE = 0.1`, `LAMBDA_INV = 0.1`,
`LAMBDA_GRAPH = 0.01`, `LAMBDA_GENE = 0.5`, `DE_CALL_ZSCORE = 1.645`, `DONOR_INVARIANCE = True`,
`DONOR_INVARIANCE_SAMPLES = 2`, `CHECKPOINTS_ROOT`, `LOGS_ROOT`.

## Public interface

- `from tcell_pipeline.training import StageALoss, StageBCalibrationLoss, DEHead, PerturbationDataset,
  Trainer`
- Real-data Stage A run: `PYTHONPATH=src python -m tcell_pipeline.training.run_train`

## Verification (synthetic tests + real-data smoke)

- `src/tests/test_training.py` (11 synthetic tests, tiny fixture marts incl. a donor-profiles fixture +
  zero embedding stores): Stage A component shapes + gradient flow (reaching `h_do`, the DE head, and
  `f_shared` via the donor variants), the graph-gate penalty (confidence lowers the unsourced term),
  Stage B Gaussian-NLL + gradient, `DEHead` probabilities in `[0,1]`, the learnable `λ` mixture, the
  dataset contract (correct keys, the **q_post fence**, `program_response` vs out-of-fold `z@B`
  projection), the **donor pool + resampler** (distinct real donors), a 2-epoch checkpointed run, a
  parameter-update check, and the **real donor-invariance signal** (non-zero + trains `f_shared` when on,
  a clean 0 when off). Pins `torch.set_num_threads(1)`.
- Real-data orchestrator smoke: `run_train --expr-only --n-max 128 --epochs 2` trains, back-props, writes
  atomic checkpoints, and the **real 4-donor invariance term falls 1.30 → 0.25 in one epoch**; the
  full-graph `--n-max 4 --epochs 1 --no-donor-invariance` exercises `L_graph` on real `edge_gates`.
- `./init.sh` green at **92 tests** (79 prior + 13 new — incl. the graph-penalty batch-normalization and
  the empty-split guard added in the post-review round below).

## Post-review round 2 (xhigh `/code-review` of `6dcf196..HEAD`)

An xhigh workflow review (6 finder angles → per-(file,line) verify) surfaced 15 verified defects; the
confirmed correctness ones were fixed:

- **`L_invariance` was degenerate (critical).** Penalising `Var(f_shared(Δz))` with a free `f_shared`
  is trivially minimised at `W=0` (weight decay + the objective both drive it there), so the donor term
  decayed back to inert without pressuring the encoder — the same inertness, in a new guise. **Fixed:**
  dropped `f_shared`, penalise `Var(Δz)` directly (no collapsible projection; the encoder is forced to
  emit donor-invariant `Δz`). Verified it now trains via the encoder (2.15 → 0.19).
- **Stochastic donor term leaked into validation** → `val_total` non-deterministic → early-stopping /
  best-checkpoint partly RNG. **Fixed:** donor variants are computed in **train only**; `val` invariance
  is 0 and the val metric is deterministic.
- **Silent no-op when `control_donor_profiles.parquet` absent** (not in the `required` gate) while the
  log printed `True`. **Fixed:** the profile parquet is in `required` when donor invariance is on
  (fail-fast), and the log reports the actual on/off state + warns if requested-but-unavailable.
- **`L_graph` summed over batch×edges** while other terms are means → batch-size/density-dependent
  weight. **Fixed:** averaged over the batch.
- **`torch.manual_seed()` global reseed** in `Trainer.__init__`. **Fixed:** dedicated `torch.Generator`s
  (donor resample + a seeded DataLoader shuffle); no global-RNG side effect.
- **Empty split → opaque crash.** **Fixed:** a clear `ValueError` on a 0-example training set.
- **DEHead sized from the global `H_DO_DIM`**, not the wrapped decoder. **Fixed:** `h_do_dim =
  model.decoder.h_do_dim`. Plus cheap guards (de_obs↔pc row-count check) and DRY (`DONOR_COLS` reused).
- **Deferred / flagged:** the train (`program_response` sparse-PCA score `A`) vs out-of-fold (`z@B`
  projection) `Δz_true` mismatch is a **feat-005 modeling decision** (persist the fitted sparse-coder to
  give val rows a consistent sparse target, or switch to `z@B` everywhere) — not silently changed here.
  `edge_confidences` are still unwired (the unsourced `L_graph` term stays a plain L2 until per-edge
  source confidence is threaded from the graph output). Refuted: a claimed trainer device mismatch (the
  encoders self-place their sub-outputs).

## Post-review finding + resolution (adversarial workflow review of this diff)

A 3-dimension adversarial review (loss-math / data-leakage / training-loop) → per-finding verify
produced **1 confirmed finding**, 1 refuted, loss-math clean:

- **Confirmed → fixed properly, not just documented: the original `L_invariance` (group rows by
  `(target, condition)`) was inert.** The mart's `donor_pc` is the condition-level *mean* of the donors
  (`control_profiles`), so grouping the mart rows found no donor pair and the term was ~always zero (the
  only firings were paralogue HGNC collisions like GPR89A/GPR89B → `GPHRA`, an upstream feat-002 artefact).
  **Intelligent fix (this diff):** the individual donor vectors *do* survive in
  `control_donor_profiles.parquet` (4 real donors × 3 conditions — the mart just averaged them away). The
  loss was reformulated to **donor resampling**: re-run the encoder under distinct real donor PC vectors
  and penalise the variance of `f_shared(Δz)` across them (see §Objective item 3). This is a real, dense
  donor-generalisation signal that trains (1.30 → 0.25 in one epoch on real data), not an inert term.
  The paralogue HGNC collision is no longer relevant (grouping is gone). `ponytail:` each donor variant
  re-runs the donor-independent graph message passing; cache node states + re-run only readout+decoder if
  Stage A becomes graph-bound (the graph path is CPU-bound, so donor resampling ~triples its per-step cost
  — opt out with `--no-donor-invariance`; expression-only is cheap).
- **Refuted:** a claimed trainer device mismatch — moot CPU-only (spec), and the encoders self-place
  sub-outputs so a GPU run is correct too; `Δz_true/Δx_true` are moved to the model device explicitly.

## Non-goals / ceiling markers

- **No Stage-B fit loop** (calibration or rationale) — loss modules only, by design (H1 freezes first).
- **Donor resampling ~triples the (CPU-bound) graph Stage-A cost** — each step re-runs the encoder under
  the donor variants. Upgrade: cache the donor-independent graph node states and re-run only
  readout+decoder per donor (`ponytail:` in `trainer._donor_variants`). Opt out: `--no-donor-invariance`.
  Only **4 real donors** exist, so the invariance signal is bounded by that donor count.
- **DE labels are a z-score proxy** for `adj_p < 0.1` (`DE_CALL_ZSCORE` is the tuning knob) — the raw
  `adj_p` layer is deliberately outside the dataset's `__getitem__` contract.
- **`L_graph` unsourced term defaults to `conf = 0`** (full L2 on gates) — wire `edge_confidences`
  through if per-edge source confidence should down-weight well-supported edges.
