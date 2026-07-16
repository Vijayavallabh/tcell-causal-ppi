# Module 5 (Loss + Training) — code-review record (feat-008)

Date: 2026-07-16 · Scope: `src/tcell_pipeline/training/` (`losses.py`, `dataset.py`, `trainer.py`,
`run_train.py`), the Module 5 `config.py` constants, `src/tests/test_training.py`, and the
`edge_confidences` wiring in `graph/typed_graph_encoder.py` + `model.py`. As-built spec:
`docs/specs/2026-07-16-module5-training.md`.

Module 5 went through **three review passes**; each surfaced a real defect in the donor-invariance term
that the previous "fix" had only relocated. The final state penalises the **variance of `Δz` across real
donors directly** and is verified to train via the encoder on real data.

## Pass 1 — adversarial workflow review of the initial diff (`912c4ab`)

3-dimension review (loss-math / data-leakage / training-loop) → per-finding verify.

- **Confirmed: `L_invariance` was inert.** The naive term grouped mart rows by `(target, condition)` and
  penalised the within-group spread — but Module 0 averages donor PCs to condition level
  (`control_profiles`), so the mart has no per-donor rows and every group is a singleton. The donor-pair
  objective was vacuously satisfied; the only groups that ever fired were paralogue HGNC collisions
  (GPR89A/GPR89B → `GPHRA`), an upstream feat-002 id_mapping artefact.
- **Refuted:** a claimed trainer device mismatch (the encoders self-place their sub-outputs).
- **Resolution (`8f83e05`):** the individual donors *do* survive in `control_donor_profiles.parquet`
  (**4 real donors** × 3 conditions). Reformulated to **donor resampling** — re-run the encoder under
  distinct real donor PC vectors and penalise the variance of `f_shared(Δz)` across them.

## Pass 2 — xhigh `/code-review` of `6dcf196..HEAD` (`912c4ab` + `8f83e05`)

6 finder angles + an independent verifier per (file, line) → **15 verified defects**. Confirmed
correctness defects fixed in `efbf00f`:

1. **CRITICAL — the donor term was degenerate.** Penalising `Var(f_shared(Δz))` with a *free*
   `f_shared = Linear(K,K)` is globally minimised at `W=0` (AdamW weight-decay + the variance objective
   both drive it there), so the term decayed back to inert without pressuring the encoder — the same
   inertness as Pass 1, in a new guise. **Fixed:** dropped `f_shared`; penalise `Var(Δz)` **directly** (no
   collapsible projection). Re-verified: trains *via the encoder*, 2.15 → 0.19 over 3 epochs on real data.
2. **Stochastic donor term leaked into validation** → `val_total` non-deterministic for frozen weights →
   early-stopping / best-checkpoint partly RNG. **Fixed:** donor variants computed in **train only**; val
   invariance is 0 and the val metric is deterministic.
3. **Silent no-op** when `control_donor_profiles.parquet` is absent (omitted from `run_train`'s `required`
   gate) while the log printed `donor_invariance=True`. **Fixed:** the profile parquet is in `required`
   when donor invariance is on (fail-fast) + an honest on/off log + a warning when requested-but-absent.
4. **`L_graph` sum-reduced** over batch×edges while every other term is mean-reduced → its weight scaled
   with batch size and subgraph density. **Fixed:** averaged over the batch.
5. **`torch.manual_seed()` global reseed** in `Trainer.__init__`. **Fixed:** dedicated `torch.Generator`s
   (donor resample + a seeded DataLoader shuffle); no global-RNG side effect.
6. **Empty split → opaque crash.** **Fixed:** a clear `ValueError` on a 0-example training set.
7. **DEHead sized from global `config.H_DO_DIM`** not the wrapped decoder. **Fixed:** `h_do_dim =
   model.decoder.h_do_dim`. Plus cheap guards (de_obs↔pc row-count check) + DRY (`DONOR_COLS` reused).

Refuted: the same false device mismatch. The lowest-severity log nit was dropped to the cap.

## Pass 3 — implement the 3 items Pass 2 flagged (`1300204`)

Pass 2 flagged three items as design decisions / out-of-scope; all were then implemented:

- **`Δz_true` train-vs-val mismatch → `z@B` for every row.** The old design trained against the sparse-PCA
  score `A` (`program_response`, train rows only) but measured validation against `z@B`, so the
  early-stopping metric wasn't the trained objective. Now one consistent fold-local target across splits;
  `program_response` is no longer a Stage-A training dependency (dropped from `PerturbationDataset` + the
  `run_train` required-gate). The `program_response` artifact still exists as feat-005 output.
- **`edge_confidences` wired into `L_graph`.** The per-edge source confidence (edge-feature score column,
  clipped [0,1]) is threaded `TypedGraphEncoder.forward` (now returns `(h_graph, edge_gates,
  edge_confidences)`) → `EGIPGModel.forward` (`out["edge_confidences"]`) → Trainer → `StageALoss._graph`,
  so the unsourced-reliance term down-weights well-sourced edges instead of treating every edge as
  unsourced. Real data: L_graph 21148 → 17977 once confidences are applied.
- **`Subset` silent-disable** → `Trainer._resolve_donor_pool` walks wrapper `.dataset` chains so a wrapped
  training set doesn't silently drop donor invariance.

## Verification

`./init.sh` green at **92 tests** (zero warnings) after each pass. Full real-data run on an A100: Module 1
33,983 rows @24.7k/s; Modules 2/3 on GPU; Module 4 on the real PPI graph; Module 5 Stage A on the **full
train fold** (21,262 / 4,400), 3 epochs — donor-invariance 0.113 → 0.0016 (encoder learning it), val
invariance 0.0; full-graph run exercised the wired `edge_confidences`.

## Still open (by design, not defects)

- A shared/nuisance decomposition (`Δz = Δz_shared + Δz_nuisance(d)`) that preserves *legitimate*
  donor-specific signal needs a paired nuisance head — deferred; the direct `Var(Δz)` penalty is the
  correct non-degenerate realisation for a single-output predictor on donor-averaged data.
- Stage-A training is data-loading / CPU-bound (the per-subgraph graph loop; donor resampling ~triples
  the graph cost). PyG mini-batching + a batched loader + a donor node-state cache are the documented
  `ponytail:` upgrades to make training GPU-bound.
