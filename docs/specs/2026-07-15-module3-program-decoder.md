# Module 3 — Program Decoder (feat-005 extraction + feat-008 decoder slice) (design + as-built)

Date: 2026-07-15 · Depends on: feat-003 (splits), feat-014/015 (Module 1 → `h_do`),
feat-016 (Module 2 → `h_graph`). Consumers: feat-008 (full model + Module 4), feat-009 (metrics).

Design source: `EG_IPG_architecture_walkthrough.md` §6 (Module 3: Program Decoder) and README
§Method → *Program decoder* / *Target representations*. Where the walkthrough and the feature name
disagree, the walkthrough wins (it is the more-recent authoritative plan).

## Purpose

Turn the two fused representations — the intervention vector `h_do` (Module 1) and the graph-context
vector `h_graph` (Module 2) — into the model's actual prediction targets: a latent **program-level
delta** `Δz`, a decoded **gene-level delta** `Δx`, and a per-program **uncertainty** `σ`. The program
axis is defined by a *fold-local* basis learned from the training-fold DE matrix only, so nothing
response-derived leaks across the split (README §Training splits, §q_pre vs q_post).

## Scope (walkthrough §6-justified)

- **In:** the fold-local program basis `Z_train ≈ A·Bᵀ` (§6.1), the two-pathway program-delta
  predictor with the graph/expression mixture gate (§6.1, §6.3), residual gene-level decoding (§6.2),
  uncertainty (§6.4), and the `EGIPGModel` that wires Modules 1+2+3 including the expression-only
  nested variant (§10.6).
- **Out (deferred):** Module 4 sparse predictive-rationale head, training losses, and the
  train/calibration loops (§6.5 method×K comparison and the shallow-VAE basis are also future work —
  the basis machinery supports them but the study itself is not run here).

## Architecture — `src/tcell_pipeline/programs/` + `src/tcell_pipeline/model.py`

### ProgramBasis — `programs/program_basis.py` (fold-local, NOT neural)

`fit_program_basis(Z_train, method, K) -> (B (G,K), A (N,K))` factorises the training-fold z-score
matrix `Z_train ≈ A·Bᵀ` (§6.1). `B` are gene→program loadings, `A` the per-perturbation program scores.
Method dispatch (§6.5): `sparse_pca` (default — `MiniBatchSparsePCA`, the scalable sparse variant),
`nmf`, `fastica` (ICA), `svd`. The gene axis of `B` is the **full** `de_var` order (10,282 genes); only
*rows* are subset for fold-locality, so `B` drops straight into the decoder buffer without realignment.

- `train_row_indices(split, pc, role="train")` — the fold-locality gate: train-role genes → DE row
  indices. The orchestrator asserts zero overlap with the `challenge` rows before fitting.
- `save_program_basis` / `save_program_response` — atomic Parquet writers (`gene_program_loadings`,
  `program_response`); `load_program_basis(gene_order=…)` reindexes `B` to a fixed gene axis (0-fill).

### ProgramDecoder — `programs/program_decoder.py` (`nn.Module`)

Inputs `h_graph (B,256)`, `h_do (B,256)`:

1. graph path `Δz_graph = Linear(512,K)([h_graph‖h_do])` (§6.1)
2. expression-only path `Δz_expr = Linear(256,K)(h_do)` (§6.3)
3. mixture `λ = σ(Linear(512,1)([h_graph‖h_do])) ∈ [0,1]`; `Δz = λ·Δz_graph + (1−λ)·Δz_expr` (§6.3)
4. gene decode `Δx = B·Δzᵀ + r`, `r = Linear(256,G)(h_graph)`; **B is a frozen `register_buffer`,
   not a `Parameter`** (§6.2)
5. uncertainty `σ = sqrt(softplus(Linear(512,K)([h_graph‖h_do])))` (§6.4)

Output dict `{delta_z (B,K), delta_x (B,G), sigma (B,K), lambda (B,1)}`. Passing `h_graph=None` runs
the expression-only nested variant: `λ` is pinned to 0, `Δz = Δz_expr`, and `r` collapses to its bias.

### EGIPGModel — `model.py` (`nn.Module`)

Wraps `PerturbationEncoder` + `TypedGraphEncoder` + `ProgramDecoder`. `forward(batch, target_genes,
conditions)` returns the decoder dict plus `h_do`, `h_graph`, `edge_gates`. `graph_encoder=None` selects
the expression-only nested member (§10.6): no graph pass, `h_graph`/`edge_gates` are `None`, `λ=0`.
`EGIPGModel.from_saved_basis(gene_order, path)` loads `B` from the loadings Parquet aligned to a fixed
gene axis. B stays frozen (buffer) so the whole model is one `.to(device)` from CPU or CUDA.

## Fold-locality (leakage fence)

The program basis is a response-derived transform, so it may see **train rows only** (README §q_pre vs
q_post: "all response-derived transformations … fit inside training folds only"). `train_row_indices`
derives the eligible rows from `blocked_target_ood.csv`; `run_program_basis` asserts no `challenge`
overlap before fitting. Val/calibration/challenge responses never enter the fit.

## Config additions (`config.py`)

`PROGRAM_DIM = 128` (K; §6.5 sweep 64/128/256/512), `PROGRAM_METHOD = "sparse_pca"`,
`PROGRAM_LOADINGS_PATH` / `PROGRAM_RESPONSE_PATH` (under `data/intermediate/`, gitignored),
`PROGRAM_COL_PREFIX = "program_"`. The decoder's gene axis is derived from the loaded basis
`B.shape[0]`, not a config constant, so it always matches the fold-local loadings.

## Public interface

- `from tcell_pipeline.programs import fit_program_basis, train_row_indices, save_program_basis,
  save_program_response, load_program_basis, ProgramDecoder`
- `from tcell_pipeline.model import EGIPGModel`
- Orchestrator: `PYTHONPATH=src python -m tcell_pipeline.programs.run_program_basis [--method M] [--K K]`
- Real-data smoke: `python src/tcell_pipeline/run_module3_smoke.py`

## Verification (synthetic tests + real-data smoke)

- `src/tests/test_programs.py` (12 synthetic tests, dataless): basis shapes across all 4 methods,
  fold-local row selection, decoder output shapes, `λ∈[0,1]`, `σ>0`, `B` is a buffer (not a Parameter),
  `Δx = B·Δzᵀ + r`, expression-only variant, and a full `EGIPGModel` forward on the synthetic graph.
- Real-data smoke `run_module3_smoke.py`: fits a fast fold-local **SVD** basis on 21,262 real train
  rows (the `sparse_pca` default is a deliberate ~15-min run via `run_program_basis`), then forwards
  M1→M2→M3 on 4 real perturbations — all outputs finite, `λ∈[0.46,0.55]`, `σ>0`; expr-only `λ==0`.
- `./init.sh` green at 69 tests (57 prior + 12 new).

## Non-goals / ceiling markers

- **`sparse_pca` cost:** `MiniBatchSparsePCA` ≈ 90 s per 2 k rows → ~15 min on the full 21 k train set.
  The smoke and tests use `svd` (seconds) for speed; `run_program_basis` with no `--method` fits the
  paper-default sparse basis for the frozen production loadings.
- **`nmf` sees the positive part only** (`np.maximum(Z,0)`): z-scores are signed, so down-regulation
  programs are dropped. Split into signed ± channels if down-regulation modules matter (marked
  `ponytail:` in `program_basis.py`).
- **§6.5 comparison + shallow VAE** (feat-005 done-criterion) are not run here — only the extraction
  machinery that a comparison harness would call.
- **Module 4 / losses / training** (feat-008 remainder) are out of scope by design.
