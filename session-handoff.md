# Session Handoff

## Current Objective

- Goal: Build the EG-IPG model for T cell perturbation response prediction
- Current status: **Module 0 pipeline done + Module 1 Perturbation & Context Encoder done (feat-014).**
  feat-001, feat-002, feat-004, feat-014 done. Next: feat-003 (leakage-safe splits).
- Branch / commit: main @ HEAD (this session's Module 1 commit)

## Completed This Session (Module 1 — Perturbation & Context Encoder, feat-014)

New package `src/tcell_pipeline/encoders/` — five nn.Modules fused into `h_do` in R^256, q_pre inputs only:

- [x] `PluggableEmbeddingStore` (`embedding_store.py`): frozen PLM (1280) / PINNACLE (512) lookup by
  UniProt accession; loads parquet if present else returns zero vectors, in-memory cache, validates dim.
  NOT an nn.Module (data loader, not a trainable parameter) — real embeddings plug in by dropping the
  file at `data/intermediate/{plm,pinnacle}_embeddings.parquet`.
- [x] `TargetEncoder`: NO trainable gene-ID embedding (H1 prohibited). h_target R^1796 = PLM 1280 +
  PINNACLE 512 + ppi_degree_physical/functional/complex + control_baseline_expr.
- [x] `ContextEncoder`: trainable `nn.Embedding(3,64)` for Rest/Stim8hr/Stim48hr + donor_pc_00..31 through
  `Linear(32,32)` (NO free donor-ID embedding, so leave-one-donor-out stays valid). h_context R^96.
- [x] `QualityEncoder`: n_guides + single_guide_estimate + zeros(64) guide-seq placeholder. h_quality R^66.
- [x] `PerturbationEncoder`: `forward(batch_dict) -> (B,256)`; fusion `Linear(1958->256)` + LayerNorm;
  rejects any q_post column at the module boundary (ValueError) — leakage fence. 503,264 trainable params.
- [x] NaN guard (`_tensor.as_float_vector` nan_to_num): missing control_baseline_expr (1558/33983) and
  n_guides can't poison the LayerNorm'd h_do. Ponytail: upgrade to fold-fit imputation in Module 3 loader.
- [x] config: PLM_EMBED_DIM/PINNACLE_EMBED_DIM/GUIDE_SEQ_EMBED_DIM/H_DO_DIM/CONDITIONS + embedding paths.
- [x] 10 tests (`test_encoders.py`) + real-data smoke feeding perturbation_condition/de_obs head -> finite.

Prior sessions: ~100 GB aggregate download, `examples/` inspectors, README, Module 0 implementation +
xhigh code-review fixes, UniProt disambiguation, HuRI/CORUM download fixes, 2026-07-14 report refresh.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Compile + tests | `./init.sh` | Pass | 36 passed; compileall clean (incl. 3 post-review fence-hardening tests) |
| Encoder unit tests | `pytest src/tests/test_encoders.py` | Pass | 10 passed (dims, zero-fallback, q_post rejected, NaN guard) |
| Encoder real-data smoke | head of perturbation_condition/de_obs -> PerturbationEncoder | Pass | h_do (4,256) finite with PLM/PINNACLE absent (zero-fallback), incl. NaN-baseline rows |
| Module 0 full run (prior) | `python src/tcell_pipeline/run_module0.py` | Pass | all 7 steps on real data; 7.98M edges; leakage fence disjoint |

## Files Changed

- `src/tcell_pipeline/encoders/` (NEW): `_tensor.py`, `embedding_store.py`, `target_encoder.py`,
  `context_encoder.py`, `quality_encoder.py`, `perturbation_encoder.py`, `__init__.py`
- `src/tcell_pipeline/config.py` — Module 1 constants
- `src/tests/test_encoders.py` (NEW, 10 tests)
- `feature_list.json` (feat-014 added, done), `progress.md`, `session-handoff.md`

## Decisions Made

- Module 1 batch contract: `PerturbationEncoder.forward` takes a dict with keys `uniprot_id` (list),
  `ppi_degree_physical/functional/complex`, `control_baseline_expr`, `culture_condition` (str names or
  long indices), `donor_pc` (a single (B,32) tensor — loader stacks donor_pc_00..31), `n_guides`,
  `single_guide_estimate`. Any q_post key raises. The Module 3 data loader builds this dict.
- Module 1: embeddings are FROZEN and pluggable — no PLM/PINNACLE parquet on disk yet, so the encoder
  runs on zero target-embeddings; drop the parquet at the config paths to activate, no code change.
- UniProt: reviewed-canonical pick; flag only equal-evidence ties; gene is the perturbation unit
- CORUM host has a broken TLS chain -> per-source verify skip for `corum` only
- Data scope: aggregate layer only; donor key = physical CE codes; controls from pseudobulk
- Near-null-signal regime (2026-07-14): confirm above-mean signal before freezing H1; negative result is valid
- Stable-Shift (feat-010): first-party code unconfirmed; plan a row-compatible reimplementation

## Blockers / Risks

- `data/raw` ~101 GB near the 105 GiB soft cap; derived marts now also on disk — watch before feat-005
- Near-null-signal regime: H1 superiority not guaranteed on this CD4+ screen
- (Resolved this session: HuRI + CORUM downloads; id_mapping UniProt/Entrez online pass)

## Next Session Startup

1. Read `AGENTS.md`.
2. Read `feature_list.json` and `progress.md`.
3. Review this handoff.
4. Run `./init.sh` before editing.

## Recommended Next Step

- Start **feat-003 (leakage-safe train/val/test splits)**: block gene families, protein complexes, and
  close graph neighborhoods from leaking train->test; hash + freeze split files. All inputs are present
  (id_mapping, protein_edges, complex_membership, perturbation_condition). Before freezing H1, run the
  near-null-signal check on development data.
