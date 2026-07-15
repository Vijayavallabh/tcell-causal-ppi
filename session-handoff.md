# Session Handoff

## Current Objective

- Goal: Build the EG-IPG model for T cell perturbation response prediction
- Current status: **Module 0 done + Module 1 encoder (feat-014) done + real PLM/PINNACLE embeddings
  ingested on GPU (feat-015).** feat-001, feat-002, feat-004, feat-014, feat-015 done.
  Next: feat-003 (leakage-safe splits).
- Branch / commit: main — this session's work sits on top of the prior session (ended at 31c507a):
  feat-015 feature commit (a5bcf1d) + doc follow-ups (requirements/README, embedding_store docstring);
  a docs-sync commit closes the session on top. Latest is always `git log -1` on main.

## Completed This Session (feat-015 — real PLM + PINNACLE embeddings, on GPU)

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

Prior session (feat-014): `src/tcell_pipeline/encoders/` package — five nn.Modules fused into `h_do` R^256,
q_pre inputs only, no trainable gene-ID embedding, no free donor-ID embedding, leakage fence at the boundary,
NaN guard. Earlier: ~100 GB download, `examples/`, README, Module 0 + code-review fixes, UniProt/HuRI/CORUM.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Compile + tests | `./init.sh` | Pass | 37 passed on torch cu126; compileall clean |
| Encoder tests (real data) | `pytest src/tests/test_encoders.py` | Pass | 10 passed; real PLM+PINNACLE parquets, real marts — no synthetic parquets |
| PLM generation (GPU) | `python -m tcell_pipeline.embeddings_plm` | Pass | 11419/11419 proteins, 1280-d, finite; A100, 100% util |
| PINNACLE ingestion | `python -m tcell_pipeline.embeddings_pinnacle` | Pass | 1119 embeddings (128-d), 1070/11419 mart coverage (CD4 helper context) |
| Encoder real-data e2e | head of perturbation_condition/de_obs -> PerturbationEncoder | Pass | h_do (8,256) finite; real PLM+PINNACLE vectors flow through |
| Module 1 full-mart smoke | `python src/tcell_pipeline/run_module1_smoke.py` | Pass | all 33,983 real rows finite; PLM 33796, PINNACLE 3135 coverage; q_post rejected |
| Module 0 full run (prior) | `python src/tcell_pipeline/run_module0.py` | Pass | all 7 steps on real data; 7.98M edges; leakage fence disjoint |

## Files Changed (this session, feat-015)

- `src/tcell_pipeline/embeddings_plm.py` (NEW): ESM-2 650M generator (resumable, GPU-aware)
- `src/tcell_pipeline/embeddings_pinnacle.py` (NEW): PINNACLE CD4-context -> UniProt mapper (Figshare download)
- `src/tcell_pipeline/config.py` — PINNACLE_EMBED_DIM 512->128; +PINNACLE_RAW_DIR/FIGSHARE_URL/CONTEXT
- `src/tests/test_encoders.py` — rewritten to real PLM+PINNACLE data (no synthetic parquets); 1796->1412
- `requirements.txt` — +fair-esm, +pyyaml (was undeclared), +cu126 torch install note
- `README.md` — GPU/cu126 setup note + "Precompute target embeddings" step; PINNACLE 128-d detail
- `src/tcell_pipeline/encoders/embedding_store.py` — docstring refresh (embeddings now generated)
- `src/tcell_pipeline/run_module1_smoke.py` (NEW): full-mart real-data smoke (Module 1 analogue of run_module0.py)
- `feature_list.json` (feat-015 added, done), `progress.md`, `session-handoff.md`
- Prior session (feat-014): `src/tcell_pipeline/encoders/` package + config Module 1 constants + test_encoders.py

## Decisions Made

- Module 1 batch contract: `PerturbationEncoder.forward` takes a dict with keys `uniprot_id` (list),
  `ppi_degree_physical/functional/complex`, `control_baseline_expr`, `culture_condition` (str names or
  long indices), `donor_pc` (a single (B,32) tensor — loader stacks donor_pc_00..31), `n_guides`,
  `single_guide_estimate`. Any q_post key raises. The Module 3 data loader builds this dict.
- **Embeddings are real (feat-015)**: PLM = ESM-2 650M (1280-d, mean-pooled), 100% mart coverage;
  PINNACLE = real published 128-d contextual vectors (`cd4-positive helper t cell`, config.PINNACLE_CONTEXT),
  1070/11419 coverage. Frozen + pluggable; artifacts gitignored, regenerate via the two embeddings_* modules.
- **GPU**: use `torch==2.13.0+cu126` on this host (CUDA-12.2 driver can't run the default cu13x wheel); the
  5x A100s are otherwise invisible to torch. Embedding generation runs on GPU.
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

- Start **feat-003 (leakage-safe train/val/test splits)**: block gene families, protein complexes, and
  close graph neighborhoods from leaking train->test; hash + freeze split files. All inputs are present
  (id_mapping, protein_edges, complex_membership, perturbation_condition). Before freezing H1, run the
  near-null-signal check on development data.
