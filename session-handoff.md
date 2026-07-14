# Session Handoff

## Current Objective

- Goal: Build the EG-IPG model for T cell perturbation response prediction
- Current status: Module 0 data pipeline implemented (`src/tcell_pipeline/`, 9 modules + orchestrator + 12 tests). feat-001 done, feat-002 done (ID mapping), feat-004 in-progress (PPI harmonizer built, real fetch pending).
- Branch / commit: main (Module 0 implementation this session)

## Completed This Session

- [x] Implemented Module 0 data pipeline in `src/tcell_pipeline/`: config, id_mapping, de_extraction,
  perturbation_table, ppi_graph, complex_membership, control_profiles, feature_availability, run_module0.
  Each exposes `run()`; the orchestrator sequences them in dependency order.
- [x] Wrote `src/tests/` (5 files, 12 tests) exercising the pure builders on synthetic fixtures:
  id-map columns/ambiguity, DE clip+sparse/dense round-trip, contiguous row_index + unmapped-kept,
  PPI score∈[0,1]/binary flags/≥2 sources/dedup, q_pre⊥q_post disjointness.
- [x] `conftest.py` puts `src/` on the path; added `pytest` to `requirements.txt`.
- [x] Applied all 15 xhigh code-review findings (correctness + cleanup): NaN-safe NTC masking, donor-PCA
  full-rank single-pass fix, `-log10(p)` storage, BioGRID ZIP + HuRI coverage, PPI score floor, real
  `ppi_degree_*` from the graph (reordered ppi_graph before perturbation_table), DE schema asserts,
  NaN-safe id_mapping guard, metadata/donor guards. Added `test_control_profiles.py`.
- [x] Verified: `./init.sh` green (compileall + 18 pytest); `id_mapping.run()` on the real 16.8 GB DE file
  produced a 12311-row mapping + ambiguity report; `control_profiles` demo self-check passes.

Prior sessions (49663b1 / f2794dd / 67124cd): ~100 GB aggregate download, `examples/` inspectors,
README download steps, and the 2026-07-14 report literature-freshness revision (report is gitignored).

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Compile | `./init.sh` (compileall) | Pass | includes examples/ scripts |
| Tests | `pytest` | Skipped | no tests yet |
| Env imports | anndata/mudata/h5py | Pass | 0.13.1 / 0.3.10 / 3.16.0 |
| Inspectors | `python examples/inspect_*.py` | Pass | self-checks (asserts) green on real data |
| Inventory | `python examples/dataset_overview.py` | Pass | aggregate complete; 2 suppl tables unpublished |
| Module 0 tests | `./init.sh` (compileall + pytest) | Pass | 18 passed; compileall clean (post code-review fixes) |
| id_mapping real run | `python -m tcell_pipeline.id_mapping` | Pass | 12311 Ensembl (11526 targets / 10282 measured / 9497 both); all HGNC offline-resolved |
| control_profiles demo | `python -m tcell_pipeline.control_profiles` | Pass | NTC \bNTC\b spares KNTC1; PCA embed shape (3, 32) |

## Files Changed

- `src/tcell_pipeline/` (NEW) — 9-module Module 0 data pipeline + `run_module0.py` orchestrator
- `src/tests/` (NEW) — 5 test files (12 tests) on synthetic fixtures
- `conftest.py` (NEW), `requirements.txt` — pytest wiring
- `feature_list.json` — feat-002 done, feat-004 in-progress
- `progress.md`, `session-handoff.md` — state sync

## Decisions Made

- Data scope: aggregate layer only for first paper; cell-level (~1.6 TiB) excluded (storage-blocked)
- Suppl tables: 3 from S3, 12 from GitHub analysis repo; 2 named tables are unpublished/unobtainable
- Donor key = physical CE codes; independent controls come from pseudobulk (DE has none)
- Near-null-signal regime (2026-07-14): confirm a detectable above-mean signal before freezing H1; accept a negative benchmark as a valid outcome
- Stable-Shift first-party code unconfirmed; `Sajib-006/PerturbGraph` hosts PerturbGraph, not Stable-Shift (affects feat-010)

## Blockers / Risks

- `data/raw` ~101 GB near the 105 GiB soft cap — watch before the heavy Module 0 marts land (DE layers ~a few GB, control profiles)
- id_mapping UniProt/Entrez are `requires_online_lookup` — needs an online mygene.info pass to fill
- PPI/CORUM source downloads not yet fetched (network + large files); harmonizer is unit-tested but unrun on real edges
- Near-null-signal regime: models may not beat the mean on this CD4+ screen — demonstrable H1 superiority is not guaranteed (2026-07-14 finding)

## Next Session Startup

1. Read `AGENTS.md`.
2. Read `feature_list.json` and `progress.md`.
3. Review this handoff.
4. Run `./init.sh` before editing.

## Recommended Next Step

- Run the heavy Module 0 steps on real data via `python src/tcell_pipeline/run_module0.py` (or step-by-step
  `python -m tcell_pipeline.<step>`): `de_extraction` -> `perturbation_table` -> `ppi_graph` (fetch sources)
  -> `complex_membership` -> `control_profiles` -> `feature_availability`. Then do the online mygene.info
  pass to fill UniProt/Entrez, and move to feat-003 (leakage-safe splits).
