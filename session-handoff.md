# Session Handoff

## Current Objective

- Goal: Build the EG-IPG model for T cell perturbation response prediction
- Current status: Data downloaded + inspected; feat-001 done, feat-002 inspection done (ID mapping remaining)
- Branch / commit: main / 49663b1

## Completed This Session

- [x] Downloaded the ~100 GB aggregate layer to `data/raw/` (4 HDF5 + 15 suppl tables + 12 jsonld); cell-level excluded
- [x] Extended README download steps (S3 sync + GitHub fetch for the 12 analysis tables not on S3)
- [x] Added `examples/` — one self-checking inspector per artifact + `dataset_overview.py` + `inspect_analysis_tables.py`
- [x] Committed code/docs (data/raw is gitignored) as 49663b1
- [x] Marked feat-001 done and feat-002 in-progress with evidence

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Compile | `./init.sh` (compileall) | Pass | includes examples/ scripts |
| Tests | `pytest` | Skipped | no tests yet |
| Env imports | anndata/mudata/h5py | Pass | 0.13.1 / 0.3.10 / 3.16.0 |
| Inspectors | `python examples/inspect_*.py` | Pass | self-checks (asserts) green on real data |
| Inventory | `python examples/dataset_overview.py` | Pass | aggregate complete; 2 suppl tables unpublished |

## Files Changed

- `README.md` — supplementary-table download steps + analysis-tables section
- `examples/` — 8 inspection scripts + README
- `feature_list.json`, `progress.md`, `session-handoff.md` — state update

## Decisions Made

- Data scope: aggregate layer only for first paper; cell-level (~1.6 TiB) excluded (storage-blocked)
- Suppl tables: 3 from S3, 12 from GitHub analysis repo; 2 named tables are unpublished/unobtainable
- Donor key = physical CE codes; independent controls come from pseudobulk (DE has none)

## Blockers / Risks

- `data/raw` ~101 GB near the 105 GiB soft cap — watch before derived marts
- ID-mapping table (Ensembl-HGNC-UniProt-Entrez) not yet built — feat-002 remainder

## Next Session Startup

1. Read `AGENTS.md`.
2. Read `feature_list.json` and `progress.md`.
3. Review this handoff.
4. Run `./init.sh` before editing.

## Recommended Next Step

- Finish feat-002: build the Ensembl-HGNC-UniProt-Entrez ID-mapping table with a one-to-many /
  unmapped / deprecated ambiguity report for perturbed targets.
