# Session Handoff

## Current Objective

- Goal: Build the EG-IPG model for T cell perturbation response prediction
- Current status: Data downloaded + inspected; feat-001 done, feat-002 inspection done (ID mapping remaining). Planning report refreshed via a 2026-07-14 deep-research literature-freshness pass (uncommitted).
- Branch / commit: main / 67124cd (doc sync committed; the report revision lives on-disk and is gitignored)

## Completed This Session

- [x] Ran an adversarially-verified deep-research literature-freshness pass (26 primary sources; 24 confirmed / 1 refuted)
- [x] Applied inline edits throughout `perturbation_informed_causal_protein_program_graphs_report.md`:
  TxPert 4-graph Exphormer architecture + 8-25% (self-reported), concrete Stable-Shift numbers +
  gene-space collapse + code caveat, PerturbGraph, the PertAdapt "effectively-linear" finding, the CD4
  near-null-signal regime, Wasserstein/Energy-distance unreliability, and the two-sided deep-vs-simple
  debate; +10 references, 3 new limitations, 2 new §F consequences
- [x] Synced `README.md` (near-null-signal note + comparator-availability caveat), `progress.md`, and this handoff
- [x] Committed the doc sync as 67124cd (the report itself is gitignored and stays on-disk)

Prior session (committed as 49663b1 / f2794dd): downloaded the ~100 GB aggregate layer, added the `examples/`
inspectors, extended README download steps, and marked feat-001 done / feat-002 in-progress.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Compile | `./init.sh` (compileall) | Pass | includes examples/ scripts |
| Tests | `pytest` | Skipped | no tests yet |
| Env imports | anndata/mudata/h5py | Pass | 0.13.1 / 0.3.10 / 3.16.0 |
| Inspectors | `python examples/inspect_*.py` | Pass | self-checks (asserts) green on real data |
| Inventory | `python examples/dataset_overview.py` | Pass | aggregate complete; 2 suppl tables unpublished |
| Report revision | deep-research adversarial verify + `grep` consistency check | Pass | 24/25 claims confirmed; all 2026-07-14 additions present; Stable-Shift code caveat reconciled across report, README, and limitation 20 |

## Files Changed

- `perturbation_informed_causal_protein_program_graphs_report.md` — 2026-07-14 deep-research literature-freshness revision
- `README.md` — near-null-signal feasibility note + Stable-Shift/TxPert comparator-availability caveat
- `progress.md`, `session-handoff.md` — state sync (`feature_list.json` unchanged: no feature status changed)

## Decisions Made

- Data scope: aggregate layer only for first paper; cell-level (~1.6 TiB) excluded (storage-blocked)
- Suppl tables: 3 from S3, 12 from GitHub analysis repo; 2 named tables are unpublished/unobtainable
- Donor key = physical CE codes; independent controls come from pseudobulk (DE has none)
- Near-null-signal regime (2026-07-14): confirm a detectable above-mean signal before freezing H1; accept a negative benchmark as a valid outcome
- Stable-Shift first-party code unconfirmed; `Sajib-006/PerturbGraph` hosts PerturbGraph, not Stable-Shift (affects feat-010)

## Blockers / Risks

- `data/raw` ~101 GB near the 105 GiB soft cap — watch before derived marts
- ID-mapping table (Ensembl-HGNC-UniProt-Entrez) not yet built — feat-002 remainder
- Near-null-signal regime: models may not beat the mean on this CD4+ screen — demonstrable H1 superiority is not guaranteed (2026-07-14 finding)

## Next Session Startup

1. Read `AGENTS.md`.
2. Read `feature_list.json` and `progress.md`.
3. Review this handoff.
4. Run `./init.sh` before editing.

## Recommended Next Step

- Finish feat-002: build the Ensembl-HGNC-UniProt-Entrez ID-mapping table with a one-to-many /
  unmapped / deprecated ambiguity report for perturbed targets.
