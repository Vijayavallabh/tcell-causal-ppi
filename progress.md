# Session Progress Log

## Current State

**Last Updated:** 2026-07-14
**Active Feature:** feat-002 - Data Inspection & ID Mapping (inspection done; ID mapping remaining)

## Status

### What's Done

- [x] feat-001 Environment & Data Download — **DONE**
  - Env imports OK (anndata 0.13.1, mudata 0.3.10, h5py 3.16.0)
  - Aggregate layer downloaded to `data/raw/` (~101 GB): 4 HDF5 + 15 suppl tables + 12 jsonld
  - README download steps extended: S3 sync (on-S3 suppl + metadata) + GitHub fetch (12 analysis tables not on S3)
  - Cell-level files intentionally excluded (storage-blocked)
- [x] Data inspection scripts added under `examples/` — one self-checking inspector per artifact,
  plus `dataset_overview.py` (provenance + local-vs-expected inventory) and
  `inspect_analysis_tables.py` (maps derived tables to EG-IPG roles)
- [x] README.md: supplementary-table download steps + "Supplementary analysis tables" section
- [x] init.sh passes (compileall clean, no tests yet)

### What's In Progress

- [ ] feat-002: Data Inspection & ID Mapping
  - Inspection portion: DONE (see `examples/`)
  - Remaining: build Ensembl-HGNC-UniProt-Entrez identifier mapping with ambiguity report
    (one-to-many, unmapped, deprecated IDs)

### What's Next

1. Build the ID-mapping table for perturbed targets (feat-002 remainder)
2. Then feat-003 (leakage-safe splits) and feat-004 (PPI graph construction)

## Blockers / Risks

- [ ] `data/raw` at ~101 GB sits right at the 105 GiB immutable-artifact soft cap — watch before derived marts land
- [ ] Two suppl tables (`QC_summaries_per_sample_lane.csv`, `Th1Th2_validation_summary.suppl_table.csv`) are unpublished on S3/GitHub — unobtainable

## Decisions Made

- **Data scope**: first paper uses the ~100 GB aggregate layer only; cell-level (~1.6 TiB) excluded
- **Suppl tables**: 3 come from S3, 12 from the GitHub analysis repo (not on S3); documented in README
- **Donor key**: physical CE codes, not batch-relative D1-D4 labels (mapping verified consistent)
- **Control source**: independent NTC controls come from pseudobulk (DE has no control rows)

## Files Modified This Session

- `README.md` — supplementary-table download steps + analysis-tables section
- `examples/` — new: 8 inspection scripts + README (data inspection)
- `feature_list.json`, `progress.md`, `session-handoff.md` — state update

## Notes for Next Session

- `examples/` scripts double as data-understanding docs; run any of them to re-verify a fact
- Read the experiment plan report for detailed specs on any feature
- Start feat-002 remainder (ID mapping) before splits/graph work
