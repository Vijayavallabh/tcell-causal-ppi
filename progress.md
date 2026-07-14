# Session Progress Log

## Current State

**Last Updated:** 2026-07-14 (report literature-freshness revision)
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
- [x] Report literature-freshness revision (2026-07-14) — ran an adversarially-verified deep-research
  pass (26 primary sources; 24 confirmed / 1 refuted) and applied inline edits throughout
  `perturbation_informed_causal_protein_program_graphs_report.md`: sharpened TxPert (peer-reviewed
  4-graph Exphormer architecture, +8-25% self-reported), added concrete Stable-Shift numbers +
  gene-space collapse + code caveat, added PerturbGraph, the PertAdapt "effectively-linear" finding,
  the CD4 near-null-signal regime, the Wasserstein/Energy-distance unreliability caveat, and the
  two-sided deep-vs-simple debate; +10 references, 3 new limitations, 2 new §F consequences. Not committed.

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
- [ ] **Near-null-signal regime (2026-07-14 finding):** a July 2026 tabular benchmark reports this CD4+ screen as near-null-signal (models barely beat the mean). H1 superiority may not be demonstrable — confirm a detectable above-mean, target-specific signal on development data before freezing H1, and treat a rigorous negative benchmark as a valid outcome

## Decisions Made

- **Data scope**: first paper uses the ~100 GB aggregate layer only; cell-level (~1.6 TiB) excluded
- **Suppl tables**: 3 come from S3, 12 from the GitHub analysis repo (not on S3); documented in README
- **Donor key**: physical CE codes, not batch-relative D1-D4 labels (mapping verified consistent)
- **Control source**: independent NTC controls come from pseudobulk (DE has no control rows)
- **Stable-Shift code (feat-010)**: first-party code unconfirmed as of 2026-07-14; the `Sajib-006/PerturbGraph` repo hosts the related PerturbGraph method, not Stable-Shift — plan for a row-compatible reimplementation
- **Distributional metrics**: do not use Wasserstein/Energy distance as a sole headline metric (unreliable in high-dim gene space); fold a dynamic-range check into the G2-MQ gate

## Files Modified This Session

- `perturbation_informed_causal_protein_program_graphs_report.md` — 2026-07-14 deep-research literature-freshness revision (inline edits throughout)
- `README.md` — near-null-signal feasibility note + Stable-Shift/TxPert comparator-availability caveat in Baselines
- `progress.md`, `session-handoff.md` — state sync for the report revision

(The prior session's data-inspection changes — `examples/` and the README data section — are already committed in 49663b1 / f2794dd. `feature_list.json` unchanged: the report revision alters no feature's status.)

## Notes for Next Session

- `examples/` scripts double as data-understanding docs; run any of them to re-verify a fact
- Read the experiment plan report for detailed specs on any feature — now carries a 2026-07-14 literature refresh
- feat-010 (external comparators): expect to reimplement Stable-Shift (first-party code unconfirmed); TxPert-public covers only the STRING/GO subset
- Before feat-011 screening / freezing H1, run the near-null-signal check (detectable above-mean signal on development data)
- Start feat-002 remainder (ID mapping) before splits/graph work
