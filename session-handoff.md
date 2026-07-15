# Session Handoff

## Current Objective

- Goal: Build the EG-IPG model for T cell perturbation response prediction
- Current status: **Module 0 data pipeline complete and run end-to-end on real data.** feat-001,
  feat-002, feat-004 done. Next: feat-003 (leakage-safe splits).
- Branch / commit: main @ eab027e

## Completed This Session (commits e453964..eab027e)

- [x] UniProt one-to-many disambiguation (`id_mapping.choose_uniprot`): reviewed human canonical via
  UniProt REST (gene_exact + reviewed), pick by annotation-score then lexical. 33 multi-accession genes
  -> 23 resolved, 10 genuine multi-product loci flagged `uniprot_ambiguous`; alternatives preserved in
  `uniprot_alternatives`. Gene stays the perturbation unit. Added `uniprot_alternatives`/`uniprot_ambiguous`
  columns. `mygene` declared in requirements.txt.
- [x] HuRI download fixed: apex host `interactome-atlas.org` (TLS cert invalid for `www.` subdomain).
- [x] CORUM download fixed: old `coreComplexes.txt.zip` path is gone (CORUM 5.x -> SPA + fastapi). Migrated
  to `fastapi-corum/public/file/download_current_file?file_id=human&file_format=txt`; handle new
  `subunits_gene_name` schema via shared `_corum_gene_col`; per-source TLS-verify skip (broken cert chain).
- [x] feature_availability: `config.KNOWN_METADATA_COLS` allowlist so the leakage-fence REVIEW warning
  fires only on genuinely-unexpected metadata.
- [x] Ran full Module 0 (`run_module0.py`) on real data — all 7 steps green (see Evidence).
- [x] 23 pytest tests (added test_complex_membership.py); `init.sh` green.

Prior sessions: ~100 GB aggregate download, `examples/` inspectors, README, Module 0 implementation +
xhigh code-review fixes, 2026-07-14 report literature refresh.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Compile + tests | `./init.sh` | Pass | 23 passed; compileall clean |
| Module 0 full run | `python src/tcell_pipeline/run_module0.py` | Pass | all 7 steps completed on real data |
| id_mapping | step 1 | Pass | 12311 Ensembl; 23 UniProt resolved / 10 flagged; 6 no-hit (HGNC-resolved) |
| de_extraction | step 2 | Pass | 6 layers (zscore/log_fc NPZ; neglog10_p/adj_p, baseMean, lfcSE NPY); de_obs 33983 / de_var 10282 |
| ppi_graph | step 3 | Pass | 7,980,907 edges from 5 sources |
| complex_membership | step 4 | Pass | 18,932 memberships / 5,628 complexes (CORUM 5.3) |
| perturbation_table | step 5 | Pass | 33983 rows; 187 without UniProt |
| control_profiles | step 6 | Pass | 11018 NTC rows; 32425/33983 rows have a target baseline |
| feature_availability | step 7 | Pass | q_pre=43 / q_post=13 / metadata=2; leakage fence disjoint |

## Files Changed

- `src/tcell_pipeline/id_mapping.py`, `ppi_graph.py`, `complex_membership.py`, `feature_availability.py`,
  `config.py` — see Completed This Session
- `src/tests/test_complex_membership.py` (NEW); test_id_mapping / test_feature_availability cases
- `requirements.txt` (`mygene`), `feature_list.json`, `progress.md`, `session-handoff.md`

## Decisions Made

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
