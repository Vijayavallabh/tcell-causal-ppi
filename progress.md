# Session Progress Log

## Current State

**Last Updated:** 2026-07-14
**Active Feature:** feat-001 - Environment & Data Download

## Status

### What's Done

- [x] README.md updated with EG-IPG naming and revised framing
- [x] README.md expanded with full experiment plan details: target representations, graph schema, loss function, training splits, baselines, evaluation metrics, DE stats field listings, q_pre/q_post distinction, pseudobulk/guide/donor MuData details, storage budget, derived tables, loading rules, PPI source table, sanity checks, responsible use
- [x] requirements.txt: renamed EG-CProG to EG-IPG in comments
- [x] Harness scaffold created and validated (100/100 on all five subsystems)
- [x] Harness Engineering skill install instructions added to README
- [x] .gitignore updated to exclude .claude-private, .claude, .agents, skills-lock.json, and claude-me
- [x] Harness re-validated (100/100) and init.sh passes

### What's In Progress

- [ ] feat-001: Environment & Data Download
  - Details: Verify uv environment installs cleanly and data download script runs
  - Blockers: Data files (~100 GB) not yet downloaded

### What's Next

1. Run `uv pip install -r requirements.txt` and verify imports
2. Start data download from S3 (see README)
3. Verify AnnData can open the DE stats file with backed reads

## Blockers / Risks

- [ ] Data download requires ~100 GB of free disk space on the remote node
- [ ] No Python source code exists yet — first feature builds the initial scripts

## Decisions Made

- **Model name**: Renamed from EG-CProG to EG-IPG across README and requirements
- **Causal language**: Removed "causal" and "counterfactual" framing per report revision
- **Feature availability**: Added q_pre/q_post distinction to README for leakage prevention
- **Raw cell-level size**: Corrected from ~1.58 TiB to ~1,617 GiB per report figures

## Files Modified This Session

- `.gitignore` — added .claude, .agents, skills-lock.json to exclusions
- `progress.md` — updated session log
- `session-handoff.md` — updated handoff with current commit

## Notes for Next Session

- Read the experiment plan report for detailed specs on any feature
- Start with feat-001 (environment check) before any modeling work
