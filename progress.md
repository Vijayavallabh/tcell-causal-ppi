# Session Progress Log

## Current State

**Last Updated:** 2026-07-14
**Active Feature:** feat-001 - Environment & Data Download

## Status

### What's Done

- [x] README.md updated with EG-IPG naming and revised framing
- [x] Experiment plan roles and expertise document created
- [x] Harness scaffold created (AGENTS.md, feature_list.json, progress.md, init.sh, session-handoff.md)

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

## Files Modified This Session

- `README.md` — updated idea, description, and method sections
- `AGENTS.md` — created project harness instructions
- `feature_list.json` — created 13 features from experiment plan phases
- `progress.md` — created session log
- `init.sh` — created verification script
- `session-handoff.md` — created handoff template

## Notes for Next Session

- Read the experiment plan report for detailed specs on any feature
- The `experiment_plan_roles_and_expertise.md` file defines who should work on what
- Start with feat-001 (environment check) before any modeling work
