# AGENTS.md

Project harness for reliable agent-assisted development in a python codebase.

## Startup Workflow

Before writing code:

1. **Confirm working directory** with `pwd`
2. **Read this file** completely
3. **Read `README.md`** for project overview, data sources, and setup instructions
4. **Run `./init.sh`** to verify environment is healthy
5. **Read `feature_list.json`** to see current feature state
6. **Review recent commits** with `git log --oneline -5`

If baseline verification is failing, repair that first before adding new scope.

## Project Context

This project builds the EG-IPG model (Evidence-Gated Intervention-Informed Protein-Program Graph)
for predicting T cell perturbation responses. See `README.md` for the full description and
`perturbation_informed_causal_protein_program_graphs_report.md` for the detailed experiment plan.

Key facts:
- Python 3.12 with [uv](https://docs.astral.sh/uv/) for environment management
- Data lives under `data/raw/` (gitignored, ~100 GB) — see README for download instructions
- Derived artifacts go under `data/intermediate/`, `data/graphs/`, `data/splits/`, `data/results/`, `data/checkpoints/`
- Only `data/manifests/` and `data/splits/` are tracked in git
- The model name is **EG-IPG**, not EG-CProG (legacy name in some older comments)
- Feature availability is split into `q_pre` (prediction-time, eligible) and `q_post` (response-derived, prohibited as H1 input) — see README
- All response-derived transformations (program bases, scaling, feature selection) must be fit inside training folds only

## Working Rules

- **One work-unit at a time** (one feature at a time by default): Pick exactly one unfinished feature from `feature_list.json`. A single
  architecture module may legitimately span more than one feature entry (e.g. Module 3 = feat-005
  program extraction + feat-008 decoder scaffold) — keep them to one session/commit and mark each
  `in-progress` until its own done-criteria are met. Don't open a second, unrelated module in parallel.
- **Verification required**: Don't claim done without running `./init.sh`
- **Update artifacts**: Before ending session, sync `progress.md`, `feature_list.json`, AND
  `session-handoff.md` — all three must match committed reality (a structurally valid but stale
  state file silently misroutes the next session)
- **Stay in scope**: Don't modify files unrelated to the current feature
- **Leave clean state**: Next session must be able to run `./init.sh` immediately

## Required Artifacts

- `feature_list.json` — Feature state tracker (source of truth)
- `progress.md` — Session continuity log
- `init.sh` — Standard startup and verification path
- `session-handoff.md` — For multi-session work

## Definition of Done

A feature is done only when ALL of the following are true:

- [ ] Target behavior is implemented
- [ ] `./init.sh` passes (compiles and tests run)
- [ ] Evidence recorded in `feature_list.json` or `progress.md`
- [ ] Repository remains restartable from standard startup path

## End of Session

Before ending a session:

1. Update `progress.md` with current state
2. Update `feature_list.json` with new feature status
3. Update `session-handoff.md` (completed work, evidence, blockers, recommended next step)
4. Record any unresolved risks or blockers
5. Commit with descriptive message once work is in safe state
6. Leave repo clean enough for next session to run `./init.sh` immediately

Cross-check: whenever `feature_list.json` status changes, `progress.md` and `session-handoff.md`
must change in the same commit. Structural validators pass on stale docs — content drift is on you.
- In the handoff's commit pointer, name the work-commit **range or state**, not a single HEAD hash —
  the docs-sync commit invalidates any exact hash you write (don't create a "fix commit hash" churn commit).
- Per-feature `evidence` in `feature_list.json` is a point-in-time completion snapshot; do not retro-edit
  its test counts when later, unrelated work changes the live total.

## Verification Commands

```bash
# Full verification (recommended)
./init.sh
```

Required checks:
- `uv run python -m pytest` (if tests exist)
- `uv run python -m compileall .`

## Command Safety

- **Safe to re-run anytime** (idempotent / deterministic): `./init.sh`, `pytest`, `compileall`,
  `run_module1_smoke.py`, `run_module2_smoke.py`, `run_module3_smoke.py`, `python -m tcell_pipeline.splits`,
  `python -m tcell_pipeline.programs.run_program_basis` (fixed seed; writes only the gitignored
  `data/intermediate/{gene_program_loadings,program_response}.parquet`).
- **DESTRUCTIVE — do NOT run to "test":** `run_module0.py` (and its steps) re-download multi-GB
  PPI DBs and **overwrite the frozen marts in `data/`** that all downstream work depends on. Run
  only when deliberately regenerating source data.
- **Direct module runs need `PYTHONPATH=src`** (e.g. `PYTHONPATH=src python -m tcell_pipeline.splits`).
  `pytest`/`compileall` don't — `conftest.py` handles them.

## Escalation

If you encounter:
- **Architecture decisions**: Consult README.md and the experiment plan report
- **Unclear requirements**: Check the experiment plan report, otherwise ask user
- **Repeated test failures**: Update progress, flag for human review
- **Scope ambiguity**: Re-read `feature_list.json` for definition of done
