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

- **One feature at a time**: Pick exactly one unfinished feature from `feature_list.json`
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

## Verification Commands

```bash
# Full verification (recommended)
./init.sh
```

Required checks:
- `uv run python -m pytest` (if tests exist)
- `uv run python -m compileall .`

## Escalation

If you encounter:
- **Architecture decisions**: Consult README.md and the experiment plan report
- **Unclear requirements**: Check the experiment plan report, otherwise ask user
- **Repeated test failures**: Update progress, flag for human review
- **Scope ambiguity**: Re-read `feature_list.json` for definition of done
