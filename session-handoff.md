# Session Handoff

## Current Objective

- Goal: Build the EG-IPG model for T cell perturbation response prediction
- Current status: README and harness fully updated, no implementation code yet
- Branch / commit: main / e8df55d

## Completed This Session

- [x] Updated README.md with EG-IPG naming and revised method framing
- [x] Expanded README with full experiment plan details (data, method, layout, responsible use)
- [x] Fixed legacy EG-CProG references in requirements.txt
- [x] Created and validated harness scaffold (100/100 across all five subsystems)
- [x] Added Harness Engineering skill install instructions to README

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Compile | `uv run python -m compileall .` | Pass | No Python source yet |
| Tests | `uv run python -m pytest` | Skipped | No tests yet |
| Harness | `validate-harness.mjs` | 100/100 | All five subsystems pass |

## Files Changed

- `README.md` — expanded idea, method, data, repository layout, responsible use
- `requirements.txt` — renamed EG-CProG to EG-IPG in comments
- `AGENTS.md` — created and updated
- `feature_list.json` — created
- `progress.md` — created and updated
- `init.sh` — created with uv
- `session-handoff.md` — created and updated

## Decisions Made

- Renamed model from EG-CProG to EG-IPG throughout
- Removed causal/counterfactual language per report revision
- Added q_pre/q_post feature availability distinction to README
- Corrected raw cell-level size from ~1.58 TiB to ~1,617 GiB

## Blockers / Risks

- Data not yet downloaded (~100 GB)
- No implementation code exists yet

## Next Session Startup

1. Read `AGENTS.md`.
2. Read `feature_list.json` and `progress.md`.
3. Review this handoff.
4. Run `./init.sh` or the documented verification command before editing.

## Recommended Next Step

- Start feat-001: verify uv environment installs, then begin data download from S3
