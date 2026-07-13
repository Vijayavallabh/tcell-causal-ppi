# Session Handoff

## Current Objective

- Goal: Build the EG-IPG model for T cell perturbation response prediction
- Current status: Harness scaffold created, no implementation code yet
- Branch / commit: main / dcbacea

## Completed This Session

- [x] Updated README.md with EG-IPG naming and revised method framing
- [x] Created experiment plan roles breakdown document
- [x] Created harness scaffold (AGENTS.md, feature_list.json, progress.md, init.sh, session-handoff.md)

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Compile | `python -m compileall .` | Pending | No Python source yet |
| Tests | `python -m pytest` | Pending | No tests yet |

## Files Changed

- `README.md` — updated idea, description, and method sections
- `AGENTS.md` — created
- `feature_list.json` — created
- `progress.md` — created
- `init.sh` — created
- `session-handoff.md` — created

## Decisions Made

- Renamed model from EG-CProG to EG-IPG throughout
- Removed causal/counterfactual language per report revision

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
