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
- **Defer the measurement, not the fix**: when deferring a perf/correctness task, record what you
  MEASURED and how you measured it — never the fix you guessed. A handoff that said "mini-batch the
  graph encoders" prescribed a cure for 5% of the real bottleneck (sampling was 95%); the measurement
  would have routed the next session correctly, the prescription sent it the wrong way. Measure again
  before acting on an inherited diagnosis. A **sub-component benchmark is not a whole-pipeline
  estimate**: a 0.36 h/epoch encoder-only bench hid the real 3× per-step cost (donor-invariance
  re-forwards the whole model), so a 22.7 h campaign nearly launched as ~7 h. Time the real end-to-end
  path — one true step — before committing multi-hour compute.
- **Presence is not freshness — check a results file's provenance before reading it as this run's.** A
  parquet / log / checkpoint at the expected path may be a PRIOR run's; treating it as the current
  result silently reports stale numbers as fact. This session, stale `stage_a_history.json` and
  screening parquets sat at the live paths and twice nearly produced a false status report. Gate on
  mtime-vs-launch or a run-id recorded IN the artifact (the `--merge`/`--promote` guard cross-checks the
  registry), never on the file merely existing. And **provenance is not comparability**: a metric read
  from a frozen artifact (the full-fold H1 in `promoted.json`) is comparable to a freshly-scored one only
  if they share the fold/basis. A `--n-max` comparator run scored a capped fold and would have published
  its systema against the full-fold H1 as an authoritative same-fold verdict — apples-to-oranges. Gate the
  A-vs-B on a fold/basis match (`fold_comparable`), not on both numbers merely existing.
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
- [ ] **Every test pinning a correctness claim has been watched FAILING** — break the thing it
      guards, see red, restore, see green. A test you have not seen fail is not evidence, and
      `./init.sh` green is not sufficient on its own: it has certified broken code here twice
      (Module 8 pass-3 — "fixes that satisfied their own regression tests"; and the sampler's
      `_grow` sort, deletable with all 237 tests green while the neighbourhood silently changed
      for 35 of 60 targets). Mutate the ONE line the claim rests on, not just the ones you
      thought of. Hand-picked probe cases are a way of choosing what the test cannot see.
- [ ] **For a correctness-critical fix, try to CONSTRUCT an input that defeats it — mutation testing
      is necessary, not sufficient.** Watching a test fail and mutating the guarded line prove your
      test is load-bearing for the code you WROTE; neither can surface an input class your tests never
      build. This session every fix was watched-failing AND mutation-tested 10/10, and a 5-agent
      adversarial pass still found a real bug (a `tensor.data` write bypasses `_version`, so cache
      invalidation served a stale subgraph) — because no mutation of the code reaches an input the
      tests never construct. Before calling a correctness fix done, spend one adversarial pass whose
      job is to break it (or spawn agents to), especially on cache/staleness/concurrency invariants.
      A value **guarded** to be `None`/sentinel on a degenerate input is only guarded if EVERY consumer
      honors it: this session a verdict `print` did `None:+.4f` right after computing the guarded value,
      crashing on the exact degeneracy the guard existed for — trace the value from guard to output/JSON/
      downstream gate. And in a verdict, **`None` ≠ negative**: encoding "nothing to compare" as `False`
      ("H1 lost") misreports a converging negative — keep undecidable distinct from decided-against.
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
- Evidence is **append-only**, so a claim that later goes stale is superseded by a dated
  `CORRECTION (YYYY-MM-DD): ...` append naming what changed — never by editing the original line.
  (feat-008 carried "feat-007 still not-started" for two days after feat-007 shipped.) Check with:
  every feature's HEAD evidence must remain a strict prefix of its new evidence —
  `git show HEAD:feature_list.json` and compare.
- When you edit a tracked data file **programmatically**, match its existing serialization (encoding,
  indent, trailing newline) and confirm the diff is minimal BEFORE staging. A `json.dump(..., ensure_ascii=
  False)` rewrote every `\uXXXX` escape across the WHOLE `feature_list.json` — a 193 KB diff masquerading
  as a one-line append, silently breaking append-only integrity even though the text was byte-identical in
  meaning. `git diff --stat` is the tell: a data-file append should touch ~1 line, not the whole file.

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
