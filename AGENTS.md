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
  path — one true step — before committing multi-hour compute. **A scaling exponent fitted on two
  points is not a cost model either.** feat-005 fitted sparse_pca at 1.08 in K from K=64→128; it held
  at K=256 and broke at K=512, predicting 52 min for a fit that really took 96 — so that cell blew a
  90 min cap and returned nothing. The error runs both ways: fastica was *assumed* linear and is
  actually FLAT in K (0.08 — ~90% of its cost is a K-independent whitening SVD), and acting on the
  assumption would have needlessly stripped two cells of their resamples. Extrapolate only within the
  range you measured, state the extrapolated cells AS extrapolations, and give any unattended sweep a
  per-unit cap plus a total budget so a wrong model costs one cell instead of the night.
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
- **A comparison you did not compute is not a result — and simultaneous tests need a multiplicity
  correction.** This session published "the frozen H1 sits BELOW no-graph" by reading two marginal
  per-config means off a ranking table. That pair was never in `CONTRASTS`; run as an actual paired
  contrast it was p=0.085 with the CI crossing zero — *indistinguishable*, not below. If you state "A
  beats / trails B", a test of A-vs-B must exist in the code that produced the report: overlapping
  summary intervals are not a substitute, and every honest-frame guard is silent on a comparison that
  was never made. Then, when several contrasts are tested at once, raw alpha inflates the family-wise
  error — report the corrected p and require it before calling anything "reliable" (a raw p=0.0208 hit
  here failed Bonferroni yet had been published as a resolved positive). Record BOTH correction methods:
  picking the one that rescues the claim after seeing the numbers is the look-elsewhere effect in a lab
  coat. Related: a baseline published as a floor the result must CLEAR is bounded by its own fit
  quality — "more regularisation only weakens it" is the safe direction for a *competitor* and the wrong
  one for a *floor*, because an under-fit bar inflates the very margin it exists to bound. Record its
  convergence/sparsity evidence instead of arguing the direction.
- **A correction that passes everything has told you nothing — check that it CAN discriminate.** The
  rule above says require the corrected p; that is necessary, not sufficient. In the feat-005 basis
  study all 16 paired contrasts ran over 21,262 rows, every raw p UNDERFLOWED to exactly 0.0, and so
  Bonferroni and Holm both "passed" all 16 — including one worth **−0.14 pp**. At large n,
  significance is automatic and "survives Bonferroni" becomes a decoration that reads as strength.
  Before leaning on a correction, ask what result it would have REJECTED; if the answer is none,
  say so and make the effect size the load-bearing column. Flag an underflowed p as a floor
  (`p_underflow`), never as certainty. And note WHICH axis carries the test: there the only axis with
  a real sampling distribution (in-sample reconstruction) was also the one most confounded by model
  capacity, while the informative axis had 3 resamples and could only be described — say that out
  loud rather than letting the tested axis pass for the important one.
- **Re-derive every number in a claims block from the artifact before you publish it — the refuting
  number is usually already in your own table.** feat-005 handed over an evidence block asserting
  sparse_pca was "the ONLY method producing any sparsity"; NMF had **55.7%** exact zeros against its
  22.7%, and that number was sitting in the study's own K=128 table two sections above the sentence.
  A reviewing session caught it. This is the same class as the uncomputed comparison above, one step
  worse: the comparison HAD been computed, recorded, and then contradicted in prose. Prose drifts
  from the table it came from. Before handing off, check the whole class programmatically (that block
  then verified 18/18), not just the claim someone happened to question — and grep for the claim's
  *meaning*, not its wording: the same falsehood sat in a second file phrased differently and a
  literal-string grep missed it.
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
      downstream gate **and to the process exit code** (`main()` later printed `FOLD MISMATCH … NOT
      comparable` and then `return 0`, so an unattended campaign and any exit-status CI gate recorded it
      green). And in a verdict, **`None` ≠ negative**: encoding "nothing to compare" as `False`
      ("H1 lost") misreports a converging negative — keep undecidable distinct from decided-against.
- [ ] **RUN THE REAL COMMAND. A test that MOCKS the thing it tests proves nothing about that thing.**
      A driver's test monkeypatched `main` and stayed green while the actual command was broken:
      `main(argv=None)` lets argparse fall back to `sys.argv`, so calling it from inside another driver
      inherited THAT driver's flags and died with `unrecognized arguments: --part repro`. Red-first,
      32/32 mutants and a 5-agent code review were all green first; only a second caller executing it for
      real found the bug — the fourth defect that day found by running rather than testing. Before done,
      invoke the feature the way a caller actually invokes it, end to end, and check the exit code.
- [ ] **A guard whose input is a CONSTANT can only confirm, and absence of evidence must never read as
      a pass.** Ask of every fence: *what input would make this FIRE?* If none can exist, it is
      decoration — the fold gate here compared a registry `split` field that the producer hardcodes, so
      no fold change could ever trip it, and a `--n-max` capped seed sailed through (the fix keyed it on
      recorded `n_train`/`n_val`, a value that actually varies). Then check the empty case: `set() <=
      {expected}` is vacuously true, so a missing registry published "single frozen fold: True" with
      zero evidence — unknown must be `None`, never green. The same inversion at the statistics layer is
      worse than a crash: zero variance across seeds was reported as `p=0.0, "CI excludes zero"`, turning
      the one condition that proves the inputs carry no information into the strongest possible evidence.
      Two further shapes of the same fault: a guard whose **expected value is derived from the thing under
      test** can also only confirm — hashing an artifact now and "checking" it against itself shows the file
      is readable, not that it reproduced, and a config check that hashed today's config against itself
      could never trip (fix: key expectations on an INDEPENDENTLY frozen record, and LABEL anything
      self-derived so it cannot read as a pass). A reference another live process rewrites is not
      independent either — an expected row count read from a concurrent session's results file follows that
      session silently. And `any([])` is False, so a whitelist-shaped verdict built from three `any()` calls
      returned REPRODUCIBLE on ZERO checks: a checkout that did not exist was certified reproducible.
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

## Committing from one of several concurrent sessions

When more than one session shares a checkout, commit ONLY your own files, by explicit path. Never
`git commit -a`, never `git add .` — both sweep up other sessions' half-finished work, and the tree may
hold thirty dirty paths belonging to four other agents.

    git commit -F <message-file> -- <path> <path> ...

**`-m` must come BEFORE `--`.** `git commit -- <paths> -m "msg"` fails with
`pathspec '-m' did not match any file(s)`: everything after `--` is a pathspec, so the message and its text
are read as filenames. It fails safely (nothing is committed, staging survives), but `-F <file>` or an
`-m` placed before the separator avoids it entirely.

Verify after committing, not before: `git show --name-only HEAD` contains nothing outside your lane, and
`git show HEAD:feature_list.json | sha256sum` is unchanged from `HEAD~1` — the DoD triad is merged once, by
the integrating session, so your commit must leave that anchor untouched for everyone else's evidence
blocks to append against.

## Claims about PROCESS are invisible to test discipline

Red-first tests, mutation testing and adversarial inputs all validate claims about **code**. They are
structurally blind to claims about the **world**: what another session holds, what a run finished, what a
file contains, what a number actually measured. Those are asserted in comments, docstrings, handoffs and
status reports, where nothing executes them — so they must be checked against the world (git, `ps`, the
filesystem, the artifact) at the moment they are written.

Every cross-session defect in the five-session night of 2026-07-20 was of this kind, and each was caught by
another session reading a justification, never by a test:

- *"a concurrent session held `config.py`, so `STAGE_B_ROOT` lives elsewhere"* — asserted as fact in a code
  comment; `config.py` was unmodified in git and claimed by nobody. The workaround existed for a lock that
  was never verified to exist.
- *"the frozen H1's gate mean is **exactly** 0.000000"* — `0.000000` was the 6-dp rendering of ~1.3e-07.
  The word "exactly" was added in a relay, converting a formatting artifact into a bit-zero claim that
  would have made every ablation identically zero, contradicting two sessions' measured residuals.
- *"the evidence block is ~4,651 chars"* — that was a different session's block, carried across without
  re-measuring. The real one was 7,665.
- *"TabICL is not decision-relevant"* — inferred from 3 of 128 probe outputs, using prediction MAGNITUDE as
  evidence on a metric that is scale-INVARIANT. It became the decisive bar.
- *"all the runs finished"* — asserted from having watched the logs, not from checking artifact integrity.
  One had been killed mid-flight; a cache-vs-table cross-check was what actually established it.

Practical rules: quote a number only from the artifact you just read, not from a sibling claim; prefer a
RATIO to a rendered value (a collapse factor survives formatting, `0.000000` does not); and when you
justify a decision by the state of the repo or another session, run the command that shows it and paste
the output. A justification nobody can execute is the one place this harness cannot help you.

### ...and the instrument that checks process state has its own blind spots

Two failures here were in the CHECK, not the claim — both silent, both in the direction that reads as
success:

- **Never poll for a process by matching its command line. Resolve the PID once, then ask the kernel.**

      PID=$(pgrep -f '[p]ython3 -m package.module' | head -1)      # match ONCE, at arm time, and eyeball it
      until ! kill -0 "$PID" 2>/dev/null; do sleep 60; done         # kernel: is THIS process alive?

  `kill -0` is immune to quoting, to sibling watchers, to diagnostics, and to the watcher's own cmdline.
  Everything else on this box failed, in escalating order, and each fix looked correct until the next
  session arrived:
  1. `until ! pgrep -f "job"` — matches the watcher itself (`pgrep` excludes its own PID, not its parent's).
     Three of four watchers here had it: permanently silent, whether the job finished, crashed or hung.
  2. `pgrep -f '[j]ob'` — defeats self-match ONLY. Every *other* watcher carrying the plain string still
     matches, so with N watchers naive and bracket converge (measured: naive 5, bracket 5).
  3. `pgrep -f '[p]ython3 -m package.module'` — still counted a shell that merely *mentioned* the string in
     an `echo` (measured: 2, one of them the diagnostic asking the question). **Any process that talks
     about the pattern joins the count**, and a count that never reaches zero is silence-on-crash again.
  4. `pkill -f 'SOME MESSAGE'` killed the shell issuing it, because the message was in its own cmdline.

  The premise is what is broken: **cmdline matching cannot distinguish a process from a process that talks
  about it**, and each layer of quoting makes it worse. Treat the bracket trick as a trap, not a fix — it
  works in the single-watcher case anyone would test it in, and degrades silently as sessions multiply.

  **Use the two branches for what each can actually decide.** `kill -0` answers *is it dead*, never *did it
  succeed* — a crashed process and a finished one are both gone. An artifact check
  (`until [ -s out.json ]`) answers *did it succeed*, never *is it dead*; one watcher here paired a sound
  `-s` success test with a `pgrep`-based crash test and so would still have polled forever on a crash. So:
  **artifact = success branch, `kill -0` = death branch, both in the same loop**, and report which fired.

  `kill -0` trades string ambiguity for PID reuse, and that is a BOUND, not a mechanism: nothing stops the
  kernel recycling a PID onto an unrelated process. Here `pid_max` is 4194304 against ~2.03M currently
  allocated, so wraparound is hours-to-days away and the risk is negligible for a watcher resolved at arm
  time. State the bound when you rely on it; on a box with a small `pid_max` or a very long wait, hold a
  file descriptor / `waitpid` on a child instead.
- **`torch`'s device numbering is NOT `nvidia-smi`'s.** On this box `torch cuda:3` is an A100 80 GB while
  `nvidia-smi` index 3 is the T400 4 GB, and the two swap again at index 4 — CUDA enumerates fastest-first
  by default, `nvidia-smi` by PCI bus. A session told "GPU 4 is free" by `nvidia-smi` and passing
  `--device cuda:4` gets the 4 GB card and OOMs. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` so the two agree, pin
  with `CUDA_VISIBLE_DEVICES`, and confirm what you actually got with
  `torch.cuda.get_device_properties(i).name` — never from the index alone.
- **`ps -eo args` truncates at terminal width.** Grepping its output for a distinctive string late in a
  long command reported a LIVE monitor as gone. `/proc/<pid>/cmdline` (NUL-separated) is the full text.

And the framing error underneath both: asked "what are you monitoring?", the check ran was `ps | grep
<the compute I expected a monitor to be attached to>`. A `tail -f` is not that, so the search was
structurally incapable of finding the answer — it had been running 19 h, ~12 of them on a finished log.
**A query shaped by the expected answer cannot falsify it.** To establish that nothing is running, enumerate
what you started; do not grep for what you think it would be.

## Cheap preconditions that stand in front of expensive runs

- **Before ANY Stage-B / rationale / faithfulness compute on a checkpoint, read its GATE MEAN and compare
  it to init.** Three minutes, and it decides a 4-8 hour run. The frozen H1's edge gates sit at ~1.3e-07
  against ~0.61 at init — a 4.5e+06x collapse — because `StageALoss._graph` normalises by BATCH SIZE and
  not by EDGE COUNT, so at ~40k edges/sample the penalty is ~103x the response term, its gradient on the
  gates is up to ~3.3e+06x the task's, and `GRAD_CLIP=1.0` then rescales the whole update by ~1/695. The
  gate dies inside epoch 0 and message passing is multiplied by ~0 thereafter. `RationaleHead` computes
  `importance = gate x sigmoid(scorer)`, so top-k then ranks a quantity that is ~1e-07 everywhere, every
  deletion is a float32 no-op, the noise floor drops 100% of cases, and every contrast returns UNDECIDABLE.
  Letting the freeze gate reach that verdict is correct but costs hours; the gate read reaches it in minutes.
  Reproducer: `PYTHONPATH=src uv run python -m tcell_pipeline.probe_graph_gradients --n-max 8 --batch-size 2
  --steps 1` — read the **collapse factor** it prints ("gate mean fell 4.51e+06x"), NOT the rendered gate
  mean, which prints `0.000000` for ~1.3e-07 and is exactly the trap the next rule describes.
- **A rendered number is not a measured one.** `0.000000` is how ~1.3e-07 prints at six decimal places.
  That rounding was relayed between sessions as "exactly 0.000000", i.e. bit-zero, which is a materially
  different claim: bit-zero would make every ablation identically zero, whereas the real residuals
  (1e-02 on h_graph, 1e-07..1e-05 on delta_z) ARE the surviving magnitude. Before writing "exactly",
  "zero", or "identical", check the unrounded value — and prefer reporting a ratio (the collapse factor)
  which survives formatting.

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
  `run_module1_smoke.py`, `run_module2_smoke.py`, `run_module3_smoke.py`, `python -m tcell_pipeline.splits`.
- **OVERWRITES THE FROZEN PROGRAM BASIS — treat as destructive:**
  `python -m tcell_pipeline.programs.run_program_basis`. This entry previously sat in the "safe to re-run"
  list above, which is true only in the narrow sense that the DEFAULT `sparse_pca`/K=128 fit is
  seed-deterministic and therefore rewrites the same bytes. Run it with any other `--method` or `--K` — as
  a basis STUDY naturally would — and it silently replaces
  `data/intermediate/gene_program_loadings.parquet` with an incompatible basis. Every result in this
  project is expressed in that basis's coordinates (the frozen H1, `promoted.json`, the 5-seed campaign,
  every baseline), so a swap invalidates all of them at once and the damage is invisible until numbers
  stop reproducing. A study fits candidate bases IN MEMORY and writes elsewhere; regenerating the
  production basis is a deliberate, human decision. Verify with `sha256sum` before and after.
- **DESTRUCTIVE — do NOT run to "test":** `run_module0.py` (and its steps) re-download multi-GB
  PPI DBs and **overwrite the frozen marts in `data/`** that all downstream work depends on. Run
  only when deliberately regenerating source data.
- **Direct module runs need `PYTHONPATH=src`** (e.g. `PYTHONPATH=src python -m tcell_pipeline.splits`).
  `pytest`/`compileall` don't — `conftest.py` handles them.
- **Set `OMP_NUM_THREADS` LOW (4-8), never to the core count.** The sklearn baselines nest parallelism:
  `MultiOutputRegressor(n_jobs=8)` forks 8 workers and each one then opens `OMP_NUM_THREADS` BLAS threads.
  On this 64-core box `OMP_NUM_THREADS=64` produced ~830 threads and a load average of 600; the same
  gradient-boosting fit took 87 min there versus ~4 min at `OMP_NUM_THREADS=8`, for byte-identical output.
  The failure mode is pure thrash — no error, no warning, just a run that looks hung. It is also a SHARED
  box: starving other users' jobs is the more expensive half of the mistake.

## Escalation

If you encounter:
- **Architecture decisions**: Consult README.md and the experiment plan report
- **Unclear requirements**: Check the experiment plan report, otherwise ask user
- **Repeated test failures**: Update progress, flag for human review
- **Scope ambiguity**: Re-read `feature_list.json` for definition of done
