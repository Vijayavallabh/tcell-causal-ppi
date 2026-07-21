# Agent lessons — the long form

Detailed write-ups of the rules summarised in `AGENTS.md` under "Hard-won rules". Each was earned by a
real defect during the five-session run of 2026-07-20/21. `AGENTS.md` carries the operative rule; this
file carries the evidence, so the instruction file stays short enough to actually be read.

---

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
  against ~0.61 at init — a ~4.5e+06x collapse — because `StageALoss._graph` normalises by BATCH SIZE and
  not by EDGE COUNT, so at ~40k edges/sample the penalty is ~103x the response term and its gradient on
  the gates is ~3.1e+06x the task's. The gate dies inside epoch 0 and message passing is multiplied by ~0
  thereafter.

  > **CORRECTED 2026-07-21.** This paragraph previously continued "and `GRAD_CLIP=1.0` then rescales the
  > whole update by ~1/695, so ~99.98% of every step drives gates to zero". **That mechanism is wrong.**
  > AdamW is scale-invariant per parameter: scaling every gradient by a constant `c` scales the first
  > moment by `c` and the second by `c²`, leaving `m̂/√v̂` unchanged — and a uniform clip factor is exactly
  > such a constant. Verified under the real settings: gradient 1e-4 → θ=-0.299969; 1e-1 → -0.300000;
  > ‖g‖=695 clipped to 1.0 → -0.300000. The clip changes nothing. What actually kills the gate is
  > DIRECTION dominance — the penalty's gradient is ~100% of the total (`g_total/g_penalty` =
  > 0.999994–1.000315), so every step marches the gate the same way at ~`lr`.
  > **Magnitude sets the rate of collapse; direction sets whether it happens.** The measurements were
  > always right; only the causal story was wrong. Confirmed prospectively: per-edge normalisation makes
  > the penalty 400x SMALLER than the task and the gates still collapse 2,108x. `RationaleHead` computes
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

