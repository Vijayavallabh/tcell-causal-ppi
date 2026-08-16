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


---

# Lessons from the 2026-08-03/06 autonomous multi-fold campaign

Four days, three folds x four arms x five seeds plus three split re-draws, ~250 GPU-hours, 0 lane
failures after the first hour. The defects below cost real time or nearly reached print. `AGENTS.md`
carries the operative rules; this section carries the evidence.

## A broken NVML kills every CUDA lane, and the error names PyTorch

Every `--device cuda` lane died in ~50 s with

    NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED at PeerToPeerAccess.cpp:83

while a bare `torch` matmul on the same card succeeded. "CUDA works" was true and useless. The chain:
`run_screening.run()` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for cuda; expandable
segments go through the CUDA driver API; PyTorch's `DriverAPI::get()` calls `nvmlInit_v2_()`; NVML fails
because the loader binds `libnvidia-ml.so.1` to a third-party 535.309.01 driver tree another user left
on the system library path while the running kernel module is 580.173.02. **Unsetting
`LD_LIBRARY_PATH` does not help** — the stray tree wins anyway. Diagnose with
`ldd $(which nvidia-smi) | grep nvidia-ml`; fix per-process with

    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02

Two traps this laid. The same assert was already sitting in a 2026-07-29 registry as `status: failed`
rows, so reading history as "transient, it retried fine" was wrong. And the tempting fix — setting
`PYTORCH_CUDA_ALLOC_CONF` yourself to dodge the code path — is a symptom fix that silently changes the
allocator config the landed seeds trained under, breaking comparability. Match the preload, not the flag.

**General rule: when a GPU job dies fast and the message names a PyTorch internal assert, ask what the
userspace driver libraries resolve to before touching any code.**

## A long-running `>` redirect eats lines that other jobs append

`run_c075c15_n5.sh` was launched with `> data/logs/n5.nohup.log` and ran 15 h. Refill lanes launched
later appended to the same file with `>>`. By morning their two `[c080] ... START` lines were gone: a
`>` redirect gives the long-running shell its own file offset, and when it next wrote — hours later, at
an offset behind EOF — it overwrote what had been appended past it.

Nothing was lost from the experiments, only from the record, which is worse than it sounds: **the log
showed a card as idle while a 7.6-hour lane was running on it.** Fixes: append-mode for the
long-running script too, or one file per launcher. Regardless, the per-lane log each worker writes with
its own `>` is the trustworthy record; the aggregate launcher log is a convenience. To establish what
is running, enumerate `/proc/<pid>/environ` and `/proc/<pid>/cmdline` for PIDs you actually started.

## A guard on a list's truthiness does not guard its contents

The campaign's final gate-health block did

    g = [e["train"]["gate_mean"] for e in d if "gate_mean" in e["train"]]
    if not g: continue

For `untyped_gnn`, which returns `edge_gates=None` by design, the key EXISTS with a `None` value, so `g`
is `[None, None, ...]` — truthy — and the next f-string died with `unsupported format string passed to
NoneType.__format__`, aborting the report for every lane sorting after it. It failed loudly, so nothing
was misreported, but the block silently covered only 7 of 10 lanes. Filter on the value
(`e.get("train", {}).get("gate_mean") is not None`) and give the no-gate case its own printed line, so
its absence is visible rather than inferred.

## Every interim value drifted in the favourable direction

Three numbers were published into working docs and later retracted, all from small-n reads on
high-variance arms, and **all three made the result look stronger than it was**:

| Claim at interim | n | Converged | What it would have supported |
|---|---|---|---|
| h2a `−0.0214` on the intermediate fold | 2 | `−0.0122` | a false "h2a is non-monotone in difficulty" |
| h2b `survives_family_wise: True` | 2 | Bonf. 0.146, does not survive | a positive contrast we do not have |
| h1 `survives_family_wise: True` | 5 | Bonf. 0.101, does not survive | a contradiction-stop that never fired |

The third is the subtlest and generalises furthest. The aggregator applies Bonferroni and Holm over the
contrasts that are *testable*; with the other arms unrun, `family_size` collapsed to **1**, so the
correction multiplied a raw p by one and reported `True`. The same run printed `INCOMPLETE COVERAGE` and
`UNBALANCED`. **Before believing any FWER verdict, read `family_size` out of the artifact and confirm it
matches the pre-registered family.** This is the "a correction that passes everything has told you
nothing" hazard arriving through missing lanes rather than through large n.

Operational consequence: hold every number to the pre-registered integration bar (n>=4 both arms) even
when an interim looks decisive, and *especially* when it looks decisive in the direction you expect.

## The split-partition seed moved difficulty further than the difficulty knob did

Three re-draws of one split specification (identical threshold 0.80 and cap 0.10, only `SPLIT_SEED`
varied) gave median train-to-challenge cosines of **0.759 / 0.793 / 0.862** — a spread of 0.103, against
0.056 for the entire designed range across all three thresholds (0.796 → 0.741). The third re-draw is
*easier* than the frozen fold it was built to sit below, the re-draws share only 20% of their validation
targets, and the no-graph baseline moves 0.099 → 0.085 with them, further than any effect under study.

This invalidated a difficulty-curve framing that was already drafted into the paper. **Before
interpreting any split sweep, re-draw one setting at a different partition seed and compare that spread
to the range your knob spans.** It is cheap, and here it overturned the framing.

A related trap found the same day: the statistic is `sequence_train_to_challenge_cosine` — train to the
*sequestered* split, not to the validation fold being evaluated. A key-fallback lookup landed on it and
it was described as "train-to-held-out" in two documents before being caught.

## Diagnose a page overflow before trimming prose

Six rounds of prose cuts moved the LaTeX main-text boundary almost not at all. The cause was an
`\fbox{\begin{minipage}}` checklist: an unbreakable box that, when it did not fit at the bottom of a
page, shipped whole to the next one and stranded ~26 lines of whitespace. Converting it to
`\hrule`-delimited text reclaimed more than every prose cut combined. Check lines-per-page first:

    for p in 6 7 8; do pdftotext -f $p -l $p main.pdf - | grep -vcE '^\s*[0-9]+\s*$|^\s*$'; done

A page holding ~26 lines where its neighbours hold ~50 is a layout problem, not a verbosity problem.
The general form is the one this file keeps re-learning: **measure where the cost actually is before
paying to reduce it.**

## A context-swap helper that always writes to the production path

`embeddings_pinnacle.run()` takes a `context` argument but always writes to
`config.PINNACLE_EMBEDDINGS_PATH`. Calling it for a replication dataset's cell type would have
overwritten the CD4 embedding store that every reference lane reads at training time, silently swapping
the node features of every running and future run — a destructive default wearing a parameterised
signature. The replication path got its own module writing per-context files, and the CD4 store was
sha256-verified byte-identical before and after. **A function that takes a parameter but writes to a
fixed path is not parameterised; check the write target, not the signature.**

## The split-partition seed changes the answer, not just the split statistic (2026-08-10)

The 2026-08-06 entry above recorded that re-drawing a split moves the *difficulty statistic* further
than the difficulty knob does. Running the full comparison on all three re-draws finished the thought,
and the result is worse than the statistic suggested. Same specification (threshold 0.80, cap 0.10),
only `SPLIT_SEED` varied, full nested family, five paired seeds each:

| re-draw | h1 $\Delta$ | uncorrected 95% CI | raw p | Bonferroni x4 |
|---|---|---|---|---|
| seed 0 | +0.00824 | [+0.00168, +0.01480] | 0.025 | 0.101 |
| seed 1 | +0.00260 | [+0.00048, +0.00472] | 0.027 | 0.109 |
| seed 2 | +0.00206 | [-0.00237, +0.00650] | 0.266 | 1.000 |

**The estimate spans fourfold and the qualitative verdict flips.** Two re-draws give an interval
excluding zero at p about 0.03; the third does not, at p = 0.27. An author who happened to draw the
first would report an apparent benefit; one who drew the third would report nothing. Neither is wrong
and neither is reproducible.

The scale is the point. Across the three genuinely *different* folds, h1 was -0.0009 / +0.0082 /
+0.0005 — so **re-draw noise is the same size as the between-fold differences we had been tempted to
interpret as an effect of difficulty.** A dose-response framing was already drafted into the paper
before this ran.

What survived all three: the sign (all positive) and the corrected verdict (none clears Bonferroni
over the pre-registered family of four). That asymmetry is the practical lesson — the corrected
verdict was stable under re-draw while the uncorrected one was not, which is an argument for leaning
on multiplicity control that has nothing to do with the usual false-positive story.

Two further details worth carrying: the re-draws silently changed the validation set from 3,632 to
7,216 targets at identical parameters, and the no-graph baseline tracked *that* (0.0991 / 0.0942 /
0.0845, spread 0.0146, larger than any contrast in the study) rather than the sequence-cosine
statistic — the nominally hardest re-draw scored highest. If a "difficulty" knob silently changes
sample composition, it is confounding difficulty with sample size.

**Rule: before interpreting any split sweep, re-draw one setting and run the actual comparison on it,
not just the split diagnostic.** The diagnostic understated the problem here.

## A shared GPU box: match the arm to the free memory, and never evict the co-tenant

Another user's job held 41.5 GB on one A100 and grew to 48.6 GB mid-campaign. Measured per-lane peaks
in this project: `expression_only` ~2.5 GB, `untyped_gnn` ~2-4 GB, `condition_gated` and
`typed_static` **47-51 GB** (measured from `nvidia-smi` during real lanes, not estimated — an earlier
guess of ~41 GB was low). A graph arm launched into the ~33 GB that remained would have failed,
plausibly hours in rather than at allocation, since memory grows with subgraph size.

Three things worked. First, **check free memory at launch and match the ARM to it**: the constrained
card ran `expression_only` seeds productively for hours while it could not have held a graph arm.
Second, when a chain script runs a cheap arm then an expensive one on the same card, **stop it by
process group after the cheap arm lands** rather than letting the expensive one allocate; the finished
work survives, only the chain dies. Third, **re-queue rather than evict**. The co-tenant was there
first, and starving another user is the more expensive half of the mistake; it freed the card on its
own several hours later and the held lane then ran normally.

The instrument note: `nvidia-smi --query-compute-apps=pid,used_memory` maps memory to PIDs, which is
how you tell your own lane's footprint from a co-tenant's. Per-card totals alone will mislead you
about whose memory it is.


---

## 2026-08-10/16 — the replication and L4 campaigns

Each of these cost real time or a wrong conclusion. The operative one-liners are in `AGENTS.md`; the
evidence is here.

### A file existing is not the same as it being current

`stage_a_history.json` from a killed lane is indistinguishable from a live one by presence alone. This
produced two wrong readings in one session: first an apparent 25x speedup after a relaunch (the "5-7
epochs in 11 minutes" were leftovers from lanes killed an hour earlier, mtime 00:42 against a 02:11
read), then an in-flight lane that appeared stalled. **Read mtime, or pair START/exit in the queue
log.** The relaunched lanes in fact had no completed epoch at all.

### `nvidia-smi` does not explain why you are slow

These lanes are CPU-bound on row-by-row subgraph sampling — the encoder's own docstring calls it "the
throughput floor" — and use about one core each. GPU utilisation reads 100% throughout and carries no
information. The same `typed_static` lane measured ~38 min/epoch on a quiet box, ~45 with three of ours
running, ~80 once a co-tenant's `load_data.py` took 5001% CPU (50 of 64 cores, load average 174), and
~90 when six of ours competed. I attributed the slowdown to my own scheduling before measuring; it was
mostly external. **Check `/proc/loadavg` and `ps -eo pcpu` before blaming your own design.**

### A free GPU does not stay free

`run_l4_card2.sh` preflighted card 2 at 78 GiB free and started a lane. 2h47m later a co-tenant
reclaimed the card: the running lane was SIGTERMed (exit 143) and the next four died within a minute
each with CUDA OOM reporting 161 MiB free. Preflight is necessary and not sufficient on shared
hardware. What kept this cheap was releasing the claim file on failure, which left all five lanes
schedulable by the main scheduler; the loss was four epochs of one lane.

### CUDA index is not `nvidia-smi` index

On this box smi index 3 is a T400, not an A100, and the four A100s are smi 0, 1, 2, 4. A preflight that
only asks "is this an A100" will still schedule a 50 GiB lane onto a card with 1 GiB free. Read
`torch.cuda.mem_get_info` per `CUDA_VISIBLE_DEVICES` and match on free memory. The user's remark that
"gpu ids 0,1,3 have ample vram" was the exact mapping, and I first read it as a general remark.

### Concurrency reduces throughput when the bottleneck is CPU

Six lanes on four cards ran ~90 min/epoch against ~50 for four lanes: packing turned one queue into six
slow ones and cost five OOM kills. Going from three to four of our lanes costs one core out of 64 and
gains a whole GPU; going from four to six loses outright. Saturating VRAM is not the objective when the
lane is single-threaded on the host.

### A partially-filled cell gives a different answer, not a preliminary one

The L4 variance decomposition run with one re-draw at n=1 concluded that partition noise exceeded the
difficulty effect and the difficulty knob had no detectable effect. At n=5 the same cell moved from
−0.0189 to −0.0051, within-level variance collapsed to indistinguishable-from-zero, and the ranking
inverted. Nothing in the pipeline changed — only the number of seeds behind one cell mean. Report
per-cell n and treat components estimated from unequal-n cells as provisional.

### Two tests that passed against buggy code

Worth recording because a green suite was not evidence. (1) A test for a `ZeroDivisionError` guard
passed against the unguarded version, because the sampled data never happened to trigger the clamp it
needed; rewritten to construct cells with identical means so the clamp is guaranteed. (2) A biased
variance estimator (`ddof=0`) survived mutation testing at 8 seeds, where the error is only 6.5% —
inside the tolerance; a dedicated test at 3 seeds, where the bias is 18%, kills it. **Verify the mutant
dies, don't infer it from a passing suite.**

### The estimator error that ran in the flattering direction

The decomposition subtracted `sigma2_seed / mean(n_j)` where the expectation is
`mean(sigma2_seed / n_j)`. With unequal cells (n = 5, 3, 1) those differ by 50%, and the wrong form
inflates the re-draw component — which is the denominator of the verdict. It was handing out "the
difficulty knob has no detectable effect" for free. Errors that flatter the more quotable conclusion
deserve the check the conclusion itself would get.
