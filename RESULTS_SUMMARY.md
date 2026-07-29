# Autonomous OOD-difficulty robustness campaign — results summary

**Status: COMPLETE.** Ran ~12 h on 4 A100s (T0 = 2026-07-29 04:14 IST / 22:44 UTC). The h1 headline
was taken to n=3 (you chose not to wait ~4 h for the straggler 4th seed); the residual budget then went
to the Q4 figure upgrade (below). Everything is UNCOMMITTED on disk for your review.

> **No CONTRADICTION-STOP fired.** The graph never helped: on the stricter split the evidence-gated
> graph stayed at parity with no-graph (below), consistent with the frozen result. No paper claim was
> reversed.

---

## The result (one line)

**The headline null holds on a stricter target-OOD split.** Re-screening the nested family at
`lambda_graph=0` on a harder blocked-target-OOD split (sequence-similarity threshold 0.85 -> 0.75,
family-size cap 5% -> 15%, which lowers median train-to-held-out sequence similarity from ~0.79 to
~0.73), the evidence-gated graph is again statistically indistinguishable from expression-only.

| Split | h1 = condition_gated - expression_only | 95% CI | p | verdict |
|---|---|---|---|---|
| Frozen 0.85 (n=5, the paper) | -0.0009 | [-0.0072, +0.0054] | 0.71 | parity |
| **Harder 0.75/0.15 (n=3, this run)** | **-0.0035** | **[-0.0262, +0.0191]** | **0.57** | **parity** |

Per-arm \textsc{systema} on the harder split: untyped_gnn 0.0805 (n=2) ~ expression_only 0.0805 (n=4)
> condition_gated 0.0769 (n=3). The graph arm sits at or below no-graph — no benefit, same ordering
sign as the frozen fold. Report: `data/results/screening_c075c15/robustness_hard_c075c15.{json,md}`.

**Honest caveat (why this is a supporting check, not a headline):** n=3 makes the paired CI wide
([-0.026, +0.019]); it is "consistent with the null but underpowered", not a tight null. The frozen
n=5 result remains the paper's headline; this run corroborates its direction under harder OOD.

---

## What ran (14 screening lanes + the Q4 lambda sweep, ~35 GPU-hours, 0 failures)

| Split (thr/cap) | condition_gated | expression_only | untyped_gnn | typed_static |
|---|---|---|---|---|
| c075c15 (0.75/0.15, primary) | **3** (s0,s1,s3) | 5 | 2 | 0 |
| c080c10 (0.80/0.10, bonus) | 0 | 4 | 0 | 0 |

- **h1 (graph vs no-graph)** is the only fully-formed contrast: n=3 at c075c15 (above).
- **promotion_margin** (untyped - expr) at c075c15: n=2, -0.0001, p=0.76 (parity, but n=2 — ignore).
- **h2a (typed_static worse)** was NOT re-tested: typed_static costs ~8 GPU-h/seed and did not fit the
  budget after the primary wave (the fit-gate correctly skipped it). So the frozen fold remains the
  only place "static graph is worse" is measured.
- c080c10 got only its expression_only baseline (4 seeds); condition_gated did not fit, so there is no
  second difficulty-curve point for h1.

### Why only n=3 (the straggler)
condition_gated seeds s1 and s3 early-stopped at epoch 11-13 and s0 at 13, all within budget. Seed s2
had a late validation-loss improvement at epoch 8 that reset its early-stop counter (patience 10), so
it would not have converged until ~epoch 18 (~4 h past the budget). At your call we wrapped up rather
than extend, so s2 was stopped without a result. Getting to n=4 (the rail-6 integration threshold) just
needs s2 (or a fresh seed) to finish: ~6-7 GPU-h.

---

## Paper: one additive edit (Q4 figure upgrade); the harder-split null was NOT integrated

**One `% AUTO` edit to paper/main.tex:** Figure 2 (empirical gate-vs-lambda sweep) was swapped for a
deeper 22-epoch version and its caption updated (audited with the no-ai-slop skill). The default weight
now drives the mean gate to $\sim 10^{-6}$ (vs the previous three-epoch $\sim 3\times10^{-5}$), and the
curve reaches $\sim 10^{-7}$ by $\lambda{=}0.1$; $\lambda{=}0$ still reproduces the live gates (0.76).
The original figure is at `paper/figures/lambda_sweep_orig.pdf`; the edit is marked `% AUTO 2026-07-29`
at the figure. Revert: restore that PDF and the three-epoch caption numbers. Sweep data:
`data/results/q4_lambda_sweep_22ep.json`.

**The harder-split null was NOT integrated** (rail 6 needs n>=4 of BOTH arms; we have n=3) — it is
documented above for review. To add it: let one more condition_gated seed finish -> n=4, then a
one-paragraph additive edit slots in after "The null survives a second metric." (sec:null, ~line 195),
e.g.: "Under a stricter target-OOD split (sequence-similarity threshold lowered 0.85 to 0.75), the
evidence-gated graph remains at parity with no-graph (Delta = ..., 95% CI [...], p = ..., n = 4)." The
integration point is pre-scouted; wiring it is ~30 min once n=4 lands.

**Paper compiles clean** (full build = `pdflatex; bibtex main; pdflatex; pdflatex`): 8 pages, 0 LaTeX
errors, 0 undefined, 0 overfull >2pt, abstract 248 words, 0 banned words, 0 em/en dashes.

---

## The only code change (additive, reversible, tagged) — for your review
- **`src/tcell_pipeline/config.py`** (~lines 132-138): made `SEQ_SIM_COSINE_THRESHOLD` and
  `GROUP_SIZE_CAP` env-overridable via `os.environ.get(..., <original default>)`, matching the pattern
  every path constant in that file already uses. **Defaults are unchanged (0.85 / 0.05)** — any process
  that does not set the env sees the frozen fold byte-for-byte. Tagged `AUTO 2026-07-29`. Revert:
  `git checkout src/tcell_pipeline/config.py`.
- No other source or paper files were changed. Frozen artifacts (`data/splits/`,
  `data/results/screening_lambda0/`, `screening/`, `promoted.json`, the architecture figure) are
  untouched. The sealed split was never opened; `sealed_eval.py` was never run; nothing was committed.

## New artifacts (all in fresh roots)
- `data/results/splits_c075c15/`, `data/results/splits_c080c10/` — the harder splits (+ leakage reports).
- `data/results/screening_c075c15/` — the primary re-screen (cond/expr/untyped parquets) +
  `robustness_hard_c075c15.{json,md}` (the n=3 aggregation).
- `data/results/screening_c080c10/` — the 2nd split's expr baseline.
- `data/results/q4_lambda_sweep_22ep.json` — the deeper 22-epoch gate-vs-lambda sweep (Q4); the paper's
  Figure 2 was regenerated from it (original figure kept at `paper/figures/lambda_sweep_orig.pdf`).
- `data/logs/campaign/` — scheduler log, per-job logs, `STATUS.json`, `campaign_plan.json`, `monitor.sh`,
  `calibrate.log` (the (threshold,cap) sweep). Scheduler + helpers: session scratchpad
  `campaign_scheduler.py`, `calibrate_splits.py`.

---

## Ops notes worth keeping (paid for in debugging time)
1. **NVML is broken on this box** (stale 535 userspace vs 580 kernel; `nvidia-smi` fails). It does NOT
   just break tooling: `expandable_segments:True` calls `nvmlInit()` and HARD-ASSERTS, crashing every
   training job. **Fix without touching the machine:** `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02`
   forces the correct NVML so `expandable_segments` (the frozen allocator, ~40 GB/seed) works. Clearing
   `LD_LIBRARY_PATH` alone does NOT fix it; only the preload does. Monitor GPUs via
   `torch.cuda.mem_get_info`, not nvidia-smi.
2. **CUDA fastest-first**: `CUDA_VISIBLE_DEVICES` 0-3 map to the four A100s; index 4 is the T400 (unused).
3. **The cards are shared.** A parallel job occupied them during part of the run; the scheduler is
   memory-aware (launches a lane only where >=52 GB is free) so it coexists without OOMing anyone.
4. **The plan's premise was wrong.** "Lower sequence threshold => harder OOD" is FALSE under the fixed
   family cap (the cap refuses the bigger merges a lower threshold wants, leaking paralogs across roles).
   The real lever is lower threshold + higher cap together; calibration (`calibrate.log`) picked the
   (0.75, 0.15) operating point.

## To strengthen next (in rough value order)
1. Finish n=4 for h1 at c075c15 (one condition_gated seed, ~6-7 GPU-h) -> then the paper paragraph.
2. Add typed_static at c075c15 (~4 seeds) to re-test "static graph is worse" under harder OOD (h2a).
3. Add condition_gated at c080c10 for a genuine 2-point difficulty curve of the null.

---

## Paper revision — 7-skill audit (2026-07-29, evening)

After the campaign, `paper/main.tex` was revised with a peer-review / scientific-critical-thinking /
statistical-power / citation-management / scientific-writing / humanizer / no-ai-slop pass. The result and its
strength are unchanged; the four claims most likely to draw reviewer fire were made defensible:

1. **The "reproduces the confounded null" argument was reframed.** A dead-gate run has the graph switched off,
   so its parity with no-graph is expected and is uninformative about whether a working graph helps. The
   **live-gate** null now carries the claim; the confounded run is described only as a control on the fix (the
   paragraph is now "The correction does not manufacture the null"). Propagated to the abstract, intro,
   contribution 3, and conclusion.
2. **The power/bounded-null claim was re-anchored on the 95% CI upper bound (+0.0054 systema)** rather than the
   noncentral-t MDE. The MDE (0.0086 uncorrected / 0.0126 Bonferroni, verified numerically) is retained but
   flagged as itself uncertain at n=5 (its own 95% span is ~0.005 to 0.025). "Actively rules out" was dropped.
3. **"Rule out the graph-prior gains reported elsewhere"** was softened to a CI-bounded statement, attached to
   the actual graph-win citations, with a note that those gains are not measured in systema (not strictly
   commensurable).
4. **The untyped-GCN tension is now confronted** in the Discussion: the minimal untyped GCN is the lone hint of
   signal (+0.0045, significant only under the lower-variance metric, inside the inconclusive band); the
   engineered typed/gated priors add nothing.

Also: softened the confound mechanism (loss-magnitude to Adam-direction) to a hypothesis backed by the
dose-response sweep and the restoration at lambda=0; removed a Figure 2 double-cite; fixed two colon-reveals and
one copula avoidance. Paper compiles clean: 8 pages, abstract 244 words, 0 errors / undefined / overfull, 0
banned words, 0 dashes. Backup at `paper/main.tex.pre_revise_bak`.

Docs brought into consistency: `README.md`, `docs/aaai-title-abstract.md`, `docs/feat011-rescreen-notes.md`
(in-place reframe), and the two gitignored reports `EG_IPG_architecture_walkthrough.md` /
`perturbation_informed_causal_protein_program_graphs_report.md` (a dated 2026-07-29 resolution block was
prepended; the historical CONFOUND blocks are preserved). Dated archival material under `docs/history/`,
`docs/reviews/`, and dated `docs/specs/` was left untouched as historical record.
