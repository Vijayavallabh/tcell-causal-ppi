# UNEXPECTED — NEEDS HUMAN REVIEW (2026-08-10)

**The graph HELPS on Replogle RPE1, and it survives both Bonferroni and Holm.** Rail 4 of the standing
goal says a positive replication is a finding and the null does not get rewritten around it. It has not
been. This banner is the flag; the numbers below are reported as measured.

| | | |
|---|---|---|
| contrast | `promotion_margin` = untyped_gnn − expression_only | |
| dataset | ReplogleWeissman2022_rpe1 (2,122 targets, RPE1, knockdown) | |
| effect | **+0.0675 systema**, 95% CI [+0.0316, +0.1033] | |
| per seed | +0.0356, +0.0724, +0.0887, +0.0731 — all four positive, none dropped | |
| correction | p=0.0093, Bonferroni 0.0186, Holm 0.0186, family_size **2** | survives BOTH |
| gate health | n/a — untyped_gnn has no learnable edge gate by construction | |

`family_size: 2` is honest, not a collapse. RPE1 is single-condition, so `condition_gated` is
arithmetically identical to `typed_static` and was not run (prereg Amendment 3.2); only `h2a` and
`promotion_margin` are testable there, and 0.0093 x 2 = 0.0186 is the arithmetic you would expect.
This is NOT the `family_size: 1` bug that produced three false "survives" earlier in the project.

**What is and is not established.**

The arm that wins is `untyped_gnn` — PPI topology with NO edge typing and NO condition gating. The
typed arm on the same dataset goes the other way (`h2a` = −0.0196). So the finding is not "the graph
helps"; it is narrower and more interesting: **on RPE1 the graph's raw topology helps, and this
project's typing-and-gating machinery is what discards the benefit.** That is a statement about the
architecture, not about biological priors, and it is the first direct evidence for it.

Pooled across all four datasets the untyped effect does NOT hold up: fixed-effect +0.0139
[+0.0072, +0.0206] excludes zero, but random-effects +0.0326 [−0.0105, +0.0758] does not, with
**I² = 88.2%** and Cochran's Q p < 0.001. At that heterogeneity the random-effects reading is the
honest one, so the pooled claim is heterogeneity, not benefit: real and large on RPE1 (+0.0675), small
on Frangieh (+0.0194), Tian (+0.0281) and Replogle K562-essential (+0.0081). Reporting the
fixed-effect interval alone here would be the single most misleading thing this report could do.

**The main null is unchanged and is now tighter than it has ever been.** `h2a` (typed_static −
expression_only) pooled over four datasets: fixed-effect **+0.0031, 95% CI [−0.0004, +0.0065]**,
random-effects +0.0042 [−0.0105, +0.0189], I² = 25.6%, Q p = 0.26. A half-width of ±0.0035 systema
across four cell types is a *bounded* null rather than an absence of evidence — the thing the
single-dataset result could never be.

**Human review is wanted on one question:** whether the RPE1 result warrants a follow-up lane
(untyped_gnn on the remaining datasets at more seeds, or an untyped arm on the reference screen) before
the paper commits to the architecture reading above.

## POST-HOC (2026-08-10): the untyped graph is positive on 5 of 5 independent datasets

Not pre-registered. The prereg fixes a PER-DATASET contrast family; a pooled meta-analytic test is not
in it. Hypothesis-generating, not confirmatory. Recorded because it is the strongest pattern in the
data and hiding it until it could be dressed as confirmatory would be worse.

The reference screen already had an `untyped_gnn` arm — nobody had read it in this light.

| dataset | n | untyped_gnn - expression_only | 95% CI | note |
|---|---|---|---|---|
| reference (CD4 T cell, frozen fold) | 5 | **+0.0045** | [+0.0011, +0.0079] | 5/5 seeds positive; Holm 0.042 but Bonferroni 0.083, so it FAILS the both-corrections rule |
| Replogle K562-essential | 4 | +0.0081 | [-0.0034, +0.0197] | ns |
| Frangieh melanoma | 4 | +0.0194 | [-0.1220, +0.1608] | ns |
| Tian iPSC neuron | 4 | +0.0281 | [-0.0929, +0.1491] | ns |
| Replogle RPE1 | 4 | **+0.0675** | [+0.0316, +0.1033] | survives BOTH corrections |

Pooled over the five: fixed-effect +0.0056 [+0.0033, +0.0078]; **random-effects +0.0209
[+0.0048, +0.0370], p=0.011**; I2 = 87.5%, tau2 = 0.00018, Q = 31.9, Q p < 0.001. Sign test 5/5,
p = 0.0625. Fold re-draws (c075c15, c080c10) are EXCLUDED — they re-draw the same dataset and are not
independent units; both are also positive (+0.0028, +0.0094).

Read it carefully. The random-effects interval excludes zero, which is a stronger statement than the
four-dataset replication pool alone supported (that one included zero). But I2 = 88% means the
MAGNITUDE does not transport: the pooled estimate says a benefit exists somewhere in this family of
datasets, not that +0.02 is what you would get on yours.

**Why this matters more than the number.** Set it beside the typed arm on the same datasets: h2a
pooled is +0.0031 with a 95% CI half-width of 0.0035 — a tight bounded null. Raw PPI topology carries
a small consistent positive signal; the typed, gated encoder built to exploit that signal removes it.

This FLIPS candidate cause C (architecture) in the paper from "Refuted within the searched space" to
**Survives**. The 14-cell architecture search that refuted it varied per-relation normalisation,
confidence pruning, edge-weighted convolution and attention — every cell varied HOW the typed gated
encoder message-passes, and not one removed the typing and gating themselves. The search excluded its
own premise, so it could never have found this.

---

# Session 2026-08-03 — ICBINB-BIO retarget, harder-split n=5, multi-dataset replication

Branch `icbinb-multidataset-2026-08-03`. Everything uncommitted on disk. The 2026-07-29 campaign
report follows below, unchanged.

## L5 RESULT (2026-08-10): the "no graph" baseline is NOT graph-free, and the graph part WORKS

Pre-specified analysis (`analyze_feature_ablation.py`, written before the lanes landed). Frozen fold,
paired by seed against the unablated `expression_only` baseline (n=5, mean $0.085676$). Channels are
ZEROED not removed, so out_dim and parameter count are identical and the contrast isolates
information rather than capacity. Family of 3, Bonferroni and Holm.

| variant | dropped | n | delta systema | 95% CI | bonf | verdict |
|---|---|---|---|---|---|---|
| nograph | PINNACLE + PPI degrees | 5 | $-0.003565$ | $[-0.004272,-0.002858]$ | **0.0005** | **SURVIVES: carried signal** |
| nodegree | PPI degrees only | 4 | $-0.003465$ | $[-0.004298,-0.002631]$ | **0.0028** | **SURVIVES: carried signal** |
| nopinnacle | PINNACLE only | 5 | $+0.000006$ | $[-0.000006,+0.000018]$ | 0.641 | inert |

**Three readings, in order of importance.**

1. **The graph information in the baseline is REAL and survives correction.** Removing it costs
   $0.0036$ systema. For scale, h1 (message passing over the full ~8M-edge typed graph) is $-0.0009$
   on this same fold and survives nothing. So a three-scalar summary of the network carries a
   measurable, correction-surviving effect while the topology on top of it does not.
2. **It is entirely the PPI degree scalars.** `nodegree` ($-0.00347$) reproduces `nograph`
   ($-0.00357$) to within the CIs, so degrees account for essentially the whole effect.
3. **PINNACLE is inert** ($+0.000006$, p=0.21). Consistent with the coverage audit: it is a zero
   vector for 90.8% of rows, so a 128-d feature contributing nothing is the expected result, not a
   surprise. The paper's cause-D argument should rest on degrees, not on PINNACLE.

**This is the first interim in this project that did NOT drift favourably on converging**: nograph
went $-0.00339$ (n=4) to $-0.00357$ (n=5), p $0.0008$ to $0.0002$. Recorded because the pattern has
been the opposite four times and noting only the convenient direction would be selective.

**Cost:** 15 lanes, ~5.4 GPU-h total. The paper currently claims this experiment "cannot be done
without starving the graph arm too" — that sentence is wrong and must be replaced with the result.

**Caveats.** One fold (frozen). The absolute effect is small ($0.0036$ on a baseline of $0.0857$,
~4% relative); it clears correction because `expression_only` is the low-variance arm (sd $0.0005$).
Comparator lanes predate a schema change and lack n_train/n_val, so fold identity is verified
indirectly (ablated runs report n_train=21262, the frozen fold).

## RESOLVED (2026-08-05 05:20): the contradiction stop did NOT fire once the family was complete

The escalation below was raised at `family_size = 1` and is now settled at `family_size = 4`, with
all four contrasts testable. **h1 on c080c10: n=5, $+0.00824$, raw p $=0.0252$, Bonferroni $=0.1006$,
Holm $=0.0560$, `survives_family_wise = False`.** That is the value predicted when the collapsed
family was spotted ($0.0252 \times 4 = 0.1006$). Rail 4's condition (positive AND survives both) is
**not met**, so no human adjudication is required and the campaign continues normally.

**What still needs to reach the paper, because it is not nothing:**

1. **The bounded-null claim is fold-specific and the paper currently overstates it.** The paper says
   the $95\%$ CI upper limit on graph benefit is $+0.0054$ (frozen fold). On c080c10 the point
   estimate is $+0.0082$ with CI up to $+0.0148$. Across folds the data are compatible with a
   benefit roughly $3\times$ larger than the frozen-fold bound. The bound must be stated per fold,
   not as a general ceiling.
2. **The corrected verdict is "indistinguishable" on all three folds, but the point estimates are
   not similar:** $-0.0009$ (frozen), $+0.0005$ (c075c15), $+0.0082$ (c080c10, raw CI excludes
   zero). Reporting only the two folds where the estimate sits near zero would be selective.
3. **CORRECTED 2026-08-05 (this entry previously said h2a was NON-monotone, from an n=2 interim).**
   At n=5 on every fold, h2a is $-0.0131$ (frozen), $\mathbf{-0.0122}$ (c080c10), $-0.0012$
   (c075c15). The earlier $-0.0214$ was the c080c10 value at **n=2** and should not have been
   compared against n=5 numbers; the non-monotonicity it implied does not exist. The real structure
   is more informative: the deficit **replicates in size** on c080c10 (within 7% of frozen) but
   fails correction there ($p=0.23$) because typed_static's across-seed SD is $0.0169$ vs $0.0049$
   on the frozen fold, widening the CI ~4x; on c075c15 the **effect itself** collapses to $-0.0012$.
   So "h2a did not replicate" is two different failures, one from added noise and one from a
   vanished effect, and only the second is a statement about the effect. The paper separates them.
4. h2b on c080c10 at n=5 is $+0.0204$, CI $[+0.0021,+0.0387]$, p=0.036, bonf=0.146 -> does NOT
   survive. (At n=2 it read $+0.0265$ with `fwer=True`; another interim that would have misled.)

**c080c10 family, FINAL at n=5** (promotion_margin n=3, untyped_gnn s3/s4 pending): h1 $+0.0082$
(bonf 0.101), h2a $-0.0122$ (bonf 0.920), h2b $+0.0204$ (bonf 0.146), promotion $+0.0094$ (bonf
0.652). **Nothing survives on this fold either.** Per-arm: untyped_gnn $0.1090$, condition_gated
$0.1073$, expression_only $0.0991$, typed_static $0.0869$.

### The reseed experiment: the difficulty axis is not stable (2026-08-05)

`SPLIT_SEED` is now env-scoped (same verified-identical-default pattern), which made this measurable.
Regenerating c080c10 at **identical threshold and cap** with `SPLIT_SEED=1`:

| | median train-to-challenge cosine | val targets |
|---|---|---|
| splits_c080c10 (seed 0) | $0.7591$ | 1,230 |
| splits_c080c10_r2 (seed 1) | **$0.7928$** | 2,438 |

The difficulty statistic moves **0.0337 under a reseed alone**, against a total designed span of
$0.0557$ across all three thresholds ($0.7964 \to 0.7407$). Realization noise is ~60% of the range
we were varying, and the redrawn "intermediate" fold is as easy as the frozen one on this measure.
The two realizations share only 20% of their val targets at 2x different val size. **This kills the
difficulty-curve framing**, and the paper now says so explicitly and cautions against reading a
difficulty trend out of any single-realization split sweep.

Also corrected: the statistic is train-to-**challenge** cosine (the sequestered split), not
train-to-val. An earlier key-fallback lookup landed on it and it was described as "train-to-held-out"
in both this file and the paper. Fixed in both.

**OBSERVATION (2026-08-08, n=3 realizations, NOT a claim): the cosine statistic does not predict the
model's own difficulty; validation-set size does.** All three realizations of the same 0.80/0.10 spec:

| realization | difficulty (cosine) | n_val | expression_only systema |
|---|---|---|---|
| seed 0 | $0.7591$ (nominally hardest) | 3,632 | $0.09906$ (n=5, sd $0.0027$) |
| seed 1 | $0.7928$ | 7,216 | $0.08449$ (n=5, sd $0.0009$) |
| seed 2 | $0.8624$ (nominally easiest) | 5,096 | $0.09444$ (n=1) |

The nominally *hardest* draw yields the *highest* no-graph score, so the sequence-cosine statistic is
not the operative axis. The baseline is instead monotone (inverse) in `n_val`, which plausibly reflects
`systema` being a correlation-type metric over a more heterogeneous target set. **Three points and one
of them n=1, so this is a lead, not a result** — but it matters for interpretation: if val size drives
the score, then a split "difficulty" knob that silently changes val size (ours ranged 3,632 to 7,216 at
IDENTICAL parameters) is confounding difficulty with sample composition. Worth testing properly before
any future split sweep is interpreted. Baseline spread across realizations is $0.0146$, still larger
than h1 ($+0.0082$) or h2a ($-0.0131$).

**The reseed also moves the model's own difficulty, by more than any effect in the paper.** The
no-graph baseline is the cleanest performance-based difficulty measure available (it is the most
stable arm by far). Across the two identical-parameter realizations:

| realization | expression_only systema | n_train / n_val |
|---|---|---|
| splits_c080c10 (seed 0) | $0.09906$ (n=5, sd $0.0027$) | 21,496 / 3,632 |
| splits_c080c10_r2 (seed 1) | $\mathbf{0.08530}$ (n=2, sd $0.0006$) | 19,630 / **7,216** |

A shift of **$0.0138$ from the partition seed alone**, against h1 $=+0.0082$ and h2a $=-0.0131$.
**Between-realization variation in the baseline exceeds every contrast effect under discussion**,
and the validation set doubles in size. The paired contrast is still valid within each split, so
this does not invalidate any result; what it does is bound how much meaning a cross-split comparison
of contrasts can carry. Recorded at n=2 for r2, but expression_only is the low-variance arm
(sd $0.0006$), so the baseline estimate is well determined.

### COMPLETE (2026-08-10): the re-draw experiment, 3 realizations x 2 arms x 5 seeds, 0 failures

Same specification (thr 0.80 / cap 0.10) three times, varying ONLY `SPLIT_SEED`. All gates alive.

| re-draw | h1 $\Delta$ | uncorrected 95% CI | raw p | Bonf x4 | no-graph baseline | n_val | cosine |
|---|---|---|---|---|---|---|---|
| seed 0 | $+0.00824$ | $[+0.00168,+0.01480]$ | $0.025$ | 0.101 | $0.0991$ | 3,632 | 0.7591 |
| seed 1 | $+0.00260$ | $[+0.00048,+0.00472]$ | $0.027$ | 0.109 | $0.0845$ | 7,216 | 0.7928 |
| seed 2 | $+0.00206$ | $[-0.00237,+0.00650]$ | $0.266$ | 1.000 | $0.0942$ | 5,096 | 0.8624 |

**The qualitative verdict FLIPS between re-draws of an identical specification.** Two of three give
an interval excluding zero at p~0.03; the third does not, at p=0.27. Estimate spans **4.0x**
($+0.00824$ to $+0.00206$, spread $0.00618$). For scale, h1 across the three genuinely DIFFERENT
folds was $-0.0009 / +0.0082 / +0.0005$ — i.e. **re-draw noise is the same size as the
between-fold differences we were tempted to interpret.**

What IS stable: the sign (all three positive) and the corrected verdict (none survives). That is why
the paper leans on the corrected verdict and presents the uncorrected intervals as a cautionary
exhibit. Baseline spread $0.0146$ likewise exceeds both h1 and |h2a|, and tracks n_val rather than
the cosine statistic (the nominally hardest re-draw scores highest).

This is now in the paper. **The experiment is finished; do not re-run it.**

### (earlier) THREE realizations (2026-08-06): the split seed moves difficulty ~2x more than the threshold does

Same specification (thr 0.80 / cap 0.10), only `SPLIT_SEED` varied:

| realization | median train-to-challenge cosine |
|---|---|
| splits_c080c10 (seed 0) | $0.7591$ |
| splits_c080c10_r2 (seed 1) | $0.7928$ |
| splits_c080c10_r3 (seed 2) | $\mathbf{0.8624}$ |

**Spread $0.1033$ from partition noise alone, against $0.0557$ for the entire designed range across
all three thresholds** ($0.7964 \to 0.7407$). Redrawing one split moves the difficulty statistic
**1.85x further than deliberately changing the threshold**. The seed-2 draw is EASIER than the
frozen fold it was meant to sit below. This is fully converged (a property of the splits, no
training involved) and is now in the paper plus a new checklist item.

**h1 contrast across realizations** (the reason the reseed was run):

| realization | h1 | 95% CI | raw p |
|---|---|---|---|
| seed 0 (n=5) | $+0.00824$ | $[+0.0017,+0.0148]$ | $0.0252$ |
| seed 1 (n=3, PRELIMINARY) | $+0.00348$ | $[-0.0004,+0.0074]$ | $0.0616$ |

The contrast roughly halves and loses even nominal significance under a reseed. **Held at n=3 and
NOT yet in the paper**: r2 seeds 3,4 are running for a like-for-like n=5 comparison. Both
realizations are positive, so the direction is consistent even as the magnitude is not; that nuance
needs the full n before it goes in print. Three interim values have already had to be retracted this
session (h2a $-0.0214$, h2b `fwer=True`, h1 family-of-1 "survives"), all in the favourable
direction, so preliminary contrast numbers stay out of the paper.

**RUNNING:** r2 seeds 3,4 (-> h1 at n=5 on realization 1) and r3 seeds 0,1 (a THIRD realization:
two realizations give one difference, three give a spread). Superseded note below:

**(was) RUNNING:** expression_only + condition_gated seeds 0-2 on `splits_c080c10_r2` to measure whether
the CONTRAST moves as much as the difficulty statistic does. If h1 there differs from c080c10's
$+0.0082$ by a comparable amount, the between-fold differences are fold noise, not difficulty.

Original escalation, kept as the record:

## (was) UNEXPECTED — NEEDS HUMAN REVIEW: h1 is POSITIVE at n=5 on the intermediate split

**Rail 4 fired on the n=5 read; nothing was integrated and no paper claim was altered while it stood.**

`data/results/screening_c080c10_h1/robustness_5seed.json`, split c080c10 (thr 0.80 / cap 0.10,
median train-to-val cosine $0.759$), all five seeds, both arms n=5, gates alive throughout:

| | value |
|---|---|
| h1 = condition_gated - expression_only | **$+0.00824$** |
| 95% CI | **$[+0.00168,+0.01480]$** (excludes zero) |
| raw p | $0.0252$ |
| p_bonferroni / p_holm **as recorded** | $0.0252$ / $0.0252$, `survives_family_wise = True` |
| per-seed | $+0.00440$, $+0.00829$, $+0.00579$, and two more, all positive |
| per-arm | condition_gated $0.1073$ vs expression_only $0.0991$ |

**Read the correction before believing it.** `family_size` in that artifact is **1**, not 4, because
h2a and h2b have n=1 and promotion_margin n=0 on this split (typed_static and untyped_gnn had not
been run). So Bonferroni and Holm multiplied by ONE and the "survives" verdict is an artifact of
incomplete coverage, not a strong effect. **Corrected over the pre-registered family of 4, the same
family applied to both other folds, p = 0.1006, which does NOT survive.** The aggregator flagged
`INCOMPLETE COVERAGE` and `UNBALANCED` on the same run. This is the "a correction that passes
everything has told you nothing" hazard in AGENTS.md, arriving through missing lanes.

**Also confounded:** each split is ONE realization, so fold composition and OOD difficulty vary
together. A between-split difference cannot be attributed to difficulty rather than to which targets
landed in val.

**Status: undecided, escalated, not resolved autonomously.** typed_static seeds 1-4 and untyped_gnn
seeds 0-4 are now running on all four cards to make the family size 4 and the correction comparable
to the other folds. Until that lands, the honest statement is: *a positive h1 on one fold whose raw
CI excludes zero, whose family-wise verdict is currently vacuous, and which the pre-registered
correction does not rescue.* A human should adjudicate whether this is a real fold-specific benefit,
a fold-realization fluctuation, or evidence that the null is difficulty-dependent.

For contrast, h1 is $-0.0009$ (frozen, n=5, p=0.71) and $+0.0005$ (c075c15, n=5, p=0.904).

---

## Superseded watch note (kept as the record of what n=3 showed)

All three landed seeds on c080c10 favour the graph, consistently:

| seed | condition_gated | expression_only | diff |
|---|---|---|---|
| 0 | $0.10280$ | $0.09841$ | $+0.00440$ |
| 1 | $0.10675$ | $0.09846$ | $+0.00829$ |
| 2 | $0.10759$ | $0.10180$ | $+0.00579$ |

n=3, mean $+0.00616$, 95% CI $[+0.00126,+0.01106]$, **raw p $=0.0325$**, **Bonferroni$\times$4 $=0.130$**.

**The raw interval excludes zero. The corrected one does not, and the corrected one is the test.**
The pre-registered bar is n$\geq$4 AND survival of BOTH Bonferroni and Holm; at n=3 that bar is not
met. Leaning on the raw p here would be exactly the multiplicity error this project's own harness
rules were written against. No paper claim has been touched.

**Comparability verified (2026-08-04), because the expression_only lanes are NOT mine** — they came
from the 2026-07-29 campaign, so an unnoticed settings difference would have manufactured this
entire effect. Both arms match on n_epochs cap (20), lambda_graph (0), n_train (21,496) and n_val
(3,632). The pairing is valid; this is not a settings artifact.

**Caveat that cuts both ways: fold and difficulty are confounded across the three splits.** Each
split is ONE realization, so its val set differs in composition as well as in difficulty. A
difference between splits cannot be attributed to difficulty rather than to which targets happened
to land in val. That weakens any "the graph helps as difficulty rises" reading of this result, and it
equally weakens the h2a "dose-response" framing below. Separating them needs several realizations per
difficulty level, which has not been run.

h1 is $-0.0009$ (frozen, n=5) and $+0.0005$ (c075c15, n=5), both parity.

**Pre-committed rule (rail 4), stated BEFORE the remaining seeds land:** if h1 on c080c10 comes out
POSITIVE and survives BOTH Bonferroni and Holm at n>=4, that is a CONTRADICTION STOP. Do not rewrite
the null, do not integrate it, do not re-run hoping it reverses. Snapshot the artifacts, replace this
section with an `UNEXPECTED — NEEDS HUMAN REVIEW` banner carrying the numbers, and leave every paper
claim untouched for a human to adjudicate. If it lands as parity, it becomes the third point of the
difficulty curve and is integrated normally under rail 5.

condition_gated s2/s3/s4 are running; expression_only is already n=5 there. typed_static s0 started
09:21 to begin the h2a dose-response (frozen $-0.0131$ survives -> c080c10 ? -> c075c15 $-0.0012$
does not).

## 0. Blocker fixed first: every CUDA lane was dying in under a minute

`nvidia-smi` and every `--device cuda` lane failed with
`NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED at PeerToPeerAccess.cpp:83`,
while a bare `torch` matmul on the same card succeeded. Cause: `run_screening.run()` sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for cuda (run_screening.py:117-120); expandable
segments use the CUDA driver API, whose PyTorch wrapper calls NVML; and the loader binds
`libnvidia-ml.so.1` to a **third-party 535.309.01 driver tree another user left on the system library
path** while the running kernel module is 580.173.02. Clearing `LD_LIBRARY_PATH` does not help.
Fix, no root needed: `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02`.
The same assert is sitting in `screening_c075c15/experiment_registry.yaml` as `failed` rows from
2026-07-29, so this had already cost lanes once. Written up in AGENTS.md.

## 1. Lane A: harder-split n=5 (RUNNING at time of writing)

`./run_c075c15_n5.sh` -> fresh root `data/results/screening_c075c15_n5/`, seeded with copies of the 10
landed lanes; the source root is sha256-manifested before and after. Launched 07:31 IST, ~50 min/epoch,
expected to finish ~18:30.

| card | lane | then |
|---|---|---|
| 0 | condition_gated seed 2 | typed_static seed 2 |
| 1 | condition_gated seed 4 | typed_static seed 3 |
| 2 | typed_static seed 0 | untyped_gnn seeds 2, 4 |
| 3 | typed_static seed 1 | untyped_gnn seed 3 |

This takes h1 on the harder split from n=3 to **n=5**, and starts h2a (the one contrast that survives
correction) on that split, which the 2026-07-29 campaign never reached.

**Seed 2 is RE-RUN, not resurrected.** `screening_c075c15/condition_gated/2/` holds a `stage_a_best.pt`
from a lane that trained 13 epochs and was then killed by the campaign wind-down. Its peers all stopped
by `EARLY_STOP_PATIENCE=10`, exactly 10 epochs after their best; seed 2's best was epoch 7, so it was
killed 5 epochs BEFORE its stopping rule could fire. Its checkpoint is an argmin over a truncated range,
so scoring it would bias `condition_gated` downward, which is the direction that manufactures the null.

Gate health after epoch 0: condition_gated s2 $0.4672$, s4 $0.5255$ (live); typed_static pinned at
$1.0$ by design. No collapse.

### 1a-final. RESULT (n=5, 16:47 IST): the harder-split null at full strength

`condition_gated` seed 2 landed (12 epochs, gate $0.4672 \to 0.2218$ ALIVE, 9.26 GPU-h,
systema $0.083641$), giving **n=5 for both condition_gated and expression_only** on the harder split,
matching the frozen fold's design exactly.

| Fold | Δ systema | 95% CI | p | n | verdict |
|---|---|---|---|---|---|
| frozen 0.85 | $-0.0009$ | $[-0.0072,+0.0054]$ | 0.71 | 5 | parity |
| **harder 0.75/0.15** | $\mathbf{+0.0005}$ | $\mathbf{[-0.0105,+0.0115]}$ | **0.904** | **5** | **parity** |

Per-seed paired differences: $-0.0003$, $+0.0035$, $+0.0032$, $-0.0138$, $+0.0100$.
Per-arm: condition_gated $0.0809$ (n=5), untyped_gnn $0.0805$ (n=2), expression_only $0.0804$ (n=5),
typed_static $0.0794$ (n=2).

**NO CONTRADICTION STOP.** The point estimate is nominally *positive* here, unlike the frozen fold's
$-0.0009$, but it clears neither correction (Bonferroni and Holm both $1.000$), the interval straddles
zero almost symmetrically, and one seed ($-0.0138$) moves the mean more than the estimate itself. The
sign is undetermined at this budget. This is parity, not a reversal, and the paper says so explicitly
rather than quietly reporting a positive point estimate.

### 1c. THE BIG ONE (22:54 IST): h2a, our only corrected-significant contrast, does NOT replicate

The c075c15 campaign finished at 22:54 (0 failures, all gates alive: condition_gated $0.22$ to $0.51$,
typed_static pinned $1.0$). Full family, both corrections:

FINAL, all four arms at n=5 on both folds (03:21 IST, typed_static s4 landed; no INCOMPLETE or
UNBALANCED warnings, so this is a clean comparable family):

| Contrast | frozen 0.85 (n=5) | harder 0.75/0.15 (n=5) | survives? |
|---|---|---|---|
| h1 (cg - eo) | $-0.0009$, p=0.71 | $+0.0005$, CI $[-0.0105,+0.0115]$, p=0.904 | neither, both folds |
| **h2a (ts - eo)** | $\mathbf{-0.0131}$, p=0.0036, **SURVIVES both** | $\mathbf{-0.0012}$, CI $[-0.0074,+0.0050]$, **p=0.614** | **NEITHER** |
| h2b (cg - ts) | $+0.0122$, p=0.023 | $+0.0017$, p=0.503 | neither |
| promotion (ug - eo) | $+0.0045$, p=0.021 | $+0.0028$, p=0.337 | neither |

Per-arm (harder, n=5): untyped_gnn $0.0831$, condition_gated $0.0809$, expression_only $0.0804$,
typed_static $0.0792$. **On the harder split NOTHING in the family survives correction.**

h2a per-seed: $-0.0010$, $-0.0013$, $-0.0033$, $-0.0070$, $+0.0066$. **Seed 4 reverses the sign**
(typed_static $0.08656$ > expression_only $0.07996$), which is what collapses the mean. An
**elevenfold** shrink from $-0.0131$, not the fourfold the n=4 interim suggested.

Variance ratios also firm up (harder fold, all n=5): condition_gated $31.7\times$, untyped_gnn
$20.6\times$, typed_static $17.4\times$ the no-graph SD of $0.00028$. Every graph arm is far noisier,
and more so than on the frozen fold ($5.9$ to $10.8\times$).

**"The typed static graph is reliably worse" is a claim about ONE fold.** On the harder split the
deficit shrinks about 4x and loses significance. Per-seed h2a diffs are heterogeneous ($-0.0010$,
$-0.0013$, $-0.0070$), and the seed dominating h2a is the same seed dominating h1, so this reads as a
seed-level effect on this fold, not an arm-level one. h2b and the promotion margin move the same way.

This is NOT a contradiction stop (rail 4 covers the graph HELPING; it still does not). It is the
opposite: a claimed NEGATIVE effect failing to replicate. It is integrable under rail 5 (typed_static
n=4, expression_only n=5, result is a null). **The paper has been revised in six places** (abstract,
intro bullet, sec:null h2a sentence, a new "our one significant contrast did not replicate"
paragraph, cause E, the causes table, and the conclusion), plus a new checklist item: *replicate the
contrast that survived, not only the one that failed*. A version reporting only the frozen fold would
have overstated it.

typed_static seed 4 is still running -> h2a at n=5; the number will be refreshed when it lands.

**WATCH (superseded by 1c above; kept as the record of what n=2 suggested):** h2a was $-0.0012$ ($p=0.075$ at n=2), against
$-0.0131$ on the frozen fold: the direction replicates but the magnitude is about **10x smaller**.
Because the per-seed spread is tiny ($-0.00104$, $-0.00132$), it may still clear both corrections at
n=4 while being a much smaller effect, so effect size and significance would part company. Needs
typed_static seeds 2 and 3 (queued) before rail 5 allows integration.

### 1a. INTERIM (n=4, 14:15 IST): the harder-split null holds and tightens

`condition_gated` seed 4 landed clean (12 epochs, best epoch 1, gate $0.5255 \to 0.5090$ ALIVE,
6.71 GPU-h, systema $0.089951$). With seeds 0, 1, 3, 4 the harder split now has **n=4 for
condition_gated and n=5 for expression_only, which clears rail 5.**

| | Δ systema | 95% CI | p | n |
|---|---|---|---|---|
| harder split, previous | $-0.0035$ | $[-0.0262,+0.0191]$ | 0.57 | 3 |
| **harder split, now** | $\mathbf{-0.0002}$ | $\mathbf{[-0.0161,+0.0158]}$ | **0.978** | **4** |

Per-seed paired differences: $-0.000250$ (s0), $+0.003475$ (s1), $-0.013826$ (s3), $+0.009991$ (s4).
Point estimate nearer zero than at n=3, interval narrower (width 0.045 -> 0.032).
**No CONTRADICTION STOP: the graph did not help.** Seed 2 still running (best epoch 1, so it should
early-stop at epoch 11) -> n=5. typed_static s0/s1 still running -> h2a on this split, which the
2026-07-29 campaign never reached.

### 1b-bis. Difficulty is monotone on the RIGHT measure, and absolute scores are not comparable

Verified from the three leakage reports (2026-08-04). Median train-to-held-out sequence cosine:

| Split | thr/cap | families | **median train-to-val cosine** | absolute systema (eo) |
|---|---|---|---|---|
| frozen | 0.85/0.05 | 5,141 | **0.7964** (easiest) | $0.0857$ |
| c080c10 | 0.80/0.10 | 4,155 | **0.7591** | $0.0984$ |
| c075c15 | 0.75/0.15 | 3,282 | **0.7407** (hardest) | $0.0804$ |

Difficulty is monotone in the median cosine, so a difficulty *curve* is a defensible framing. But
**absolute systema is NOT monotone and must never be compared across splits** — the val sets differ
in size and composition (3,632 to 4,400 rows), and c080c10 scores highest while sitting in the
middle on difficulty. Only the paired within-split contrast is comparable across folds. The paper
now says this explicitly, and the earlier "~0.79 to ~0.73" in this file was imprecise: it is
0.796 to 0.741.

### 1b. NEW FINDING: the graph changes the variance, not the mean

Across seeds the no-graph arm is very stable and every graph arm is not. Per-seed SD of systema:

| Fold | expression_only | condition_gated | typed_static | untyped_gnn |
|---|---|---|---|---|
| frozen 0.85 (n=5) | $0.00051$ | $0.00545$ (**10.8x**) | $0.00488$ (9.6x) | $0.00301$ (5.9x) |
| harder 0.75/0.15 | $0.00028$ | $0.00997$ (**36x**) | pending | n=2, not read |

Adding the graph does not move the expected score; it inflates run-to-run spread at the same or a
lower mean. This is self-defeating for anyone trying to demonstrate a benefit, because the variance
the graph adds is the variance that widens the paired interval a benefit would have to clear. It
reproduces on both folds and is now a paragraph in the paper (sec:null) plus a checklist item.

## 2. Multi-dataset replication: gated, built, and honestly short of the target

Two documents were written before any replication compute, and are the gate on it:
`docs/replication-dataset-survey.md` (every number read from the actual data file, not the paper) and
`docs/replication-prereg.md` (contrast family, per-dataset primary contrast, kill criteria, integration
rules, frozen; one dated amendment pins the replicate unit per dataset).

New standalone package `src/tcell_pipeline/replication/` turns a harmonised scPerturb `.h5ad` into the
DE-statistics matrix the pipeline consumes. It touches no module a running lane has loaded. Its
self-check is mutation-tested (3/3 detected) and carries a regression case for a real bug found on real
data: Datlinger and Shifrut label controls only in `perturbation` and leave `target` NaN, so reading
controls off the target column silently dropped every control.

**Six candidates downloaded (Zenodo 7041849, CC-BY-4.0, checksums recorded) and built. Four fall out.**

| Dataset | Targets kept | Conditions | Replicate unit | Verdict |
|---|---|---|---|---|
| FrangiehIzar2021 | **216**/248 | **3** | sgRNA | usable, the only one that can test h1 |
| NormanWeissman2019 | **102**/105 | 1 | gemgroup | usable, h2a only; no PINNACLE context -> ESM-2-only |
| PapalexiSatija2021 | 24/25 | 1 | hto | preliminary, below the 50-family floor |
| ShifrutMarson2018 | 20/21 | 1 (lost one) | donor | preliminary, below the floor |
| ReplogleWeissman2022 RPE1 | **12**/2,393 | 1 | batch | **DROPPED by prereg** |
| DatlingerBock2017 | 6/32 | 2 | replicate | unusable |

Replogle is the important one. It has the best target count available (2,393 against the reference's
11,526) but ~103 cells per target over 56 batches is ~1.8 cells per pseudobulk, so only 12 targets clear
the 25-cell floor with >=2 replicates, and no other replicate axis exists (`guide_id` is a dual-guide
construct at ~1.1 per target). The pre-registration says such a dataset is **dropped, not switched to a
different test**, so it was not rescued with random pseudo-replicates. That rule was written before
these numbers existed. **The goal asked for >=4 replication datasets; the data supports 2 at this
protocol strength, and manufacturing 4 would require weakening exactly the discipline this paper is
about.**

**Adapter validated against biology it cannot fake** (Frangieh): knockouts reduce their own transcript
(own-gene log2FC mean $-0.632$, 89% negative, vs $-0.014$/61% for random genes), and IFNGR1 knockout
blunts ISGs (STAT1, GBP1, IRF1, CXCL10, HLA-B) by 1.0 to 3.0 log2 units under IFN-gamma and co-culture
but not under Control (within $\pm0.17$) — which also proves the condition axis is correctly assigned.

**NOT trained.** These are DE matrices only. Running the four arms on them needs
`config.CONDITIONS`, `config.PINNACLE_CONTEXT` and the DE paths made dataset-scoped instead of
module-level constants, deliberately not attempted while a GPU campaign was in flight.

## 3. ICBINB-BIO paper

`paper/icbinb/main.tex` (new; `paper/main.tex` untouched). Stock `neurips_2026.sty`, single column,
`\usepackage[dblblindworkshop]` + `\workshoptitle`. **Compiles clean: 0 errors, 0 undefined
references, 0 overfull boxes >2pt, 0 banned words, 0 dashes; body 7 pages** (references start p8,
appendix p10), so it is inside the 8-page limit with about a page spare for the harder-split n=5 result.

Restructured rather than reformatted: it leads with the instrument failure, then the null, then a
**five-cause decomposition of why the graph fails** with a verdict per cause (the rubric weights this
above the number), then a boxed checklist; the architecture moved to the free appendix. Required LLM
disclosure written and honest about agent-written code. New finding promoted into the body: the
`expression_only` baseline is **not graph-free** — `TargetEncoder` gives it a 128-d PINNACLE embedding
(learned on a PPI network) plus three PPI-degree scalars, so h1 measures the marginal value of message
passing over a graph summary, not graph versus no graph.

---

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

## Multi-dataset replication — the measured result (2026-08-10)

Four datasets, four seeds per arm, 20-epoch cap with early stopping, `lambda_graph=0`, blocked
target-OOD split refit per dataset, program basis refit inside each dataset's own train fold.
Every lane listed landed; none was dropped. Arms per prereg Amendment 3.2.

| dataset | cells | rows | K | PINNACLE ctx | arms x seeds |
|---|---|---|---|---|---|
| FrangiehIzar2021_RNA | melanoma, 3 conditions | 702 | 128 | melanocyte | 4 x 4 |
| ReplogleWeissman2022_rpe1 | RPE1, 1 condition | 2,122 | 128 | RPE cell | 3 x 4 |
| ReplogleWeissman2022_K562_essential | K562, 1 condition | 2,003 | 128 | none | 3 x 4 |
| TianKampmann2021_CRISPRi | iPSC neuron, 1 condition | 184 | 32 *(deviation)* | none | 3 x 4 |

### Per-dataset contrasts (n, per-seed deltas, both corrections)

| dataset | contrast | n | mean | 95% CI | p | Bonf | Holm | family | survives | per-seed |
|---|---|---|---|---|---|---|---|---|---|---|
| FrangiehIzar2021_RNA | h1_vs_no_graph | 4 | +0.0033 | [-0.0829, +0.0894] | 0.9111 | 1.0000 | 1.0000 | 4 | no | +0.0076, +0.0409, -0.0746, +0.0392 |
| FrangiehIzar2021_RNA | h2a | 4 | +0.0256 | [-0.0494, +0.1005] | 0.3572 | 1.0000 | 1.0000 | 4 | no | +0.0935, -0.0146, +0.0069, +0.0164 |
| FrangiehIzar2021_RNA | h2b | 4 | -0.0223 | [-0.1371, +0.0926] | 0.5809 | 1.0000 | 1.0000 | 4 | no | -0.0858, +0.0555, -0.0815, +0.0228 |
| FrangiehIzar2021_RNA | promotion_margin | 4 | +0.0194 | [-0.1220, +0.1608] | 0.6920 | 1.0000 | 1.0000 | 4 | no | -0.1127, +0.0782, +0.0488, +0.0633 |
| ReplogleWeissman2022_rpe1 | h2a | 4 | -0.0196 | [-0.0759, +0.0367] | 0.3490 | 0.6981 | 0.3490 | 2 | no | -0.0115, +0.0138, -0.0111, -0.0697 |
| ReplogleWeissman2022_rpe1 | promotion_margin | 4 | +0.0675 | [+0.0316, +0.1033] | 0.0093 | 0.0186 | 0.0186 | 2 | **YES** | +0.0356, +0.0724, +0.0887, +0.0731 |
| ReplogleWeissman2022_K562_essential | h2a | 4 | +0.0030 | [-0.0027, +0.0087] | 0.1932 | 0.3864 | 0.2230 | 2 | no | +0.0070, +0.0021, -0.0015, +0.0044 |
| ReplogleWeissman2022_K562_essential | promotion_margin | 4 | +0.0081 | [-0.0034, +0.0197] | 0.1115 | 0.2230 | 0.2230 | 2 | no | +0.0177, +0.0090, +0.0005, +0.0052 |
| TianKampmann2021_CRISPRi | h2a | 4 | +0.0332 | [-0.0457, +0.1121] | 0.2727 | 0.5453 | 0.5453 | 2 | no | +0.0861, +0.0627, +0.0046, -0.0205 |
| TianKampmann2021_CRISPRi | promotion_margin | 4 | +0.0281 | [-0.0929, +0.1491] | 0.5138 | 1.0000 | 0.5453 | 2 | no | +0.1414, -0.0041, -0.0224, -0.0026 |

### Pooled across datasets

| contrast | k | fixed-effect | random-effects | RE p | I2 | tau2 | Q p |
|---|---|---|---|---|---|---|---|
| h1_vs_no_graph | 1 | +0.0033 [-0.0498, +0.0563] | +0.0033 [-0.0498, +0.0563] | 0.9034 | 0.0% | 0.00000 | None |
| h2a | 4 | +0.0031 [-0.0004, +0.0065] | +0.0042 [-0.0105, +0.0189] | 0.5749 | 25.6% | 0.00008 | 0.258 |
| h2b | 1 | -0.0223 [-0.0930, +0.0485] | -0.0223 [-0.0930, +0.0485] | 0.5372 | 0.0% | 0.00000 | None |
| promotion_margin | 4 | +0.0139 [+0.0072, +0.0206] | +0.0326 [-0.0105, +0.0758] | 0.1385 | 88.2% | 0.00136 | 0.0 |

`h1` and `h2b` pool over k=1 because Frangieh is the only qualified multi-condition dataset;
they are single-dataset results reported as such, not pooled claims.

### Gate health — the thing that made the frozen H1 undecidable

| dataset | arm | seed | epochs | gate first | gate last | verdict |
|---|---|---|---|---|---|---|
| FrangiehIzar2021_RNA | condition_gated | 0 | 16 | 0.6788 | 0.3630 | ALIVE |
| FrangiehIzar2021_RNA | condition_gated | 1 | 13 | 0.4786 | 0.4461 | ALIVE |
| FrangiehIzar2021_RNA | condition_gated | 2 | 12 | 0.5098 | 0.4682 | ALIVE |
| FrangiehIzar2021_RNA | condition_gated | 3 | 11 | 0.5522 | 0.5752 | ALIVE |
| FrangiehIzar2021_RNA | typed_static | 0 | 18 | 1.0000 | 1.0000 | ALIVE |
| FrangiehIzar2021_RNA | typed_static | 1 | 11 | 1.0000 | 1.0000 | ALIVE |
| FrangiehIzar2021_RNA | typed_static | 2 | 12 | 1.0000 | 1.0000 | ALIVE |
| FrangiehIzar2021_RNA | typed_static | 3 | 12 | 1.0000 | 1.0000 | ALIVE |
| ReplogleWeissman2022_K562_essential | typed_static | 0 | 20 | 1.0000 | 1.0000 | ALIVE |
| ReplogleWeissman2022_K562_essential | typed_static | 1 | 20 | 1.0000 | 1.0000 | ALIVE |
| ReplogleWeissman2022_K562_essential | typed_static | 2 | 20 | 1.0000 | 1.0000 | ALIVE |
| ReplogleWeissman2022_K562_essential | typed_static | 3 | 20 | 1.0000 | 1.0000 | ALIVE |
| ReplogleWeissman2022_rpe1 | typed_static | 0 | 11 | 1.0000 | 1.0000 | ALIVE |
| ReplogleWeissman2022_rpe1 | typed_static | 1 | 13 | 1.0000 | 1.0000 | ALIVE |
| ReplogleWeissman2022_rpe1 | typed_static | 2 | 12 | 1.0000 | 1.0000 | ALIVE |
| ReplogleWeissman2022_rpe1 | typed_static | 3 | 11 | 1.0000 | 1.0000 | ALIVE |
| TianKampmann2021_CRISPRi | typed_static | 0 | 12 | 1.0000 | 1.0000 | ALIVE |
| TianKampmann2021_CRISPRi | typed_static | 1 | 15 | 1.0000 | 1.0000 | ALIVE |
| TianKampmann2021_CRISPRi | typed_static | 2 | 16 | 1.0000 | 1.0000 | ALIVE |
| TianKampmann2021_CRISPRi | typed_static | 3 | 13 | 1.0000 | 1.0000 | ALIVE |

Every graph lane with a learnable gate ended ALIVE. That is the difference between this
campaign and the frozen H1, whose gates were annihilated inside epoch 0 by an unnormalised
`L_graph` roughly 103x the response term. With `lambda_graph=0` there is no such term, so these
nulls are decidable experiments rather than instrument failures. `untyped_gnn` has no learnable
edge gate by construction and is absent from the table by design.

### What was excluded, and why

- `condition_gated` was NOT run on the three single-condition datasets. The condition gate needs
  >= 2 contexts; with one it is arithmetically `typed_static`. Pre-registered, not attrition.
- Datlinger 2017 and Shifrut 2018 failed on-target QC (51% / 50% direction consistency) and are
  out of the pool entirely. They were the only T-cell screens, so **this replication contains no
  T-cell replication of the reference result.**
- One lane (rpe1 / typed_static / s2) OOM'd when another user's process took 66 GiB of the card;
  it was re-run to completion and is included. No lane is missing from any reported n.
