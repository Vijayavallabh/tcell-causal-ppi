# B2 CLOSED (2026-08-19): the graph's harm is confined to the twenty genes each perturbation moves hardest

No training. Pre-registered as Amendment 8 before the numbers were read. Artifact
`data/results/b2_deciles/deciles.json`, module `screening/rank_deciles.py`, n=5 seeds both arms.

A3 left the paper saying the untyped contrast "crosses zero between the 250th and 500th DE gene". That
is a fact about a CUMULATIVE statistic and it is misleading about the thing a reader cares about. Every
top-k set contains all smaller ones, so a sign change in the running average only bounds where the
per-gene effect turned. Re-scoring the same stored predictions on DISJOINT rank intervals separates them.

## promotion_margin (untyped_gnn - expression_only), disjoint intervals, family of 36 cells

| rank interval | mean | 95% CI | bonf | holm | seeds positive | survives |
|---|---|---|---|---|---|---|
| 1-20 | **-0.0424** | [-0.0485, -0.0363] | 0.0015 | 0.0014 | 0/5 | **YES (negative)** |
| 21-50 | -0.0043 | [-0.0097, +0.0010] | 1.0000 | 0.6210 | 1/5 | no |
| 51-100 | -0.0013 | [-0.0050, +0.0023] | 1.0000 | 1.0000 | 2/5 | no |
| 101-250 | +0.0074 | [+0.0038, +0.0110] | 0.1643 | 0.1055 | 5/5 | no |
| 251-500 | +0.0095 | [+0.0061, +0.0128] | 0.0497 | 0.0414 | 5/5 | **YES** |
| 501-1000 | +0.0126 | [+0.0097, +0.0155] | 0.0093 | 0.0083 | 5/5 | **YES** |
| 1001-2500 | +0.0137 | [+0.0108, +0.0166] | 0.0068 | 0.0062 | 5/5 | **YES** |
| 2501-5000 | +0.0147 | [+0.0126, +0.0167] | 0.0014 | 0.0013 | 5/5 | **YES** |
| 5001-10282 | +0.0096 | [+0.0084, +0.0107] | 0.0008 | 0.0008 | 5/5 | **YES** |

Both directions clear correction over the SAME 36-cell family, which is the form Amendment 8.4 permits:
the contrast is corrected-significant NEGATIVE on ranks 1-20 and corrected-significant POSITIVE on every
interval from 251 to 10,282.

**The cumulative crossover at 250-500 is an artifact of accumulation.** The per-interval effect is
already positive in 5/5 seeds by rank 101-250; the running average stays negative to k=250-500 only
because it carries the top-20 deficit forward until enough positive mass cancels it. The paper's
appendix now says so, and states the reading as "the graph helps everywhere except the handful of genes
each perturbation moves hardest" rather than "the graph helps beyond the 500th gene".

## The resolution of the binning decides whether the harm is visible AT ALL

At decile resolution the top decile spans ranks 1-1028 and reads **+0.0064, clearing nothing**: the
twenty damaged genes are diluted by a thousand that benefit. A decile analysis of the identical
predictions would have concluded the graph never hurts. This is why both binnings are reported, and it
is the transferable methodological point: a claim about where a prior helps is only as sharp as the
resolution it was measured at.

## h2a is uniform, not localised

typed_static - expression_only is negative in EVERY interval, shrinking monotonically from -0.0463
(ranks 1-20) to -0.0051 (ranks 5001-10282), and clears nothing over the 36-cell family. That is NOT a
contradiction of tab:ksweep, where h2a stars at every k: there the family is four cells, here it is
thirty-six. The typed encoder's deficit is spread across the whole ranking, where the untyped arm's
benefit is blocked only at the very top.

## What is confounded, and conceded in advance (Amendment 8.2)

Genes are selected into an interval by the same observed response that is the correlation's target, so
range restriction and selection on a noisy statistic both bias a bin's LEVEL. No claim rests on
comparing one interval's level to another's. The within-interval CONTRAST is clean: both arms are scored
on the identical genes of the identical rows, and neither arm's prediction enters the selection.

# A2(a) CLOSED (2026-08-19): the detection floor is at or below 0.02 response SDs, and the control works

48 lanes, 6 conditions, 4 paired seeds each, 0 failures. Pre-registered in Amendment 6 before any lane
ran. Artifact `data/results/a2_ladder/floor.json`.

A known graph-dependent component was added to the real responses: for each target, delta times the mean
response of its PPI neighbours, computed from TRAIN ROWS ONLY, scaled so delta reads as a fraction of a
response SD. One rung is a NEGATIVE CONTROL in which each target gets some OTHER target's neighbourhood
at the same magnitude.

## The pre-registered primary (family of six, Bonferroni AND Holm, both required)

| condition | delta | gap | 95% CI | bonf | holm | recovered |
|---|---|---|---|---|---|---|
| no injection | 0 | +0.0048 | [-0.0000, +0.0097] | — | — | zero point, not in the family |
| injected | 0.02 | **+0.0088** | [+0.0071, +0.0105] | 0.0029 | 0.0029 | YES |
| injected | 0.05 | +0.0081 | [+0.0052, +0.0110] | 0.0175 | 0.0117 | YES |
| injected | 0.10 | +0.0041 | [+0.0026, +0.0056] | 0.0198 | 0.0117 | YES |
| injected | 0.20 | +0.0090 | [+0.0072, +0.0109] | 0.0033 | 0.0029 | YES |
| injected | 0.40 | +0.0183 | [+0.0112, +0.0254] | 0.0225 | 0.0117 | YES |
| **scrambled** | 0.40 | **-0.0001** | [-0.0021, +0.0018] | 1.0000 | 0.8333 | **NO** |

**MEASURED FLOOR = 0.02 response SDs**, the smallest size injected. Every injected rung is recovered;
the control is not.

## The control is what makes the number readable

At the same injection magnitude, a scrambled neighbourhood gives a gap of -0.0001 with a CI of width
0.004 around zero. So the gaps above are the graph being READ, not the injection being large.

The absolute scores say it from the other side:

| condition | expr_only | untyped | vs the un-injected reference |
|---|---|---|---|
| delta=0 | 0.0856 | 0.0904 | — |
| delta=0.40 | 0.1679 | 0.1862 | expr **+0.0823**, untyped **+0.0958** |
| **scrambled 0.40** | 0.0830 | 0.0828 | expr **-0.0027**, untyped **-0.0076** |

Injected structure that FOLLOWS the graph makes the task easier for both arms. Injected structure of
identical magnitude that does NOT follow it makes the task slightly harder for both. Nothing here is a
magnitude artifact.

## POST-HOC: what the injection bought over the benefit already there

Committed while two rungs were still training, so it cannot have been shaped by its own numbers.

| condition | increment over delta=0 | 95% CI | p |
|---|---|---|---|
| 0.02 | +0.0040 | [-0.0024, +0.0104] | 0.14 |
| 0.05 | +0.0033 | [-0.0030, +0.0095] | 0.19 |
| 0.10 | -0.0007 | [-0.0068, +0.0054] | 0.73 |
| 0.20 | +0.0042 | [-0.0017, +0.0102] | 0.11 |
| 0.40 | **+0.0135** | [+0.0022, +0.0248] | **0.03** |
| **scrambled 0.40** | **-0.0050** | [-0.0083, -0.0016] | **0.02** |

The primary tests each rung against ZERO, and this fold already carries a real +0.0048 with no injection,
so a rung can clear on the pre-existing benefit alone. Subtracting it: the pipeline **detects
graph-dependent structure at 0.02**, and at four seeds it **cannot distinguish an ADDED benefit from the
one already present until delta=0.40**. Both numbers are reported; the first is the pre-registered rule.

**The control carries its own finding.** Its increment is significantly NEGATIVE (-0.0050): injecting
graph-shaped structure that follows the WRONG graph does not merely fail to help, it destroys the real
benefit that was there. A confidently wrong prior is worse than no prior, measured rather than asserted.

## Consistency with A2(b)

The simulation predicted an MDE of 0.0075 systema at n=7 on the frozen fold. The delta=0.02 rung produced
a gap of +0.0088 and was detected at n=4. The empirical instrument does about what the modelled one said
it would.

---

# A1 CLOSED (2026-08-18): edge typing hurts, and NEITHER of the two candidate routes is why

Ten lanes, n=5 both arms, frozen fold, pre-registered in Amendments 4, 4a and 4b BEFORE any of them ran.
83 GPU-hours. Artifact `data/results/screening_a1/a1_mechanism.json`.

The n=7 family established that edge typing costs -0.0120 systema. Two explanations were confounded
inside that number: the relation PARTITION being the wrong inductive bias, and typed message passing
simply carrying 4x the message parameters over the same edges. Two arms separate them.

| contrast | what it removes | mean | 95% CI | p | bonf | holm | survives |
|---|---|---|---|---|---|---|---|
| **D1** typed_shared − typed_static | the per-relation parameter multiplicity (4 message modules -> 1; 1.80M fewer parameters, 34% of the model) | **+0.0004** | [-0.0048, +0.0057] | 0.83 | 1.000 | 0.829 | no |
| **D2** typed_permuted − typed_static | the typing's information (labels reassigned at random, counts preserved exactly) | **+0.0065** | [-0.0017, +0.0147] | 0.091 | 0.182 | 0.182 | no |

Family of two, Bonferroni AND Holm, survival requires both. **Both are null**, which is the
("null","null") cell of the 2x2 fixed in Amendment 4.4:

> The typed STRUCTURE hurts, and neither its parameter count nor its labels is the route.

**Per-arm means at n=5 on the frozen fold:**

    untyped_gnn      0.0902      the best graph arm
    expression_only  0.0857      no graph
    condition_gated  0.0838
    typed_permuted   0.0791      random relation labels
    typed_shared     0.0730      one message module for all relations
    typed_static     0.0726      the true typing

**What this rules out.**

- **Not capacity.** Cutting the encoder's message parameters by a factor of four — 2,396,160 to 599,040,
  a 34% cut in the model's total parameters (5,254,884 to 3,457,764) — moves systema by +0.0004. If the deficit were overfitting, this is the
  intervention that would have fixed it.
- **Not the annotation's content.** Giving every edge a random relation label, with each relation's edge
  count preserved exactly and the sampled neighbourhood bit-identical, does not significantly change the
  result. The point estimate is POSITIVE (+0.0065): if anything the true partition is slightly worse than
  a random one. That is suggestive and NOT established — it does not clear correction and is reported as
  a null under the rule fixed in advance.

**What it leaves.** Both diagnostic arms remain far below the untyped GCN (shared -0.0172
[-0.0199,-0.0145]; permuted -0.0111 [-0.0177,-0.0044]) and below no-graph (-0.0126 and -0.0065). So the
damage lives in what all three typed arms SHARE and the untyped one lacks: signed messages
(tanh x relu), the per-edge feature term, complex-membership nodes, the residual FFN, and `add`
aggregation instead of GCN symmetric normalisation.

**A4 makes that last item live again.** The 14-cell architecture search is what "refuted" the
normalisation lever. Its entire spread across all 14 cells was 0.0089, while the untyped-minus-typed gap
we measure at n=5 is 0.0176 — the search's single-seed cells were not resolving differences of the size
that actually separate these arms. "Refuted within the searched space" was never established, for
normalisation or for anything else it tried. This is a correction to a standing instruction in
`NEXT_ACTIONS.txt` ("do not re-propose per-relation normalisation... a known dead end"), and it is on
the record rather than acted on: nothing has been re-launched.

**No contradiction stop.** Neither diagnostic arm beats expression_only; both are below it. Amendment
4.5's ban stands anyway — a diagnostic arm may not be used to promote a graph claim.

---

# CONTRADICTION STOP FIRED — A3, 2026-08-16

**The sign of "does the graph help" DEPENDS ON WHICH REPORTED METRIC YOU CHOOSE, and both signs clear a
twenty-test family-wise correction under Bonferroni AND Holm.** Same predictions, same five seeds, same
frozen fold, same held-out rows. Nothing was retrained: this is the stored screening output re-scored.

| endpoint | who reports it | `promotion_margin` = untyped_gnn - expression_only |
|---|---|---|
| `pearson_delta` | TxPert | **+0.0089**, 95% CI [+0.0070, +0.0107], across-metric p 0.0037 — the graph HELPS |
| `pearson_delta_top20` | GEARS | **-0.0424**, 95% CI [-0.0485, -0.0363], across-metric p 0.0008 — the graph HURTS |

Both survive the ACROSS-METRIC bar (m=20, five endpoints times four contrasts, Bonferroni and Holm
both), which is the strictest bar this project applies anywhere. The correction was fixed in
Amendment 5.3 of the pre-registration BEFORE the full fold was read, precisely so that neither of these
could be the one that got reported.

**Rail 4 applies and has been honoured.** `pearson_delta/promotion_margin` is a POSITIVE graph benefit
surviving both corrections. The null has NOT been rewritten around it. Artifact snapshotted at
`data/results/a3_external/rescored.json`; every one of the twenty cells is in it.

Everything that survived the across-metric bar, with its sign:

| endpoint / contrast | mean | reading |
|---|---|---|
| `pearson_delta` / promotion_margin | +0.0089 | untyped graph beats no-graph |
| `pearson_delta_top20` / promotion_margin | -0.0424 | untyped graph loses to no-graph |
| `pearson_delta_top20` / h1_vs_no_graph | -0.0454 | the frozen H1 loses to no-graph |
| `energy_distance` / h2a | -2.0102 | typed_static matches the response DISTRIBUTION worse than no-graph |
| `energy_distance` / h2b | +5.3028 | condition gating matches it BETTER than typed static |

**WHERE THE SIGN FLIPS, measured (added 2026-08-16 20:20).** The two endpoints are the same statistic
over different gene sets, so the disagreement has a location. Sweeping the scored set from each
perturbation's top-20 observed DE genes to all 10,282 (`data/results/a3_external/k_sweep.json`):

| top-k DE genes | h2a | h2b | untyped − none | h1 |
|---|---|---|---|---|
| 20 | -0.0463* | +0.0009 | **-0.0424*** | -0.0454* |
| 50 | -0.0437* | +0.0120 | -0.0236* | -0.0317* |
| 100 | -0.0434* | +0.0189 | -0.0143* | -0.0244* |
| 250 | -0.0397* | +0.0240 | -0.0032 | -0.0157 |
| 500 | -0.0376* | +0.0263 | +0.0019 | -0.0113 |
| 1000 | -0.0345* | +0.0274 | **+0.0063*** | -0.0071 |
| 2500 | -0.0299* | +0.0264 | +0.0097* | -0.0034 |
| 5000 | -0.0249* | +0.0239* | +0.0108* | -0.0010 |
| all | -0.0197* | +0.0190* | +0.0089* | -0.0007 |

`*` = survives Bonferroni AND Holm within that k's family of four.

The untyped contrast is corrected-significant NEGATIVE up to k=100 and corrected-significant POSITIVE
from k=1000, crossing zero between the 250th and 500th DE gene. **Both effects are real.** The untyped
graph makes the broad response more accurate and the genes each perturbation moves hardest LESS
accurate. Which one a paper reports is a choice of k, and k is a convention: 20 because GEARS reports
20, all genes because TxPert reports all.

**h2a is the one row that does not depend on the choice** — negative and corrected-significant at every
k tried. The typed graph's deficit is not a metric artifact.

**Why this is a result and not a nuisance.** These are not exotic endpoints. `pearson_delta` is
TxPert's headline and `pearson_delta_top20` is GEARS'. A field in which the two disagree in sign, at
this significance, on the same predictions, cannot settle "does a graph prior help" by reporting one
number — and neither can we. It bounds what OUR null means too: our null is stated on SYSTEMA,
and SYSTEMA is one endpoint among several that do not agree.

**What it does NOT establish.** Nothing about splits. Outside results also use different folds, and
re-scoring our own predictions cannot speak to that; the split half of the commensurability hedge
stands. And nothing about single-cell distributions: this pipeline predicts one pseudobulk response per
(target, condition), so both distance endpoints compare the DISTRIBUTION OF RESPONSES across held-out
perturbations, never cell populations.

**A trap this run walked past, recorded so it is not walked into.** scPerturb's E-distance, computed the
way scPerturb computes it (squared euclidean distances), collapses algebraically to
`2*||mean(X) - mean(Y)||^2` — a difference of means. It cannot see a spread difference at all, and a
test pins that by feeding it two populations with identical means and 4x spread, where it reads zero.
It is reported for commensurability; `energy_distance` (plain euclidean, Szekely) is the distributional
evidence. Reading the squared form as distributional evidence would have been wrong in the paper's
favour.

Human review is wanted on how much of this to put in the paper. The material is in
`data/results/a3_external/rescored.json` and the endpoint definitions, with sources, are in
`src/tcell_pipeline/evaluation/external_metrics.py`.

---

# A5 CLOSED (2026-08-16): the rationale audit's ratios, given their sizes — and a correction against us

The audit compares every quantity against a matched random control. Right design, and every one of them
is a RATIO, in a regime where the denominator can be numerically zero. Absolute sizes, computed from the
landed audit (`data/results/a2_power/rationale_bound.json`, module
`tcell_pipeline.screening.rationale_audit_bound`):

| quantity | rationale | random control | gap | as fraction of control |
|---|---|---|---|---|
| sufficiency | 1.473 | 1.462 | **+0.0110** [+0.0063, +0.0158], p=2.6e-05, n=50 | **+0.8%** |
| necessity | 4.98e-07 | 3.34e-07 | +1.64e-07, p=5.4e-05 | **+49%** |
| GInX @ 20% sparsity | 0.084 | 0.030 | **+0.054** | +182% |

So: sufficiency is reliable and small. Necessity's 49% improvement is over a quantity of order 1e-7 —
deleting the selected edges moves the prediction by about **one part in ten million**, which makes the
reported "92% of cases beat random on necessity" true and nearly empty. GInX is the one comparison with
a substantive absolute gap.

**THE CORRECTION, and it goes against us.** The paper's cause-E evidence is "rationale sufficiency mass
sits almost entirely on that tier (Δ=0.365 for STRING against ≤0.0021 for BioPlex, HuRI and CORUM)".
STRING is **85.4%** of the graph's 8,029,296 protein-protein edges (bioplex 1.47%, biogrid 11.27%,
corum 1.21%, huri 0.64%), so ablating it removes most of the graph. Per 1% of edges removed:

| source | edges | share | Δ sufficiency | Δ per 1% of edges |
|---|---|---|---|---|
| string | 6,857,702 | 85.41% | 0.36469 | **0.00427** |
| huri | 51,773 | 0.64% | 0.00162 | 0.00251 |
| bioplex | 118,162 | 1.47% | 0.00208 | 0.00141 |
| corum | 96,778 | 1.21% | 0.00002 | 0.00002 |

STRING still leads, by **1.7x over HuRI and 3.0x over BioPlex** — not the ~175-225x the raw deltas
suggest. Only CORUM stays far behind. Cause E remains **Partly**; the verdict does not change, but the
evidence behind it is weaker than the paper stated and `app:causes` now says so.

Edge counts derived once from the frozen graph (source one-hot column of every PP edge_attr) and held
as a constant with its provenance; `--recount` re-derives and reports drift.

---

# A4 CLOSED (2026-08-16): the architecture search could not have found anything

The paper already says the 14-cell search was the wrong SHAPE — every cell varied HOW the typed encoder
passes messages and none asked whether the typing belonged. With A2(b)'s floor in hand there is a second
and simpler objection, and it is quantitative. Artifact:
`data/results/a2_power/arch_search_bound.json`, re-derivable with
`PYTHONPATH=src .venv/bin/python -m tcell_pipeline.screening.arch_search_power`.

| | |
|---|---|
| cells | 14, each ONE run at ONE seed, 5 epochs |
| best | `condition_gated__norm-add__thr-0.0__scale-0` (the DEFAULT), **+0.0051** vs no-graph |
| worst | `condition_gated__norm-gcn__thr-0.7__scale-0`, **-0.0037** |
| observed spread | **0.0089** |
| spread 14 draws of PURE SEED NOISE would give (sd 0.00431) | **0.0147** |
| verdict | the search's ENTIRE spread is INSIDE what re-seeding one configuration produces |
| seeds the best cell would need to clear our own correction rule at 80% power | **12** |
| seeds it was run at | **1**, where this pipeline emits no p-value at all |

**What this licenses the paper to say.** No variant in the searched space improves on the baseline by
more than the floor we can detect. The best cell is +0.0051 against a measured MDE of 0.0075 at the
seven seeds we ran, and the search ran at one.

**What it says about us.** We ranked 14 architectures on differences we had no power to resolve and
then treated the ranking as a refutation. That is the same failure as the rest of the paper, in a place
we had not looked. It is now in `app:archsearch` as a paragraph, not a footnote.

**Not re-proposed, per the standing instruction:** per-relation normalisation and confidence pruning are
already refuted and re-running them is a known dead end. This bound does not resurrect them; it says the
search that refuted them was underpowered too, so "refuted within the searched space" means less than it
sounded like. The A1 arms now running are the lever that search never pulled.

---

# A2(b) CLOSED (2026-08-16): the detection floor, simulated over the variance this project MEASURED

The paper carried an MDE from a normal approximation on five paired per-seed differences: "0.0085
uncorrected, roughly 0.013 under Bonferroni", with its own 95% span running 0.005 to 0.025. That is now
replaced by a simulation that runs the pipeline's OWN rule -- paired t, then Bonferroni AND Holm over a
family of four, survival requiring both -- 2,000 replicates per point, over variance components read
from landed artifacts. Artifact: `data/results/a2_power/power_simulation.json`.

**Detecting Δ=0.0043 (the size of the one real graph benefit we have), at 80% power:**

| generalise over | variance component | sd | units needed | power as run |
|---|---|---|---|---|
| seeds, this fold | seed, frozen fold | 0.0043 | **15 seeds** | 0.32 at n=7 |
| seeds, any re-draw of it | seed, pooled over levels (L4) | 0.0107 | **73 seeds** | 0.05 at n=7 |
| a fresh partition | level + re-draw + seed | 0.0058 | **24 levels** = 240 training lanes | — |
| datasets, typed arm | between dataset (τ) | 0.0032 | **10 datasets** | 0.58 at k=7 |
| datasets, untyped arm | between dataset (τ) | 0.0205 | **252 datasets** | 0.02 at k=7 |

**Measured MDE at the seeds we actually ran (n=7, 80% power):** 0.0075 on the frozen fold, 0.0185 if
the claim must also survive re-drawing the fold. The paper's old 0.013 sat between the two, understating
one and overstating the other.

**The quotable one.** The between-dataset spread of `promotion_margin` is τ=0.0205, five times the
effect anyone reports, because the datasets disagree in SIGN. Detecting a +0.0043 pooled benefit against
that spread needs on the order of **250 datasets**. scPerturb has 44. Cross-dataset perturbation studies
use single digits. Power at the k=7 we have is **2%**.

**A correction the simulation forced.** The paper said "the between-re-draw component is at least as
large as the between-seed one". Measured, it is not: for h2a the seed contributes 0.0107 against the
re-draw's 0.0003. The noise blamed on re-drawing the split was mostly the seed. Fixed in `app:power`.

**Two calibrations, so this is not a free-floating model.**
- delta=0 survives at 0.013, the corrected rate, not the nominal 0.05.
- `promotion_margin`'s known-true +0.0043 needs 9 seeds at its own measured sd of 0.0028; we ran 7,
  which the simulation puts at 70% power, and it did survive both corrections. The one true positive
  lands where the model says it should.

**What is NOT closed.** A2(a), the empirical floor: inject a known graph-dependent signal at a ladder of
effect sizes and report the smallest one recovered. The simulation says what the answer should be; only
the injection shows whether the pipeline achieves it. Still open in NEXT_ACTIONS.txt.

**Mutation-tested, because this project has shipped two tests that passed against buggy code.** Killed:
a dropped family-wise correction (the null rate moves 0.013 -> 0.058), a Holm-only survival rule, a
level component treated as shrinkable by compute, and a bisection returning the point below the
crossing. `src/tests/test_power_simulation.py`.

---

# UNEXPECTED — NEEDS HUMAN REVIEW (updated 2026-08-11)

**On the REFERENCE screen, at n=7, the untyped graph beats the no-graph baseline and survives BOTH
Bonferroni and Holm over the FULL pre-registered family of four.** This is the strongest form the
result can take in this project: the reference dataset, the frozen fold, the complete family, no
dropped seeds. Rail 4 applies — the null has not been rewritten around it.

| | |
|---|---|
| contrast | `promotion_margin` = untyped_gnn − expression_only |
| dataset | reference CD4+ T-cell screen, frozen blocked-target-OOD fold |
| n | **7** (seeds 0-6, none dropped) |
| effect | **+0.0043 systema**, 95% CI [+0.0017, +0.0069] |
| per seed | +0.0090, +0.0050, +0.0022, +0.0031, +0.0033, +0.0066, +0.0008 — all seven positive |
| correction | p=0.0068, **Bonferroni 0.0273, Holm 0.0205**, family_size **4** — survives BOTH |
| fold | comparable: registry evidence present, fold sizes [21262, 4400] consistent, single frozen fold |
| gate health | n/a — untyped_gnn has no learnable edge gate by construction |

At n=5 this same contrast was +0.0045 with Bonferroni 0.0832 and it FAILED. Two more paired seeds did
not change the estimate (+0.0045 -> +0.0043); they tightened it (CI half-width 0.0034 -> 0.0026). That
is what an underpowered true effect looks like when you add power, and it is the opposite of what a
fragile result looks like.

`family_size: 4` is the full family — expression_only, untyped_gnn, typed_static, condition_gated —
not the collapse-to-1 bug that produced three false "survives" earlier in this project.

**What this establishes.** The arm that wins carries PPI topology with NO edge typing and NO condition
gating. On this same fold the typed arms do not: typed_static is reliably worse and condition_gated is
indistinguishable. Combined with the replication (untyped outright winning on Replogle RPE1, +0.0675,
where the typed arm goes negative), the conclusion is no longer hedged and is no longer post-hoc on the
reference dataset:

> RETRACTED IN THIS PARAGRAPH, 2026-08-16: this block was written on 2026-08-11 and said "untyped
> positive on 5/5 independent datasets". The sixth dataset reversed that the same day — Norman,
> −0.0790, surviving both corrections — and over eight datasets the pooled random-effects estimate is
> +0.0091 [−0.0024, +0.0207], p=0.12, with three per-dataset contrasts clearing correction and
> DISAGREEING IN SIGN. See the CORRECTION section below. The reference-screen result in the table above
> is unaffected; only the cross-dataset sentence was wrong.

> The protein-interaction prior is not what failed. The typed, gated encoder built to exploit it is
> what discards the signal the raw topology carries.

This is now a pre-registered family contrast surviving both corrections on the dataset the project is
about, so the paper's cause-C claim no longer needs its post-hoc caveat there. The POOLED
cross-dataset statement remains post-hoc and keeps its caveat.

**Human review is still wanted** on whether to promote this from a diagnosis in a failure paper to a
positive claim in its own right. Everything needed to decide is in
`data/results/screening_untyped_n7/robustness_5seed.json`.

## n=7 on the frozen fold — the full family, FINAL (2026-08-11 14:51)

All four balance lanes landed. Every arm is now at n=7 on the frozen fold; the report is BALANCED
(common seeds 0-6) with no dropped seeds and no incomplete coverage.

| contrast | n | mean | 95% CI | p | Bonf | Holm | survives BOTH | per-seed |
|---|---|---|---|---|---|---|---|---|
| h2a `typed_static − expression_only` | 7 | **−0.0120** | [−0.0160, −0.0080] | 0.0003 | 0.0013 | 0.0013 | **YES** | −0.0075, −0.0140, −0.0143, −0.0199, −0.0098, −0.0106, −0.0083 |
| promotion_margin `untyped_gnn − expression_only` | 7 | **+0.0043** | [+0.0017, +0.0069] | 0.0068 | 0.0273 | 0.0205 | **YES** | +0.0090, +0.0050, +0.0022, +0.0031, +0.0033, +0.0066, +0.0008 |
| h2b `condition_gated − typed_static` | 7 | +0.0077 | [+0.0009, +0.0146] | 0.0327 | 0.1308 | 0.0654 | no | +0.0048, +0.0118, +0.0098, +0.0194, +0.0100, +0.0016, **−0.0033** |
| h1 `condition_gated − expression_only` | 7 | −0.0043 | [−0.0084, −0.0002] | 0.0424 | 0.1695 | 0.0654 | no | −0.0027, −0.0022, −0.0045, −0.0005, +0.0002, −0.0089, −0.0116 |

**CORRECTION to the interim n=6 entry.** At n=6, h2b was +0.0096 with Bonferroni 0.0491 and it
SURVIVED. The seventh seed came in at −0.0033, the only negative in that contrast, and moved it to
+0.0077 with Bonferroni 0.1308 — it no longer survives. Anything written from the n=6 interim saying
the gate's repair survives correction is wrong; this table is the record. Gate health on that lane was
fine (0.4076 -> 0.4324, ALIVE, 12 epochs), so it is a real seed, not a broken one.

This is the fifth time in this project an interim value has moved on the last seed, and the fourth time
it moved AGAINST the more interesting reading. That is the reason for the >=4-seed rule.

Arms ordered by systema at n=7:

    untyped_gnn      +0.0904   plain topology, no typing, no gating
    expression_only  +0.0861   no graph
    condition_gated  +0.0818   typed graph + condition gate
    typed_static     +0.0740   typed graph, gate pinned to 1

What survives correction, and what does not:

- **Plain topology beats no graph** (+0.0043, both corrections). Established.
- **Edge typing is actively harmful** (−0.0120, all seven seeds negative, both corrections at p=0.001).
  Established, and it is the largest effect in the family.
- The condition gate **partly** repairs typing's damage (+0.0077) but this does NOT survive correction.
- The gated graph is **worse than no graph at all** (h1 = −0.0043; raw CI excludes zero, fails
  correction).

So the honest statement is narrower than the interim one but points the same way: the graph's raw
topology carries signal, per-relation edge typing destroys more than that signal is worth, and the
condition gate — the component this project is named for — does not reliably win it back.

## CORRECTION 2026-08-11 (later): the untyped graph is NOT uniformly positive. Norman reverses it.

Two datasets the 2026-08-11 audit found built-but-untrained have now landed, and one of them overturns
the post-hoc claim recorded earlier today.

| dataset | n | untyped_gnn − expression_only | 95% CI | Bonf | survives | per-seed |
|---|---|---|---|---|---|---|
| **NormanWeissman2019_filtered** (K562, **activation**) | 4 | **−0.0790** | [−0.1351, −0.0229] | 0.0414 | **YES, NEGATIVE** | −0.0606, −0.0391, −0.1016, −0.1147 |
| TianKampmann2021_CRISPRa (neuron, activation) | 4 | +0.0226 | [−0.0621, +0.1073] | 0.9165 | no | +0.1003, +0.0084, +0.0022, −0.0204 |

Norman is a real result, not a broken lane: all four seeds negative, none dropped, `family_size` 2
(correct for single-condition), gates alive on every graph lane, 14-20 epochs each.

**What this does to the pooled claim.** Over seven independent datasets:

    fixed-effect   +0.0049 [+0.0029, +0.0069]   excludes zero
    random-effects +0.0093 [−0.0083, +0.0268]   p=0.30, INCLUDES ZERO
    I2 = 89.2%, Cochran Q p < 0.001
    sign test 6/7 positive, p = 0.125

This SUPERSEDES the five-dataset pool reported earlier today (+0.0208 [+0.0047, +0.0369], p=0.011),
which excluded zero. It no longer does. At I2 = 89% the random-effects reading governs, so the honest
pooled statement is now **no pooled benefit**, and the earlier "positive on 5 of 5" is a fact about
which five datasets had been trained, not a property of the method.

    Norman            −0.0790   ← survives correction, NEGATIVE
    reference (n=7)   +0.0043   ← survives correction, positive
    K562-essential    +0.0081
    Frangieh          +0.0194
    Tian CRISPRa      +0.0226
    Tian CRISPRi      +0.0281
    Replogle RPE1     +0.0675   ← survives correction, positive

**What survives unchanged.** The three per-dataset contrasts that clear both corrections are
pre-registered per-dataset tests and none of them is affected by the pooling: reference +0.0043,
RPE1 +0.0675, Norman −0.0790. What changed is that they no longer agree in sign.

**The honest reading is stronger than the one it replaces, not weaker.** A structure-only graph arm has
a REAL effect — three datasets clear family-wise correction — but its SIGN is dataset-dependent and
cannot be predicted in advance. That makes the checklist item sharper: run the structure-only arm not
because it will help, but because you cannot know from the typed arm alone whether your graph is
helping or hurting.

Both activation datasets do not behave alike (Norman −0.0790, Tian CRISPRa +0.0226), so this is NOT
simply "activation flips the sign". With n=2 activation datasets no direction claim is supportable.

**PENDING: genome-wide Replogle K562 (gwps, 9,730 targets) is training now.** It is 4.6x the largest
dataset here and will be the highest-weight point in the fixed-effect pool. These numbers are the
seven-dataset state and will need recomputing when it lands.

## EIGHT DATASETS — the genome-wide screen lands (2026-08-11 20:00)

ReplogleWeissman2022_K562_gwps: 9,730 targets, 12 lanes, 4 seeds per arm, none dropped. It is 4.6x the
next largest replication and the best-powered dataset in the study after the reference screen.

| contrast | n | mean | 95% CI | Bonf | survives |
|---|---|---|---|---|---|
| h2a `typed_static − expression_only` | 4 | **+0.0008** | [−0.0042, +0.0058] | 1.0000 | no |
| promotion_margin `untyped_gnn − expression_only` | 4 | +0.0069 | [−0.0024, +0.0163] | 0.1997 | no |

Where power is highest, BOTH contrasts are null and the intervals are the tightest single-dataset
values anywhere in the project (h2a half-width 0.005).

### Pooled, all datasets

    h2a, seven replication datasets   FE +0.0018 [-0.0005, +0.0040]
                                      RE +0.0018 [-0.0028, +0.0065]   I2 = 39.2%, Q p = 0.13
    untyped, eight datasets (incl. reference)
                                      FE +0.0051 [+0.0032, +0.0070]
                                      RE +0.0091 [-0.0024, +0.0207] p=0.12   I2 = 87.5%
                                      sign test 7/8 positive, p = 0.07

**The h2a null is now the strongest claim in the project.** Seven replication datasets, five cell
types, both perturbation directions, target counts from 100 to 9,730 — and I2 = 39.2% with Cochran's
Q p = 0.13, i.e. *no significant heterogeneity*. Every earlier pooled h2a carried I2 near 88%; this one
does not, because adding the well-powered datasets shrank the between-dataset spread rather than
widening it. A random-effects interval of [-0.0028, +0.0065] across that range is a bounded null in the
proper sense.

**The untyped-graph picture is unchanged by gwps.** +0.0069, positive but not significant, and the
pooled random-effects interval still includes zero. Three datasets clear family-wise correction on
their own pre-registered test and still disagree in sign: reference +0.0043, RPE1 +0.0675,
Norman -0.0790.

Per-dataset untyped effect, all eight:

    Norman            -0.0790   survives correction, NEGATIVE
    reference (n=7)   +0.0043   survives correction, positive
    K562 genome-wide  +0.0069
    K562-essential    +0.0081
    Frangieh          +0.0194
    Tian CRISPRa      +0.0226
    Tian CRISPRi      +0.0281
    Replogle RPE1     +0.0675   survives correction, positive

## L4 CLOSED (2026-08-14): difficulty vs partition noise vs seed noise, measured

16/16 lanes landed, 0 failures. Three roots complete at 5 expression_only + 5 typed_static each:
screening_c070 (0.70/0.05), screening_c075c15_r2 and _r3 (0.75/0.15, SPLIT_SEED 1 and 2). The paper
asked for exactly this design — "several realizations per level, so that between-level differences can
be read against the within-level spread" — and it is now run.

### h2a (typed_static − expression_only): 6 cells, 4 difficulty levels

    0.70/0.05  s0  -0.0073
    0.75/0.15  s0  -0.0012      three re-draws at ONE difficulty
    0.75/0.15  s1  -0.0107
    0.75/0.15  s2  -0.0051
    0.80/0.10  s0  -0.0122
    0.85/0.05  s0  -0.0120   (n=7)

    seed   (within re-draw)  sd = 0.01067   df 26
    level  (between levels)  sd = 0.00329   df  3
    redraw (within level)    sd = 0.00034   df  2

### h1 (condition_gated − expression_only): 5 cells, 3 levels (0.80/0.10 has the 3 re-draws)

    seed   sd = 0.00524   level sd = 0.00371   redraw sd = 0.00250

### What this establishes

1. **The training seed is the largest term for both contrasts**, and by a wide margin for h2a:
   0.01067 against a difficulty effect of 0.00329, a factor of 3.2. This is the robust, quotable
   finding — it has 26 df behind it, unlike the other two components.
2. **Partition re-draw noise is contrast-dependent, and the paper's existing claim is about h1.**
   For h1 it is substantial (sd 0.00250, consistent with the published +0.0082 / +0.0026 / +0.0021
   spread whose verdict flips). For h2a it is negligible (sd 0.00034). Re-drawing a split at fixed
   difficulty barely moves h2a and moves h1 a lot.
3. **A difficulty effect IS detectable above partition noise** — but only once you have several
   realisations per level to measure against, and only for h2a decisively (level/redraw about an order
   of magnitude; for h1 it is 1.5x, which at 2 df is not a claim).

### What must NOT be over-read

Both the level and re-draw components carry 2-3 df. The "9.7x" the script prints is not a stable
quantity and should be reported as "roughly an order of magnitude", which is why the module prints its
own CAUTION line. Only the seed component (26 df) is precisely estimated.

Reproduce: `PYTHONPATH=src .venv/bin/python -m tcell_pipeline.screening.variance_decomposition
--contrast h2a` (and `--contrast h1_vs_no_graph`). Artifacts in data/results/l4/.

### Correction on record

An interim run of this decomposition, taken while re-draw r3 had only n=1, reported re-draw noise
(sd 0.00588) EXCEEDING the difficulty effect and concluded the difficulty knob had no detectable
effect. That was an artifact of one cell mean resting on a single seed (-0.0189; at n=5 it is
-0.0051). The n=5 answer above supersedes it. Nothing in the pipeline changed — only the seeds behind
one cell.

---

<details>
<summary>Superseded 2026-08-10 banner (RPE1 only) — kept verbatim</summary>

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
| reference (CD4 T cell, frozen fold) | **7** | **+0.0043** | [+0.0017, +0.0069] | 7/7 seeds positive; Bonferroni 0.027, Holm 0.021 — **now SURVIVES both** (was n=5, +0.0045, Bonferroni 0.083, failed) |
| Replogle K562-essential | 4 | +0.0081 | [-0.0034, +0.0197] | ns |
| Frangieh melanoma | 4 | +0.0194 | [-0.1220, +0.1608] | ns |
| Tian iPSC neuron | 4 | +0.0281 | [-0.0929, +0.1491] | ns |
| Replogle RPE1 | 4 | **+0.0675** | [+0.0316, +0.1033] | survives BOTH corrections |

Pooled over the five: fixed-effect +0.0051 [+0.0031, +0.0071]; **random-effects +0.0208
[+0.0047, +0.0369], p=0.0112**; I2 = 87.7%, Q p < 0.001. Sign test 5/5, p = 0.0625.
Reproduce with: `PYTHONPATH=src .venv/bin/python -m tcell_pipeline.replication.pool --with-reference`.
(These supersede the +0.0209 [+0.0048,+0.0370] first reported on 2026-08-10, which used the reference
at n=5; the script uses the n=7 result. The conclusion is unchanged.) Fold re-draws (c075c15, c080c10) are EXCLUDED — they re-draw the same dataset and are not
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


</details>

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

Per-arm SYSTEMA on the harder split: untyped_gnn 0.0805 (n=2) ~ expression_only 0.0805 (n=4)
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
