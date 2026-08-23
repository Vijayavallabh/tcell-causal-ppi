# Session Handoff

## STOP — READ FIRST (2026-08-21): D-series and C1a CLOSED. Rail 4 fired. Only C1b's lanes remain.

**Nothing is running and no card is held by us.** Verified 2026-08-21 00:54, not assumed: no
`run_screening`, no runner, no finaliser, no monitor. Cards 0, 1 and 3 (CUDA indexing) were free at
78/78/75 GiB; card 2 was a co-tenant holding 63 GiB and card 4 is the T400. **Load average was 102**
on 64 cores from other users' CPU jobs. These lanes are CPU-bound on row-by-row subgraph sampling, so
that is what sets per-epoch time, not the free VRAM. Price any C1b estimate at the slow end of the
measured 38-90 min/epoch, and RE-CHECK the cards before launching: a free card does not stay free.

### *** RAIL 4 FIRED. Read the top block of `RESULTS_SUMMARY.md` before touching the paper. ***

D4's amendment audit found that Amendment 3.1 had pre-registered a pooled estimate over the K=128
subset alongside the all-datasets one, that it was **never computed**, and that the paper nonetheless
claimed it *was* reported. Running it is free — it re-pools landed numbers. The two disagree:

| pool | k | random effects | 95% CI | p |
|---|---|---|---|---|
| all datasets (what the paper reports) | 8 | +0.0091 | [-0.0024, +0.0207] | 0.121 |
| **K=128 subset (the pre-registered second estimate)** | **5** | **+0.0141** | **[+0.0041, +0.0241]** | **0.0056** |

All five K=128 datasets are positive, and at m=4 both Bonferroni and Holm give 0.0224. **The graph
helps in a pre-registered analysis nobody had run.** Snapshotted, flagged, null NOT rewritten around
it. It does **not** touch the headline null — h1 is unchanged at k=1, +0.0033, p=0.90. It **does**
undercut the body's "sign we cannot predict" sentence: the only negative anywhere is Norman, and
Norman ran at K=16, an eight-fold capacity reduction the paper never labelled. Whether the body
argument changes is a **human** call, not an agent's. Full reasoning: `docs/prereg-audit-2026-08-21.md`.

**The whole D-series is closed** (commits c4c8eb6, 23330d6, 55d3325, 51b4c69, 9a55284). The paper now
has seven gates instead of five: anonymity over the paper *and* all 245 tracked files (the venue
blinds linked material), and artifact agreement over all eleven tables instead of three. Two real
errors were found and fixed — a Bonferroni p that rounded the wrong way, and an appendix table still
reporting a superseded rule's counts — plus a harness bug in which `check_paper.sh` reported FAIL on
its most important gate and exited 0 anyway.

**C1a is closed: Amendment 9 has landed**, so rail 7 no longer blocks C1b. Writing it caught a setting
that would have voided the campaign: `run_a2_ladder.sh` uses the config-default `LAMBDA_GRAPH=0.01`,
which is harmless for Amendment 6's arms but suppresses the gate of `condition_gated` — the only arm
whose gate is live, and the arm the headline null is about. All 24 lanes would have measured a floor
for a different model and tripped Amendment 3.4's own kill criterion. **Two prerequisites before any
C1b lane:** `LAMBDA_GRAPH=0` must reach the lane, and `ladder_report.py` must take the arm and
reference root as parameters, with tests, before the lanes land (Amendment 9.11).

### C1b IS RUNNING (launched 2026-08-21 02:08) (first launch 01:53 was stopped at 02:07 and RESTARTED at 02:08 after editing the runner mid-flight: bash reads a script incrementally by byte offset, so shifting lines under a running shell can garble everything after the current statement. Only three cheap lanes were in flight. The restart also exposed the stale-claim trap now reaped automatically)

`run_c1_ladder.sh`, 48 lanes into a FRESH root `data/results/c1_ladder` — 24 `expression_only` and 24
`condition_gated` over the six existing rungs at seeds 0-3, on cards 0/1/3 (card 2 is a co-tenant
whose CUDA context could not even be created). Watch `data/logs/c1_ladder.nohup.log`.

**It is a separate script from `run_a2_ladder.sh` for one reason, and it is the reason Amendment 9.2
exists.** `config.LAMBDA_GRAPH` is a plain module constant, so `export LAMBDA_GRAPH=0` does
**nothing**; the override is the CLI flag `--lambda-graph 0`, and `run_a2_ladder.sh` does not pass it.
`run_screening --help` states the consequence itself: at the default `0.01` "the unnormalised per-edge
sum is ~103x the response term and annihilates the gates inside epoch 0". `condition_gated` is the
only arm here with a live gate, so reusing the A2 runner would have spent 180-380 GPU-hours measuring
a floor for a model the paper's null is not about, and tripped Amendment 3.4's own kill criterion.
Verified on the live command line after launch.

**Job order is seed-major, deliberately.** A stopped campaign then holds a *balanced* ladder at
whatever n landed. Rung-major would leave three rungs at n=4 and three at n=0, which the floor rule
cannot read at all — the floor is the smallest rung that clears AND is cleared by every larger one.
Amendment 9.9 anticipates being stopped short; this makes that outcome readable rather than wasted.

**FIRST MEASUREMENTS, 2026-08-21 02:34.** Two things are already verified rather than assumed.
`lambda_graph = 0.0` is recorded in the landed lane's own parquet, so the flag reached the artifact and
not merely the command line. And the same rung/seed lane exists in the A2 ladder at `lambda_graph=0.01`:
the two agree to `+0.000076` against a per-seed spread of `0.003`, i.e. forty times smaller than seed
noise. That confirms both that lambda genuinely does not touch a graph-free arm and that this campaign
reproduces the landed lanes correctly.

**MEASURED, 2026-08-21 11:31 — Amendment 9.9's requirement is now discharged.** The first
`condition_gated` lane, `d020_s0`, ran **03:00:14 to 11:31:44 = 8 h 31 m wall, 8.51 GPU-h**, completing
12 of 20 epochs on early stopping, `systema` 0.0871. `expression_only` lanes measure ~26 min (0.41
GPU-h) at load ~95.

**THE GATE STAYED ALIVE: min 0.7027, running 0.7027 -> 0.7442 over 12 epochs, against a dead threshold
of 1e-3.** That is the empirical validation of the whole Amendment 9.2 decision. At the config default
`lambda_graph=0.01` the same gate is annihilated inside epoch 0 and collapses toward 1e-7; had this
campaign reused `run_a2_ladder.sh` it would have burned days measuring a model the paper's null is not
about, and every lane would have been UNDECIDABLE under Amendment 3.4.

**Per-epoch, which is the comparable measure** (wall time alone confounds contention with how many
epochs early stopping allowed), across the first three gated lanes:

| lane | wall | epochs | min/epoch | GPU-h |
|---|---|---|---|---|
| d020_s0 | 8.53 h | 12/20 | 42.6 | 8.51 |
| d050_s0 | 10.60 h | 13/20 | 48.9 | 10.58 |
| d100_s0 | 14.69 h | 14/20 | 63.0 | 14.67 |

Both drivers are benign. Early stopping allowed more epochs each time (12, 13, 14 of 20), and per-epoch
time rose with contention — all inside the 38-90 min/epoch this box is documented to deliver.

**Schedule, central and pessimistic.** At the observed ~51 min/epoch and ~13 epochs, a gated lane is
~11 h, so 24 of them over 3 rolling cards plus ~4 h of cheap lanes is **~3.9 days from the 02:08 start,
finishing around 26 Aug**. The pessimistic bound matters and is written down rather than hoped away: if
later lanes run the full 20 epochs at the slowest observed 63 min/epoch, a lane is ~21 h and the
campaign needs **~7 days, i.e. 28 Aug** — inside the 29 Aug deadline but with no margin. So re-check
against the deadline rather than assuming, and invoke the stop rule (Amendment 9.9, rail 5) rather than
quietly extending. Load fell from ~95 to ~50 on 21 Aug afternoon, which should help.

Card 2 was re-checked at 18:05 on 21 Aug and is STILL held by its co-tenant, so no fourth worker is
available; cards 0/1/3 are ours. Workers take the next job as each card frees, so this is a rolling
pipeline, not lock-step waves.

**If card 2 frees up**, adding a fourth worker takes 8 waves to 6 and cuts roughly a quarter off the
wall clock. The runner is idempotent and reaps stale claims, so a second invocation with `CARDS=2` is
safe. Treat it as best-effort only: `run_l4_card2.sh` preflighted card 2 at 78 GiB, started a lane, and
lost it to a returning co-tenant 2h47m later.

### Recorded 2026-08-22 at n=1, BEFORE any contrast is computable — not a result

Two observations from seed 0's landed lanes. Both are written down now, while the primary needs n=4 and
has n=1, precisely so neither can look like a post-hoc excuse once the numbers exist. This is the same
discipline that made Amendment 6's post-hoc increment worth reading: it was committed while three rungs
were still unrun.

**1. The two ladders are not matched on training length, and the reason is early stopping.** Amendment
6's untyped ladder ran essentially every lane to the 20-epoch cap (19-20, one 16). C1's gated lanes
early-stop far sooner: 12, 13, 14, 14 epochs on d020-d200, and only d400 reached 20.

This is NOT a protocol difference. Both ladders run the identical pre-registered protocol — 20-epoch
cap, the same `EARLY_STOP_PATIENCE`, the same fold, the same batch size (Amendment 6.6, held fixed by
9.1). `condition_gated` simply stops improving on validation sooner than `untyped_gnn` does. That is a
property of the arm, not a handicap imposed on it.

It is recorded because someone will eventually ask whether a higher gated floor is an artifact of
shorter training. The answer is that both arms trained until their own early-stopping criterion under
one pre-registered protocol — defensible, but it should be stated rather than discovered. It is also a
concrete reason why Amendment 9.4 is right to call any comparison between the two floors DESCRIPTIVE.

**2. A gate story I told from one seed, and seed 1 refuted it. CORRECTED 2026-08-23.**

On seed 0 the permuted control's gate came in at **0.0240** against 0.57-0.70 on every real rung, and I
wrote that this was mechanistically what a working gate should do, since a scrambled neighbourhood
gives it nothing worth keeping open. It was labelled "at n=1 this is one seed; do not read a trend into
it", and that caution turned out to be the load-bearing part. **Seed 1's permuted control is 0.4654 —
mid-range among its own rungs, not an outlier at all.**

| rung | s0 | s1 |
|---|---|---|
| d020 | 0.7027 | 0.4138 |
| d050 | 0.6495 | 0.4600 |
| d100 | 0.6997 | 0.4804 |
| d200 | 0.6366 | 0.1573 |
| d400 | 0.5735 | 0.3400 |
| **permuted_d400** | **0.0240** | **0.4654** |

**RESOLVED 2026-08-23 15:18.** Seed 1's control completed at **0.465407**, unchanged from the partial
0.4654, so the correction below stands on completed data. Both controls are now final: seed 0 at
0.0240, seed 1 at 0.4654.

**That correction was originally made against a PARTIAL reading, and this is the caveat on it.** When I
compared them, `permuted_d400_s1` was still RUNNING: 19 epochs of history and no parquet. Its 0.4654
was a minimum-so-far that could still fall. The comparison put a completed seed-0 lane beside a
running seed-1 one — the same mistake in miniature as reading a result off an unfinished campaign.
`gate_health` now EXCLUDES lanes with a history but no parquet and names them under `in_flight_lanes`,
so the report can no longer mix the two, and re-check seed 1's control once it lands.

Subject to that, there is no "the gate closes on the control" effect. Seed 0's 0.0240 is a single low
lane, and seed 1's lowest is d200 at 0.1573 — a different rung entirely. What the two seeds DO show is that gate
minima vary widely lane to lane (0.02 to 0.70) and that seed 1 sits systematically below seed 0. Every
lane remains above the 1e-3 dead threshold, the closest by 24x, so Amendment 3.4 still does not bind.

**The guard that observation prompted is still right, and for a reason that does not depend on it.** A
control whose lanes are dropped leaves the rung present with n=0, and the veto then passes silently.
That is a defect regardless of which rung happens to have the lowest gate; the seed-0 reading only made
it salient. `ladder_report` refuses on an untestable control now, and the fix stands on its own.

**If you pick this up mid-flight:** re-measure per-epoch cost on the first `condition_gated` lane
rather than trusting 7.5-16 GPU-h, and if the measured rate puts it past 29 Aug, **stop and report at
whatever n landed, labelled preliminary with its n** (rail 5, Amendment 9.9). Do not quietly extend.
Every gated lane echoes `gate_mean_min`; below 1e-3 the lane is UNDECIDABLE under Amendment 3.4 and
must shrink n rather than count as evidence. Read the result with:

    PYTHONPATH=src .venv/bin/python -m tcell_pipeline.screening.ladder_report \
      --root data/results/c1_ladder --arm condition_gated \
      --reference-root data/results/screening_lambda0 \
      --out data/results/c1_ladder/floor_condition_gated.json
They came from auditing what the 2026-08-20 rewrite left ungated, and D1 is the one that matters most.
**Nothing in the harness checks anonymity** — `check_paper.sh` gates the build, the page budget and
abstract drift, and none of its checks looks for a name. The repo scrub is dated 2026-08-11; the whole
paper was rewritten on 2026-08-20 with no re-check after. Verified today and currently clean: `main.tex`,
the rendered PDF text, and an empty `/Author` and `/Title` in `pdfinfo`. So D1 is a gate to keep a clean
thing clean, and it is the only failure on the list that cannot be repaired after the deadline. D2 extends
artifact agreement from 3 of the paper's 11 tables to all of them, D3 re-verifies venue facts last checked
on 2026-08-11, D4 audits all eight prereg amendments the way the devil's advocate audited one. C1 also
split: **C1a is its pre-registration, free, and rail 7 needs it before any lane**; C1b is the 180-380 GPU-h.

**The paper is submission-ready and was heavily reworked on 2026-08-20.** Every section rewritten,
audited against the ICBINB content spec, and restructured to Problem / Proposed approach / Observed
outcome / Reason for failure. Body exactly 8pp, References open page 9, 24 pages, all six gates pass.

**Two new tools, use them.** `paper/icbinb/verify_numbers.py --check` diffs every numeric literal in
`main.tex` against a snapshot and re-derives the tables from artifacts; run it after ANY paper edit and
re-snapshot only once a change is justified. `./check_paper.sh` now has a sixth gate on
`abstract_plain.txt` drift, and its body-page gate counts body lines above the References heading
rather than asking whether the word appears on page 9, which is how a 0.9-page overrun hid for weeks.

**Corrections made on 2026-08-20, do not re-reverse.** I^2 for the eight-dataset pool is 87.5% (89% was
the seven-dataset value); condition_gated at n=7 is 0.0818 (0.0829 contradicted the paper's own
h1=-0.0043); the study spans eight datasets and the replication panel four cell types; h1 is testable
on exactly one replication dataset, so the seven-dataset null belongs to the STATIC arm.

---

## (superseded 2026-08-19 12:30): every autonomous item is closed; nothing is running

The GPUs are free. What is left in `NEXT_ACTIONS.txt` is the NOT AUTONOMOUS section: the OpenReview
submission and the sealed split, both needing a human.

**A2(a) landed and the negative control worked.** Measured floor <= 0.02 response SDs — every injected
rung recovered under both corrections — while the scrambled-neighbourhood control at the same magnitude
gives -0.0001, a CI of width 0.004 around zero. At delta=0.40 both arms gain 0.08-0.10 systema; under
the scrambled injection of identical magnitude both arms LOSE a little. So the ladder measures the graph
being read, not the injection being large.

Post-hoc, subtracting the +0.0048 already on this fold, the injection's own contribution is significant
only at delta=0.40. And the control's increment is significantly negative (-0.0050): a confidently wrong
prior destroys the real benefit rather than merely failing to help.

**A1 landed the day before.** Both diagnostic contrasts null: cutting the message parameters fourfold
does nothing (+0.0004), randomising every relation label does nothing that clears correction (+0.0065).
The damage is in what all three typed arms share, not in the typing.

**Where the paper stands.** 21 pages, body still exactly 8pp, all four gates passing via
`./check_paper.sh`. Four new appendices and three body clauses since 2026-08-16. Amendments 4, 4a, 4b, 5
and 6 all pre-date the runs they govern.

**Read in this order:** `RESULTS_SUMMARY.md` top to bottom — it is newest-first and the top four
sections are A2(a), A1, A5, A4, then the A3 contradiction stop.

---

## STOP — READ FIRST (2026-08-18 13:50): A1 closed; only the ladder is still running

**RUNNING.** `run_a2_ladder.sh` since 13:46 on all four A100s — 48 lanes, ~66 GPU-h, about 17 wall-hours.
`run_ladder_finalise.sh` is chained and writes `data/results/a2_ladder/floor.json` when they land.
That is the LAST autonomous item; everything else in `NEXT_ACTIONS.txt` is closed.

**A1's answer, and it is a clean negative.** Neither candidate route explains why edge typing costs
$-0.0120$:

    D1  typed_shared   - typed_static  = +0.0004  [-0.0048, +0.0057]  bonf 1.000   NULL
    D2  typed_permuted - typed_static  = +0.0065  [-0.0017, +0.0147]  bonf 0.182   NULL

Cutting the message parameters fourfold (34% of the whole model) does nothing, and randomising every
relation label does nothing that clears correction — its point estimate is positive, i.e. the curated
partition may be slightly worse than a random one, which is suggestive and NOT established. Both arms
stay far below untyped and below no-graph, so the damage is in what all three typed arms share: signed
messages, the per-edge feature term, complex nodes, the residual FFN, `add` instead of normalisation.

**Eleven and a half hours of idle GPU, and why.** The first ladder launch refused at 02:06 because ONE
card had 23 GiB free; the preflight exited rather than using the other three. Both runners now select
usable cards and refuse only if none is usable.

**A standing instruction is now wrong.** `NEXT_ACTIONS.txt` says not to re-propose per-relation
normalisation because it was refuted. A4 measured that the search which refuted it spans 0.0089 across
all 14 cells while the gap under study is 0.0176. Recorded, not acted on.

---

## STOP — READ FIRST (2026-08-16 20:15): campaigns ARE running; A2-A5 closed today

**RUNNING RIGHT NOW.**

    ./run_a1_mechanism.sh        typed_shared seeds 0-3 on all four A100s since 19:06, seed 4 queued.
                                 ~40 min/epoch, early stop expected around epoch 11-13, so first
                                 landings roughly 02:00-04:00. Log data/logs/a1_mechanism.nohup.log.
    inject_signal --ladder       CPU. Building the six A2(a) rung roots under data/intermediate/inject.

**READY TO LAUNCH, WAITING ON CARDS.**

    ARMS="typed_permuted" ./run_a1_mechanism.sh     A1's tie-breaker arm; implemented and tested.
    ./run_a2_ladder.sh                              the injection ladder, ~66 GPU-h.

Both preflight-refuse a card under 40 GiB free, so starting one early fails fast instead of fighting
A1 for memory. Start them when A1's lanes have landed.

## What closed today, and what it says

    A2(b)  detection floor, simulated over the MEASURED variance     data/results/a2_power/
    A3     the null under TxPert / GEARS / scPerturb endpoints       data/results/a3_external/
    A4     the architecture search was narrower than its seed noise  data/results/a2_power/
    A5     the rationale audit's ratios, given their sizes           data/results/a2_power/

**A3 fired the CONTRADICTION STOP and it is the biggest result of the day.** On identical predictions,
fold and seeds, `untyped_gnn - expression_only` is **+0.0089 under TxPert's Pearson-delta** and
**-0.0424 under GEARS' top-20 DE correlation**, both clearing Bonferroni AND Holm over all twenty
endpoint-by-contrast cells. The sign of "does the graph help" depends on which reported metric you
choose. The null was NOT rewritten around the positive; it is flagged at the top of RESULTS_SUMMARY.md.

**A2(b)'s quotable number.** Detecting the untyped arm's own +0.0043 across datasets needs about **250
datasets**, because the between-dataset spread is tau=0.0205 and the datasets disagree in sign.
scPerturb has 44. On the frozen fold the measured MDE at n=7 is 0.0075, or 0.0185 if the claim must
also survive re-drawing the fold.

**A4.** The 14-cell architecture search spans +0.0051 to -0.0037. Fourteen draws of seed noise alone
would span 0.0147. The whole search is inside what re-seeding ONE configuration produces.

**A5, and it corrects us.** Cause E's source ablation is largely an edge-count effect: STRING is 85.4%
of the graph, and per 1% of edges it leads HuRI by 1.7x, not the ~200x the raw deltas suggest. And the
audit's "92% of cases beat random on necessity" is a 49% improvement over a quantity of order 1e-7.

## Three corrections made today, all against us

1. `paper/icbinb/abstract_plain.txt` — the text SUBMISSION.md tells a human to paste into OpenReview —
   was a full campaign out of date and still claimed "positive on all five independent datasets", the
   exact claim Norman retracted. Regenerated from main.tex; SUBMISSION.md now says it is derived.
2. The paper said the between-re-draw variance component was at least as large as the between-seed
   one. Measured: 0.0003 against 0.0107. Fixed in `app:power`.
3. The injection module's scaling constant was computed over train AND validation rows, which is a
   leak. Its own leakage test caught it before any lane ran, and it was watched to fail against a
   deliberately leaky variant first. See Amendment 6.2.

## Paper state

Body still EXACTLY 8pp (References open page 9), 0 errors, 0 undefined, 0 overfull >2pt, 19 pages
total. Run `./check_paper.sh` — it checks all four gates in one command, from any directory.
Added: `app:metrics` (A3), rewritten `app:power` (A2b), an A4 paragraph in `app:archsearch`, an A5
paragraph in `app:causes`, and one sentence in Limitations/Scope. Pre-registration amendments 4, 4a, 5
and 6 all went in BEFORE the runs they govern.

---


## STOP — READ FIRST (2026-08-16): all campaigns are done; only the submission is left

**Nothing is running.** Eight replication datasets, the n=7 reference family and the L4 decomposition
are all complete. The paper is at 8pp with everything integrated, 0 errors / 0 overfull / 0 undefined.

**The three headline numbers.**

    h2a pooled, 7 replication datasets   +0.0018  RE 95% CI [-0.0028, +0.0065]  I2 = 39.2%, Q p = 0.13
    untyped arm, reference screen n=7    +0.0043  [+0.0017, +0.0069]  survives Bonferroni AND Holm
    edge typing, reference screen n=7    -0.0120  [-0.0160, -0.0080]  7/7 seeds, survives BOTH

The typed graph is a bounded null with NO significant heterogeneity across five cell types, both
perturbation directions and 100-to-9,730 targets. Edge typing is actively harmful. The untyped arm
clears correction on three datasets **in disagreeing directions** (+0.0043, +0.0675, -0.0790), so its
sign is not predictable — that is the paper's central claim, not "the graph helps".

**L4 is closed.** h2a variance components: seed 0.01067 (26 df) > level 0.00329 (3 df) > redraw
0.00034 (2 df). The training seed moves the contrast 3.2x more than the difficulty knob. Only the seed
component has enough df to quote precisely.

**Corrections that are on the record and must not be silently re-reversed:**
- Norman (-0.0790, survives both) overturned the "untyped positive on 5/5 datasets" pooled claim.
- h2b survived at n=6 (bonf 0.0491) and does NOT at n=7 (bonf 0.1308).
- An interim L4 run with a cell at n=1 gave the opposite verdict to the n=5 result.
- The 0.80/0.10 re-draws identify h1's within-level variance, NOT h2a's.

**The only thing left needs a human.** The ICBINB-BIO submission: create the anonymous.4open.science
mirror, paste the URL at paper/icbinb/main.tex line ~238, rebuild, confirm 8pp, then upload main.pdf on
the full-paper track. Deadline 29 Aug 2026 AoE. Everything else is in paper/icbinb/SUBMISSION.md,
including a probe table showing this machine has no OpenReview credentials, no client library and no
gh CLI.

**Rails held throughout.** Sealed split never opened, `evaluation/sealed_eval.py` never run, reference
roots verified byte-identical, every campaign in a fresh root, branch only.

**Where things live.** Campaign scripts `run_replication_*.sh`, `run_l4_*.sh`; analysis
`tcell_pipeline.replication.pool` and `tcell_pipeline.screening.variance_decomposition`; artifacts
`data/results/replication/`, `data/results/screening_untyped_n7/`, `data/results/l4/`. Read
NEXT_ACTIONS.txt for what is still open (OPEN A submission, OPEN B why typing hurts, OPEN C/L2 floor,
OPEN D/L3 metrics, OPEN F/L6, OPEN G sealed split stays shut).

---

## SUPERSEDED 2026-08-11 handoff — kept for the record

## STOP — READ FIRST (2026-08-11): the replication is done, and the paper's conclusion changed

**The multi-dataset null exists.** Four datasets, 52 lanes: pooled h2a fixed-effect +0.0031, 95% CI
[-0.0004, +0.0065], I2 = 25.6%, Q p = 0.26. A bounded null across four cell types. h1 is Frangieh-only
(the sole multi-condition dataset) at +0.0033 [-0.0829, +0.0894] and is underpowered, not null.

**The paper no longer argues that the protein-interaction prior fails.** At n=7 on the frozen fold,
every arm balanced, nothing dropped:

    untyped_gnn      +0.0904   plain topology, no typing, no gating
    expression_only  +0.0861   no graph
    condition_gated  +0.0818   typed graph + condition gate
    typed_static     +0.0740   typed graph, gate pinned to 1

  promotion_margin  untyped - expression_only    +0.0043 [+0.0017,+0.0069]  bonf 0.027  SURVIVES BOTH
  h2a               typed_static - expression_only -0.0120 [-0.0160,-0.0080] bonf 0.001  SURVIVES BOTH
  h2b               condition_gated - typed_static +0.0077 [+0.0009,+0.0146] bonf 0.131  fails
  h1                condition_gated - expression_only -0.0043 [-0.0084,-0.0002] bonf 0.170 fails

Plain topology helps. Per-relation edge typing is ACTIVELY HARMFUL and is the largest effect in the
family. The gate repairs only part of that damage and not significantly. So the prior carries signal
and our encoder discards it. Cause C flipped from Refuted to Survives: the 14-cell architecture search
varied HOW the typed encoder passes messages and never asked whether the typing belonged, so it
excluded its own premise.

**Retraction on record.** h2b SURVIVED at n=6 (bonf 0.0491) and does NOT at n=7 (bonf 0.1308). Seed 6
came in at -0.0033 with a healthy gate. Anything quoting h2b as surviving is from the interim.

**The one thing that needs a human.** The ICBINB-BIO submission. The paper is finished — 8pp, clean
build, style file byte-identical to the workshop's, repo scrubbed of six deanonymising strings,
deadline verified 29 Aug 2026 AoE. Filing needs an OpenReview account, which this machine does not
have (probed directly: no env vars, no ~/.openreview, no openreview-py, no gh CLI). Steps, in order,
are in paper/icbinb/SUBMISSION.md: create the anonymous.4open.science mirror, paste the URL at
main.tex:238, rebuild, then upload on the full-paper track.

**In flight as of this handoff (2026-08-11 15:45).** A repo audit found three QC-passing DE matrices
built and never trained — genome-wide Replogle K562 gwps (9,730 targets), Norman and Tian CRISPRa
(both ACTIVATION, where every trained dataset is knockdown) — plus a fourth difficulty split (c070)
generated and untrained. All are queued; see NEXT_ACTIONS.txt OPEN B0/B1 and data/logs/repl/campaign2.log.

**Rails held.** Sealed split never opened, sealed_eval.py never run, reference roots verified
byte-identical, every run in a fresh root, branch only.

---

## SUPERSEDED 2026-08-06 handoff — kept for the record

## STOP — READ FIRST (2026-08-06): venue changed, three folds are done, and one published claim was retracted

Everything below dated 2026-07-23 or earlier is SUPERSEDED. It is kept as the record, not as routing.

**Where the work is.** Branch `icbinb-multidataset-2026-08-03`, all uncommitted (rail 6). Live plan:
`NEXT_ACTIONS.txt`. Full numbers: `RESULTS_SUMMARY.md`. Read those two before anything else.

**Venue: ICBINB-BIO (NeurIPS 2026 workshop), not AAAI.** Deadline 29 Aug 2026 AoE (site marks it
tentative — RE-CHECK). 8pp main text, refs and appendices free, double-blind, non-archival, OpenReview
`NeurIPS.cc/2026/Workshop/ICBINB-BIO`. New paper at `paper/icbinb/main.tex`; `paper/main.tex` (AAAI) is
untouched and still valid for dual submission, which the user has cleared.

**BEFORE ANY GPU WORK ON THIS BOX:**

    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02

Without it every `--device cuda` lane dies in <60s on an NVML assert naming PyTorch. Cause is a stray
535.309.01 driver tree on the library path against a 580.173.02 kernel module. See AGENTS.md.

### The science moved: three folds, and NOTHING survives correction on any of them

| Fold (thr/cap) | h1 (cg − eo) | h2a (ts − eo) | survives FWER |
|---|---|---|---|
| frozen 0.85/0.05 | −0.0009 (p=0.71) | **−0.0131 (p=0.0036)** | h2a only |
| intermediate 0.80/0.10 | +0.0082 (bonf 0.101) | −0.0122 (bonf 0.920) | none |
| harder 0.75/0.15 | +0.0005 (p=0.904) | −0.0012 (p=0.614) | none |

All four arms, n=5, gates alive, 0 failures. **The one contrast that ever cleared Bonferroni+Holm —
"the typed static graph is reliably worse" — does NOT replicate.** That is now stated in the paper
rather than buried. Note the two failures differ: on the intermediate fold the effect size replicates
(−0.0122 vs −0.0131) and dies from 3.5× variance; on the hardest fold the effect itself collapses.

### The finding that reframes the split sweep (COMPLETE 2026-08-10 — do not re-run)

Three re-draws of the SAME split spec (0.80/0.10, only `SPLIT_SEED` varied), full family, n=5 each,
0 failures, all gates alive. Roots: `screening_c080c10_h1` / `_r2` / `_r3`.

| re-draw | h1 | uncorrected 95% CI | raw p | Bonf x4 | baseline | n_val | cosine |
|---|---|---|---|---|---|---|---|
| seed 0 | +0.00824 | [+0.00168,+0.01480] | 0.025 | 0.101 | 0.0991 | 3,632 | 0.7591 |
| seed 1 | +0.00260 | [+0.00048,+0.00472] | 0.027 | 0.109 | 0.0845 | 7,216 | 0.7928 |
| seed 2 | +0.00206 | [-0.00237,+0.00650] | 0.266 | 1.000 | 0.0942 | 5,096 | 0.8624 |

**The qualitative verdict FLIPS between re-draws of an identical specification.** Two give an
interval excluding zero at p~0.03; the third does not, at p=0.27. The estimate spans **4x**. For
scale, h1 across the three genuinely DIFFERENT folds was −0.0009 / +0.0082 / +0.0005 — so **re-draw
noise is the same size as the between-fold differences we were tempted to read as difficulty.**
The difficulty statistic itself spans 0.759–0.862 across re-draws against 0.056 for the whole
designed threshold range, and does not even order the baseline (the nominally hardest re-draw scores
highest; the baseline tracks `n_val`, which the re-draws moved 3,632 → 7,216 unbidden).

Stable across all three: the sign (all positive) and the corrected verdict (none survives). The
paper leans on the corrected verdict and presents the uncorrected intervals as a cautionary exhibit.

### DO NOT repeat these four mistakes

1. **Do not quote interim n.** Three interim values were retracted this session — h2a `−0.0214`
   (n=2, true −0.0122), h2b `fwer=True` (n=2, does not survive at n=5), and h1 `survives=True`
   (family_size had collapsed to 1). **All three were in the favourable direction.** Small-n reads on
   high-variance graph arms drift toward looking decisive. Wait for the rail-5 bar.
2. **Check `family_size` before believing any FWER verdict.** With missing arms the aggregator
   corrects over a family of 1 and reports `survives_family_wise: True` on a raw p. It also prints
   `INCOMPLETE COVERAGE` / `UNBALANCED` on the same run — read those lines.
3. **The aggregate launcher log is not the record.** A script launched with `>` holds its own offset
   and overwrites lines other jobs append with `>>`. Two `[c080]` lines vanished and the log implied
   an idle card while a 7.6h lane ran. Enumerate `/proc/<pid>/environ` + `cmdline`, or read the
   per-lane logs under `data/logs/n5/`.
4. **Do not trim prose to fix a page overflow before checking layout.** Six rounds of cuts moved the
   boundary barely at all; the cause was an `\fbox{\begin{minipage}}` that could not break across
   pages and stranded 26 lines. `pdftotext -f N -l N main.pdf - | grep -vc '^\s*$'` per page first.

### State of the deliverables

- `paper/icbinb/main.tex` — compiles clean (0 errors, 0 undefined, 0 overfull >2pt, 0 dashes), main
  text exactly 8pp. Required LLM-usage disclosure written and honest about agent-authored code.
- `docs/replication-prereg.md` — FROZEN. Amend only by dated append; never edit above the amendments.
- `docs/replication-dataset-survey.md` — six scPerturb datasets measured from the files themselves.
  Only **two** survive the pre-registered rules (Frangieh, Norman); Replogle RPE1 was **dropped**
  (12 of 2,393 targets clear the 25-cell floor). The goal asked for ≥4; the data supports 2, and
  manufacturing 4 would require weakening the discipline the paper is about.
- `src/tcell_pipeline/replication/` — h5ad → DE-stats adapter, mutation-tested self-check, PINNACLE
  context extractor. NEVER use `embeddings_pinnacle.run("melanocyte")` for a context swap: it takes a
  context argument but always writes to `config.PINNACLE_EMBEDDINGS_PATH`, clobbering the CD4 store
  every reference lane reads.
- `config.py` — 8 constants now env-scoped, defaults verified byte-identical, 549 tests pass.
  **`CONDITIONS` is read at IMPORT time** into `_COND_INDEX` in two encoders, so it must be set in the
  environment before first import; patching config afterwards silently does nothing.

**Next:** E1 in `NEXT_ACTIONS.txt` — train the replication arms. The config blocker is cleared; the
remaining work is CPU prep (ID-map Frangieh's 43 unmapped targets, per-dataset splits and program
basis) before any GPU. **Nothing is in flight as of 2026-08-10 15:30; all four cards are idle.**

### Fifth mistake to avoid: this is a SHARED box, and memory is the constraint

On 2026-08-08 a co-tenant (another user's text-to-3D job) held 41.5 GB on one A100 and grew to
48.6 GB, while `condition_gated` needs **47–51 GB** (measured, not estimated). A lane launched there
would have died, possibly hours in. `run_realization.sh` runs both arms on one card, so the fix was
to let the cheap arm finish, then stop the chain by process group before the expensive arm allocated,
and re-queue that seed when the card actually freed. Do NOT kill a co-tenant; it was there first, and
starving another user is the more expensive half of the mistake. Do check free memory at launch AND
match the ARM to what is available — `expression_only` needs ~2.5 GB and ran happily on the
constrained card while `condition_gated` could not.

---

## SUPERSEDED 2026-07-23 handoff — kept for the record

## (2026-07-23): the re-screen RAN; the negative is now VALID, and feat-011/012 are DONE

The confound below (2026-07-21) was real, and it has now been REPAIRED and RE-MEASURED. `condition_gated`
was re-run at `lambda_graph=0` (live gates, mean 0.57–0.77) across 5 paired seeds on the frozen fold.

**Official n=5 result** (`data/results/screening_lambda0/robustness_5seed.{json,md}`, exit 0,
single_frozen_fold=True): `h1_vs_no_graph` = **−0.0009, CI [−0.0072,+0.0054], p=0.71 → PARITY** (does not
survive FWER). Per-config: untyped_gnn 0.0902 > no-graph 0.0857 > condition_gated 0.0848 > typed_static
0.0726. `h2a` (typed_static−no-graph) −0.0131 SURVIVES FWER (reliably worse). **The corrected numbers
nearly reproduce the confounded campaign (h1 was −0.0019), so a 4.5e6× gate swing changed no conclusion —
the graph negative is ROBUST, not an artifact.** The inner-holdout selection lead (+0.0051, n=1) did NOT
replicate.

**Feature state:**
- **feat-011 DONE** — valid multiplicity-controlled graph-vs-no-graph comparison exists (parity). Plus the
  14-cell architecture ablation (`docs/feat011-arch-search-notes.md`): no encoder lever beats the current
  design. `promoted.json` stays frozen; the deliverable is the separate `robustness_5seed`.
- **feat-012 DONE** — 50-case rationale audit on the seed-3 live-gate checkpoint
  (`data/results/rationale_audit_lambda0/audit_report.json`): frac_sufficiency_below_random 0.04,
  frac_necessity_above_random 0.92, minimality 0.888, stability 1.0, STRING dominates source ablation
  (0.365). First valid audit on a functioning graph; correctly refuses the dead-gate frozen H1.
- **feat-013 in-progress at its CORRECT terminal** — run_repro_real gives CANNOT_VERIFY (confirmatory
  decision is on the sequestered split, steward-only). Cannot reach `done` from an agent session by
  governance; that is the right verdict, not a shortfall.

**AAAI paper:** null/parity is the headline; abstract `[RESULT]` is filled (`docs/aaai-title-abstract.md`,
recommended **title A** — the balanced confound-diagnosis framing that survives either outcome; that doc
marks A as recommended and `NEXT_ACTIONS.txt` agrees). The paper is the confound diagnosis + the robust
valid null + the 14-cell ablation + the faithfulness-audited rationale (feat-012).

**Next actions = experiments, in `NEXT_ACTIONS.txt` (E1–E6, prioritised for the AAAI main-conf submission):**
E1 MDE/power analysis of the null (cheap, no GPU — do first) · E2 λ_graph sweep → confound-collapse figure ·
E3 edge-source decomposition · E4 second Perturb-seq dataset (external validity; needs new data ingestion) ·
E5 live-gate Stage-B rationale head + re-audit · E6 sealed-split confirmation (steward-only). Triage: E1
alone if time for one; E1+E2+E5 if three (all runnable on current artifacts).

**Remaining (docs sweep, unchanged from before):** README / the report / the walkthrough still frame the
graph negative as a *finding*; that framing is now further out of date — they should cite the valid n=5
null. `./init.sh` green at 549. Commits this session: 3b2c12e, 7b8179e (+ the triad + doc updates here).

---

## SUPERSEDED 2026-07-21 handoff (the confound, before the repair above) — kept for the record

## STOP — READ FIRST (2026-07-21): the graph negative is CONFOUNDED, not a result

Five concurrent sessions ran 2026-07-20/21. The decisive finding supersedes the framing in every
correction below, and in README / the report / the walkthrough, none of which have been updated yet.

**The EG-IPG graph arms were trained with their message passing switched off, by their own regulariser.**

`StageALoss._graph` is an unnormalised `sum` over edges divided only by BATCH SIZE, while every other loss
term is mean-reduced. At ~40k edges/sample the penalty is ~103x the response term, and its gradient on the
edge gates is ~3.1e+06x the task's — so the gates' gradient *direction* is ~100% the penalty's
(`g_total/g_penalty` = 0.999994–1.000315). AdamW then marches them at ~`lr` per step, the same way, for all
2,127 steps of an epoch, and the gate dies inside epoch 0. The frozen H1's gate mean is ~1.3e-07 against
~0.61 at init (NOT bit-zero: `max` 3.54e-07, 0 of 117,174 gates exactly zero).

> **Corrected 2026-07-21 (session E, correcting its own earlier claim).** This paragraph previously read
> "`GRAD_CLIP=1.0` then rescales the whole update by ~1/695 — so ~99.98% of every step drives gates to
> zero." That mechanism is **wrong** and contradicted the direction-dominance paragraph further down.
> **AdamW is scale-invariant per parameter**: scaling every gradient by a constant `c` scales the first
> moment by `c` and the second by `c²`, so `m̂/√v̂` is unchanged — and a uniform clip factor is exactly
> such a constant. Verified on one parameter under the real settings: gradient 1e-4 → θ=-0.299969;
> gradient 1e-1 → -0.300000; ‖g‖=695 clipped to 1.0 → -0.300000; ‖g‖=0.17 unclipped → -0.300000. The clip
> changes nothing. Magnitude sets the *rate* of collapse, direction sets *whether* it happens. The
> measurements (103x, 3.1e+06x, 695.53) were always right; only the causal story was wrong.

Established by a controlled three-arm pilot (`pilot_lambda_graph`, seed 0, 2 epochs), read from artifacts:

All arms share one init (0.678556, seed 0). Gate mean is over all 289,974 edges; neighbourhood sensitivity
(`sensitivity.rel_delta`) is measured on a 33,754-edge subsample:

| arm | lambda | loss | gate mean: init -> epoch 1 | dead % | nbhd sens | gates survived |
|---|---|---|---|---|---|---|
| baseline | 0.01 | `StageALoss` | 0.678556 -> **1.9882e-07** | 100.0% | 9.41e-04 | NO |
| zero | 0.0 | `StageALoss` | 0.678556 -> **0.897950** | 0.7% | **9.03e-01** | **YES** |
| normalised | 0.01 | `EdgeNormalisedStageALoss` | 0.678556 -> **3.2190e-04** | 98.2% | 4.06e-03 | NO |

> **Corrected 2026-07-21.** This table previously read 0.6138 / 2.74e-07 / 0.845 / 2.42e-04. Every one of
> those was `sensitivity.gate_mean` — the 33,754-edge probe subsample — not `gates.mean` over all 289,974
> edges. One misread field, four wrong numbers. Session E caught it. Use `gates.mean`. Response figures were
> never affected (different code path). Verified against `pilot_{baseline,zero,normalised}.json`.

With the penalty OFF the gates do not merely survive, they RISE — the prediction task wants the edges open.
**And the cheap repair FAILS**: per-edge normalisation lowers the penalty/response ratio to 0.008 (a
penalty 400x SMALLER than the task) and the gates still collapse 2,108x, because AdamW is scale-invariant
per parameter so only the penalty's direction-dominance matters, not its size. This was PREDICTED on the
record before the run — with one part wrong, which E named: `normalised` collapses about one epoch *slower*,
because the real penalty shrinks as the gates do. **Magnitude sets the rate; direction sets the outcome.**

**LATER THE SAME DAY — the negative SURVIVES the repair.** The pilot's third arm gives held-out response
3.354460 (baseline, gates dead) / **3.356810 (zero, gates ALIVE — the worst)** / 3.355367 (normalised). A
4.52e+06x swing in gate magnitude — equivalently **960x** in how neighbourhood-dependent `h_graph` actually
is — moves the response 0.07%, and the working-graph arm is marginally last. Quote the 960x for scientific
claims: gate magnitude is the knob, neighbourhood sensitivity is the effect.
So: the campaign's comparison was invalid AND fixing it does not change the answer. Both findings stand.
Scope: pilot only — seed 0, 2 epochs, inner holdout, response loss; NOT the frozen fold, NOT 5 seeds, NOT
systema. It cannot close feat-011; it is a strong prior on what a re-screen would find.

**Do not call the pilot "the strongest evidence the negative is real"** — an earlier revision here did, and
E's objection is accepted: n=1 seed cannot outrank a 5-seed multiplicity-controlled result. But the 5-seed
screening is not confirmatory on the graph question either, since its graph arms had dead gates. No valid
powered test of the graph exists yet; the pilot supplies **admissibility, not strength**. Exact wording is
UNRESOLVED between sessions A and E — see `next_goal_after_gate_collapse.txt`.

**What this does and does not license.** It is NOT evidence the graph helps. It means the experiment did
not test the hypothesis. A redesigned regulariser could re-run and still produce a negative — that would
then be a real one. Every graph claim (frozen H1, 5-seed campaign, all screening, feat-011, feat-012) rests
on runs where the graph was switched off.

**Consequence: fixing this is a REDESIGN, not a one-line change** — a new objective plus a full re-screen,
which invalidates the frozen H1, the 5-seed campaign, every screening result, and moves the config hash
feat-013's committed manifest pins. That decision is the user's and is OPEN.

## feat-006 (2026-07-21): H1 no longer beats the strongest tabular comparator

Margin over the strongest bar: **+0.0492 -> +0.0140 -> +0.0051 -> +0.0000147** (H1 0.08340653 vs
`tabicl_qpre` 0.08339185) — 1/680th of the 0.01 noise band, and flagged an UPPER BOUND. Every step came
from fitting the BAR more honestly (convergence -0.0001; giving it the q_pre covariates H1 already gets
-0.0351; a val-blind CatBoost depth -0.0089), none from a better model class. The graph premise is
untouched: no-graph `expression_only` (0.0861) clears that bar by +0.0027, 180x more than H1 does.
Also fixed: `systema` scored collapsed-to-the-mean predictors on floating-point dust (+0.0129 for
`perturbed_mean`); blast radius measured at exactly one baseline, frozen results reproduce to 3.5e-09.

## State at handoff

- COMMITTED: feat-013 `bdc1f56` (D), feat-008 `ac1cbcd` (B). `./init.sh` green at 514.
- feat-006 is DONE (status flipped; description fully satisfied, all DoD criteria met). UNCOMMITTED: its
  code (12 files) + `feature_list.json` carrying THREE merged evidence blocks
  (feat-006 / feat-008 / feat-013), append-only, diff 3 lines, prefixes verified.
- feat-005 (session C) COMPUTE COMPLETE 2026-07-21 19:31 — the `sparse_pca K=512` stability backfill
  finished in 16,283 s (4h31m) under its 6 h cap. **All 17 cells now carry all three axes, no gaps**;
  both CSVs regenerated 19:32; `./init.sh` green at 514. K=512 stability landed at **0.2250**, against
  C's pre-registered ~0.2, completing a monotone collapse 0.841 -> 0.612 -> 0.350 -> 0.225. Headline
  unchanged and now fully supported: at K=128 held-out explained runs fastica 15.59% / svd 15.55% /
  **frozen sparse_pca 15.41%** / vae 14.11% / nmf 13.11% — the frozen basis is 0.18 pp off the best and
  is the only method combining that accuracy with sparsity — 22.7% exact zeros against 0.0% for
  svd/fastica/vae; **NMF yields more zeros still (55.7%)** but pays 2.3 pp of held-out (13.11%) and is
  decidably unsuited to a signed target. **Nothing justifies changing it.**
  (This line previously read "the only method delivering any sparsity (22.7% vs 0%)" — false, and the
  third instance of the same claim. C caught two in its files; this one was mine.)
  Frozen basis verified byte-identical end-to-end; `data/intermediate` never written.
  **feat-005 is DONE** (2026-07-21 20:2x). C delivered its evidence block in
  `docs/feat005-handoff-to-session-A.md` §2 and committed its six paths as `7c3e0bf` (no triad touched).
  Block merged append-only into `feature_list.json` (+4,067 chars, strict prefix verified), status
  flipped `in-progress` -> `done`. `./init.sh` green at **514**, exit 0.
  Merge review caught a false claim in the block — "sparse_pca is the ONLY method producing any
  sparsity", when NMF K=128 has **55.66%** exact zeros against sparse_pca's 22.69%, in a cell the block
  itself calls decidable. C fixed it in both its files with a dated `CORRECTION:`. The same claim had
  propagated to **five** places in total; the two in C's files, one here, one in README, and one in
  `docs/specs/2026-07-15-module3-program-decoder.md` — all corrected. The instance in
  `docs/history/progress-archive-2026-07.md` is left as-is deliberately: archives record what was said.
- BLOCKED, not startable: feat-011 and feat-012 — both need a comparison that actually tested the graph.
- feat-013 CANNOT reach done from an agent session: the sealed confirmatory step is test-steward-only and
  `CANNOT_VERIFY` is the correct verdict.

## Blockers / Risks

> **RESOLVED 2026-07-23 (see top section):** the regulariser decision was taken (option C, `lambda_graph=0`),
> the re-screen ran on the frozen fold, and **feat-011 + feat-012 are DONE**. The next two bullets are kept
> for the record but are no longer live blockers. Still live: feat-013's governance terminal and the
> sealed-split / frozen-basis invariant.

- **The regulariser decision (options A-D) is the user's and gates everything.** feat-008's last evidence
  line, feat-011 and feat-012 all wait on it. Do not pick one on the project's behalf.
- **feat-011 and feat-012 are BLOCKED, not merely unfinished.** Both need a graph comparison that does not
  exist. Do NOT mark them done to satisfy a completion hook — every arm was screened with dead gates.
- **feat-013 cannot reach done from an agent session.** The sealed confirmatory step is test-steward-only;
  `CANNOT_VERIFY` is the correct verdict, not a defect.
- **Unresolved between sessions A and E:** what kind of evidence the n=1 pilot is. Current wording — no
  valid powered test of the graph exists yet; the pilot supplies *admissibility*, not strength. Needs the
  user's sign-off before the docs sweep, because it determines what the corrected text says.
- **The sealed challenge split (5,608 rows) stays SEQUESTERED.** Never run `evaluation/sealed_eval.py` or
  `run_module0.py`. Never regenerate the frozen program basis.

## Files

- **The DoD triad — `feature_list.json`, `progress.md`, `session-handoff.md`** — is merged by ONE
  integrating session. If you are not it, do not touch these; hand your evidence block over as text.
- **Evidence blocks are append-only.** Verify the existing string stays a strict prefix before writing,
  and dump with `json.dumps(d, indent=2, ensure_ascii=True)` — non-ASCII is escaped inline, so an
  em-dash is not a reason to block a merge.
- **Forward plan / experiment backlog:** `NEXT_ACTIONS.txt` (E1–E6 for the AAAI submission). The old goal
  spec `next_goal_after_gate_collapse.txt` was spent and deleted in `692dd7f`; recoverable from history.
- **Harness:** `AGENTS.md` (routing + invariants) → `docs/agent-lessons.md` (the long form).
- **Per-feature notes:** `docs/feat00{5,6,8,13}-*.md`, `docs/h1-optimization-notes.md`.

## Next Session Startup

1. Read this file's top section, then `NEXT_ACTIONS.txt` (the E1–E6 experiment backlog).
2. `./init.sh` — expect the last green baseline (**549 passed** at `692dd7f`); this session changed only
   docs, so the count is unchanged. If it differs, a parallel session moved it — reconcile before building.
3. Pick the next experiment from `NEXT_ACTIONS.txt`: **E1** (MDE/power analysis, cheap, no GPU) is the
   highest-leverage first move; **E2** (λ_graph sweep) and **E5** (live-gate Stage-B rationale head) are the
   other two runnable on current artifacts. **E4** (second dataset) needs new data ingestion — start early.
   feat-011/012 are DONE; do NOT re-open the regulariser question (resolved: option C, `lambda_graph=0`).
4. Docs sweep still pending: README, the report and the walkthrough still frame the graph negative as a
   *finding* rather than the valid n=5 null — update them to cite h1_vs_no_graph = −0.0009, p=0.71 (parity).
5. **feat-013 and the sealed split are steward-only** — never run `evaluation/sealed_eval.py`,
   `run_module0.py`, or regenerate the frozen program basis.
6. Before reporting any process state — runs, monitors, whether something finished — RUN THE COMMAND.
   Answering from memory was the single most repeated error of 2026-07-20/21.

---

## Retracted phrasings — do not resurrect these

Two claims were published and withdrawn after an xhigh code review of `f1a00dd`. The paired-t math was
verified CORRECT and no number changed; two *conclusions* did not follow from the numbers.

- **"the frozen H1 sits BELOW no-graph"** — never tested. That pair was read off two marginal per-config
  means and was not in `CONTRASTS`. Run properly: `condition_gated − expression_only` = **−0.0019,
  CI [−0.0042, +0.0004], p=0.0847** — the CI crosses zero. H1 is at **statistical parity** with no-graph:
  it does not beat it, and it cannot be called below it either.
- **"the only graph variant that reliably beats no-graph is the untyped GNN (+0.0045)"** — RETRACTED.
  Nominally positive but not robust to multiplicity: Bonferroni 0.0832 fails while Holm 0.0416 passes, and
  `survives_family_wise` requires BOTH so the correction method cannot be shopped after the fact.

Corrected bottom line: after multiplicity control **no graph variant reliably beats no-graph**. And per
the section at the top of this file, that whole comparison was measured on models whose graph could not
contribute — so it is a confounded measurement, not a finding.

---

Earlier handoffs are archived: [`docs/history/session-handoff-archive-2026-07-21.md`](docs/history/session-handoff-archive-2026-07-21.md)
and [`docs/history/session-handoff-archive-2026-07.md`](docs/history/session-handoff-archive-2026-07.md).
