# Pre-registration audit — every amendment against what was actually done

**Date: 2026-08-21.** Audits `docs/replication-prereg.md`, which is FROZEN (rail 7). Nothing here
edits it. Where a promise was not kept, the fix is a PAPER fix, and this file says which.

**Why this exists.** The devil's-advocate pass of 2026-08-20 found one unkept promise — Amendment 6.5
said the detection floor would be labelled as the pipeline's rather than the typed encoder's, and until
that day it was not — by reading one amendment. Nobody had read the rest. This reads all of them.

**THE KEPT ONES ARE RECORDED TOO.** This project's credibility rests on four self-reported
retractions. An audit that finds nothing is evidence only if it could have found something, so the
verdicts below are stated with the artifact or line that settles each one, kept and unkept alike.

**A count correction first.** `NEXT_ACTIONS.txt` says "all eight amendments" and "Amendments 1 to 8".
There is no Amendment 1, and there are **ten** amendment sections, because 4a, 4b and an unnumbered
2026-08-03 one are separate registrations with their own promises. All ten are audited here.

## Verdicts at a glance

| # | Amendment | Verdict |
|---|---|---|
| 1 | **2** (2026-08-10) replicate-pseudobulk rule relaxed | **KEPT in rule, BROKEN in reporting** — the paper's survey table still shows the superseded counts |
| 2 | **(2026-08-03)** replicate unit pinned per dataset | KEPT |
| 3 | **3.1** K is not portable | **BROKEN, twice** — a promised analysis was never run, and a deviation is unlabelled |
| 4 | **3.2** primary contrast fixed by condition count | KEPT |
| 5 | **3.3** PINNACLE context per dataset | KEPT |
| 6 | **3.4** gate-collapse kill criterion | KEPT |
| 7 | **4 / 4a / 4b** the A1 diagnostic arms | KEPT |
| 8 | **5** the A3 re-scoring | KEPT |
| 9 | **6** the injected-signal ladder | KEPT (6.5 was broken until 2026-08-20; now kept) |
| 10 | **7** `typed_gcnnorm`; **8** rank bins | KEPT (7.4's "descriptive only" label is implicit, not written) |

Two findings are material. Both are in **Amendment 3.1**, and one of them fired the contradiction
stop. They are at the bottom, in full.

---

## 1. Amendment 2 (2026-08-10) — the replicate-pseudobulk rule was relaxed

| promised | verdict |
|---|---|
| DE unit becomes all cells per (target, condition) against pooled same-condition controls | **KEPT** — `src/tcell_pipeline/replication/de_amendment2.py`, and the `*.DE_stats_v2.*` artifacts are the rebuilt matrices |
| 25-cell floor retained, applied per (target, condition) | **KEPT** — `de_amendment2.py:31` `MIN_CELLS = 25`, applied at line 197 |
| replicate axis recorded in the provenance, and used as a covariate *if the DE method supports one* | **KEPT, conditionally** — it is recorded in every provenance file. It is NOT used as a covariate, and the conditional is why: the cell-level Welch fit has no replicate level. The module's own docstring concedes this is anti-conservative rather than burying it |
| the 50-group target-axis floor still bars Shifrut, Datlinger and Papalexi from any headline or pooled estimate | **KEPT** — `pooled_with_reference.json` contains exactly eight members and none of those three |
| single-condition datasets still take h2a as primary | **KEPT** — see Amendment 3.2 below |

### BROKEN IN REPORTING: `tab:repl` is the pre-amendment table

Amendment 2 states its own practical consequence: under the corrected rule the targets clearing the
floor are "Replogle RPE1 **2,122** (was 12), Frangieh **246** (was 216), Norman **105**, Papalexi 25,
Datlinger 31, Shifrut 21", and "Replogle RPE1 moves from *dropped* to the best-powered candidate
available".

The paper's `tab:repl` reports the **was** column, not the corrected one. Its numerators are, digit for
digit, the pre-amendment provenance files:

| dataset | `tab:repl` | pre-amendment provenance | post-amendment `DE_stats_v2` |
|---|---|---|---|
| FrangiehIzar2021 | 216 | 216 | **238** |
| NormanWeissman2019 | 102 | 102 | **105** |
| PapalexiSatija2021 | 24 | 24 | **25** |
| ShifrutMarson2018 | 20 | 20 | 20 |
| ReplogleWeissman2022 | 12, **"dropped"** | 12 | **2,122** (rpe1) |
| DatlingerBock2017 | 6 | 6 | **30** |

The table's caption reads "Replication candidates after applying the pre-registered rules … only
Frangieh and Norman clear the target-axis floor." The paper's own results section, two paragraphs
below it, says seven of nine candidates passed and were trained, naming three Replogle datasets and
two Tian datasets that this table either marks dropped or does not list at all. The pooled estimate
the paper reports includes all five of them.

So the table contradicts its own section, and it presents the superseded rule as the pre-registered
one. **This is a paper fix, and it is in the appendix** (`main.tex`, section
"Replication protocol and dataset survey", after `\appendix`), so correcting it cannot disturb the
8-page body gate. Fixed on 2026-08-21; see the commit for this audit.

Note also that Amendment 2's own projected counts are not exactly what the build produced — it
predicted Frangieh 246, Datlinger 31 and Shifrut 21 where the built matrices give 238, 30 and 20. The
amendment was written before the rebuild, so this is a projection missing its target by a few percent
rather than a broken promise. The paper should quote the **built** numbers, which is what the fix does.

## 2. Amendment (2026-08-03) — the replicate unit, pinned per dataset

**KEPT.** The table pins a unit and a column for five datasets, and the built provenance agrees with
the column for every one. The paper's `tab:repl` replicate-unit column is consistent with it.

One trap this audit walked into and is recording so nobody repeats it: the provenance JSON's key
`replicate_unit` holds `spec.replicate_col`, a **column name**, not the unit. Shifrut's unit is the
donor and its column is `replicate`; Datlinger's column is *also* `replicate` for a different unit
entirely. A verifier that trusted that field reported the paper's correct "donor" as an error.
`adapter.py` now carries a comment at the emit site, and `verify_numbers.py` reads the units from this
amendment's table instead.

## 3. Amendment 3.1 — K is not portable across datasets

**BROKEN, twice.** In full at the bottom of this file.

## 4. Amendment 3.2 — the primary contrast is fixed by condition count

**KEPT.** `run_replication_campaign.sh:22-34` runs four arms on `FrangiehIzar2021_RNA` alone and three
on every other dataset, with `condition_gated` absent by construction. `pooled_with_reference.json`
carries `h1_vs_no_graph` at k=1 (Frangieh only) and `h2a` at k=8, pooled separately and never merged,
exactly as registered. The paper states the k=1 limit rather than hiding it.

## 5. Amendment 3.3 — PINNACLE context assignment

**KEPT.** The campaign passes `melanocyte` to Frangieh and `retinal_pigment_epithelial_cell` to
Replogle RPE1, and `none` to K562-essential, K562-gwps, both Tian sets and Norman. The amendment
allows `none` for K562 explicitly and for the neuron sets conditionally ("matching context if present,
else none"); the campaign's comment records that no neuron context exists upstream. The provenance
files record the context per dataset, and `esm2_only_ablation` is set where it is `none`.

## 6. Amendment 3.4 — the gate-collapse kill criterion

**KEPT, and it is the paper's opening result.** A lane whose mean edge gate falls to `<= 1e-3` is an
undecidable experiment and must be reported as such, never as evidence against the graph. That is
exactly what happened on the reference screen, and the paper leads its Observed Outcome section with
it ("The instrument was off"), reporting the repaired `screening_lambda0` root as the headline family.
This is the strongest KEPT verdict in the audit: the promise cost the project its original result and
was honoured anyway.

## 7. Amendments 4, 4a and 4b — the A1 diagnostic arms

**KEPT, all three.**

- **4.4/4.5**: `a1_mechanism.json` carries `family_size: 2` with Bonferroni and Holm both present and
  `survives_family_wise` requiring both. D1 `+0.0004` (Bonferroni 1.000) and D2 `+0.0065`
  (Bonferroni 0.182) are both null, which is the 2x2's "D1 null / D2 null" cell: the typed structure
  hurts and neither its parameters nor its labels are the route. That is what the paper concludes.
- **4.5's ban**: neither diagnostic arm is used anywhere to promote a graph claim.
- **4a**: the post-sampling relabelling, the globally consistent permutation and the exact per-relation
  edge counts are all pinned by named tests, including
  `test_permuted_relations_leave_the_sampled_neighbourhood_untouched`.
- **4b**: the sign convention (positive = the intervention improved on the typed encoder) is the one
  used in `a1_report.py` and carried unchanged into Amendment 7.2.

## 8. Amendment 5 — the A3 re-scoring

**KEPT, including the part that could have been gamed.** `rescored.json` carries all twenty cells with
`family_size: 20` on the across-metric bar and `family_size: 4` within each metric, both computed and
both reported, and the paper's `tab:metrics` marks survival on the across-metric bar as registered.

5.4's hardest clause — "endpoints that DISAGREE IN SIGN on the same contrast are a result, not a
nuisance, and are reported as one" — is kept against interest. The promotion margin is positive under
`pearson_delta`, `edistance_scperturb` and `energy_distance` and negative under the two GEARS
endpoints, and the paper reports that disagreement as a finding in the body rather than choosing a
kind endpoint. 5.5's restriction (this closes the metric half of commensurability, never the split
half) is respected: no sentence uses the re-scoring to adjudicate another paper's claim.

## 9. Amendment 6 — the injected-signal ladder

**KEPT.** `floor.json` carries `family_size: 6`, four seeds, `floor: 0.02`, `floor_status: measured`
and `control_clears: false`, which is the 6.7 rule executed without discretion — the permuted control
did not clear, the floor is the smallest rung that clears with every larger rung also clearing, and the
artifact's own notes state both. 6.2's leakage guard was watched to FAIL against a deliberately leaky
variant before being trusted. 6.3's rail-1 promise is asserted on the real 33,983-row matrix.

**6.5 was the one previously-found breach and it is now kept.** The amendment closed with "this is a
bound on the pipeline's sensitivity, not on the typed encoder's specifically, and it will be labelled
that way", and until 2026-08-20 it was not. The paper now says so in three places, and `floor.json`'s
own notes carry the sentence. Recorded here because a fixed breach is still a breach that happened.

## 10. Amendments 7 and 8

**KEPT.**

- **7.1/7.2**: seeds 0-4, paired against the landed `typed_static` lanes, D3 signed per the 4b
  convention.
- **7.3**, the anti-laundering rule: every B1 contrast is corrected at the number of B1 arms run as of
  the moment the result is read. B1b-d were declined on power and never ran, so m=1 stands, and the
  artifact records `family_size: 1` accordingly. The paper carries the m=1 caveat explicitly,
  including that Bonferroni and Holm are the identity there and that the p would not clear at m=4.
  This is the promise most exposed to quiet abuse and it was kept in writing.
- **7.6**: rail 2 honoured. `screening_b1/src_sha256.{before,after}.txt` hash the 61 seeded artifacts
  of the frozen source root and are **byte-identical to each other**, so the read-only root is
  demonstrably unmodified.
- **8.2/8.3/8.4**: both binnings exist in `deciles.json` with `family_size` 40 and 36, corrected as two
  separate families rather than pooled, and the level-versus-contrast caveat is stated in the paper.

**One soft spot, not a breach.** 7.4 says the recovery share "is DESCRIPTIVE only: it is a ratio of two
estimated quantities and carries no interval. The inferential statement is the CI on D3 itself." The
paper reports 79% twice and gives D3's CI immediately beside it at the appendix occurrence, so the
substance holds — but the words "descriptive only" and "carries no interval" appear nowhere, and the
abstract-adjacent occurrence at `main.tex:358` gives the share with no interval near it. Worth one
clause if the paper is touched again; not worth a page-budget fight on its own.

---

# The two material findings, both Amendment 3.1

Amendment 3.1 fixed a rule and a safeguard:

> RULE: K = 128 wherever train rows >= 256; otherwise the largest power of two <= train_rows/2. Any
> dataset run at K != 128 is a **deviation from the reference architecture** and must be labelled as
> such wherever it appears. The pooled estimate is reported TWICE - over all datasets, and over the
> K=128 subset alone - and if those two disagree, the K=128 subset is the one that speaks to the
> reference architecture.

Its stated rationale is worth quoting too, because it is exactly the failure that occurred: "silently
shrinking capacity on some datasets and not others is exactly the kind of unlogged weakening that
manufactures a null".

## Finding 1 — the K deviations are mislabelled

Measured from the built program bases (`program_response.parquet` column count, minus the index):

| dataset | K actually used | deviation? | labelled in the paper? |
|---|---|---|---|
| FrangiehIzar2021_RNA | 128 | no | n/a |
| ReplogleWeissman2022_rpe1 | 128 | no | n/a |
| ReplogleWeissman2022_K562_essential | 128 | no | n/a |
| ReplogleWeissman2022_K562_gwps | 128 | no | n/a |
| TianKampmann2021_CRISPRi | 32 | **yes** | yes, as "Tian runs at K=32" |
| TianKampmann2021_CRISPRa | **16** | **yes** | **no** — folded into "Tian at K=32" |
| NormanWeissman2019_filtered | **16** | **yes** | **NO, nowhere** |

The paper's only statement is "Tian runs at $K{=}32$ and is labelled a deviation". Norman is not
mentioned, and Tian CRISPRa is at 16 rather than 32. The campaign logs independently corroborate the
three distinct values (`K=128` nine times, `K=16` six times, `K=32` three times).

**Why Norman specifically matters.** Norman is the dataset carrying the paper's sharpest replication
claim. The body argues that raw topology has a real effect "whose sign we cannot predict", and the
only negative supporting that is Norman at $-0.0790$. It ran at one eighth of the reference program
capacity, undeclared.

**Be fair about what this is not.** Low K does not by itself explain Norman's sign: the other two
low-K datasets are both POSITIVE (Tian CRISPRa $+0.0226$ at K=16, CRISPRi $+0.0281$ at K=32). "Low K
goes negative" is not a rule this data supports. The defensible statement is narrower and still
serious: the paper's sign-disagreement claim is confounded with an undeclared capacity deviation, and
the pre-registration both anticipated that and prescribed the check.

## Finding 2 — the promised second pooled estimate was never computed, and it fires the contradiction stop

The K=128-subset pooled estimate does not exist anywhere: `pool.py` has no option for it, no artifact
contained it, and no number for it appears in the paper — **while the paper states that it is
reported.** That sentence was false.

Running it is free, since it re-pools landed per-dataset numbers and trains nothing:

| pool | k | random effects | 95% CI | p | I2 |
|---|---|---|---|---|---|
| all datasets (what the paper reports) | 8 | $+0.0091$ | $[-0.0024, +0.0207]$ | 0.121 | 87.5% |
| **K=128 subset (the pre-registered second estimate)** | **5** | **$+0.0141$** | **$[+0.0041, +0.0241]$** | **0.0056** | 87.7% |

The two disagree, which is the case Amendment 3.1 legislated for: the subset is the one that speaks to
the reference architecture. In it, all five datasets are positive with no sign disagreement at all —
reference $+0.0043$, Frangieh $+0.0194$, Replogle RPE1 $+0.0675$, K562-essential $+0.0081$, K562-gwps
$+0.0069$ — and under this project's family of four, raw $p=0.0056$ gives Bonferroni and Holm both
$0.0224$, clearing both.

**Rail 4 therefore fired.** Snapshotted to `data/results/replication/pooled_k128_subset.json`, flagged
at the top of `RESULTS_SUMMARY.md`, and the null was NOT rewritten around it.

**What it does not touch.** The headline null is h1, `condition_gated - expression_only`. In this
subset h1 is unchanged at k=1, $+0.0033$, $p=0.90$, because only Frangieh can test it at all. The
finding is about the UNTYPED arm, which was already the paper's one corrected-significant positive.
What changes is its consistency, and the standing of the sentence built on Norman.

**Two honest caveats.** $I^2$ is 87.7% in the subset too, so this is a random-effects reading over
datasets that still disagree in magnitude. And the pre-registration fixes Bonferroni-and-Holm at m=4
for the per-seed contrasts and says nothing explicit about correcting POOLED estimates, so "clears
both" is applied under this project's own convention rather than a rule the amendment spells out for
pooled numbers.

## What was fixed in the paper, and what was left for a human

Fixed, because they were false rather than arguable, and both in the appendix so the body gate is
untouched:

1. `tab:repl` now reports the post-Amendment-2 built counts and the datasets that actually trained.
2. The claim that the pooled estimate is reported over the K=128 subset is now true: the number is
   there, with its disagreement stated.

Left for a human: whether the sign-disagreement argument in the body should change. That is a
scientific call on the paper's central replication claim eight days from a deadline, and rail 4's
instruction is to flag and continue, not to rewrite.
