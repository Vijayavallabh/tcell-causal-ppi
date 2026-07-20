# feat-013 Reproducibility Verification — evidence notes

**Date:** 2026-07-20 · **Status: IN-PROGRESS — this feature cannot reach `done` from an agent session.**
Evidence file kept separate from the DoD triad (`feature_list.json` / `progress.md` /
`session-handoff.md`) because four sessions were live in this checkout; merge at commit time.

**Headline:** the reproduction axis is **`CANNOT_VERIFY`**, and that is the *correct* answer, not a defect
to fix. The confirmatory decision is defined on the sequestered challenge split (5,608 rows), which is
UNOPENED, and opening it is **test-steward-only**. Everything in this repository is DEVELOPMENT-fold
evidence. Nothing here was tuned to change that.

---

## 1. What was built

| File | Role |
| --- | --- |
| `reproducibility/manifest.py` | **new** — builds the manifest from this checkout's real frozen artifacts, and authors the 11 fallacy probes from its real diagnostics |
| `reproducibility/run_repro_real.py` | **new** — entry point; two-axis report, JSON, non-zero exit code |
| `reproducibility/verify.py` | **changed** — hash entries now carry `provenance`; a self-derived hash can no longer read as a passed check |
| `reproducibility/__init__.py` | **changed** — exports |
| `src/tests/test_reproducibility.py` | **changed** — 54 tests (was 37) |

`run_module8_real.py::run_repro()` (session A's older driver) was **not touched**. The new entry point
lives in this package; session A can rewire `run_repro()` to call `run_repro_real.main()` when convenient.

Run it with:

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src uv run python -m tcell_pipeline.reproducibility.run_repro_real
```

---

## 2. The manifest, from a real run

Real output, `data/results/reproducibility/manifest_real.json`:

| artifact | path | provenance | expected hash came from |
| --- | --- | --- | --- |
| `splits` | `data/splits/blocked_target_ood.csv` | **`independent-frozen`** | `data/splits/manifest.json`, published when the split was cut |
| `id_mapping` | `data/intermediate/id_mapping.parquet` | `self-derived` | hashed now — no frozen record exists |
| `de_layers` | `data/intermediate/de_layers/zscore.npz` | `self-derived` | hashed now — no frozen record exists |

**The distinction is enforced, not merely labelled.** A label nothing consumes is decoration, so
`verify._check_hashes` now downgrades a self-derived *match* from `pass` to `incomplete`, with the reason
in the report:

> *expected hash is not from an independently frozen record, so the artifact was compared against itself —
> this shows it is readable and internally stable, NOT that it reproduced*

Consequences, all deliberate:

* An **unlabelled** entry counts as self-derived. Unknown provenance is not evidence.
* A **mismatch still fails** whatever the provenance — the fire path is unchanged
  (`test_independently_frozen_hash_mismatch_still_fails`).
* A checkout whose hashes are *all* self-derived can no longer reach `REPRODUCIBLE`. Only `splits` is a
  genuine reproduction test today, so **1 of 3** deterministic artifacts is actually being reproduced.

`data/checkpoints/` was deliberately **not** hashed: session B was writing there during this work, so a
hash would have raced a half-written file.

### The config check now has a live input

Hashing today's config and comparing it to itself always passes — a guard whose input is a constant can
only confirm. The expected value is therefore taken from the **independently frozen**
`data/splits/manifest.json`, which recorded four knobs when the split was cut:

`SPLIT_SEED=0`, `SPLIT_FRACTIONS={train .60, val .15, calibration .10, challenge .15}`,
`SEQ_SIM_COSINE_THRESHOLD=0.85`, `GROUP_SIZE_CAP=0.05`.

All four still match `config.py`. *What input would make it fire?* Editing any of them:
`test_config_check_fires_when_live_config_drifts_from_the_freeze` bumps
`SEQ_SIM_COSINE_THRESHOLD` to 0.9 and watches the check go `fail` → verdict `NOT_REPRODUCIBLE`.

### Prediction table

`data/results/predictions/perturbed_mean/val/0.parquet`, 4,400 rows, checked against `val_rows: 4400`
published by a *different* artifact (`comparators/tabular_baselines_vs_h1.json:feature_coverage`) — an
independent count, not a self-measurement. **If that independent count is unavailable the block is not
declared at all**, because `verify` treats `n_rows: None` as vacuously satisfied and would emit a `pass`
for a row check that never ran. Absent → `missing` → `CANNOT_VERIFY`, never green.

---

## 3. The eleven probes, authored from real diagnostics

Sources (all read-only): `data/results/screening/*/*.parquet` (21 runs),
`screening/robustness_5seed.json` (the 4-contrast paired campaign), `screening/promoted.json`,
`comparators/*.json`.

**6 authored from real numbers · 5 Unevaluable with a stated reason · 3 flagged.**

The probe→input mapping was fixed *before* any outcome was computed, and every predicted outcome matched
what the run produced. No input was swapped after seeing whether it flagged. Coverage was later *reduced*
from 7 to 6 by the adversarial pass in §7 — `correlation_not_causation` was inflating it.

### Authored (6)

| probe | real input | outcome | evidence |
| --- | --- | --- | --- |
| `garden_of_forks` | the two pre-registered estimates of the **same** estimand "does the graph beat no-graph?": `promotion_margin` (untyped_gnn − expression_only) `+0.004513` and `h1_vs_no_graph` (condition_gated − expression_only) `−0.001909` | **FLAGGED** | sign flip; spread `0.006422` vs mean `0.001302` |
| `regression_to_mean` | baseline = per-config systema on the **selection seed 0** (the seed `promoted.json` promoted from); followup = mean over the **retest seeds 1–4** | **FLAGGED** | the seed-0 winner sat **1.551 SD** above the field on the fold it was selected on, and only **0.974 SD** above on the independent retest |
| `survivorship` | systema of all 20 runs; `survived` = used the full 20-epoch budget (10 of 20) | **FLAGGED** | survivor-only mean `0.08793` vs full `0.08306` |
| `look_elsewhere` | **all four** simultaneously-tested contrast p-values: `0.003555, 0.009225, 0.020791, 0.084719` | not flagged | m=4, Bonferroni α `0.0125`; min p `0.003555` survives |
| `simpson` | per-config `(epochs_run, systema)` — the early-stopping confound | not flagged | pooled `+1`, within `[+1, 0, +1, 0]` — no reversal |
| `ecological` | same 20 runs, aggregate vs individual | not flagged | individual `0.7239` vs aggregate `0.7632`, 4 groups |

Notes on four of these, so they are not over-read:

* **`simpson` / `ecological` are weak evidence, by construction.** `epochs_run` is **98.8 %** explained by
  the arm label (between-arm SS 361.4 of 365.8) and is *constant* within two of the four arms, so those two
  contribute no within-trend and the "aggregate" correlation is 4 points on a tied x. It is also the
  *outcome* of early stopping on val loss — a descendant of model quality, not a treatment. And it does not
  survive a change of budget proxy: under `gpu_hours` the pooled trend is **negative** (−0.534) with all
  four within-arm trends negative. Neither probe flags under either proxy, so nothing was shopped, but a
  clean non-reversal here should not be quoted as evidence that the budget confound is dead.
  Also: **`untyped_gnn` is a graph arm**, it used the full budget, and it posts the highest 5-seed systema
  (0.0902). `expression_only` is the *only* no-graph arm. An earlier draft of this note said "the no-graph
  arms run all 20", which is wrong and would have supported "the graph arms simply trained less" — the
  data do not.

* **`look_elsewhere` does not flag, and feeding it the whole family is the honest call.** The detector is
  family-shaped (`any` raw hit vs `any` Bonferroni survivor); with all four contrasts, h2a at p=0.003555
  clears α/4, so the family retains a survivor. That is *not* a clean bill for the promotion margin:
  `promotion_margin`'s raw p=0.020791 → Bonferroni 0.0832, `survives_family_wise: false`, and it had been
  published as a resolved positive. Feeding the detector that single p-value would have made it
  arithmetically incapable of flagging (α/1 == α) — a guard whose input is a constant — so the builder
  **refuses** any p-family smaller than the recorded `family_size`
  (`test_look_elsewhere_is_dropped_rather_than_understating_the_family`).
* **`survivorship`'s flag is a reporting hazard, not a causal result.** `survived` is *collinear with the
  arm label* — zero within-arm variance, two arms always finish and two always early-stop. So the flag says
  "the full-budget subset reports a higher mean" (0.08793 vs 0.08306), **not** "the budget caused it". That
  is still a real hazard: any statistic restricted to full-budget runs silently drops two arms. The same
  collinearity *is* fatal to `berkson`, which asks a causal question (does selection *induce* association?),
  which is why that one is refused rather than authored on the same mask.
* **`regression_to_mean` bears directly on the promotion**, but read it as a *directional diagnostic, not a
  significance test*. `promoted.json` records `basis: "single-seed screening on systema (seed 0)"` and
  `pinned_rank: 3`, so seed 0 genuinely is the selection fold. The series is 4 configs wide, so
  `select_frac=0.1` reduces to top-1 and the statistic is one config's deviation in cross-config SD units.
  It says the screening winner shrank toward the field on an independent retest; it does not put a p-value
  on that.

### Unevaluable (5) — with an honest reason each

Absent-with-a-reason beats both a fabricated input and a silent omission (a dropped probe loses its reason,
and `run_fallacy_scan` reports it identically to "not evaluated"). **Three of these reasons were wrong in
the first draft and were corrected after the §7 adversarial pass** — the corrections are the substance
here, so they are stated, not quietly patched.

| probe | why not, concretely |
| --- | --- |
| `base_rate` | **NOT a lack of evidence — and the probe fires.** See §3.1 below. It is not authored only because the label array is 4,400 × 10,282 ≈ 45 M entries and `base_rate` takes *arrays*, so it cannot be carried as literal kwargs in a frozen manifest. **Unlocked by:** a confusion-matrix entry point on `base_rate`, or a manifest that may reference a derived artifact. |
| `correlation_not_causation` | **refused for being decoration.** The detector flags only on \|corr\| ≥ 0.3 *and* absent interventional support; the study's headline association is 0.083 (systema; its Pearson is 0.113), so it returns "not flagged" for *either* value of the support flag — a guard that can only confirm, which is exactly what this module refuses to author elsewhere. The support argument is worse: it was derived from a hardcoded split filename, i.e. a compile-time constant, not a reading of any artifact. **Unlocked by:** a headline association above 0.3. |
| `berkson` | **deferred, not impossible.** Both per-row inputs *are* derivable — the off-graph mask from `data/graphs/protein_edges.parquet` (the published 385/91 counts come from exactly that derivation, `run_module8_real.py:491`), and a per-row metric from `metrics._rowwise_pearson`. At 4,400 rows they would even fit in a manifest. What is missing is the derived artifact, which this session did not compute. The run-level selections that exist cannot substitute: full-budget completion is a deterministic function of one of the variables (range restriction, not a common effect), and registry completion excludes 3 runs whose metrics are empty, so they carry no `y`. |
| `collider` | needs a third variable measured independently of the two. `best_val` — the natural candidate — sits on two incommensurable scales (~3.47 vs ~490 for `typed_static`), and sub-setting to the comparable arms *after seeing the numbers* is itself a fork. **Correction:** the first draft claimed `best_val` was the *only* run-level candidate. False — `pearson`, `prog_cos`, `mae`, `rmse`, `topk`, `sign`, `centroid`, `gpu_hours` are all single-scale. The real objection is that they are not independent of the outcome: `pearson` and `prog_cos` correlate **0.973** with `systema`, so conditioning on them conditions on the outcome. |
| `reverse_causation` | needs a cross-lagged pair. Single-timepoint design: Rest/Stim8hr/Stim48hr are experimental arms, not repeated measures on the same units. **Unlocked by:** a longitudinal panel this dataset does not contain. |

### 3.1 `base_rate` — a measured result, reported rather than dropped

The first draft claimed no per-row truth existed. That was false, and the adversarial pass caught it. Val
truth *is* on disk: `data/intermediate/de_layers/zscore.npz` is **(33,983 × 10,282)** keyed by the same
`row_index` the prediction parquets carry, and `training/dataset.py.__getitem__` already slices it that way.

I performed the join and validated it before drawing anything from it — it reproduces `condition_gated`
seed 0's recorded metrics **exactly**:

| metric | recomputed from the join | recorded in the screening parquet |
| --- | --- | --- |
| `topk` (k=20) | 0.012807 | 0.012807 |
| `sign` | 0.507814 | 0.507814 |
| `mae` | 0.815549 | 0.815549 |

`base_rate` on the study's own top-20 DE call (`config.METRICS_TOP_K`, the same rule `topk_recall` uses —
no invented threshold) then **FLAGS**:

```
flagged = True   accuracy 0.99616   precision 0.012807   prevalence 0.0019451   n_positive_predictions 88000
```

That precision *is* the published `topk` metric. 99.6 % accuracy with 1.3 % precision on a 0.19 %-prevalence
class is textbook base-rate neglect, on a number the screening table publishes for all 21 runs. I computed
it, so I report it — burying it because it will not fit in a manifest would be the mirror image of the
dishonesty this feature exists to prevent.

The goal brief pointed at the off-graph rows (385 train / 91 val, all-zero feature vector) as material for
`survivorship`/`base_rate`. Those counts are real, but neither detector consumes them as shaped: both need
a per-row *value or predictor*, and pairing the off-graph mask with a constant gives a guard that can only
confirm. `survivorship` was therefore authored at the *run* level, and `base_rate` on the DE call above.

---

## 4. The verdict, and its exit code

Real run (`data/results/reproducibility/repro_real_report.json`):

```
[repro] verdict (all checks)   = NOT_REPRODUCIBLE
[repro] reproduction axis      = CANNOT_VERIFY   <- confirmatory_decision: missing
[repro] inference axis         = FLAGGED: regression_to_mean, survivorship, garden_of_forks
                                 (6/11 probes evaluated)
PROCESS EXIT CODE = 1
```

**Two axes are reported because one word would misreport.** `verify._verdict` ranks a fallacy flag above
an unperformed check, so `NOT_REPRODUCIBLE` alone would read as *"we ran the reproduction and it failed"* —
false; it was never run. So the report carries both, from the same `_verdict` function (the reproduction
axis is `_verdict(checks minus fallacy_scan)`, not a re-implementation):

* **reproduction = `CANNOT_VERIFY`**, cause `confirmatory_decision: missing`. The cause is picked in
  `_verdict`'s own precedence (`fail` → `missing` → `incomplete`); naively taking the first non-passing
  check would have blamed a self-derived hash for a verdict the missing decision determined.
* **inference = FLAGGED** on 3 real traps in development-fold analysis. Not evidence about reproduction.

**Exit code:** `0` only for exactly `REPRODUCIBLE` — a whitelist, so `CANNOT_VERIFY`, `None`, `""` and any
future verdict all exit non-zero. An unattended run or a CI status gate cannot record this as green.
(`test_exit_code_is_zero_only_for_the_reproducible_verdict`.)

---

## 5. What remains, and WHO must do it

| remaining | who | how |
| --- | --- | --- |
| **The confirmatory decision** — the blocker for `done` | **the test steward.** Opening the sealed split is steward-only; no agent session may do it. | Run `evaluation/sealed_eval.py` **once** on the sequestered challenge split (5,608 rows). Publish its result as `manifest.decision` (`h1_confirmed` plus at least one of `lcb_95` / `rho_egipg` / `delta_vs_best`) and the re-run's as `manifest.observed.decision`, then re-run this verifier. |
| **Independent hashes for `id_mapping` and `de_layers`** | whoever cuts the next freeze | Publish a sha256 for every deterministic preprocessing artifact *at the moment it is frozen*. Then a later checkout's hashes become genuine reproduction tests instead of readability checks. |
| **4 Unevaluable probes** | analyst, optional | The table in §3 names the artifact each one needs. A joined per-row prediction-vs-truth table on val unlocks two of the four. |

`sealed_eval.py` was **never executed**, and the sealed split remains sequestered and unopened. So does
`run_module0.py` and `programs.run_program_basis` (the frozen basis was only ever hashed, never
regenerated). `promoted.json` was read, never modified. All writes went to
`data/results/reproducibility/`.

---

## 6. Verification

* `./init.sh` green, **exit 0 — 490 passed** (411 at session start). `test_reproducibility.py` went
  **37 → 85** collected tests; the rest of the delta arrived from the three other sessions live in this
  checkout, so the total is a snapshot, not this feature's number.
* One transient failure was seen in a single full-suite run and did not reproduce across three subsequent
  runs. The collected count changed 489 → 490 between those runs, i.e. another session added a test
  mid-run — this suite is being mutated under us by three concurrent sessions. This feature's own surface
  is deterministic: `test_reproducibility.py` gives 85/85 on three consecutive isolated runs. Recorded
  rather than dismissed.
* Red-first throughout: every new test was watched failing before its fix existed.
* **Mutation testing: 29/29 mutants caught.** Each breaks the one line a claim rests on; the guarding test
  goes red. Harness: `scratchpad/mutate.py` (28th mutant dropped as *equivalent* — removing the
  `usable_runs` native-coercion is unobservable now that the `bool()` sits on the survivor mask itself;
  reporting it as caught would have been gaming the number).

Mutants are grouped by the claim they attack:

| claim under attack | mutant | caught by |
| --- | --- | --- |
| a self-derived hash cannot certify | provenance check → `True` | `test_self_derived_hash_match_reads_as_incomplete_not_pass` |
| a hash mismatch is still decisive | mismatch branch → `False` | `test_independently_frozen_hash_mismatch_still_fails` |
| **zero checks is not a pass** | drop the empty-critical guard | `test_zero_checks_is_never_reproducible` |
| an empty prefix list is not a schema check | drop the prefix validation | `test_an_empty_parquet_does_not_pass_the_schema_check` |
| an unchecked row count is not a pass | `None` rows → `_PASS` | `test_an_unchecked_row_count_does_not_read_as_a_passed_schema_check` |
| **a decision compared to itself cannot certify** | attestation → `True` | `test_a_decision_compared_against_itself_never_certifies` |
| a bool is not a numeric endpoint | drop the observed-side bool exclusion | `test_a_boolean_observed_endpoint_is_not_a_numeric_comparison` |
| a null manifest returns a verdict | drop the `report = None` init | `test_a_null_manifest_returns_a_verdict_rather_than_crashing` |
| wrong-typed manifest slots return a verdict | `_as_dict` → identity | `test_a_malformed_manifest_yields_a_verdict_not_a_traceback` |
| the p-family must be complete | `== family_size` → `>= 1` | `test_look_elsewhere_is_dropped_rather_than_understating_the_family` |
| the retest excludes the selection seed | seed filter → `True` | `test_regression_to_mean_retests_on_seeds_excluded_from_the_selection` |
| survivors are the full-budget runs | mask → all `True` | `test_survivorship_survivors_are_the_runs_that_used_their_full_budget` |
| a one-sided split cannot be a check | drop `not all(survived)` | `test_survivorship_is_refused_when_every_run_survived` |
| a fork needs both arm choices | `== 2` → `>= 1` | `test_garden_of_forks_needs_both_arm_choices_not_one` |
| no probe may vanish silently | empty the reason table | `test_every_fallacy_is_authored_or_explicitly_unevaluable` |
| **an authored probe must actually evaluate** | drop `_demote_dead_probes` | `test_a_probe_that_dies_inside_the_detector_keeps_its_reason` |
| a detector *bug* stays visible as a bug | demote on any exception | `test_a_detector_bug_is_not_laundered_into_inadequate_input` |
| a probe that cannot fire is refused | drop the threshold refusal | `test_correlation_not_causation_is_refused_while_it_cannot_flag` |
| the config snapshot is not a constant | hardcode `SPLIT_SEED` | `test_frozen_config_snapshot_matches_todays_config_and_can_still_drift` |
| only the frozen split claims independence | provenance → always `INDEPENDENT` | `test_build_manifest_marks_only_the_frozen_split_independent` |
| predictions need an independent row count | drop the `n_rows` type guard | `test_a_prediction_table_with_no_independent_row_count_is_not_declared` |
| artifact paths cannot escape the checkout | drop normpath + escape guard | `test_an_artifact_outside_the_project_is_dropped_not_emitted_as_an_escape` |
| the builder hashes the checkout it is given | — | `test_manifest_hashes_the_checkout_it_is_pointed_at` |
| wrong-typed diagnostics still partition | drop the `_as_dict` filter | `test_build_fallacy_inputs_partitions_even_on_wrong_typed_diagnostics` |
| numpy scalars are not dropped | `_num` → python types only | `test_numpy_scalars_from_parquet_are_not_silently_dropped` |
| `CANNOT_VERIFY` never exits 0 | `exit_code` → `0` | `test_exit_code_is_zero_only_for_the_reproducible_verdict` |
| the cause is what *drives* the verdict | reorder the precedence walk | `test_real_run_reports_cannot_verify_on_the_reproduction_axis_and_exits_nonzero` |
| the reproduction axis excludes the scan | include the fallacy check | `test_real_run_reports_cannot_verify_on_the_reproduction_axis_and_exits_nonzero` |

**Four mutants were invisible to mutation alone** and needed a hand-built input, exactly as AGENTS.md warns
("no mutation of the code reaches an input class the tests never construct"):

1. the fork-count guard — real data carries both estimates, so nothing changed;
2. the checkout re-anchoring — `build_hashes`' own `relative_to` masked it (the mutant hashed *nothing*
   rather than the wrong thing: safe, but useless);
3. the escape guard — my first attempt pointed at a file that **did not exist**, so both versions dropped
   it for the wrong reason. The input only discriminates once the outside file is real and readable;
4. the detector-bug launderer — reachable only by monkeypatching a detector to raise.

## 7. The adversarial pass — 14 real defects, found after everything above was green

Everything in §6 was already green — red-first, 17/17 mutants, `./init.sh` passing — when two agents were
run with the sole brief of breaking this module. They found **14 real defects**, which is the point worth
recording: *watched-failing plus mutation testing did not make this correct.* One agent attacked the code,
one attacked the scientific honesty of the probe authoring.

### Defects in the code (all fixed, each with a test above)

| # | defect | why it mattered |
| --- | --- | --- |
| 1 | **`_verdict([])` returned `REPRODUCIBLE`** | Three `any()` calls over an empty list are all `False`, so it fell through to the certifying return. Every early exit in `verify_reproducibility` sets `checks: []`, so a checkout **that does not exist** got `reproduction_verdict: REPRODUCIBLE` stamped into the JSON. Absence of *all* evidence was the cleanest possible pass — the exact inversion this feature exists to prevent, sitting in my own new code path. |
| 2 | **the confirmatory decision had no provenance guard** | `decision` and `observed.decision` both arrive in one manifest and nothing required them to differ. Pointing one at the *same dict* passed. A hash got `incomplete` for being compared against itself while the H1 decision — the most load-bearing check here — got `pass`. Now `observed` must declare `provenance: independent-rerun`; unattested is `incomplete`. |
| 3 | **authored-but-dead probes lost their reason** | Probes were authored on a *shape* precondition but detectors enforce *variance* ones. On a realistic checkout where nobody early-stops (constant `epochs_run`), simpson and ecological were authored, raised `Unevaluable` inside the scan, and vanished from **both** lists — the operator saw "no trap detected (2/11)" with nothing named. Now every authored probe is run at build time and demoted with the detector's own reason. Only `Unevaluable` demotes; a real detector bug stays visible in `crashed`. |
| 4–9 | **six inputs raised instead of returning a verdict** | A manifest that is the valid JSON document `null` took the *success* path and left `report` unbound — so a truncated write crashed while pure garbage was handled. Plus `config_hashes`/`observed`/`fallacy_inputs` as a list or string, and `columns_prefixes` non-string or `5`. The module's docstring promises a verdict, not a traceback. |
| 10–11 | **the schema check was vacuously satisfiable** | `columns_prefixes: []` makes `all(...)` true, so an empty parquet passed; and a missing `n_rows` skipped the row comparison while still reading `pass`. My own `_predictions` declined to emit `n_rows: None` — but *verify* is the guard, and it accepted it from any other manifest. |
| 12 | **`build_manifest` emitted a path escaping the checkout** | `Path.relative_to` is purely lexical and does not normalise `..`, so an env root of the form `<project>/../<project>/data/splits` published `../<project>/…` into the manifest, with a hash of a file outside the declared root. |
| 13 | **`build_fallacy_inputs` crashed on wrong-typed diagnostics** | Documented as pure and callable with a constructed diagnostic set — which is precisely how an adversarial caller reaches it. |
| 14 | **a bool passed as a numeric endpoint** | `bool` was excluded on the frozen side only, so `observed.lcb_95: true` compared as `1.0`. |

Two findings were **not** fixed, deliberately, and are recorded as limitations instead:

* **A checkout that is a symlink to the original data verifies against itself.** `verify._resolve`'s
  containment is lexical *by design* (documented: `data/` may symlink to a shared store), so a "checkout"
  containing only `data -> <repo>/data` passes three checks on zero bytes of its own content. Resolving
  symlinks would reject the documented layout. **A clean-checkout verification must therefore use a real
  copy, not a symlink farm** — this is the one caveat a steward needs to know.
* **`MAX_DECISION_TOLERANCE = 0.01` does not prevent a sign flip.** Its comment claimed it did. At the cap,
  a frozen `lcb_95 = +0.005` and an observed `−0.005` compare as reproduced. The cap is on *absolute* drift,
  not drift relative to zero. I left the rule alone rather than inventing a statistical criterion nobody
  specified, but the steward should read the endpoint's sign, not only the verdict.

### Defects in the probe authoring (the honest-frame half)

The second agent's headline: **all three flags survive every defensible alternative input** — none was
shopped. `garden_of_forks` flags under every extension of the contrast set; `regression_to_mean` flags for
mean, median, or any single retest seed. Every number quoted in `manifest.py` (385/91, 21,262, 4,400,
5,608, 3.47 vs 490, 11–13 of 20, "3 runs with empty metrics") checks out against disk.

The dishonesty was entirely on the **non-flagging** side, which is the direction that inflates apparent
coverage:

1. **`base_rate` was dodged with a false reason.** Fixed — see §3.1. I verified the join myself before
   accepting the finding, and the probe flags on a published metric.
2. **`correlation_not_causation` was decoration.** It cannot flag at this study's 0.083 for *either* value
   of its second argument, and that argument was a compile-time constant. Now refused. Honest coverage
   dropped 7/11 → 6/11.
3. **Two Unevaluable reasons were factually wrong about disk** (`berkson`, `collider`). Both *conclusions*
   survive, but on different grounds; the reasons are corrected in §3 and the corrections are stated as
   corrections.
4. **`simpson`/`ecological` rest on a near-degenerate x**, and the trend reverses under `gpu_hours`. Both
   still report clean under either proxy, so nothing changes — but the note now says so rather than
   implying the budget confound was cleanly ruled out.
5. **`survivorship`'s selector is collinear with the arm label.** The flag is a reporting hazard, not a
   causal claim, and the note now distinguishes that from why `berkson` is refused on the same grounds.


---

## 8. `/code-review` pass — 4 further defects, after §7 was green

A five-agent code review (AGENTS.md compliance, shallow bug scan, git history, prior-review feedback,
comment contracts) ran over the finished diff. It found four more real defects plus one cross-call-site
regression. **Every one was reproduced by execution before being accepted**, and all are now fixed with a
red-first test and a mutant.

| # | defect | fix |
| --- | --- | --- |
| 1 | **`build_hashes` crashed on a malformed frozen split record.** `_frozen_split_record(root).get("sha256") or {}` guards a *falsy* value but not a wrong-typed one — a truncated `data/splits/manifest.json` whose `sha256` is a string raised `AttributeError` out of `build_manifest`, so `run_repro_real.main()` produced **no report at all**. Exactly the hole `_as_dict` was added to close in `verify.py`, in the one call site that didn't use it. | `_as_dict(...)`; `test_build_hashes_survives_a_malformed_frozen_split_record` |
| 2 | **`correlation_not_causation` had a live path back to decoration.** `has_interventional_support` derives from `config.BLOCKED_SPLIT_PATH.name`, and `config.py:135` hardcodes that filename — a tautology, `True` in every run. Verified: with support present the detector cannot flag for *any* correlation. The old code refused the probe only *below* the 0.3 threshold and authored it above, and the refusal text claimed crossing the threshold would make support "a real question". It would not. | refused **unconditionally**, with both reasons stated; `test_correlation_not_causation_can_never_be_authored_in_this_study` |
| 3 | **`_cause` did not walk `_verdict`'s precedence**, despite saying so. Its fallback scanned every check rather than critical ones, so a critical check with an unrecognised status (the case the whitelist verdict exists to survive) got blamed on whichever non-critical check sorted first. Verified: `_verdict` → `CANNOT_VERIFY`, `_cause` → `schema:p`. | mirrors `_verdict` exactly, including unknown statuses; `test_cause_names_the_critical_check_that_drove_the_verdict` |
| 4 | **`usable_runs`' docstring claimed the coercion was load-bearing.** It isn't — the `bool()` at the call site prevents the `np.bool_` → `"True"` hazard, and the other values are `np.float64`, which subclasses `float`. I had already established this when I dropped that mutant as *equivalent*, and left the overstated claim in anyway. | docstring corrected to name `_num`'s `np.integer` acceptance as the real guard |

### Cross-call-site regression — for session A, not fixable from here

`verify.py`'s provenance change treats an unlabelled hash as self-derived. `run_module8_real.py:591`
(session A's older driver) builds entries without `provenance`, **including `splits`, whose expected hash it
correctly reads from the frozen record**. Reproduced against the real artifacts:

```
hash:splits -> status=incomplete  provenance=self-derived
reason: expected hash is not from an independently frozen record...
...while run_repro() line 617 prints: independently-checked artifacts: ['splits']
```

The same invocation asserted both.

**CORRECTION (2026-07-21), from session A.** The sentence that stood here — *"the verdict is unaffected, so
this is a labelling contradiction, not an exit-code one"* — understated the defect, and I should have known
better: `run_repro()` printed `VERDICT = CANNOT_VERIFY` and then `return 0`, so an unattended run or any
exit-status gate recorded an unverifiable reproduction as **green**. That is the same
guard-not-honoured-to-the-exit-code failure this repo already fixed once in the multiseed campaign, and it
is the reason my own driver's `exit_code()` exists. I read that `return 0` at the start of this session and
built against it — then wrote a note that denied it. My change caused only the labelling half; the
exit-code half was pre-existing and I failed to record it.

**RESOLVED (2026-07-21).** Session A took the delegation option: `run_repro()` is now a 21-line call to
`run_repro_real.main([])`, with 54 lines of duplicate manifest-building deleted. Verified live from here —
`run_module8_real.run_repro()` returns **1**, `hash:splits` reads `pass` / `independent-frozen`, and
`id_mapping` / `de_layers` read `incomplete`. A pinned both defects with
`test_comparators.py::test_repro_part_propagates_a_nonzero_exit_on_cannot_verify`.

### Filtered as a nit

`_verdict`'s docstring attributes its reachability to `verify_reproducibility`'s early returns, which
hard-code their verdict and never call `_verdict`. The guard *is* reachable and end-to-end tested — via
`run_repro_real.reproduction_axis`, one hop away — so the sentence is imprecise rather than wrong.

### What the review did NOT find

The comment-contract agent independently recomputed every factual claim in `manifest.py` from disk —
zscore shape `(33983, 10282)`, the 0.012807 / 0.507814 / 0.815549 metric triple, the best_val scale break,
the 0.973 correlations, 385/91 off-graph, 21,262 / 4,400, the 98.8 % arm/epoch confound, "3 runs with empty
metrics" — and **all matched**. The bug-scan agent found no logic inversions, caller-data mutation,
determinism issues, or resource problems.

---

## 9. Session A's return handoff — one real bug in this module

A's delegation exercised `run_repro_real` from a second caller for the first time and immediately found a
defect that none of §7's or §8's passes had reached.

**`main(argv=None)` silently read `sys.argv`.** `argparse` falls back to `sys.argv` when `argv` is None, so
A's first `main()` call inherited the *parent driver's* flags and died with
`unrecognized arguments: --part repro --device cuda:2`. A's own test monkeypatched `main`, so it stayed
green while the real command was broken — **only running it caught this**, which is the sharpest instance
in this whole feature of a test proving something about code that was never executed.

Fixed at the source rather than by asking every caller to remember `main([])`:

* `def main(argv=())` — the default is now empty, never None, so a programmatic call can never inherit a
  parent's argv;
* `if __name__ == "__main__": sys.exit(main(sys.argv[1:]))` — the CLI passes its own argv explicitly.

Pinned by `test_main_ignores_a_parent_process_argv`, which sets `sys.argv` to A's exact failing command
line and asserts `main()` still returns a verdict; it also keeps A's current `main([])` form working.
A suggested a separate `run(root=…, out_dir=…)` library entry point; I did not add one, because the
two-line default fix removes the footgun for every caller without a second way to do the same thing.

**Also from A's handoff:**

* A deleted the stale `data/results/reproducibility/reproducibility_report.json` left by its pre-fix
  driver. Confirmed gone; the directory now holds only `manifest_real.json` and `repro_real_report.json`.
* **The spec line is NOT stale, but it is now a trap.**
  `docs/specs/2026-07-17-module8-comparators-audit-sealed-repro.md:189` says the report is written to
  `<REPRODUCIBILITY_ROOT>/reproducibility_report.json`. That remains a true statement about
  `verify_reproducibility`'s *default* `out_path` (`verify.py:345`, unchanged) — but now that both drivers
  pass an explicit path, **no run produces that filename**, so a reader following the spec looks for a file
  that never appears. Whoever owns that spec should ADD `repro_real_report.json` rather than replace the
  line. I did not edit it: it is a shared spec outside this session's ownership and the sentence is not
  wrong.
* **A second, quieter divergence in the same spec.** Its verdict tree reads *"any critical **fail** →
  NOT_REPRODUCIBLE; else any critical **missing** → CANNOT_VERIFY; else any non-critical issue →
  PARTIALLY"*. It does not contemplate a critical check in `incomplete`, which now yields PARTIALLY. That
  status predates this session (the fallacy scan used it), but this session widened it from one check to
  three classes — self-derived hashes, unattested decisions, and unchecked row counts. The code is what I
  intend; the spec's tree no longer fully describes it.
* A notes that `exit_code()`'s semantics now gate `--part repro` and `--part all` as well as this module's
  own CLI. Intended: `CANNOT_VERIFY` is the honest answer here and must not exit 0 from any entry point.

---

## 10. Session A's item 3 — a runtime dependency on another session's artifact (fixed)

A flagged that `manifest.py` read the prediction table's expected row count out of
`data/results/comparators/tabular_baselines_vs_h1.json:feature_coverage.val_rows` — **a file session A
rewrites**. A rewrote it at 01:23 with the completed feat-006 run; the values happened to be unchanged
(385 / 91 / 21,262 / 4,400, verified), so nothing was wrong today. But if that run's feature handling had
changed, the "independent" expected count would have moved with it and this probe would have followed
**silently**. That is AGENTS.md's *presence is not freshness* rule, and a check that tracks whatever another
session last wrote is not a check.

**Fixed.** The authority is now the FROZEN split: `data/splits/blocked_target_ood.csv` joined to
`data/intermediate/perturbation_condition.parquet`, both git-tracked and frozen, giving val = 4,400 in
0.06 s. A's artifact is still read, but only as a **cross-check that must agree** — recorded in the manifest
with its mtime. If the two sources disagree the prediction entry is **refused** (verify then reports
`missing` → CANNOT_VERIFY) rather than picking a winner: two independent sources disagreeing about the fold
means the row count is not known.

Three tests, three mutants:

| claim | test |
| --- | --- |
| the count comes from the frozen split, not another session's file | `test_prediction_row_count_comes_from_the_frozen_split_not_another_sessions_artifact` |
| a disagreeing cross-check refuses the entry | `test_a_row_count_that_disagrees_with_the_frozen_split_is_refused` |
| the module does not NEED another session's results to function | `test_the_row_count_survives_the_comparator_artifact_being_absent` |

Verified live: `n_rows: 4400`, source "derived from the frozen split", `cross_check.agrees: true`.

---

## Appendix — feature_list.json evidence block for feat-013

Append-only, house style (` UPDATE (YYYY-MM-DD, context): …`, joined with a single leading space, existing
text preserved as a strict prefix). **Session A merges this; session D does not edit the triad.**

```text
 UPDATE (2026-07-21, parallel session D — manifest from a real run + the 11 probes from real diagnostics): built reproducibility/manifest.py (build_manifest / load_diagnostics / build_fallacy_inputs) and reproducibility/run_repro_real.py; run_module8_real.run_repro() now delegates to it (session A). VERDICT ON THIS CHECKOUT, from a real run: NOT_REPRODUCIBLE over all checks, and the REPRODUCTION AXIS (every check except the fallacy scan, via verify._verdict) is CANNOT_VERIFY, cause confirmatory_decision=missing. That is the CORRECT answer, not a defect: the confirmatory decision is defined on the SEQUESTERED challenge split (5,608 rows), which is UNOPENED. Both axes are reported because verify's single verdict ranks a fallacy flag above an unperformed check, so NOT_REPRODUCIBLE alone would read as "we ran the reproduction and it failed" -- it was never run. Exit code is 0 ONLY for REPRODUCIBLE (whitelist), so the run exits 1 and no CI gate can record it green. MANIFEST: 1 of 3 deterministic artifacts is genuinely reproduced -- splits is independent-frozen (expected sha256 published in data/splits/manifest.json at freeze time) and PASSES; id_mapping and de_layers are self-derived and now read `incomplete`, never `pass`, because a hash compared against itself shows the file is readable, NOT that it reproduced (unlabelled provenance counts as self-derived). Config check re-keyed onto the 4 knobs the split freeze independently recorded (SPLIT_SEED / SPLIT_FRACTIONS / SEQ_SIM_COSINE_THRESHOLD / GROUP_SIZE_CAP), so editing config.py now FAILS it; previously it hashed today's config against itself and could only confirm. Prediction row count (4,400) derived from the FROZEN split, with session A's feature_coverage.val_rows kept only as a cross-check that must agree -- reading it as the authority would have let another live session's rewrite move this probe silently. FALLACY SCAN: 6 of 11 probes AUTHORED from real diagnostics, 5 Unevaluable with a stated reason, 3 FLAGGED -- garden_of_forks (the two pre-registered graph-vs-no-graph estimates, promotion_margin +0.004513 vs h1_vs_no_graph -0.001909, SIGN FLIP), regression_to_mean (the seed-0 screening winner sat 1.551 SD above the field on its selection fold and 0.974 SD on the independent seeds 1-4 retest), survivorship (full-budget-only mean 0.08793 vs 0.08306 over all 20 runs; NOTE the selector is collinear with the arm label, so this is a REPORTING hazard, not a causal claim). Clean: look_elsewhere (all four campaign p-values, m=4, min p 0.003555 clears alpha/4 -- feeding one p would make it arithmetically incapable of flagging, so a family smaller than the recorded family_size is REFUSED), simpson, ecological (both on epochs_run, which is 98.8% explained by the arm label and reverses under gpu_hours -- weak evidence, quoted as such). UNEVALUABLE, each with the artifact that would unlock it: base_rate, berkson, collider, reverse_causation, correlation_not_causation (the last REFUSED as decoration -- it cannot flag at this study's 0.083, and has_interventional_support derives from a hardcoded filename, so it is a compile-time constant). MEASURED BUT NOT AUTHORED: base_rate on the study's own top-20 DE call DOES flag -- accuracy 0.9962, precision 0.012807 (identical to the published topk), prevalence 0.0019; the join was validated by reproducing condition_gated seed 0's recorded topk/sign/mae exactly. It is not in the manifest only because the label array is 4,400 x 10,282 (~45M entries) and base_rate takes arrays; recorded rather than dropped. REMAINS, STEWARD-ONLY: run evaluation/sealed_eval.py ONCE on the sealed challenge split and publish decision + observed.decision (with provenance independent-rerun) -- an agent session CANNOT reach this feature's terminal state, and everything in this repo is DEVELOPMENT-fold evidence. Also remains: publish a sha256 for id_mapping/de_layers at the next freeze so they stop being self-derived. TEETH: red-first throughout; 32/32 mutants caught; 3 adversarial/review passes AFTER the suite was green found 19 further defects, all fixed -- 14 from a 2-agent adversarial pass (worst: _verdict([]) returned REPRODUCIBLE, so a checkout that does not exist was stamped reproducible; and decision-vs-observed had no provenance guard, so pointing one at the SAME dict passed), 4+1 from a 5-agent /code-review, 1 from session A (main(argv=None) read sys.argv, so a programmatic call inherited the parent driver's flags -- A's test monkeypatched main and stayed green while the real command was broken). feat-013 STAYS in-progress. src/tests/test_reproducibility.py 37 -> 89 collected; ./init.sh green at 504.
```
