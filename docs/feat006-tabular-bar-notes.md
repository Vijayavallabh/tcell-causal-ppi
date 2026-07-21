# feat-006 — the tabular comparator family, finished

**Date:** 2026-07-21 · **Session A** · **Status: CODE COMPLETE, uncommitted.** `./init.sh` green at 512.

The brief was to *break* H1's one surviving positive claim — "beats the strongest eligible comparator" —
not to defend it. It broke.

## The result

H1's margin over the strongest tabular bar, at each stage of making that bar honest:

| stage | strongest bar | H1 − bar | clears the 0.01 noise band? |
|---|---|---|---|
| as published | `elastic_net` 0.0342 (under-fit, node-only features) | **+0.0492** | yes, 4.9× |
| converged elastic-net | `elastic_net` 0.0343 | +0.0491 | yes |
| feature parity | `elastic_net_qpre` 0.0694 | **+0.0140** | yes, 1.4× |
| fairly-fit CatBoost | `catboost` 0.0783 (val-blind depth) | **+0.0051** | **no — half the band** |
| TabICL | **`tabicl_qpre` 0.08339** | **+0.0000147** | **no — 1/680th of the band** |

**H1 does not beat the strongest tabular comparator.** 0.08341 vs 0.08339 is a dead tie, and the under-fit
gate flags it an UPPER BOUND (`catboost_qpre` hit its cap; `tabicl_qpre` reports `converged: None`).

**The graph premise is untouched, as at every stage.** No-graph `expression_only` (0.0861) clears TabICL by
**+0.0027** — 180× more than H1's +0.0000147.

## What actually moved the number

Not model class. **Bar fit quality**, three times:

1. **Convergence: −0.0001.** `selection="random"` converges in 115 iterations against a 20,000 cap. Score
   0.0342 → 0.0343, sparsity 0.06376 → 0.06375. The prior session's "under-fit ⇒ upper bound" caveat was
   methodologically right and numerically **tight** — 6.4% non-zero was the true optimum at α=0.1, not a
   truncation artifact. Retired with evidence rather than argument.
2. **Feature parity: −0.0351.** The published bar saw *only* the target's static graph node feature — a
   function of the target alone (7,079 distinct rows over 21,262), so it could not distinguish Rest from
   Stim48hr **at all**, while H1's encoder gets the condition. Adding the q_pre covariates H1 already
   consumes (condition one-hot, 32 donor PCs, PPI degrees, control baseline expression, guide count; 41
   columns, routed through the repo's own `feature_availability.classify_columns` fence) nearly doubled it.
3. **Boosting depth: −0.0089.** CatBoost's own early stopping chose **40× too many trees**, because it
   holds out a *random* slice of train and one target spans many rows — so its holdout shares targets with
   its fit and keeps rewarding depth long after blocked-target-OOD generalisation decays. Its "converged"
   4000-tree fit scored 0.0553, *worse* than an arbitrary 1000-tree cut (0.0657). Depth chosen val-blind on
   a **target-grouped** holdout (721 disjoint targets, overlap verified 0) gives 100 trees → **0.0783**.

## A defect in the primary endpoint, found incidentally

`systema` is a Pearson correlation and therefore **scale-invariant**, so a predictor that has collapsed onto
the training mean is scored on the *direction* of its floating-point residue — at full strength, whether
that residue is 1e-5 or 1e-12 (verified identical across twelve orders of magnitude). `perturbed_mean` was
publishing **+0.0129** for pure numerical dust: above the noise band and above three genuine bars. The
existing guard only caught *bit-exact* zero, which float arithmetic never produces.

Fixed in `metrics.py` and its independent reference `metrics_ref.py`, red-first. **Blast radius measured,
not argued:** of every saved prediction, exactly two rows are numerically degenerate — both the same
`perturbed_mean` baseline. Re-scoring every frozen config under the new metric reproduces `promoted.json`
to **3.5e-09** (float32 parquet round-trip). The frozen H1, the 5-seed campaign and every margin are
provably unchanged.

This matters beyond one baseline: collapse-to-the-mean is the expected failure mode in a near-null-signal
regime, and the metric was rewarding it.

## Dependencies

`catboost` (Apache-2.0, ~97 MB, native `MultiRMSE`) and `tabicl` (BSD-3, ~114 MB checkpoint) added, both
approved. **TabPFN declined**, and the reason is governance, not feasibility: its current weights are
NON-COMMERCIAL and its own documentation says competitive benchmarking needs a commercial licence — which
is exactly this use. It is also single-target (128 refits) and needs a PriorLabs login token.

## Where I was wrong

**I called TabICL "not decision-relevant" and floated killing the run.** I inferred that from 3 of 128
probe outputs, reasoning that predictions "shrunk to std 1.49 against truth std 25.7" meant a weak bar. But
`systema` is scale-invariant — magnitude is precisely what it ignores. I had *proved* that property hours
earlier in the collapse fix and then failed to apply it to my own prediction. TabICL became the decisive
bar. Cost of the error: nearly discarding the result that finished off H1's last claim.

Also: I quoted session D's evidence-block length (4,651) at session B without re-measuring (B's was 7,665),
and I claimed session E's probe covered a reproducer after reading its printed output rather than its
return dict. Both were caught by the session the claim was about.

## Guards added, each with a firing input

- **Under-fit gate** (`flag_underfit_bars`): any bar that cannot demonstrate convergence makes the margin an
  UPPER BOUND. Verified in **both** directions — exit code 1 on a constructed truncated fit, exit 0 when
  every bar converged. `converged: None` counts against, never as a pass.
- **CatBoost convergence criterion**: `used < max_iter` read True at 999/1000; early stopping needs a full
  `od_wait` window, so the real test is `used + od_wait <= max_iter`. Verified empirically against catboost
  1.2.10, both branches.
- **Leakage fence** (`check_qpre`): refuses any q_post column, and refuses *unclassified* ones too, since
  `metadata` is that classifier's permissive fall-through and not evidence of safety.
- **Bar cache**: keyed on fold shape **plus** the source of the metric, the specific bar's class, and the
  feature construction. The last two were holes I found by probing my own guard — a whole-module hash would
  have discarded TabICL's 4.4 GPU-hour score to fix CatBoost's diagnostic, and a shape-only key would have
  served pre-change scores after an imputation swap.
- **Capped-run artifact stem**: a `--n-max` smoke run overwrote the published full-fold artifact mid-session.
  Capped runs now write to their own stem.

## Remaining for feat-006 done

Nothing in code. The evidence block below is ready for the triad merge.

---

## Appendix — `feature_list.json` evidence block (append verbatim)

Continues immediately after the existing tail (`...+2 tests. ./init.sh green at 314.`). Pure ASCII, `--`
for dashes, so appending cannot perturb the file's existing `\uXXXX` escapes.

```text
 UPDATE (2026-07-21, the tabular family finished -- and H1's comparator clause did not survive it): the three items deferred above are closed. (1) CONVERGED ELASTIC-NET: selection="random" reaches the same optimum in 115 iterations against a 20,000 cap (the shipped cyclic config had stopped at its 2,000 cap). systema 0.0342 -> 0.0343, nonzero_coef_frac 0.06376 -> 0.06375 -- so the prior "under-fit bar => the margin is an UPPER BOUND" caveat was methodologically correct and numerically TIGHT: 6.4% support is the true solution at alpha=0.1, not a truncation artifact. Retired with evidence instead of argument. (2) GRADIENT BOOSTING + CATBOOST: GradientBoostingBaseline (sklearn HistGradientBoostingRegressor, no new dependency) and CatBoostBaseline (loss_function="MultiRMSE", the only bar here that fits all K programs in ONE model, sharing tree structure across them). (3) TABICL (BSD-3) added; TABPFN DECLINED on GOVERNANCE not feasibility -- its current weights are NON-COMMERCIAL and its own docs state that competitive benchmarking requires a commercial licence, which is exactly this use; it is also single-target (128 refits) and gated behind a PriorLabs login token. A FOURTH change mattered more than any of them: the published bar consumed ONLY the target's static graph node feature -- a function of the TARGET alone (7,079 distinct rows across 21,262) -- so it could not distinguish Rest from Stim8hr from Stim48hr AT ALL, while H1's perturbation encoder gets the condition. The bars now also receive the q_pre covariates H1 itself consumes (culture_condition one-hot, 32 donor PCs, 3 PPI degrees, control_baseline_expr, n_guides, single_guide_estimate = 41 columns), every one routed through feature_availability.classify_columns, which refuses q_post AND refuses unclassified columns (metadata is that classifier's permissive fall-through, not evidence of safety). RESULT on the same frozen fold (21,262/4,400, blocked_target_ood) through the same response_metric_suite: tabicl_qpre 0.0834 > elastic_net_qpre 0.0694 > catboost_qpre 0.0657 > gradient_boosting_qpre 0.0411 > ridge_qpre 0.0396 > elastic_net 0.0343 > catboost 0.0262 > low_rank_qpre 0.0257 > gradient_boosting 0.0215 > ridge 0.0206 > zero 0.0197 > low_rank 0.0169 > nearest_neighbor_qpre 0.0117 > nearest_neighbor 0.0042 > perturbed_mean 0.0000. THE MARGIN: frozen H1 condition_gated 0.08340653 vs tabicl_qpre 0.08339185 = +0.0000147, i.e. 1/680th of the 0.01 noise band -- H1 DOES NOT BEAT the strongest tabular comparator, and the under-fit gate flags the margin an UPPER BOUND because catboost_qpre hit its iteration cap and tabicl_qpre reports converged=None (used ~14x outside its documented column range, so its score may UNDER-represent its family). Margin history: +0.0492 (published) -> +0.0491 (convergence) -> +0.0140 (feature parity) -> +0.0051 (fairly-fit CatBoost) -> +0.0000147 (TabICL). Every step came from fitting the BAR more honestly, none from a better model class. HONEST READ, unchanged at every stage: this is not about the graph -- the no-graph expression_only (0.0861) clears tabicl_qpre by +0.0027, 180x more than H1 does. CatBoost also exposed a systematic defect in both boosting bars: their early stopping holds out a RANDOM slice of train, but one target gene spans many rows, so the holdout shares targets with the rows being fitted and keeps rewarding depth long after blocked-target-OOD generalisation has decayed -- its "converged" 4000-tree fit scored 0.0553, WORSE than an arbitrary 1000-tree cut (0.0657), and depth chosen val-blind on a TARGET-GROUPED holdout (721 disjoint targets, overlap verified 0) gives 100 trees -> 0.0783. gradient_boosting_qpre is understated for the same reason. A DEFECT IN THE PRIMARY ENDPOINT was found incidentally and fixed in metrics.py + the independent metrics_ref.py: systema is a Pearson correlation and therefore SCALE-INVARIANT, so a predictor that has collapsed onto the training mean is scored on the DIRECTION of its floating-point residue at full strength -- verified identical across eps from 1e-12 to 1e-5, twelve orders of magnitude -- and perturbed_mean was publishing +0.0129 for numerical dust (above the noise band and above three genuine bars) because the existing guard caught only BIT-EXACT zero, which float arithmetic never produces. Blast radius MEASURED, not argued: of every saved prediction exactly two rows are numerically degenerate, both the same perturbed_mean baseline (H1 sits at 0.883 x ||train_mean||, expression_only 0.787, every comparator order-1), and re-scoring every frozen config under the new metric reproduces promoted.json to 3.5e-09 (float32 parquet round-trip) -- the frozen H1, the 5-seed campaign and every margin are provably unchanged. This matters beyond one baseline: collapse-to-the-mean is the expected failure mode in a near-null-signal regime and the metric was rewarding it. GUARDS, each with a verified firing input: flag_underfit_bars makes the margin an UPPER BOUND whenever any bar cannot demonstrate convergence, honored to the PROCESS EXIT CODE and verified in BOTH directions (exit 1 on a constructed truncated fit, exit 0 when every bar converged), with converged=None counting against and never as a pass; CatBoost's convergence test corrected from used < max_iter (which read True at 999/1000) to used + od_wait <= max_iter, since overfitting detection needs a full patience window, verified empirically against catboost 1.2.10 on both branches; a per-bar result cache keyed on fold shape PLUS the metric source, the specific bar's class source and the feature-construction source -- the last two were holes found by probing the guard rather than by it firing, since a whole-module hash would have discarded TabICL's 4.4 GPU-hour score in order to fix CatBoost's diagnostic, and a shape-only key would have served pre-change scores after an imputation swap; and capped --n-max runs now write to their own artifact stem after a 300-row smoke run overwrote the published full-fold table mid-session. AN ERROR OF MINE, recorded because it nearly cost the result: I called TabICL "not decision-relevant" and proposed killing the run, inferring from 3 of 128 probe outputs that predictions "shrunk to std 1.49 against truth std 25.7" meant a weak bar -- but systema is scale-invariant and magnitude is exactly what it ignores, a property I had PROVED hours earlier in the collapse fix and then failed to apply to my own prediction. TabICL became the decisive bar. feature_coverage persisted in the artifact (385 train / 91 val rows off-graph carry an all-zero node-feature vector). These remain feat-006 BASELINES: kind="baseline", no feat-010 comparator-family cap consumed, not registered. +21 tests (repo total 512 with four sessions' work co-resident); every test pinning a correctness claim was watched FAILING first, and the collapse fix, the convergence criterion and the leakage fence each have a CONSTRUCTED breaking input. promoted.json unchanged; the sealed split untouched. Artifacts: data/results/comparators/tabular_baselines_{val.parquet,vs_h1.json}, plus _probe_catboost_{budget,curve,grouped}.json for the depth study. feat-006 is CODE COMPLETE.
```
