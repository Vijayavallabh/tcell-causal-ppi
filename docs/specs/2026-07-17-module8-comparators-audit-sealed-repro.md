# Module 8 — External Comparators + Rationale Audit + Sealed Eval + Reproducibility (feat-010 + feat-012 + feat-013) (design + as-built)

Date: 2026-07-17 · Depends on: feat-004 (typed PPI graph), feat-006 (`BaseBaseline` contract + common output
schema), feat-008 (`EGIPGModel`, `RationaleHead`, `FaithfulnessTester`), feat-009 (metrics + `systema`),
feat-011 (screening harness + experiment registry). Consumers: the H1 confirmatory decision (sealed eval),
the final reproducibility gate.

Design source: `perturbation_informed_causal_protein_program_graphs_report.md` §Comparators / §protocol
(identical target IDs, fold-local basis, split hashes, ≤2 close comparator families ×16 trials), §Module 4
audit (necessity/sufficiency/minimality/stability vs matched-random + structural-OOD + source ablation +
GInX), §Phase 5 sealed challenge eval (LCB rule, 10 000 bootstrap), §reproducibility (clean-checkout hash
reproduction + 11/11 fallacy scan). Walkthrough §10.7 (hypothesis rule).

## Purpose

Everything downstream of screening: (a) **external comparators** the H1 model must beat, adapted to our data
format / splits / output schema, with a compatibility + license report; (b) the **rationale audit** that
stress-tests the frozen predictive rationale; (c) the **sealed challenge evaluation** that opens the
sequestered split once and applies the confirmatory H1 rule; (d) the **reproducibility verifier** that
re-derives the deterministic hashes, confirms the same decision, and runs the 11/11 statistical-fallacy scan.

Like Module 7, this is a **verification / comparator framework**, not a compute campaign — the real
reproduction run and the 32-trial comparator sweep are compute the harness enables. All logic is exercised on
synthetic fixtures.

**Status (2026-07-18): the gating dependency is satisfied — the frozen H1 now exists.** All four
campaigns here need a graph model trained to convergence; the compute ceiling that blocked it was lifted
(3.94 h → **0.36 h**/epoch; `docs/specs/2026-07-17-graph-throughput-minibatch.md`), and **feat-011's
full-fold screening ran on 2026-07-18 and froze the H1**: the pre-registered `condition_gated`
(`data/results/screening/condition_gated/0/ckpt/stage_a_best.pt`, `promoted.json`; a NEGATIVE result —
the graph did not beat no-graph, so this is the pre-committed H1 kept honestly at rank 3/4, see
`docs/specs/2026-07-17-feat011-full-fold-campaign.md`). So feat-010 (16-trial comparators), feat-012
(50-case audit on that frozen H1) and feat-013 (clean-checkout reproduction + the sealed opening) are now
simply *unrun work* — the model they consume is on disk. Two carry-overs before feat-012 runs:
`run_module8_real.py`'s `run_audit` still prints a STALE "the graph model cannot converge until the
mini-batch refactor lands" (false — the refactor landed and the checkpoint above is a trained graph
model); and a graph-mutating ablation control that edits via `tensor.data` must call
`invalidate_graph_caches(graph)` (the subgraph cache/index invalidate automatically only on reassignment
or a normal in-place edit). The **sealed challenge split (5,608 rows) remains UNOPENED** — write-once,
test-steward-only, against the PROMOTED model; opening it early burns the fold, the exact garden-of-forks
this module exists to prevent.

## A) External comparators (feat-010) — `comparators/`

Both are `BaseBaseline` subclasses with the graph-baseline contract (`fit(genes, z)` → `predict(genes)` →
`(delta_z, delta_x=dz@B.T)`, plus a `from_hetero_graph(..., string_only=True)` classmethod), so they drop into
the same scorer path as `NetworkPropagationBaseline`. `source_adjacency(graph, sources=("string",))` is the
shared STRING-channel adjacency builder (filters the PP relations by the source one-hot, weights by score,
symmetrises).

| Comparator | Mechanism | Public-only |
|---|---|---|
| `StableShiftAdapter` | REIMPLEMENTED (published code unconfirmed). Fold-local low-rank SVD subspace from TRAIN responses only, then predict a held-out target's shift by one STRING graph-conv (presence-weighted neighbour mean of the reduced coords) decoded through the low-rank basis. | STRING topology + train-only basis |
| `TxPertPublicAdapter` | PUBLIC-ONLY (never the proprietary TxPert graph/checkpoints). Sparse graph transformer reduced to a single STRING score-attention head: a held-out target attends over its covered neighbours' training responses, weight ∝ softmax(edge confidence). Records whether `valence-labs/TxPert` is importable; predict stays the public reimpl. | STRING PPI (GO extension deferred) |

Each carries `LICENSE` / `EXPOSURE_CLASS` / `CHECKPOINT` class attributes and its own registry `family`;
`compatibility_report.py` writes `<COMPARATORS_ROOT>/<family>/compatibility_report.yaml` (license, exposure
class, checkpoint, public-only + wrapped-upstream flags). They register as two distinct comparator families
(within `MAX_COMPARATOR_FAMILIES=2`, `MAX_COMPARATOR_TRIALS=16` each).

ponytail: single graph-conv hop / fixed SVD rank (Stable-Shift), single deterministic attention head + STRING
only (TxPert-public) — the documented upgrade paths are more hops / learned propagation / multi-head
transformer over STRING+GO or wrapping the upstream public checkpoint.

## B) Rationale audit (feat-012) — `rationale/rationale_audit.py`

`audit_rationale(model, head, dataset, n_cases=50)` runs on the **frozen** H1 model + fitted rationale head
(no training), over a stratified case set (degree × effect-size × condition × graph-coverage, round-robin so
the subset spans buckets). Per covered case it reuses the Module-4 machinery unchanged — `RationaleHead` to
extract S, `FaithfulnessTester` for the fixed-model sufficiency/necessity + structural-OOD, and
`MatchedRandomSampler` for size+relation-matched controls — and adds:

- **minimality curve** — sufficiency as the top-ranked edges are added back; the scalar is the smallest prefix
  fraction of |S| that recovers most of the rationale's reconstruction.
- **source ablation** — Δ prediction (‖dz(without source) − dz_full‖) when BioPlex / HuRI / STRING / CORUM
  edges are removed (a keep-mask that zeroes each source's PP edges, scored through `FaithfulnessTester`).
- **GInX-by-sparsity** — at several keep fractions, keeping the top-importance edges vs the same number of
  random edges (stochastic edge masks at evaluated sparsities).
- **stability** — Jaccard of the selected-edge set across train-mode (DropEdge-on) re-encodes.

Aggregates: fraction of cases more sufficient / more necessary than random, mean minimality, mean stability,
per-source mean Δ, GInX curves. Written to `<RATIONALE_AUDIT_ROOT>/audit_report.json`. Requires a graph model
(an expression-only model has no rationale → `ValueError`).

## C) Sealed challenge evaluation (Phase 5) — `evaluation/sealed_eval.py`

`SealedEvaluator(challenge_ds, train_mean).evaluate(model, baseline_predictions, split="challenge")` forwards
the frozen model over the sequestered fold (`collect_predictions`), scores EG-IPG + the supplied baseline
predictions on the primary endpoint, and applies the confirmatory rule:

> **H1 confirmed ⇔ LCB₉₅( ρ_EGIPG − ρ_best_baseline ) > `DELTA_PRED` (0.05) AND ρ_EGIPG > ρ_perturbed_mean**

ρ is the per-row systema correlation (the per-row terms `systema_pert_specific_delta` macro-averages, reused
so the sealed score can't drift from the headline metric). The LCB is the 2.5th percentile of a
**`N_BOOTSTRAP`=10 000 paired-row bootstrap** of the per-row EG-IPG − best-baseline difference (best baseline
chosen by point estimate; memory-bounded blocks). `baseline_predictions` must include `perturbed_mean` (the
reference the second clause checks) and be aligned to the challenge fold in dataset order. The result is
**write-once**: `<SEALED_ROOT>/<split>/<seed>.json` refuses to overwrite (the split is opened once) unless
`force=True`.

## D) Reproducibility verification (feat-013) — `reproducibility/`

`verify_reproducibility(checkout, manifest)` re-derives each item the frozen `manifest` published against a
clean `checkout` and returns one verdict:

- **deterministic hashes** (id_mapping / splits / de_layers) — recomputed sha256 must match (critical);
- **prediction schema + row counts** and **config hash** and **checkpoint provenance** (schema/provenance);
- **confirmatory decision** — the reproduction's `observed.decision` must match the frozen `decision` (same
  H1 call + numeric fields within `tolerance`) (critical);
- **11/11 fallacy scan** — `fallacy_scan.run_fallacy_scan` runs eleven independent detectors (Simpson,
  ecological, Berkson, collider, base-rate, regression-to-mean, survivorship, look-elsewhere, garden-of-forks,
  correlation≠causation, reverse-causation); a flagged fallacy is critical.

Verdict tree: any critical **fail** → `NOT_REPRODUCIBLE`; else any critical **missing** →
`CANNOT_VERIFY`; else any non-critical issue → `PARTIALLY_REPRODUCIBLE`; else `REPRODUCIBLE`. Written to
`<REPRODUCIBILITY_ROOT>/reproducibility_report.json`. It performs no training — the "rerun the final model +
comparators over frozen seeds" step is the sealed evaluator, whose decision the manifest carries under
`observed`.

## Config additions

`COMPARATORS_ROOT`, `RATIONALE_AUDIT_ROOT`, `SEALED_ROOT`, `REPRODUCIBILITY_ROOT` (all env-overridable, under
`DATA_DIR/results/`), `DELTA_PRED=0.05`, `N_BOOTSTRAP=10000`, `N_RATIONALE_AUDIT_CASES=50`.

## Verification

`./init.sh` green at **224 tests** (171 prior + 53 Module 8: 7 comparators, 5 rationale-audit, 10 sealed-eval,
32 reproducibility). Fully synthetic — tiny marts + a small STRING-typed PPI graph; deterministic echo model
for the sealed H1 decision; crafted fallacy inputs + a synthetic checkout for the repro verdicts. One test
runs a real CUDA audit (skipped when no GPU is present).

## As-built consequences of the three review passes

Behaviours a caller must know, all forced by review findings:

- **The sealed seal is keyed on the FOLD (`fold_fingerprint`), not on the seed or the split label.** Both
  weaker keys were shown to be walkable: `seed` only redraws the bootstrap, and a `split` label is
  caller-supplied so any alias (`"Challenge"`, `"challenge_rerun"`, `"a/../challenge"`) re-opened the same
  fold. The fingerprint is a sha256 of the fold's `row_index`; any prior sealed result anywhere under
  `sealed_root` carrying it blocks a second evaluation, backed by an atomic `O_EXCL` claim. Split labels must
  be a single path component. *Residual by construction:* pointing `sealed_root` at a fresh directory starts a
  registry with no memory — a filesystem control cannot bind a determined operator; the protocol assigns the
  test-steward role.
- **Undefined statistics raise, they are never encoded as sentinels.** `fallacy_scan._corr` raises
  `Unevaluable` rather than returning `0.0` for "too few pairs"/"constant series" — a caller cannot tell that
  sentinel from a real zero, which is how detectors came to flag studies with no fallacy present.
- **`verify._verdict` is whitelist-shaped**: a critical check certifies only on an explicit `pass`.
- **A manifest cannot widen its own bar**: `decision.tolerance` is capped at `MAX_DECISION_TOLERANCE` (0.01).
- **`verify_reproducibility` always returns a verdict**, never raises on a malformed manifest.
- **The H1 rule's second clause is structurally weak.** `ρ_perturbed_mean` is exactly 0.0 under systema, so
  `ρ_EGIPG > ρ_perturbed_mean` reduces to `ρ_EGIPG > 0`. Kept (spec-mandated) and stated in the sealed JSON's
  `perturbed_mean_reference_note`; the binding constraint is the LCB clause.
- **The verifier refuses to certify what it did not check.** Absolute/escaping manifest paths, an absent
  `hashes` block, a decision record without `h1_confirmed` or numeric fields, and a missing `config_snapshot`
  all yield CANNOT_VERIFY rather than REPRODUCIBLE. Manifest paths MUST be relative to the checkout, and
  `verify_reproducibility` MUST be given a `config_snapshot`.
- **Decision tolerance** defaults to `DEFAULT_DECISION_TOLERANCE = 1e-6` (bit-exact floats are not a
  realistic cross-machine bar); a manifest may pin its own `decision.tolerance`.
- **Degenerate detector input raises `Unevaluable`**, dropping 11/11 coverage → PARTIALLY. A detector that
  could not run never certifies.
- **`wrapped_upstream` reflects what actually runs** (always False — the public reimpl), with importability
  recorded separately as `upstream_importable`.

## Review history

Adversarial workflow review (2026-07-17): 5 finder dimensions (sealed-math / comparators / rationale-audit /
reproducibility / contract-test) → 13 candidates → per-finding skeptical verify → **6 confirmed + 3
plausible, all fixed** (see `docs/reviews/2026-07-17-code-review-module8.md`):

1. **[high] fallacy scan** — a detector that *raised* was counted toward "11/11 coverage" (`evaluated: True,
   flagged: None`), so a crashed check silently certified `REPRODUCIBLE`. Fixed: an errored detector is
   `evaluated: False` + listed in `errored`, so `complete` is False → the verifier reports `incomplete` →
   `PARTIALLY_REPRODUCIBLE`, never a silent clean pass.
2. **[medium] rationale audit** — CORUM source-ablation ignored the (100%-CORUM) `complex_membership` edges,
   understating the CORUM delta. Fixed: `_source_keep_mask` ablates PP **and** membership edges.
3. **[medium] rationale audit** — uncovered targets competed for and then burned audit-case slots. Fixed:
   uncovered targets are filtered out **before** stratified selection (reported as `n_uncovered_in_dataset`).
4. **[medium] fallacy scan** — the ecological detector flagged spuriously with ≤2 groups (a 2-point aggregate
   correlation is degenerately ±1). Fixed: requires ≥3 groups, else does not flag.
5. **[low] sealed eval** — no minimum-row guard: an empty/1-row challenge fold would seal a NaN / zero-width
   CI write-once. Fixed: `min_rows` guard raises before anything is sealed.
6. **[low] compatibility report** — `public_only` used a substring match (`"public" in "non-public…"` is
   True). Fixed: explicit `PUBLIC_ONLY` class flag.

Each fix has a regression test (every fallacy detector's flag path now exercised; errored-scan → PARTIALLY;
CORUM-ablation reaches membership; ≥3-group ecological; degenerate-fold refusal; non-constant bootstrap LCB;
explicit public-only). The sealed-math finder confirmed the per-row systema / paired bootstrap / H1 rule are
correct; the comparators finder confirmed no train/challenge leakage.

**Second pass — xhigh workflow `/code-review` of the committed module (2026-07-17, 63 agents):** **15
confirmed, all fixed** across 4 tiers — see `docs/reviews/2026-07-17-code-review-module8.md`. The theme: each
subsystem failed toward its own headline claim. The verifier certified REPRODUCIBLE on an **empty checkout**
(absolute manifest paths hashed the original run), on a manifest with **no hashes block**, on a decision
record pinning **nothing** (`None == None`), and with the config check **skipped by default**; the sealed
**write-once seal was keyed on the bootstrap `seed`**, so bumping it re-opened the sequestered fold — the
garden-of-forks this module ships a detector for; four fallacy detectors fired on clean data or passed on
undefined input; the rationale audit crashed on any non-CPU device and its `stability` was not reproducible
from the audit seed; and the TxPert provenance report asserted `wrapped_upstream` from mere importability.
Fixing the device bug surfaced a **latent Module-4 bug** (`RationaleHead._select` indexes CPU tensors with a
CUDA `topk` index — the head had never run on GPU), also fixed. +15 regression tests, red-green verified.

**Third pass — adversarial verification OF those fixes (2026-07-17, 17 agents):** re-attacking each pass-2
fix found **2 still exploitable and 9 partial** — the fixes were point patches that satisfied their own
regression tests. The seal had merely moved from `seed` to the (equally caller-controlled) `split` label; the
decision check never touched the `bool()` coercion that was the defect; and the real root cause — `_corr`
returning a `0.0` sentinel for three distinct degeneracies — was untouched, so `berkson` still false-flagged
on a constant series. Pass 3 fixed the causes: a fold-keyed atomic seal, strict bool/number comparison,
`_corr` raising `Unevaluable`, scale-invariant `regression_to_mean`, a floor on the *stronger* cross-lagged
direction, input validation across all 11 detectors, a whitelist verdict, a tolerance cap, and a conditional
H1 note. See `docs/reviews/2026-07-17-code-review-module8.md`. +9 regression tests, red-green verified.
