# feat-011 re-screen at `lambda_graph=0` — analysis plan, pre-registered

**Written 2026-07-22 00:04 IST, BEFORE any new lane produced a number.** The four lanes launched at
00:04:08 and the first epoch lands ~00:45. Everything in §1–§4 is fixed as of that timestamp; §5 is the
only section that may be written after results exist.

## 0. Why this run exists

The 5-seed screening campaign compared four arms. Its `condition_gated` arm was trained with an
unnormalised `L_graph` (a `sum` over 16k–60k edges divided only by batch size) at `lambda_graph=0.01`,
which is ~103x the response term and contributes ~100% of the edge gates' gradient direction. The gates
died inside epoch 0 in all five seeds (mean ~1.3e-07 against ~0.61 at init). That arm's message passing
was therefore switched off, and every contrast involving it measured a no-graph model with spare
parameters.

The user selected **option C** on 2026-07-21: drop the penalty for the headline family
(`lambda_graph=0`), let the prediction task learn the gates, and report the sparsity term as a separate
diagnostic. It is still computed and logged every epoch — it is simply weighted 0 in the total.

## 1. Blast radius — why only ONE arm re-runs

`L_graph` carries gradient to exactly one of the four arms. Measured by backpropagating the penalty
term ALONE and counting parameters with a non-zero gradient:

| arm | gates emitted | penalty value | `grad_fn` | params moved by the penalty alone |
|---|---|---|---|---|
| `expression_only` | 0 | 0.0000 | False | 0 / 17 |
| `typed_static` | 13 | 0.0480 | False (constant) | 0 / 165 |
| `condition_gated` | 13 | 0.0187 | True | **7 / 165** |
| `untyped_gnn` | 0 | 0.0000 | False | 0 / 29 |

`typed_static` pins its gates to 1.0 (`_gate` returns `new_ones`), so its penalty is a constant with no
`grad_fn`; `untyped_gnn` returns `edge_gates=None`; `expression_only` has no graph encoder. This is
pinned by `test_graph_penalty_carries_gradient_to_condition_gated_only`, which FAILS if any other arm's
gates ever become learnable — mutation-verified by unpinning `typed_static`'s `physical_ppi` gate
(caught, 3 params moved).

Consequence: the campaign's other 15 lanes remain valid comparators. Only `condition_gated` x 5 seeds
re-runs — 5 lanes, not 20.

## 2. What is held fixed

Same frozen fold (`blocked_target_ood`, 21,262 train / 4,400 val, loaded by name — never redrawn), same
frozen program basis, same seeds 0–4, same 20-epoch budget, same `patience=10`, same batch size 8, same
lr. **`lambda_graph` is the only variable that changes.** Fold identity is verified per lane from the
recorded `n_train`/`n_val`, not from the registry's hardcoded `split` label.

Nothing frozen is written: the run uses a separate `SCREENING_ROOT`, `PREDICTIONS_ROOT` and a COPY of
the registry. `data/results/screening/` — including `promoted.json` and the frozen H1 checkpoint it
points at — is hashed before and after (`rescreen_frozen_sha256.{before,after}.txt`) and must be
byte-identical.

## 3. The contrast family — fixed before the numbers

The confirmatory family stays the **same pre-registered four** as the published campaign, so the
multiplicity correction is comparable to the one already published:

| contrast | changes? |
|---|---|
| `h2a` = typed_static − expression_only | **no** — neither arm re-runs |
| `promotion_margin` = untyped_gnn − expression_only | **no** — neither arm re-runs |
| `h2b` = condition_gated − typed_static | yes |
| `h1_vs_no_graph` = condition_gated − expression_only | yes |

Both Bonferroni and Holm are reported, and `survives_family_wise` requires BOTH — the correction method
cannot be chosen after seeing which one rescues a claim.

**The gate-repair contrast** — `condition_gated@lambda=0` − `condition_gated@lambda=0.01`, paired by
seed — is reported as a **labelled diagnostic, not a member of the confirmatory family**, because it
measures what an instrument defect cost, not a hypothesis about the architecture. To keep that choice
from being a way of dodging correction, its p is reported BOTH raw and corrected under a family of five.
Both framings are published regardless of which is more flattering.

## 4. Control, kill criteria and known asymmetries

**Control (already run, 2026-07-22 00:1x).** The merged-root aggregation path was exercised on the OLD
lanes before the new ones landed, and reproduces the published campaign exactly: h2a −0.0131
CI[−0.0190,−0.0072] p=0.0036 bonf 0.0142; h2b +0.0112 p=0.0092; promotion_margin +0.0045 p=0.0208
bonf 0.0832 holm 0.0416 fwer=False; h1_vs_no_graph −0.0019 CI[−0.0042,+0.0004] p=0.0847. Any change in
the new report is therefore attributable to `lambda_graph`, not to the analysis path.

**Kill criteria, stated in advance:**
- If the gates COLLAPSE at `lambda_graph=0` (final `gate_mean` <= 1e-3), the Phase-1 diagnosis is wrong.
  Report that and stop — do not tune.
- If any lane crashes, `n` shrinks and the aggregator names the dropped seed. A 4-seed result is
  reported as a 4-seed result, never silently.
- If any lane's `n_train`/`n_val` is not 21,262 / 4,400, the fold moved and the contrast is not formable.

**Known asymmetry, recorded now so it cannot be presented as a finding later.** In the original
campaign the no-graph arms ran the full 20 epochs while the graph arms early-stopped at 11–13. With live
gates the re-screened arm may now use more of its budget. That is a legitimate consequence of the same
fixed budget + patience, not a confound — but the epochs-run column must be published alongside, because
`epochs_run` is 98.8% explained by the arm label and a reader will otherwise mistake it for a result.

**This is a development-fold result.** The sealed challenge split (5,608 rows) stays sequestered.

## 4b. Selection step (2026-07-22) — the architecture the campaign confirms

Before this campaign relaunched, a 14-cell architecture search on the SAME inner holdout selected the
model it tests. Full record: `docs/feat011-arch-search-notes.md`. Verdict: `condition_gated` (typed,
gated, unnormalised aggregation, full graph) with `lambda_graph=0` is the winner; every architectural
lever tried (per-relation normalisation, edge-confidence pruning, GCN edge-weights, GATv2 attention)
made it worse or neutral. So this campaign tests the SELECTED architecture, and the selection was on
train's inner holdout — val stayed closed. The winner leads on the inner holdout (n=1) by
condition_gated − expression_only = +0.0051 and − untyped_gnn = +0.0040; the campaign is what makes
that a confirmed, multiplicity-controlled statement (or refutes it — either is honest).

## 4c. Aggregation verified-ready (2026-07-22)

The post-campaign aggregation was exercised END TO END before the lanes landed: the REAL
`multiseed.py --seeds 0,1,2,3,4` was run on a synthetic root built exactly as
`run_rescreen_lambda0.sh` seeds it (the 3 valid reference arms + a copied registry + 5 fresh
condition_gated lanes carrying `n_train=21262/n_val=4400`). It produced all four contrasts with
`fold_comparable=True` and both corrections. The reference parquets lack `n_train/n_val` (they predate
that column), but condition_gated's sizes plus the registry split label give `single_fold=True` — the
same path the control reproduction used. So when the real lanes land, aggregation is one command and
will not structurally fail. (The synthetic condition_gated values were fabricated to test plumbing and
are NOT results.)

## 5. Results — n=5, official, 2026-07-23

All five `condition_gated@lambda=0` lanes landed on the frozen fold (21,262/4,400); the launcher exited
clean, `multiseed.py --seeds 0,1,2,3,4` returned exit 0, `single_frozen_fold=True`. Report:
`data/results/screening_lambda0/robustness_5seed.{json,md}`. Per-seed systema: 0.08631 / 0.08755 /
0.07555 / 0.08962 / 0.08482 (seed 2 low, not an outlier by Grubbs' eye; seed 4 central).

Per-config systema (mean, 95% CI):
- untyped_gnn      0.0902 [0.0865, 0.0939]
- expression_only  0.0857 [0.0850, 0.0863]
- condition_gated  0.0848 [0.0780, 0.0915]
- typed_static     0.0726 [0.0665, 0.0787]

Pre-registered contrasts (paired, both corrections; `survives_family_wise` requires BOTH):

| contrast | Δ | 95% CI | p_raw | Bonf | Holm | FWER |
|---|---|---|---|---|---|---|
| **h1_vs_no_graph** (condition_gated − expression_only) | **−0.0009** | [−0.0072, +0.0054] | 0.7091 | 1.0000 | 0.7091 | **no — parity** |
| h2a (typed_static − expression_only) | −0.0131 | [−0.0190, −0.0072] | 0.0036 | 0.0142 | 0.0142 | **yes — reliably worse** |
| h2b (condition_gated − typed_static) | +0.0122 | [+0.0028, +0.0215] | 0.0226 | 0.0905 | 0.0624 | no |
| promotion_margin (untyped_gnn − expression_only) | +0.0045 | [+0.0011, +0.0079] | 0.0208 | 0.0832 | 0.0624 | no |

**Headline (VALID this time — live gates, mean gate ~0.57–0.77 across seeds):** the evidence-gated typed
graph is at statistical **parity** with no-graph (h1_vs_no_graph −0.0009, CI crosses zero, p=0.71). No
graph variant reliably beats no-graph after multiplicity control; typed_static is reliably *worse*.

**The negative is robust, not an artifact.** The corrected n=5 numbers nearly REPRODUCE the confounded
campaign (which had dead gates): h1 −0.0009 vs the confounded −0.0019; per-config untyped 0.0902 (same),
expr 0.0857 (same), condition_gated 0.0848 vs 0.0838, typed_static 0.0726 (same); h2a/promotion_margin
identical. So repairing the gate-annihilating regulariser — a 4.5e6× swing in gate magnitude — moved the
headline by ~0.001 and changed no conclusion. The graph negative holds whether or not the gates function.

This CLOSES feat-011's confirmatory comparison: a valid, multiplicity-controlled graph-vs-no-graph test
now exists on the development fold. The inner-holdout selection lead (+0.0051, n=1) did NOT replicate.
`promoted.json` stays frozen; this is the SEPARATE `robustness_5seed` deliverable. Sealed split untouched.
