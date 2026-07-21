# Session Handoff

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
- **Goal spec:** `next_goal_after_gate_collapse.txt` is the only one; five spent specs were deleted in
  `de9ddf9` and are recoverable from history.
- **Harness:** `AGENTS.md` (routing + invariants) → `docs/agent-lessons.md` (the long form).
- **Per-feature notes:** `docs/feat00{5,6,8,13}-*.md`, `docs/h1-optimization-notes.md`.

## Next Session Startup

1. Read this file's top section, then `next_goal_after_gate_collapse.txt`.
2. `./init.sh` — expect **514 passed**, exit 0.
3. Answer the regulariser question (A-D) before touching feat-008/011/012.
4. Then the docs sweep: README, the report and the walkthrough still carry "no graph variant reliably
   beats no-graph" as a *finding* rather than as a confounded measurement. Large; deserves its own session.
5. Before reporting any process state — runs, monitors, whether something finished — RUN THE COMMAND.
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
