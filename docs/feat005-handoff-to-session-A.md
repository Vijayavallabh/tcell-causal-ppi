# feat-005 → session A: merge instructions

From session C, 2026-07-21. feat-005's remaining work (method × K study + shallow VAE) is **complete**.
I did not touch `feature_list.json`, `progress.md`, or `session-handoff.md` — three-way read-modify-write
on one JSON silently loses updates, so everything you need to merge is below or in
`docs/feat005-basis-study-notes.md`.

`./init.sh` green at **514 passed** as of 2026-07-21 17:0x. Nothing committed.

---

## 1. Commit these paths, and only these

```
git commit -- \
  src/tcell_pipeline/programs/basis_study.py \
  src/tcell_pipeline/programs/run_basis_study.py \
  src/tcell_pipeline/programs/program_basis.py \
  src/tests/test_basis_study.py \
  docs/feat005-basis-study-notes.md \
  docs/feat005-handoff-to-session-A.md
```

Three of those are new. **`program_basis.py` is the only pre-existing file I modified** (+24/−2) — see §4
before you stage it. Results live under `data/results/basis_study/` and are gitignored.

## 2. Evidence block — APPEND to feat-005, do not edit what is there

Evidence is append-only, so this must extend the existing string, leaving the current text a strict
prefix. It contains a `CORRECTION:` for a number in the existing block (§3).

> UPDATE (2026-07-21, session C — method × K study + shallow VAE, feat-005 DONE): ran the §6.5
> comparison over {sparse_pca, nmf, fastica, svd} × K ∈ {64,128,256,512} plus a shallow linear VAE at
> K=128 — 17 cells, all complete on reconstruction / sparsity / stability, on TRAIN rows only (21262 ×
> 10282), every candidate fit IN MEMORY with nothing written to data/intermediate (frozen basis sha256
> verified byte-identical at session start, mid-session and end against an independently recorded
> fingerprint; `find data/intermediate -newermt` returns nothing; run_program_basis never invoked).
> Harness validated by reproducing the recorded frozen cell to 4 dp BEFORE trusting any other cell:
> recon MAE 0.686475 vs 0.817409 zero-baseline (16.02% explained), 22.694% exact-zero loadings, 0 dead
> programs. Stability protocol pre-stated before any cell ran: 3 resamples × 80% of train rows without
> replacement, seed 20260720+r, factorisation seed held FIXED, components matched by Hungarian
> assignment on |cosine| (a basis is identified only up to permutation and sign; dead components score
> 0 and are counted, an all-dead pair is None/UNDECIDABLE not 0.0). Per-fit convergence recorded via a
> new optional `info` channel on fit_program_basis (`_factor` silences ConvergenceWarning, so a capped
> fit was previously unobservable). RESULT — NEGATIVE, the frozen choice is unremarkable and that is
> the successful outcome: at K=128, the only fully-decidable column (sparse_pca converged 2/100, svd
> uncapped, nmf 257/500, fastica 115/500), held-out explained fraction is fastica 15.59% / svd 15.55% /
> FROZEN sparse_pca 15.41% / vae 14.11% / nmf 13.11% — the frozen basis is 0.18 pp off the best (a
> consistent gap: paired by resample 0.00145/0.00147/0.00151) and is the only method combining that
> accuracy with sparsity — 22.7% exact zeros against 0.0% for svd/fastica/vae; NMF yields more zeros
> still (55.7%) but pays 2.3 pp of held-out (13.11%) and is decidably unsuited to a signed target.
> Raising K buys reconstruction and pays in
> reproducibility: sparse_pca stability collapses monotonically 0.841 → 0.612 → 0.350 → 0.225 across
> K=64..512 while held-out gains only ~1.5–1.8 pp per doubling. Shallow VAE is NOT competitive — worst
> dense reconstruction at K=128 (14.11% held-out) and least stable basis in the study (0.258);
> `converged` reported None since a fixed epoch budget is not a convergence criterion. NMF is decidably
> unsuited here: it CONVERGED at K=128 (257/500) and still explains −1.43% on the common centred target
> and −2.91% on its own positive-part target — its non-negative encoding cannot represent a signed
> target (its subspace is fine: +13.11% held-out via LS projection). 5 of 16 method×K cells hit
> max_iter=100 (nmf K=64/256/512, fastica K=256/512) and are reported capped and NOT RANKED
> (undecidable ≠ worse). Multiplicity: 16
> candidate-vs-frozen paired contrasts over 21262 rows, both Bonferroni AND Holm reported — but all 16
> p_raw UNDERFLOW to exactly 0.0, so every contrast "survives" both corrections including one worth
> −0.14 pp; the correction is reported as required and is explicitly NOT load-bearing, conclusions rest
> on the effect-size column. Known limitation: the paired test is on in-sample reconstruction (monotone
> in K by construction), while the informative held-out axis has only 3 resamples and is reported
> descriptively. Tests: 45 in src/tests/test_basis_study.py, each watched failing first; mutation
> testing 9/10 caught with the 1 survivor documented as an equivalent mutant. Full evidence, tables and
> per-cell convergence: docs/feat005-basis-study-notes.md; artifacts data/results/basis_study/.
> CORRECTION (2026-07-21): this entry's earlier parenthetical "sparse_pca trades reconstruction for
> sparsity vs svd ~0.61" does not reproduce. Measured at K=128, svd is 0.6876 on the common centred
> target and 0.6851 on its own raw target — within ±0.003 of sparse_pca's 0.6865, not 0.08 better. The
> same harness reproduces the sparse_pca cell exactly, so the harness is not the suspect; "~0.61"
> appears to have been an uncomputed comparison.

**Status:** feat-005's own evidence said what remained was "the method x K comparison metrics +
shallow-VAE basis" — both are delivered, so `in-progress` → **`done`** is justified. Qualify it with the
two limitations above (5 capped cells, N_RESAMPLE=3) rather than claiming an unqualified sweep.

## 3. Two things I could not do — please carry them

1. **The `CORRECTION:` above is mandatory, not optional.** feat-005's existing evidence currently
   asserts a number that does not reproduce. Per AGENTS.md it is superseded by a dated append, never by
   editing the original line.
2. **Your `session-handoff.md` is stale about me.** It records *"IN FLIGHT: feat-005 (session C) —
   `sparse_pca` K=512 retry under a 4 h cap"*. That completed: the fit took 96.3 min (it had needed 6
   min more than the original 90 min cap), and stability/held-out were backfilled in a further 4 h 31 m.
   All 17 cells are complete.

## 4. The one shared-code change — read before staging

`program_basis.py` gained an optional `info: dict | None = None` kwarg on `fit_program_basis`, threaded
into `_factor`, plus a `_record()` helper. **Purely additive**: `info` defaults to `None`, `_record`
returns immediately when it is `None`, and `run_program_basis` passes nothing — so the production
sparse_pca/K=128 path is numerically unchanged by construction. Needed because `_factor` deliberately
silences `ConvergenceWarning`, making a capped fit invisible to any caller, and the study had to record
convergence per fit. Full suite green (514) with it.

## 5. Do NOT do these

- **Do not swap the frozen basis** on the strength of this table. fastica K=128 edges the frozen cell by
  **0.18 pp** of held-out explained fraction while giving up *all* 22.7% of its sparsity, and higher-K
  cells win on reconstruction only by spending stability (down to 0.225 at K=512). Changing the basis is
  the user's call, explicitly not this study's mandate, and every result in the repo — frozen H1, the
  5-seed campaign, `promoted.json`, your baselines — is expressed in the current basis's coordinates.
- **Do not read the K=512 column as a method comparison.** Two of its four cells are capped. §9.1
  (K=128) is the fair one.
- **Do not quote `fit_seconds` as a benchmark.** Cells ran under wildly different box load (fastica
  K=64: 564 s under load ~57, 226 s in-sweep). Provenance only.
- **Do not treat NMF's −1.43% as "NMF is bad at reconstruction."** That is its non-negative *encoder*
  against a signed target; its *subspace* scores +13.11% held-out.

## 6. Open work, if anyone picks it up

Recorded in §10 of the notes: more resamples so the held-out axis can be tested rather than described;
a decidable NMF/fastica row away from K=128 (their convergence budget was spent at K=128 by design).
