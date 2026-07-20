# feat-011 — full-fold screening campaign: subgraph cache, concurrent lanes, promotion (2026-07-17)

Runs the §10.6 nested confirmatory family to a 20-epoch budget on ONE shared FULL fold (21,262 train /
4,400 val) and reports H2a/H2b on `systema_pert_specific_delta`, replacing the capped-fold numbers
(H2a +0.0010 / H2b −0.0062) that were noise at 1 epoch on 1,000 rows.

**Status: launched 2026-07-17 19:20 IST, ~12.5 h.** Results + the honest verdict land in the Results
section below; until then this file documents the method and the infrastructure it needed.

## The estimate the campaign was almost launched on was wrong

The Module 7 RESOLVED note offered **0.36 h/epoch**. That number is an *encoder-only* benchmark
(`bench.py`: forward+backward through the graph encoder). The real `Trainer` step is 3× larger, and
the campaign is 4 lanes × 20 epochs, so launching on it would have missed by ~2.5×.

Measured on the real fold (A100, bs=8), `condition_gated` is **183.4 ms/row**, and
183.4 / 61 = **3.0** — exactly `1 + DONOR_INVARIANCE_SAMPLES`. `Trainer._donor_variants` re-forwards
the WHOLE model once per donor variant, and the graph encoder samples + message-passes on every one.
`trainer.py` had carried the marker since Module 5:

> `ponytail: each variant re-runs the (donor-independent) graph message passing; cache node states +
> re-run only readout+decoder per donor if Stage A throughput becomes graph-bound.`

Stage A is now graph-bound, so the deferred note had become load-bearing.

## What was cached — and what deliberately was not

The obvious read of that marker is "cache node states across the variants". **That is wrong**, and the
reason matters: `_donor_variants` calls `self.model.eval()`, so the variants run with **DropEdge off**
while the main forward runs with it **on**. Their node states are not the same tensor and reusing one
for the other would leak a DropEdge mask into the donor-invariance signal. Node states are also a
function of the model WEIGHTS, so any cache of them that outlives a single optimiser step returns
states from stale weights.

What *is* safely reusable is the **sampled subgraph**: `sample_subgraph` is a pure function of
`(graph, gene, hops, cap)` — independent of the donor, the weights, and train/eval mode. It is
re-derived 3× per row per step and ~3× again per epoch (7,079 unique in-graph targets over 21,262
train rows) and every epoch after. This is the per-target cache the throughput spec already named as
the next ceiling.

`_SubgraphCache` (bounded LRU, `config.SUBGRAPH_CACHE_SIZE`, cached on the graph like the CSR index):

| condition_gated, real fold, bs=8 | train ms/row | val ms/row |
| --- | --- | --- |
| no cache | 183.4 | 42.2 |
| cache = batch only | 127.5 | 23.5 |
| **cache ≥ fold (9000)** | **102.8** | **13.0** |

| lane | train ms/row before → after | h/epoch | 20-epoch h |
| --- | --- | --- | --- |
| expression_only | 2.7 → 2.6 | 0.018 | 0.35 |
| untyped_gnn | 95.7 → **18.1** (5.3×) | 0.114 | 2.3 |
| typed_static | 158.3 → **80.3** | 0.490 | 9.8 |
| condition_gated | 183.4 → **102.8** | 0.623 | **12.5** |

**1.8× end-to-end: 22.7 h → 12.5 h wall clock** fanned over three A100s (~25 GPU-hours). `untyped_gnn`
gains 5.3× because its message passing is cheap — it was almost pure sampling. Cost: **38.6 GB RSS per
graph lane** (8,541 unique targets — 7,079 train + 1,462 val, disjoint by construction — at ~4.5 MB
each), ~116 GB for the three lanes against 440 GB free. Measured at 38.1 GB warm, against a 38.6 GB
prediction.

**Not done, deliberately:** the readout/decoder still re-run per donor variant (they are cheap and
genuinely donor-dependent), and the sampler is still row-by-row. GPU utilisation went from a median
46% to **99% / 84% / 43%** across the lanes, so sampling is no longer the floor.

### Two traps the cache had to be built around

- **`HeteroData.to(device)` mutates in place and returns the same object** (verified, not assumed).
  `encode_subgraph` does `sub = sub.to(device)` on whatever the sampler hands it, so serving the cached
  object itself would migrate a ~38 GB host cache onto the GPU. Hits are `clone()`d — ~0.5 ms against
  the ~28 ms saved.
- **A subgraph depends on strictly more than the CSR index does.** `_fingerprint` stamps only
  `edge_index`, but a sampled subgraph embeds `graph[PROTEIN].x[sel]` and `edge_attr[kept]`. Editing
  either leaves every `edge_index` untouched, so nothing would force a rebuild and the cache would
  serve stale features forever. Hence `_content_fingerprint` (topology + `x` + `edge_attr` + complex
  count) — the same staleness hazard the review already caught on the index, one level out.

## The sampler's thread cliff (found in passing, and it is a live trap)

`sample_subgraph` on the real graph, by `torch.get_num_threads()`:

| threads | ms/target |
| --- | --- |
| 1 | 28.3 |
| 8 | **19.9** |
| **64 (this box's default)** | **470.7** |

**17× slower at the default.** `run_screening` and `test_screening` already call
`torch.set_num_threads(1)`, which is the only reason the measured numbers are sane — but any caller
that forgets inherits a 470 ms/target sampler and will conclude the sampler is hopeless. This also
reconciles the spec's 22 ms/row with a naive 505 ms/target reading. The driver's `1` is left as-is
(known-safe, and the cache removes most sampling anyway); 8 is on the table if sampling ever matters
again.

## Infrastructure the campaign needed

- **Registry advisory lock** (`experiment_registry._locked`). Its own marker said to add locking
  "before running the four real GPU lanes concurrently against one manifest" — this campaign is that.
  Unlocked it is worse than lossy: `write_text_atomic` stages through a **fixed** `<name>.tmp`, so
  concurrent lanes collide on that path and the loser's rename raises `FileNotFoundError` — the
  campaign would have crashed partway. The whole read-check-append-write is under the lock (the cap is
  only a ceiling if two lanes cannot both read `cap - 1` and both append). Lock is on a sidecar file:
  `_save` renames the manifest, so its inode changes under any lock held on it.
- **`--only NAME`** — one config per process, so the wave fans one lane per GPU (12.5 h vs ~25 h
  sequential). A lane writes its row + predictions + registry entry but **not** `summary.json`: a
  lane's summary would claim the whole wave.
- **`--merge`** — recombines the lanes' rows into `summary.json` + H2a/H2b once they all land. A lane
  that never landed is recorded `status: missing` and drops out of the contrast; it exits non-zero.
  Silently omitting it would leave a summary that reads like full coverage of a short wave.
- **Stale-parquet guard.** `--merge` and `--promote` read `<root>/<name>/<seed>.parquet`, and the tree
  accumulates parquets across runs — confirmed live: `condition_gated`/`typed_static` parquets were
  from 15:44/15:49 (a prior run) while this campaign's lanes were still training, so a *manual* early
  `--merge` would report last run's numbers as this result. Both now take the registry and force a
  config whose latest run isn't `completed` to `missing`/skip (log_run flips status in place, so the
  newest run per config is the truth). The guard only demotes configs the registry TRACKS — the
  never-registered `network_propagation` reference falls back to parquet presence, or the guard would
  silently drop it. The campaign script itself is safe by ordering (it merges only after every `wait`),
  so this protects the human/next-session path.
- **`--promote`** (`screening/promotion.py`) — nothing existed to name the frozen H1, yet feat-010,
  feat-012 and feat-013 all consume "the promoted model" (`N_FINAL_SEEDS` was declared in config and
  referenced nowhere). Ranks on `systema` — NOT `best_val`, which is not comparable across the family
  at all: the graph members' totals carry regulariser terms expression_only has no equivalent of
  (measured this run: typed_static val 490.4 vs expression_only 3.47). Refuses to promote a row with
  no checkpoint on disk, excludes the non-neural `network_propagation` reference, breaks exact ties by
  name (reachable, not hypothetical), and flags a first-vs-second margin inside `--noise-margin` as a
  coin toss rather than a win. It is the single-seed screening promotion, **not** the report's
  five-seed promotion — that remains a separate campaign, and `promoted.json` says so in `basis`.

## Epoch budget and the early-stopping rule

**20 epochs, and the budget IS the wall clock.** `Trainer` early-stops on val total with `patience=10`,
but its min-delta is **1e-6** against improvements of ~1e-3/epoch, so patience never increments and it
will not fire. Measured again on this very run: untyped_gnn val moved −0.0047 over 5 epochs.

There was no convergence evidence to size the budget from — every prior graph run was 1 epoch, and M5
expression-only was still descending when it stopped at epoch 3 (3.4726 → 3.4697). So **if val is
still descending at epoch 20 the honest verdict is "not converged at budget", not convergence.**

## Verification

`./init.sh` green at **268** (252 + 16). Every new test was watched failing, and the diff was
mutation-tested — which caught **two tests passing for the wrong reason**:

- the edges-invalidation test appended to `edge_index` *and* `edge_attr`, so the `edge_attr` shape
  change masked a missing topology stamp: dropping topology from the fingerprint left it green. Fixed
  by adding an in-place **rewire** test (which is what an edge-ablation control actually does).
- the cache-isolation test mutated the miss path's return value, which is the object the cache stored
  a clone OF — so it was never the cached entry and dropping `.clone()` on hit stayed green. Fixed by
  mutating a **hit**.

That is the same lesson the throughput review paid for: a test can pass because it is pointed at the
one thing that cannot fail. Mutations now caught: 10/10 (6 cache, 2 harness, 4 promotion).

Fan-out smoke on real data before launch: 4 concurrent lanes + merge, all exit 0, registry holds 4 runs
with 4 distinct ids and zero loss under genuine contention.

The sealed challenge split (5,608 rows) was **not** touched and remains unopened; every lane scores
`val`.

## Results (2026-07-18, campaign finished 03:32 IST — 8.2 h wall clock)

All 5 members completed on the shared full fold (21,262 train / 4,400 val). `summary.json` +
`experiment_registry.yaml` written; every parquet verified fresh (post-launch mtime). **This is a
negative result for the EG-IPG's central premise, reported as one — not tuned.**

| rank on `systema` | model | systema | pearson | mae | rmse |
| --- | --- | --- | --- | --- | --- |
| 1 | **untyped_gnn** (homogeneous GCN) | **0.0951** | 0.1191 | 0.8138 | 1.0335 |
| 2 | expression_only (no graph) | 0.0861 | 0.1153 | 0.8147 | 1.0342 |
| 3 | condition_gated (full EG-IPG) | 0.0834 | 0.1127 | 0.8155 | 1.0361 |
| 4 | typed_static | 0.0786 | 0.1107 | 0.8309 | 1.0541 |
| — | network_propagation (reference) | 0.0319 | 0.0880 | 0.8174 | 1.0369 |

- **H2a (typed_static > expression_only): NOT supported, Δsystema = −0.0075.** Adding the typed PPI
  graph *lowers* systema below the expression-only model. The graph structure does not help on this
  fold; it hurts.
- **H2b (condition_gated > typed_static): supported, Δsystema = +0.0048.** Condition gating recovers
  part of what typing lost — but read it in context: condition_gated (0.0834) is **still below**
  expression_only (0.0861), so the full typed+gated EG-IPG does **not** beat the no-graph baseline. H2b
  is a within-family recovery, not evidence the graph pathway earns its place.
- **The best screening score is the untyped diagnostic** (a plain GCN over all edges collapsed to one
  type). The report anticipated this "untyped-graph diagnostic" as the outcome to watch for, and it is
  what happened: whatever signal the PPI graph carries is captured better by an untyped GCN than by the
  typed, provenance-aware, condition-gated architecture the project is built around.

**Magnitudes / confidence.** All neural members sit within ~0.017 systema of each other on a ~0.086
base; the promotion margin (untyped_gnn − expression_only = 0.0090) is **inside the 0.01 noise band**
and was flagged a coin toss. This is single-seed screening, so there are no formal error bars — the
report's 5-seed promotion would be needed for those. The direction, not the decimals, is the result.

**Convergence — an asymmetry that matters.** expression_only ran the full 20 epochs and was **still
descending** (best val @ep19): its 0.0861 is a NOT-converged model that would likely climb with more
budget. The two typed graph models **overfit almost immediately** (typed_static best @ep2 then
early-stopped @13; condition_gated best @ep1, early-stopped @12) — their best checkpoints are barely
trained. untyped_gnn plateaued (best @ep13 of 20). So the comparison is between a still-improving
no-graph model and graph models that stop learning at epoch 1–2. If anything, more budget widens the
gap *against* the graph.

**Compute.** 8.2 h wall clock over three A100s (early stopping ended the typed lanes before 20 epochs).
Registry gpu_hours: condition_gated 8.17, typed_static 6.89, untyped_gnn 2.31, expression_only 0.359
(network_propagation CPU, unregistered) → **17.7 GPU-hours**. GPU utilisation ran 79–99% on the graph
lanes for the bulk of the run (median 46% → ~90% after the subgraph cache), confirming the cache moved
the bottleneck off CPU sampling. Peak ~51 GB on condition_gated's A100.

**Promotion (`promoted.json`).** The *mechanical* argmax pick (rank all trainable members on `systema`,
exclude the non-neural reference) was FINAL = untyped_gnn, margin +0.0090 **within noise** — but
untyped_gnn has no typed edges or condition gate, so it **cannot support the feat-012 rationale audit**
and is not the confirmatory H1 the project defines. That made the choice a PI decision, not a
mechanical one.

**Decision taken (2026-07-18): freeze `condition_gated`, the pre-registered confirmatory H1** (via
`--promote --pin condition_gated`). The confirmatory protocol commits to the typed+gated model as H1
before seeing the fold; keeping it — rather than swapping in the argmax winner after the fact — is what
makes the negative result honest, and it is the only choice under which feat-012's audit can run.
`promoted.json` records this without dressing it up: `final = condition_gated`, `pinned_rank = 3/4`,
`screening_winner = untyped_gnn`, `runner_up = untyped_gnn`, `margin = −0.0117` (the frozen H1 is
**behind** the model that won screening), `margin_within_noise = False`. So the frozen H1 is the
pre-committed model, and the record states plainly that it lost to an untyped GCN and to the no-graph
baseline. Nothing here is tuned; the negative result stands and is carried forward intact.

The `--pin` path is pinned by tests (freeze-over-winner, reject-a-config-that-didn't-complete) and the
pin logic mutation-tested 3/3. `run_module8_real.py`'s `run_audit` still carries a stale line — "the
graph model cannot converge until the mini-batch refactor lands" — that is now false (the refactor
landed; `condition_gated/0/ckpt/stage_a_best.pt` is a trained graph checkpoint); it should be corrected
when feat-012 is picked up.

## xhigh `/code-review` of the campaign commit (2026-07-18, 19 agents) — 6 findings, all resolved

A workflow-backed review of `b875dfa` (finders per correctness angle + a cleanup sweep, each candidate
independently verified) surfaced 6 findings — 2 CONFIRMED, 4 PLAUSIBLE. All were reproduced before
fixing and each fix was watched failing then mutation-tested (277 → 278; 6 new regression tests, 6
mutations caught). The two confirmed bugs both compromised the campaign's final model selection.

- **`promote()` crashed on a non-finite primary metric (CONFIRMED).** `json.dumps(..., allow_nan=False)`
  raised on a NaN/Inf `systema` — reachable in the near-null-signal regime (a zero-variance correlation),
  and it would block promotion after a full campaign. Fixed: a non-finite primary metric is filtered from
  the ranking (a NaN-scoring model cannot be the H1) and the dump is sanitized through `_finite_or_none`,
  mirroring the merge/summary path.
- **Freshness guard discarded a completed-then-failed lane (CONFIRMED).** The guard keyed on "latest run
  ≠ completed", so a config that completed (good parquet on disk) and was later re-run and failed was
  marked `missing` — dropping it from the H2a/H2b contrast and from promotion, freezing a worse H1. The
  registry distinguishes `registered` (reserved, never executed → any parquet is foreign/stale) from
  `failed` (executed, so an earlier `completed` parquet is still this config's last good result). New
  rule (`_config_statuses` + `_is_stale`): fresh iff `completed` appears AND the latest run is not a bare
  `registered`. Pinned by BOTH the finding's regression test and its dual (a `registered`-latest with a
  prior `completed` must stay excluded — the original stale-parquet case).
- **Subgraph cache key omitted `gene_to_idx` (PLAUSIBLE, self-flagged).** The subgraph is a pure function
  of `(seed, graph, hops, cap)`; the cache now keys on the RESOLVED seed index rather than the gene
  string, so two mappings sending one gene to different seeds cannot collide on one cached subgraph. This
  removes the footgun outright rather than documenting a precondition.
- **`data_ptr` ABA in the cache/index fingerprint (PLAUSIBLE).** `(data_ptr, shape, _version)` can
  collide if a freed tensor's address is reused by a new same-shape tensor. Closed STRUCTURALLY, not just
  documented: `_TensorSet` holds the actual tensor OBJECTS and compares by `is` + `_version`. A live
  reference keeps the old object (and its address) from being freed, so a replacement is always a distinct
  object at a distinct address — `is` cannot false-match. Applied to both the subgraph cache and the CSR
  index for consistency; the equivalence tests confirm sampling stays bit-identical.
- **`promote_final` hard-coded `'systema'` (cleanup).** Now prints via the `PRIMARY_METRIC` constant.
- **`--noise-margin` default 0.0 (PLAUSIBLE footgun).** Kept the default (0.0 = "report the raw margin",
  an explicit choice) but the driver now prints a loud note when no noise band is set, so a rounding-size
  gap is not misread as a decisive result. The campaign always passes `--noise-margin 0.01`.

**Adversarial verification (5 agents, each trying to BREAK one fix with a runnable counterexample).**
Four fixes survived — F1 stood up to 8 constructed non-finite fixtures, F0 to every reachable
`register_run`/`log_run` status sequence, F4 to seed-collision attacks, F5/F3 to 10 driver scenarios. The
fifth adversary (the object-identity anti-ABA fix) found a **real hole my own tests missed**: a write
routed through `tensor.data` (`t.data.add_(...)`, `t.data[mask] = 0`) changes contents WITHOUT bumping
`_version`, so neither `is` nor the version counter sees it and a stale subgraph is served. This is
*pre-existing* — the old `data_ptr` stamp missed `.data` writes identically — and a collision-free
content check would be O(edges) on the 6.9M-edge tables *per sample call*, defeating the cache. Resolved
by making the contract explicit rather than over-claiming: the `_TensorSet` docstring now states the
limit precisely, and `invalidate_graph_caches(graph)` is the escape hatch a `.data`-editing control (e.g.
an edge-ablation using `.data[mask] = 0`) must call. Pinned by a test that shows the stale read without
it and a fresh read with it. This is the honest ceiling of O(1)-per-call invalidation; the automatic path
still catches every reassignment and every normal in-place edit.

The frozen H1 is unchanged by all of this: `--promote --pin condition_gated` still reproduces
condition_gated at rank 3/4, margin −0.0117.

## Follow-on: feat-010 comparator test (2026-07-18)

With the H1 frozen, the external-comparator half of H1's definition was scored on this same dev fold: the
frozen `condition_gated` (systema 0.0834) beats the strongest eligible public comparator `txpert_public`
(0.0321) by +0.0513 — but the no-graph `expression_only` (0.0861) beats the comparators too, so it does not
rescue the graph (H2a stays negative). Full record + the code-review hardening:
`docs/specs/2026-07-17-module8-comparators-audit-sealed-repro.md` §A-results.

## Follow-on: the 5-seed robustness campaign (2026-07-20) — the coin tosses are RESOLVED

Every margin this campaign reported sat inside the 0.01 noise band (H2a −0.0075, H2b +0.0048, promotion
margin +0.0090), so the whole negative was a single-seed coin toss. `config.N_FINAL_SEEDS=5` was declared
for exactly this and was referenced nowhere but a `promotion.py` docstring note. It has now been run.

**Design.** Seeds 1–4 retrained (seed 0 already on disk) over the same §10.6 family, on the **same frozen
fold** — `blocked_target_ood`, 21,262 / 4,400, loaded by name from `BLOCKED_SPLIT_PATH`, never redrawn;
`--seed` reseeds weight init (`seeded_init`) and Trainer data order only. One A100 per seed (GPUs 0/1/2/4),
20-epoch budget, batch 8, `SUBGRAPH_CACHE_SIZE=9000`. `./run_multiseed_campaign.sh`, then
`python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3,4`.

**Statistic.** For each seed, `d_s = systema(better, s) − systema(worse, s)`; a one-sample (paired) t on
`{d_s}` against 0. Pairing removes the shared per-seed nuisance (init + data order on one fixed fold).
Because four contrasts are tested simultaneously, each also carries a family-wise correction, and
`survives_family_wise` requires **both** Bonferroni and Holm — the conservative call, so the method cannot
be chosen after seeing which one rescues a claim.

**Result** (n=5; coverage 20/20 cells, zero dropped / non-finite / stale; single frozen fold):

| contrast | mean Δsystema | 95% CI | raw p | Bonferroni | Holm | survives FWER |
|---|---:|---|---:|---:|---:|---|
| h2a `typed_static − expression_only` | **−0.0131** | [−0.0190, −0.0072] | 0.0036 | 0.0142 | 0.0142 | **yes** |
| h2b `condition_gated − typed_static` | **+0.0112** | [+0.0046, +0.0177] | 0.0092 | 0.0369 | 0.0277 | **yes** |
| promotion_margin `untyped_gnn − expression_only` | +0.0045 | [+0.0011, +0.0079] | 0.0208 | 0.0832 | 0.0416 | **no** |
| h1_vs_no_graph `condition_gated − expression_only` | −0.0019 | [−0.0042, +0.0004] | 0.0847 | 0.3389 | 0.0847 | **no** |

Per-config mean systema: `untyped_gnn` 0.0902 [0.0865, 0.0939] > `expression_only` 0.0857 [0.0850, 0.0863]
> `condition_gated` 0.0838 [0.0810, 0.0866] > `typed_static` 0.0726 [0.0665, 0.0787].

**Reading it honestly.** After multiplicity control, **no graph variant reliably beats no-graph**. H2a
survives correction, so the typed graph is reliably *worse* than no-graph — the central negative is now a
statistically resolved result rather than a coin toss. The frozen H1 is at **statistical parity** with
no-graph: `h1_vs_no_graph` crosses zero at p=0.085, so it neither beats no-graph nor can be called *below*
it. H2b survives but only means gating **repairs** the damage typing did (0.0726 → 0.0838, still short of
0.0857). The `untyped_gnn` edge is nominally positive and does **not** survive correction, so "a plain
untyped GCN reliably beats no-graph" is not supported either.

**Convergence favours the negative.** `expression_only` and `untyped_gnn` were 5/5 **capped** at 20/20
epochs — still improving when the budget ran out. `typed_static` and `condition_gated` were 0/5 capped,
early-stopping at 11–13 epochs; with patience=10 that puts their best validation at **epoch 1–3**. The graph
models plateaued almost immediately at a worse optimum while the no-graph models were still climbing and
still won, so more epochs would widen the gap against the graph. The deficit is not a budget artifact.

**Cost / provenance.** ~30.9 h wall (Jul 18 21:25 → Jul 20 04:22 IST, vs ~17.7 h estimated) and 102.3
`gpu_hours` over 20 lanes. `gpu_hours` is a *contended* wall-time proxy on the shared box — `typed_static`
seed 1 took 11.88 h against seed 4's 5.34 h for the **same** 11 epochs — so it is not clean compute.
`promoted.json` is **unchanged** (frozen H1 still `condition_gated` seed 0 @ 0.08340652613564893, basis
still single-seed); the deliverable is the separate `data/results/screening/robustness_5seed.{json,md}`.

## xhigh `/code-review` of the 5-seed commit (2026-07-20, 55 agents) — 15 findings, 2 claim-level

The paired-t math was verified **correct** (every "the statistic is wrong" candidate was refuted) and no
reported number changed. Two published *conclusions* did not follow from those numbers, and both were
corrected in `e542d8c`:

1. **The headline claim was never tested.** "The frozen H1 sits BELOW no-graph" was read off two marginal
   per-config means; that pair was not in `CONTRASTS`. Run as a paired contrast it is −0.0019, p=0.0847 —
   indistinguishable. `h1_vs_no_graph` is now a first-class contrast so the claim cannot be read off
   marginal means again. **Rule now in AGENTS.md: a comparison you did not compute is not a result.**
2. **No multiplicity control.** The three contrasts were each tested at raw alpha=0.05; the promotion
   margin (raw p=0.0208) does not survive Bonferroni and had been published as a resolved positive. It is
   retracted above.

Thirteen further guard defects were fixed, all red-first with constructed breakers. The most instructive:
the fold gate compared the registry `split` field, which `screen_config` fills from
`cfg.get("split", "blocked_target_ood")` while `nested_family_configs` never sets it — a hardcoded literal
that can only ever **confirm**, so a `--n-max` capped seed passed it. `screen_config` now records
`n_train`/`n_val` and the aggregator keys fold identity on those. Related: `splits <= {FROZEN_SPLIT}` was
vacuously true on the empty set (absence of evidence published as proof — now `None`); zero variance
reported `p=0.0, "CI excludes zero"`, turning the one condition that proves the seeds carry no information
into the strongest possible evidence (now undecidable); and `main()` returned 0 immediately after printing
`FOLD MISMATCH … NOT comparable`, so an unattended campaign or any exit-status gate recorded it green.
