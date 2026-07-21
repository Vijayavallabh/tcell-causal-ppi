# Session Progress Log

## ⚠️ CORRECTION (2026-07-20) — two claims below were OVERSTATED; the numbers stand, the conclusions change

An xhigh workflow `/code-review` of `f1a00dd` returned 15 verified findings, two of them claim-level. The
paired-t math was **verified correct** (every "the statistic is wrong" candidate was refuted) and **no
number below changes** — but two published *conclusions* did not follow from those numbers. The section
below is left intact (append-only); this block supersedes it.

**(1) The headline H1-vs-no-graph claim was never actually tested.** "The frozen H1 `condition_gated`
(0.0838) still sits *below* no-graph `expression_only` (0.0857)" was read off two marginal per-config
means — that pair was **not in `CONTRASTS`**. Run as a proper paired contrast:

| contrast | mean Δ | 95% CI | p |
|---|---:|---|---:|
| `condition_gated − expression_only` | **−0.0019** | [−0.0042, **+0.0004**] | **0.0847** |

The CI **crosses zero**. The honest statement is that the frozen H1 is at **statistical parity** with
no-graph on this fold: it does **not** beat no-graph, and it cannot be said to sit *below* it either.
`h1_vs_no_graph` is now a first-class entry in `CONTRASTS`, so this claim can never again be read off
marginal means.

**(2) No multiplicity control.** The contrasts were each tested at raw alpha=0.05. With family-wise
correction over the (now four) simultaneous contrasts:

| contrast | raw p | Bonferroni | Holm | survives FWER |
|---|---:|---:|---:|---|
| h2a `typed_static − expression_only` | 0.0036 | 0.0142 | 0.0142 | **yes** |
| h2b `condition_gated − typed_static` | 0.0092 | 0.0369 | 0.0277 | **yes** |
| promotion `untyped_gnn − expression_only` | 0.0208 | **0.0832** | 0.0416 | **NO** |
| h1_vs_no_graph | 0.0847 | 0.3389 | 0.0847 | **NO** |

`survives_family_wise` requires **both** methods (the conservative call, so the correction method cannot
be chosen after seeing which one rescues a claim). The claim "**the only graph variant that reliably
beats no-graph is the plain untyped GNN (+0.0045)**" is therefore **RETRACTED**: that margin is nominally
positive but **not robust to multiplicity**.

**CORRECTED BOTTOM LINE.** After multiplicity control, **no graph variant reliably beats no-graph**. The
typed variant is reliably **worse** (h2a, survives correction). The frozen H1 is at **parity**. H2b
(survives) only means gating *repairs* the damage typing did. The core negative for the EG-IPG premise
**stands, and is cleaner than what was first published** — the overstatement was in claiming more
resolution than the data supports, in both directions.

`robustness_5seed.{json,md}` were regenerated and now carry raw + Bonferroni + Holm p,
`survives_family_wise`, the `h1_vs_no_graph` contrast, fold-size evidence, and a non-zero exit path.
13 further guard defects were fixed (see the feat-011 evidence block). `./init.sh` green at 314.

## ✅ DONE: the 5-seed robustness campaign (finished 2026-07-20 04:22 IST, ~30.9 h) — the negative is RESOLVED, not a coin toss

`config.N_FINAL_SEEDS=5` was declared for exactly this and referenced nowhere but a `promotion.py` note.
Built the paired aggregation (`screening/multiseed.py`) and ran seeds 1-4 (seed 0 already on disk) over the
§10.6 family on the **same frozen fold** (`blocked_target_ood`, 21,262/4,400 — reused, never redrawn;
`--seed` reseeds init + data order only), one A100 per seed, 20-epoch budget.

**Paired t on per-seed Δsystema (n=5; coverage 20/20 cells, zero dropped / non-finite / stale):**

| contrast | mean Δ | 95% CI | p |
|---|---:|---|---:|
| H2a `typed_static − expression_only` | **−0.0131** | [−0.0190, −0.0072] | 0.0036 |
| H2b `condition_gated − typed_static` | **+0.0112** | [+0.0046, +0.0177] | 0.0092 |
| promotion `untyped_gnn − expression_only` | **+0.0045** | [+0.0011, +0.0079] | 0.0208 |

Per-config mean systema: `untyped_gnn` 0.0902 > `expression_only` 0.0857 > `condition_gated` 0.0838 >
`typed_static` 0.0726.

**All three single-seed coin tosses (−0.0075 / +0.0048 / +0.0090, each inside the 0.01 band) are now
statistically resolved.** The negative **holds and sharpens**: the frozen H1 `condition_gated` (0.0838)
still sits *below* no-graph `expression_only` (0.0857); `typed_static` is reliably *worse* than no-graph;
H2b only means gating **repairs** the damage typing did (0.0726 → 0.0838, still short of 0.0857). The only
graph variant that reliably beats no-graph is the plain **untyped** GNN (+0.0045) — the one with no typed
biology and no evidence gating.

**Convergence favours the negative:** `expression_only`/`untyped_gnn` were 5/5 **capped** at 20/20 epochs
(still improving at the budget), while `typed_static`/`condition_gated` were 0/5 capped, early-stopping at
11-13 epochs — with patience=10 their best val was **epoch 1-3**. The graph models plateaued early at a
worse optimum; the no-graph models were still climbing and still won. More epochs would widen the gap.

**Cost:** ~30.9 h wall (vs ~17.7 h est) and 102.3 `gpu_hours` over 20 lanes — but `gpu_hours` is a
*contended* wall-time proxy on the shared box (`typed_static` seed 1 = 11.88 h vs seed 4 = 5.34 h for the
SAME 11 epochs), not clean compute. `promoted.json` **unchanged**; the deliverable is the separate
`data/results/screening/robustness_5seed.{json,md}`.

## ✅ DONE: feat-006 elastic-net + the H1 tabular comparator bar (2026-07-20)

`ElasticNetBaseline` uses a per-output `MultiOutputRegressor(ElasticNet)` parallel across the K programs —
the coupled-L21 `MultiTaskElasticNet` ground for >17 min without converging on 1412 features × 128
programs, while the per-output form fits the full train fold in 65 s (alpha/tol set for *convergence*, not
the score; more regularisation only weakens the model, so it cannot flatter it vs H1).

New `run_module8_real.py --part baselines` fits every simple baseline on the train fold predicting Δz from
the target's **static graph node feature** (1412-d) and scores val through the same `response_metric_suite`
as feat-010, ranked vs the frozen H1 by the shared `summarize_vs_h1`. Val targets are disjoint from train,
so it is a genuine generalisation bar; no comparator-family cap is consumed.

`elastic_net` +0.0342 > `ridge` +0.0206 > `zero` +0.0197 > `low_rank` +0.0169 > `perturbed_mean` +0.0123 >
`nearest_neighbor` +0.0042 — **H1 (0.0834) beats the strongest by +0.0492**, outside the noise band. Same
honest caveat as feat-010: a *trained-predictor* win, not graph value (no-graph `expression_only` 0.0861
beats the tabular bar too). CatBoost/TabPFN remain deferred (new dependency + weight download).

## ✅ DONE: feat-010 external-comparator campaign (2026-07-18) — H1 beats the comparators, but the graph premise still fails

Scored the two public comparators on the SAME development val fold the feat-011 campaign used (21,262 train /
4,400 val, `blocked_target_ood`) through the SAME scorer path as `network_propagation`
(`run_module8_real.py --part comparators`): fit on TRAIN responses only (leakage fence), STRING-only
adjacency, `compute_all_metrics = response_metric_suite`. **Fold identity verified** — each comparator's
prediction row-index is identical to the campaign val fold (4,400 rows). Deterministic re-run reproduced the
numbers exactly; ~65 s CPU each, **0 GPU-hours**.

**Result on `systema` (primary endpoint):** frozen H1 `condition_gated` **0.0834** > `txpert_public` 0.0321 >
`stable_shift` 0.0217 (both comparators cover 4,232/4,400; 168 targets have no covered STRING neighbour →
zero shift, counted against them). **H1 BEATS the strongest eligible comparator by +0.0513** — outside the
0.01 noise band, so H1's "beyond the strongest comparator" clause HOLDS on the dev fold.

**Honest frame — this does NOT rescue the graph.** The no-graph `expression_only` (0.0861) and `untyped_gnn`
(0.0951) beat both comparators too, so the win is *trained-neural-predictor > topology-only-public-smoother*,
not *graph > no-graph* (H2a stayed −0.0075). Consistency check: `txpert_public` 0.0321 ≈ `network_propagation`
0.0319 (both STRING smoothers, same suite) confirms the comparator path equals the campaign's non-neural
reference. **Single-seed, no error bars** (the report's 5-seed promotion, `N_FINAL_SEEDS=5`, would be needed).
Cap: 2 distinct comparator configs across 2 families = 1/16 trials each, **2/2 families (at the ceiling)**; the
other 15 slots/family (a hyperparameter sweep) are unused. New: `summarize_vs_h1()` + a red-green,
adversarial-input test. Artifacts under `data/results/comparators/` (`comparators_val.parquet`,
`comparators_vs_h1.json`, per-family `compatibility_report.yaml`). feat-010 → **done**. Committed `92ad4e8`.

**xhigh `/code-review` of `92ad4e8` (2026-07-18) — 9 findings, all fixed; reported result UNCHANGED.** Every
finding was degenerate/misused-input robustness, not the campaign numbers. Fixed test-first (+5 tests watched
failing, constructed breakers per the adversarial-input gate): a **fold guard** (a `--n-max` run no longer
compares a capped fold to the full-fold H1 — verified live, the verdict is now skipped); a **None-safe
verdict print** (`_fmt_signed`, the old `None:+.4f` crashed *after* the JSON was written); `h1_beats_strongest`
is **None not False** when there's nothing to compare (≠ a loss); robust `promoted.json` loading
(`_load_promoted_final` — corrupt/partial/bare-string no longer crash or mislabel provenance); a **None-safe
ranking tie-break**; `_finite` widened to `np.floating`; a **`margin_within_noise` flag** (0.01 band,
mirroring `promotion.py`); and an import hoist. Full-fold rerun reproduces H1 0.0834 beats txpert_public
0.0321 (+0.0513, outside noise). `./init.sh` green at **286** (281 + 5). 2 candidates refuted. **Fixes not
committed yet.**

## ✅ DONE: the feat-011 full-fold screening campaign (finished 2026-07-18 03:32 IST, 8.2 h) — NEGATIVE result

`./run_screening_campaign.sh` ran the §10.6 nested family on ONE shared FULL fold (21,262 train / 4,400
val), 20-epoch budget, bs=8, three A100s. All 5 members completed; merge + promotion done. Full write-up:
`docs/specs/2026-07-17-feat011-full-fold-campaign.md` (Results section).

**The graph does not help on this fold — a clean negative, reported not tuned.** Ranking on `systema`:
untyped_gnn 0.0951 > expression_only 0.0861 > condition_gated 0.0834 > typed_static 0.0786 >
network_prop 0.0319. **H2a NOT supported** (typed_static − expr-only = −0.0075: the typed graph *hurts*);
**H2b technically supported** (+0.0048) but condition_gated is still *below* the no-graph expression_only,
so the full EG-IPG does not beat no-graph. Best model is the untyped-GCN diagnostic. All within ~0.017 of
each other on a 0.086 base; promotion margin 0.0090 flagged WITHIN NOISE (single-seed, no error bars).
Convergence asymmetry: expression_only still descending at ep20 (best@19), the typed models overfit at
ep1–2 — more budget would widen the gap *against* the graph. 17.7 GPU-hours; GPU util ~79–99% on graph
lanes (cache worked).

**H1 FROZEN (PI's call): condition_gated, the pre-registered confirmatory H1** (`--promote --pin
condition_gated`). Not the argmax winner (untyped_gnn) — the confirmatory protocol keeps the pre-committed
typed+gated model, and it's the only one feat-012's audit can run on. `promoted.json` records it honestly:
pinned_rank 3/4, screening_winner untyped_gnn, margin −0.0117 (H1 behind the winner), within_noise False.
Frozen checkpoint: `data/results/screening/condition_gated/0/ckpt/stage_a_best.pt`. feat-010/012/013 now
unblocked (each a separate campaign — not started). Campaign committed `b875dfa`.

**xhigh `/code-review` of `b875dfa` (2026-07-18) — 6 findings, all resolved + adversarially verified.** Two
confirmed bugs (promote() crashed on a non-finite metric; freshness guard dropped a completed-then-failed
lane) + four plausible (cache key omitted gene_to_idx; data_ptr ABA; hard-coded 'systema'; noise-margin
default). Fixed test-first, mutation-tested. Object-identity invalidation (`_TensorSet`, is + _version)
replaced the data_ptr stamp on BOTH the subgraph cache and CSR index. A 5-agent adversarial pass found one
real hole my tests missed — a `tensor.data` write bypasses `_version` → stale subgraph (pre-existing; a
content check would be O(edges)/call) — closed by making the contract explicit + `invalidate_graph_caches()`
escape hatch. `./init.sh` green at **280** (273 + 7). **Review fixes not committed yet.**

Three things the next reader needs:

1. **The 0.36 h/epoch figure below does NOT size a training run** — it is an encoder-only bench. The
   real `Trainer` does 1+`DONOR_INVARIANCE_SAMPLES`=3 forwards per step (`_donor_variants` re-forwards
   the whole model), measured at **183.4 ms/row = exactly 3.0×** the bench. The campaign was 22.7 h,
   not ~7 h. A **per-target subgraph cache** (`config.SUBGRAPH_CACHE_SIZE`) took it to **12.5 h**
   (1.8×; untyped_gnn 5.3× — it was almost pure sampling) at 38.6 GB RSS per graph lane. GPU util is
   now 99%/84%/43%, up from a median 46%.
2. **Early stopping will NOT fire**: patience 10 but min-delta 1e-6 against ~1e-3/epoch improvements.
   20 epochs is a BUDGET. If val is still descending at 20, the verdict is **"not converged at
   budget"** — not convergence. A negative H2a/H2b is a valid outcome; do not tune until it turns.
3. **Stale `stage_a_history.json` files** from a 15:43 run sit at the exact paths the lanes write and
   read as plausible progress. Check mtime before reporting anything from them.

## Current State

**Last Updated:** 2026-07-17 (**graph throughput refactor: 10.9× end-to-end, `./init.sh` green at 242**, xhigh `/code-review` 15 defects fixed. Prior: Module 8 `5ea8a4b` → xhigh `/code-review` 15 defects fixed `2edb44f` → pass-3 adversarial verification OF those fixes found 2 still exploitable + 9 partial, root causes fixed `6a68882` → real-data drivers `97f8451`.)
**Active Feature:** the graph mini-batch refactor is **done** — the compute ceiling that blocked feat-010/011/012/013 is lifted (3.94 h → 0.36 h per 21,262-row epoch; GPU util median 1% → 46%). The four campaigns are now **unblocked but not yet run**: feat-011 (32-trial screening + 5-seed promotion), feat-010 (16-trial comparators), feat-012 (50-case audit on the frozen H1), feat-013 (sealed opening + clean-checkout reproduction). Also open: feat-006 (elastic-net + CatBoost), feat-008 (Stage-B calibration + rationale fit loops + freeze gate), feat-005. Next: the screening campaign → a converged/promoted model → the deferred campaigns.

## Archived: per-module build logs

The Module 6/7/8 completion records and the 2026-07-17 graph-throughput refactor (the ceiling was the
SAMPLER, not the message passing) are in
[`docs/history/progress-archive-modules-6-8.md`](docs/history/progress-archive-modules-6-8.md).
Earlier still: [`docs/history/progress-archive-2026-07.md`](docs/history/progress-archive-2026-07.md).

## Status

### What's Done

- [x] feat-001 Environment & Data Download — **DONE**
  - Env imports OK (anndata 0.13.1, mudata 0.3.10, h5py 3.16.0)
  - Aggregate layer downloaded to `data/raw/` (~101 GB): 4 HDF5 + 15 suppl tables + 12 jsonld
  - Cell-level files intentionally excluded (storage-blocked)
- [x] feat-002 Data Inspection & ID Mapping — **DONE**
  - `examples/` inspectors; `src/tcell_pipeline/id_mapping.py`
  - Ran online on real DE: 12311 unique Ensembl (11526 targets / 10282 measured / 9497 both), all HGNC
    resolved, UniProt/Entrez filled via mygene.info
  - One-to-many UniProt DISAMBIGUATED (reviewed-canonical strategy, see Decisions): 33 multi-accession
    genes -> 23 resolved to a confident canonical, 10 genuine multi-product loci flagged
    `uniprot_ambiguous` with all candidates kept in `uniprot_alternatives`
- [x] feat-004 PPI Graph Construction — **DONE**
  - `ppi_graph.py` typed-edge harmonizer + `complex_membership.py` CORUM bipartite membership
  - All 5 sources fetched + merged on real data -> `data/graphs/protein_edges.parquet`:
    **7,980,907 edges** (bioplex 118162, huri 52256, biogrid 1218142, string 13715404, corum 77696)
  - `complex_membership.parquet`: 18,932 memberships / 5,628 complexes (CORUM 5.3)
  - `ppi_degree_*` computed from the graph into perturbation_condition

- [x] **Module 0 data pipeline** (`src/tcell_pipeline/`, 9 modules + `run_module0.py`) ran end-to-end on
  real data. Derived marts written under `data/intermediate/`, `data/graphs/`, `data/manifests/`
  (all gitignored): id_mapping, DE layers (zscore/log_fc sparse NPZ; neglog10_p_value/neglog10_adj_p_value/
  baseMean/lfcSE dense NPY), de_obs/de_var, protein_edges, complex_membership, perturbation_condition
  (33983 rows; 187 without UniProt; 32425/33983 with a control baseline), control baseline + donor
  profiles, feature_availability.yaml (**q_pre=43 / q_post=13 / metadata=2, leakage fence disjoint**).
- [x] **This session's fixes** (commits e453964..eab027e):
  - UniProt disambiguation via reviewed-canonical strategy (`choose_uniprot`); `uniprot_ambiguous` flags
    only equal-evidence ties (10 loci: CDKN2A p16/p14ARF, GNAS, MOCS2, TMPO...)
  - HuRI download: apex host `interactome-atlas.org` (cert invalid for the `www.` subdomain)
  - CORUM download: migrated to CORUM 5.3 fastapi endpoint (old `coreComplexes.txt.zip` path gone);
    handles new `subunits_gene_name` schema; per-source TLS-verify skip for the broken helmholtz cert chain
  - feature_availability: `KNOWN_METADATA_COLS` allowlist so the leakage-fence REVIEW warning fires only
    on genuinely-unexpected metadata (row_index/mapping_status no longer cry wolf)
  - `mygene` added to requirements.txt
- [x] 23 pytest tests in `src/tests/` (synthetic fixtures; added test_control_profiles.py,
  test_complex_membership.py); `init.sh` green (compileall + pytest)

- [x] **Module 1 Perturbation & Context Encoder** (feat-014, `src/tcell_pipeline/encoders/`) — **DONE**
  - `PluggableEmbeddingStore` (frozen PLM 1280 / PINNACLE 512 by UniProt; zero-fallback when the
    parquet is absent, in-memory cache, NOT an nn.Module), `TargetEncoder` (no trainable gene-ID
    embedding; h_target R^1796), `ContextEncoder` (trainable condition Embedding(3,64) + donor PCs
    through Linear(32,32), no free donor-ID embedding), `QualityEncoder` (n_guides +
    single_guide_estimate + zeros(64) guide-seq placeholder; h_quality R^66), `PerturbationEncoder`
    (fusion Linear(1958->256)+LayerNorm -> h_do R^256; rejects q_post cols at the boundary).
  - 503,264 trainable params, CPU-only, batch-first. Config: PLM_EMBED_DIM/PINNACLE_EMBED_DIM/
    GUIDE_SEQ_EMBED_DIM/H_DO_DIM/CONDITIONS/PLM_EMBEDDINGS_PATH/PINNACLE_EMBEDDINGS_PATH.
  - NaN guard (`as_float_vector` nan_to_num): missing control_baseline_expr (1558/33983) and
    n_guides can't poison the LayerNorm'd h_do. Upgrade path = fold-fit imputation in Module 3 loader.
  - 10 tests (test_encoders.py) + real-data smoke on perturbation_condition/de_obs -> finite (4,256).

- [x] **Module 1 real embedding ingestion** (feat-015, `embeddings_plm.py` + `embeddings_pinnacle.py`) — **DONE**
  - `embeddings_plm.py`: real **ESM-2 650M** (1280-d, mean-pooled), UniProt-REST sequences, resumable,
    **device-aware (GPU)**. Ran on an A100 -> **11419/11419 mart proteins embedded** (100% PLM coverage), finite.
  - `embeddings_pinnacle.py`: real **PINNACLE** (Figshare 22708126) `cd4-positive helper t cell` context.
    Real dim is **128** (config placeholder was 512 -> **corrected to 128**). Gene-symbol->UniProt via id_mapping;
    **1119 embeddings, 1070/11419 mart proteins covered** (contextual — rest keep zero fallback).
  - Live encoder dims now: target.out_dim **1412** (1280+128+4), fusion Linear(1574->256), **404,960** params.
  - Tests rewritten to **real data/embeddings — no synthetic parquets** (still 10 in test_encoders.py).
  - **GPU enabled**: swapped torch cu130->**cu126** (host driver is CUDA 12.2; cu13x can't see the 5x A100s).
    requirements.txt: +fair-esm, +cu126 install note.

- [x] **Module 2 Typed Graph Encoder** (feat-016, `src/tcell_pipeline/graph/`) — **DONE**
  - `graph_builder.build_hetero_graph` -> PyG HeteroData + gene_to_idx: 25440 protein nodes
    (frozen 1412-d TargetEncoder descriptor, graph-derived degrees, zero-fallback) + 5628 complex
    nodes (index-only, learned embedding in the encoder); 4 relations (physical_ppi 1123205 /
    co_complex 48389 / functional_assoc 6857702 / complex_membership 18932) with 8-d edge features.
  - `neighborhood_sampler.sample_subgraph`: physical/co-complex-first then score-fill, cap 512
    proteins + member complexes, induced HeteroData preserving orig_idx.
  - `typed_graph_encoder.TypedGraphEncoder`: 3-layer per-relation MessagePassing with signed
    message `tanh(W_sign h_u)*relu(W_mag h_u)`, condition gate `sigmoid(w_gate[h_cond||f_e])`
    computed once/relation and returned as `edge_gates` for Module 4, residual FFN+LayerNorm,
    DropEdge 0.1. `graph_readout.GraphReadout`: 4-head cross-attention (q=h_do) -> h_graph R^256.
  - CPU **and** CUDA (device-aware). 8 synthetic tests (`test_graph.py`) + real-data smoke
    (`graph/run_module2_smoke.py`): full graph in ~18s, CD3E neighbourhood, Module 1 h_do ->
    Module 2 h_graph (4,256) finite on GPU, gates differ by condition, attention sums to 1.

- [x] **feat-003 Leakage-Safe Splits** (`src/tcell_pipeline/splits.py`) — **DONE**
  - Design in `docs/specs/2026-07-15-feat-003-leakage-safe-splits.md` (from the experiment-plan
    report). **Empirical measurement forced the algorithm**: naive connected-components collapses on
    every axis (physical 1-hop → 95% giant component, complex → 23%, raw ESM cos≥0.95 → 92%,
    Louvain → 42%). Hard block = sequence/paralog family via **representative (non-chaining)
    clustering on centered ESM-2 embeddings** (cos≥0.85 → 3.1% largest family) + CORUM co-membership
    under a 5% size cap (capped union-find). Physical-PPI neighbourhood is **audit-only** (95%
    hairball; per report G1 + Phase-1 step 6/9 "publish the similarity distribution").
  - **4-role** partition (train/val/calibration/challenge, ~60/15/10/15; realized 62.5/13/7.9/16.6)
    + random diagnostic split. Frozen + hashed to `data/splits/` (git-tracked): blocked/random CSVs,
    manifest.json, leakage_report.json. **Effectiveness validated** (corrected post-review): challenge
    genes with a ≥0.85 train paralog cut 53.8% (random) → 26.4% (blocked) = 51% reduction.
  - 8 synthetic tests (`test_splits.py`); `./init.sh` green (54 pytest).
- [x] **Post code-review fixes** (feat-016 + feat-003; committed 7760624) — **DONE**
  - Applied the verified `/code-review` findings (`docs/reviews/2026-07-15-code-review-feat-016-feat-003.md`).
  - **feat-003 leakage-safety (split CSVs byte-identical, sha256 unchanged):** audit now publishes
    cap-induced family splits via an uncapped pre-cap component pass (post-cap "no split" assertion was
    blind to families the 5% cap must break: `cap_induced_family_splits=1`); sequence residual centered
    in one global frame (was mismatched per-subset means, understating leakage → 53.8/26.4/51% vs old
    53.5/28.1/47%); `run()` fails closed when PLM embeddings absent (was silent fail-open).
  - **feat-016 bugs:** graph degree columns reordered to match Module 1 `[physical, functional, complex]`;
    `encode_one` moves `h_do` to device; flaky signed-message test seeded + false `<1.0` bound dropped.
  - **Tier 3 (all addressed):** cheap defenses (edge-feature `nan_to_num`, gene-symbol `dropna`,
    unknown-source fail-fast); dead config constants removed; OOV culture_condition raises a legible
    `ValueError` (`_condition_index`); diagnostic random split uses cumulative-boundary allocation (no
    truncated tail — `random.csv` regenerated, blocked split + effectiveness numbers unchanged);
    `edge_gates[rel]` now length E (one per original edge) for all relations, not 2E-doubled for PP
    (full gate→edge identity API still deferred to Module 4).
  - Regenerated `data/splits/`; **57 pytest** green (+3 regression checks).

### What's In Progress

- **feat-005 Latent Program Extraction** — COMPLETE 2026-07-21 (session C). The 4-method × 4-K comparison
  (reconstruction / sparsity / stability) + shallow-VAE basis are delivered: 17 cells, no gaps. Nothing
  found justifies changing the frozen basis. Also retracted a non-reproducing number in feat-005's own
  evidence and in README: "sparse_pca trades reconstruction for sparsity vs svd ~0.61" — svd is within
  ±0.003 of sparse_pca at K=128, not 0.08 better.
- **feat-008 EG-IPG Model** — M1+M2+M3 decoder/EGIPGModel + Module 4 rationale head / loss / faithfulness
  eval built; the training-loss OPTIMIZATION loop + train/calibration loops remain (and feat-007 is not-started).

### What's Next

1. feat-008 training loop: wire RationaleLoss + the decoder losses into a Stage-A (predictor) then
   Stage-B (rationale) fit; then feat-011 screening consumes it.
2. feat-006 Simple Baselines / feat-007 Graph Baselines — consume the frozen splits, unblock the feat-008 comparison.
3. feat-005 method×K comparison + shallow VAE (extraction machinery done; only the study remains).
4. Optional: near-null-signal check on development data before freezing H1 (2026-07-14 finding).

## Blockers / Risks

- [ ] `data/raw` ~101 GB near the 105 GiB soft cap; derived marts now also on disk (protein_edges ~35 MB,
  DE layers, control profiles) — watch disk before feat-005 program bases land
- [ ] **Near-null-signal regime (2026-07-14 finding):** this CD4+ screen may be near-null-signal (models
  barely beat the mean). Confirm a detectable above-mean signal before freezing H1; a rigorous negative
  benchmark is a valid outcome.
- [x] RESOLVED: HuRI + CORUM downloads (both source URLs migrated this session)
- [x] RESOLVED: id_mapping UniProt/Entrez (online mygene pass done; only 6 no-hit genes remain, HGNC-resolved)

## Decisions Made

- **UniProt disambiguation**: pick the gene's reviewed human canonical (UniProt REST gene_exact+reviewed)
  by annotation-score then lexical; flag only equal-evidence ties; keep the gene as the perturbation unit
  (CRISPRi knocks down the whole locus) — no forced single-protein pick, alternatives preserved
- **CORUM host**: broken TLS chain (certifi also fails) -> per-source verify skip for `corum` only
- Data scope: aggregate layer only; cell-level (~1.6 TiB) excluded
- Donor key = physical CE codes; independent NTC controls come from pseudobulk (DE has none)
- Distributional metrics: do not use Wasserstein/Energy distance as a sole headline metric
- Stable-Shift (feat-010): first-party code unconfirmed; plan a row-compatible reimplementation
- **Embeddings (feat-015)**: PLM = real ESM-2 650M (1280-d, mean-pooled); PINNACLE = real published
  128-d contextual vectors (NOT the 512 placeholder), `cd4-positive helper t cell` context to match the
  CD4+ screen (configurable via config.PINNACLE_CONTEXT). Frozen features; artifacts gitignored + regenerable.
- **GPU**: host has 5x A100 80GB but the CUDA-12.2 driver can't run the default cu13x torch; use the
  cu126 build (`torch==2.13.0+cu126`, minor-version compat). Embedding generation runs on GPU.
- **Module 2 (feat-016)**: protein node features reuse Module 1's frozen 1412-d TargetEncoder
  descriptor (degrees recomputed from the graph so all 25440 nodes have them); complex nodes are
  index-only (learned embedding lives in the encoder). Custom PyG `MessagePassing` per relation —
  RGCNConv/GATConv can't express the signed `tanh*relu` message or the condition gate. Condition
  gate `alpha` depends only on `h_cond` + edge features (not `h_u`), so it's computed once and
  reused across layers and returned as `edge_gates`. ponytail: per-target subgraph loop in forward
  (upgrade to PyG mini-batching if Module 3 graph-encode throughput demands it).

> **Archived** — the per-feature *Files Added* lists (feat-003/014/015/016) are in [`docs/history/progress-archive-2026-07.md`](docs/history/progress-archive-2026-07.md).

## Notes for Next Session

- `examples/` scripts double as data-understanding docs
- Read the experiment plan report for detailed feature specs (2026-07-14 literature refresh)
- Before feat-011 screening / freezing H1, run the near-null-signal check
- Module 0 marts are on disk but gitignored; rerun `python src/tcell_pipeline/run_module0.py` to regenerate

## 2026-07-20/21 — five concurrent sessions

Ran A (feat-006 tabular bar), B (feat-008 Stage B), C (feat-005 basis study), D (feat-013 reproducibility),
E (H1 optimisation) in one checkout with per-session file ownership and a single integrator for the DoD
triad. Committed: `bdc1f56` (feat-013), `ac1cbcd` (feat-008), `c52f24a` (session E — the two probes, the
Stage-1 pilot, `training/inner_split.py` + tests, and `docs/h1-optimization-notes.md`; 6 new files,
1,467 insertions, nothing else touched). `./init.sh` 314 -> 514.

**The headline is E's**: the graph arms were trained with message passing driven to ~0 by `StageALoss._graph`
(unnormalised over edges, divided only by batch size). A three-arm pilot settles it — from a shared init of
0.678556, gates collapse to 1.9882e-07 at lambda=0.01, RISE to 0.897950 at lambda=0, and still collapse to
3.2190e-04 under per-edge normalisation. The graph negative is a measurement-validity defect; the repair is
a redesign, not one line. See `session-handoff.md`.

**feat-006**: H1's margin over the strongest tabular bar fell +0.0492 -> +0.0000147 (a tie), every step from
fitting the bar more honestly rather than from a better model. A scale-invariance defect in `systema` that
scored collapsed predictors on floating-point dust was fixed, blast radius measured at one baseline.

**Harness**: AGENTS.md gained four sections, all from defects found by re-derivation rather than by tests —
concurrent-session committing, claims-about-process being invisible to test discipline, instrument blind
spots (self-matching watchers, `ps` truncation, torch-vs-nvidia-smi device numbering), and cheap
preconditions (read a checkpoint's gate mean before spending hours on it).
