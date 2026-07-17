# Module 7 — Graph Baselines + Screening Harness (feat-007 + feat-011) (design + as-built)

Date: 2026-07-16 · Depends on: feat-004 (typed PPI graph + `TypedGraphEncoder`), feat-006 (`BaseBaseline`
contract + common output schema), feat-008 (`EGIPGModel.forward`), feat-005 (frozen fold-local basis `B`),
feat-009 (evaluation metrics + G2-MQ). Consumers: feat-008 (the H1-vs-baseline comparison), feat-012
(rationale audit runs on the promoted config), feat-013 (reproducibility references read the registry).

Design source: `EG_IPG_architecture_walkthrough.md` §10.6 (nested confirmatory family) / §10.7 (hypothesis
hierarchy), `perturbation_informed_causal_protein_program_graphs_report.md` §screening (lines ~1109, 1187,
1274-1291: the 32-trial EG-IPG cap, matched budgets, the experiment registry).

## Purpose

Give the H1 predictor its **graph references** and the machinery to **screen the nested family** on a
development split under a frozen trial budget. Module 7 supplies (a) three PPI-graph baselines behind the
existing baseline / model contracts, (b) a screening harness that trains + scores the §10.6 nested family
through the existing Stage-A `Trainer` and reports the H2a/H2b contrasts on the primary endpoint, and (c) an
immutable experiment registry that enforces the report's trial caps and logs every run (including failed).

## Scope

- **In (feat-007):** `baselines/graph_baselines.py` — `NetworkPropagationBaseline` (non-neural PPI
  diffusion), `UntypedGraphEncoder` (homogeneous GCN, all edges one type), `StaticTypedGraphEncoder`
  (`TypedGraphEncoder` with the condition gate pinned to 1.0).
- **In (feat-011):** `screening/{screening,experiment_registry,run_screening}.py` — `screen_config`,
  `run_screening`, the nested-family factories, the registry, and the real-data driver.
- **Out:** external comparators (feat-010: Stable-Shift / TxPert-public adapters) and the rationale
  faithfulness audit (feat-012). The full 32-trial screening campaign + five-seed promotion is a compute
  campaign, not code — the harness is done, the campaign is feat-011's remaining work.

## The three graph baselines (feat-007)

Each isolates one variable:

| Baseline | What it uses | What it strips |
|---|---|---|
| `NetworkPropagationBaseline` | symmetric-normalised PPI diffusion `Ŵ=D^{-1/2}AD^{-1/2}`, `F ← r·S₀+(1−r)·ŴF` | no training, no evidence typing, no condition — pure topology smoothing |
| `UntypedGraphEncoder` (GCN) | homogeneous `GCNConv` over all PPI edges collapsed to one relation | evidence provenance + condition gate (report's "untyped-graph diagnostic") |
| `StaticTypedGraphEncoder` | full `TypedGraphEncoder` signed/typed message passing | the condition gate — pinned to 1.0 (§10.6 member #2, the H2b variable) |

Network propagation places each training target's mean Δz (and a presence indicator) on its protein node,
propagates both fields `n_iter` power-iteration steps, and predicts `F_signal[node]/F_presence[node]` — a
graph-proximity-weighted mean of nearby training responses, so an unseen target inherits its neighbours'
signal; an isolated or absent node returns zero. It subclasses `BaseBaseline` (reusing `B`-decoding) but
takes per-row **target symbols** in place of the opaque feature matrix. `StaticTypedGraphEncoder` reuses
**all** of `TypedGraphEncoder` and overrides **only** `_gate` → `new_ones`, so it's condition-invariant by
construction. The two neural encoders honour the `forward(target_genes, conditions, h_do) → (h_graph,
edge_gates, edge_confidences)` contract, so they drop into `EGIPGModel(graph_encoder=…)` and train through
the identical Stage-A path (`StageALoss._graph(None)` is a no-op, so the untyped GNN's absent gates don't
crash the loss).

## The screening harness (feat-011)

`screen_config(cfg, train_ds, val_ds, train_mean)` trains one config through `Trainer`, reloads its **best**
checkpoint, scores `val_ds` (predictions collected in dataset order so they align with the truths by row),
writes predictions in the common output schema + a one-row metrics table, and returns the metric suite —
the same metric spaces as `run_module6_smoke._score` (program-space Δz for pearson/systema/centroid/cosine,
gene-space Δx for mae/rmse/topk/sign; **`systema_pert_specific_delta` is the primary endpoint**). A config
supplies `{name, model_factory, n_epochs, lr, batch_size, seed}`; `nested_family_factories` builds the
family with a **fresh model per call** (fresh graph AND perturbation encoder — the review-caught weight-
sharing fix).

`run_screening(configs, …)` screens each config on the shared split and reports the two key-secondary
contrasts on `systema`: **H2a** (typed-static > expression-only) and **H2b** (condition-gated > typed-
static), writing `<screening_root>/summary.json`. It is **failure-isolating** by default: a config that
OOMs/crashes is caught, recorded as a failed result (and logged failed in the registry), and the remaining
configs still run (report §screening: four independent lanes for cleaner failure isolation).

## The experiment registry (feat-011)

`register_run(config_id, hypothesis, inputs, split, seed, budget, family="egipg")` reserves an immutable
`run-NNNN` ID in a YAML manifest and **enforces the trial caps** (report line 1187 / 1291): ≥32 registered
EG-IPG-family runs → `ValueError`; comparator families cap at 16 each. `log_run(run_id, status, metrics,
checkpoint, gpu_hours)` records the outcome — **every run is logged, including failed** — so the registry is
a complete audit trail. `load_registry` degrades a missing / empty / null-`runs:` manifest to `[]`.
ponytail: single-process sequential read-modify-write, no file lock (add advisory locking before the four
real GPU lanes write one manifest concurrently).

## Config additions

`SCREENING_ROOT`, `REGISTRY_PATH` (both env-overridable), `MAX_EGIPG_TRIALS=32`, `MAX_COMPARATOR_TRIALS=16`,
`N_SCREENING_SEEDS=1`, `N_FINAL_SEEDS=5`.

## Review history

Adversarial workflow review (2026-07-16): 6 finder dimensions → per-finding adversarial verify (11 agents).
The correctness-critical dimensions (network-propagation math, neural-encoder wiring, screening alignment)
produced **nothing that survived** verification; **3 findings confirmed, all fixed** — see
`docs/reviews/2026-07-16-code-review-module7.md`: (1) a tautological H2a test (now cross-checks the delta +
direction against the per-config `systema`), (2) `load_registry` returning `None` on a null `runs:` key, (3)
the driver artifact guard omitting `ID_MAPPING_PATH`. A pre-review fix caught the shared perturbation-encoder
that would have co-trained two configs' weights.

A second, deeper **xhigh `/code-review`** of the committed module (6 finders → 24 candidates → 21 verifiers)
surfaced **15 verified findings** (4 refuted), fixed across four tiers — see
`docs/reviews/2026-07-16-code-review-module7.md`. The as-built consequences folded into this module: the
registry cap counts **distinct configs** (dev re-runs no longer exhaust it); `summary.json` is sanitized to
valid JSON; the driver exits non-zero on a wholly-failed wave; the **network-propagation baseline now has a
scoring path** (`score_network_propagation` + `run_screening(extra_scorers=…)`); checkpoints are
seed-namespaced and `gpu_hours` is logged; a `MAX_COMPARATOR_FAMILIES=2` cap; a `seeded_init(seed)` context
manager so weight init is reproducible from the seed (the Trainer's generators cover only data shuffling);
one-pass val scoring + a CSR `train_mean`; and one shared `response_metric_suite` so screening and Module-6
scores can't drift.

## Verification

`./init.sh` green at **171 tests** (145 prior + 26 Module 7: 7 in `test_graph_baselines.py`, 18 in
`test_screening.py`, + the `seeded_init` test in `test_training.py`). Fully synthetic — tiny marts + a
small in-memory PPI graph.

**Real-data smoke** — `screening/run_screening.py --device cuda` on the real blocked-target-OOD split
(40-row bounded, 1 epoch, batch 4) trained and scored all four wave members (expression-only, untyped-GNN,
typed-static, condition-gated) on an A100, registered + logged all four `completed`, and wrote predictions
in the common schema. As-built findings:
- **Honest negative:** the graph variants did **not** separate from expression-only (systema 0.377 expr-only
  vs 0.362 typed-static vs 0.348 condition-gated; H2a Δ=−0.015, H2b Δ=−0.015, **neither supported**) — the
  near-null-signal regime the report anticipates, on 1-epoch models. Convergent training + the full split is
  the real H2a/H2b test.
- **Memory ceiling:** the typed encoder's per-edge signed messages OOM a single 80 GB A100 on **real dense
  PPI subgraphs at batch 32** (a hub's 512-node neighbourhood carries tens of thousands of STRING functional
  edges) — the graph model had never been trained on real data before (Module 5's real run was expr-only).
  Batch 4 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` fits; CPU (1 TB RAM) is the report's stated
  home for graph message passing. `run_screening`'s failure isolation means this no longer aborts the wave.

**Full real-data run (2026-07-17) — ⚠ the compute verdict below was SUPERSEDED later the same day; read
the RESOLVED note that follows before acting on any of it.** Modules 1-6 validated at full scale (M5
Stage-A `best_val` 3.4690; M6
egipg `systema` 0.0810 edging ridge 0.0806, G2-MQ PASSED). **M7 graph screening is compute-bound on the
full 21,262-row fold**: the per-target subgraph sampling + per-row message passing is single-threaded on CPU
(`torch.set_num_threads(1)`, GPU ~0% util), so the *fastest* graph config (untyped-GNN) did **not** finish
one epoch in ~11 h — full-data graph screening is not practical as-is. Tractable workaround used: the 4
nested configs + network-propagation on a **1,000-row fold, one A100 each in parallel** (~55 min vs ~2.5 h
sequential) → same-fold `systema` expr-only 0.0402 / untyped 0.0404 / typed-static 0.0412 / condition-gated
0.0350 / network-prop 0.0237; **H2a +0.0010 (nominally supported), H2b −0.0062 (not)** — noise at 1 epoch,
the near-null-signal regime. `run_full_pipeline.sh` (repo root) runs Modules 1-7 unattended under nohup.

**RESOLVED (2026-07-17) — the paragraph above is history, not current guidance.** The full fold is now
tractable: **667 → 61 ms/row (10.9×)**, a 21,262-row epoch **3.94 h → 0.36 h**, GPU util **median 1% →
46%**. The diagnosis above was half right and the prescription was wrong, which is the part worth
remembering: "single-threaded per-target subgraph sampling, GPU ~0%" correctly named the *symptom*, but
the deferred task it prescribed — mini-batch the encoders — targeted **message passing, which measured
only 5%** of GPU wall-clock. `sample_subgraph` was **95%**, because `_grow`/`_induce` each scanned the
whole ~8M-edge table *per row*. Mini-batching alone would have been Amdahl-capped at ~1.05×. Fixed in
that order: a CSR neighbour index (581 → 22 ms/row), *then* the PyG `Batch` mini-batching this entry
asked for. Both pinned by exact-equivalence tests; the Module-4 `edge_gates` contract and the sampled
subgraphs are unchanged. Full measurements + the xhigh review that followed:
`docs/specs/2026-07-17-graph-throughput-minibatch.md`.

**What this changes for screening:** run the nested family on the **full fold**, not `--n-max 1000`. The
capped-fold H2a/H2b numbers above are noise and must not be cited as a result. Use `--batch-size 8`
(bs=32 buys ~5% for 3× memory); the batch-32 OOM note above still stands for *training*, but evaluation
at `BATCH_SIZE=64` is now safe — the encoders chunk message passing at `config.GRAPH_ENCODE_CHUNK`
(default 8), bounding eval peak to 2.01 GB (from 12.53 GB) without changing results.
