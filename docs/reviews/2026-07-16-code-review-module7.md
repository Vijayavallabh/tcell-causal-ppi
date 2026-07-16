# Code review — Module 7 (Graph Baselines + Screening Harness)

Date: 2026-07-16 · Scope: the Module 7 diff (feat-007 + feat-011). Method: a dynamic multi-agent
adversarial-review workflow — 6 finder dimensions over the new files (graph baselines, screening,
experiment registry, driver, config, tests), each finding then handed to an independent verifier prompted
to **refute** it (default REFUTED unless a concrete failing input is constructed). 11 agents ran; the raw
findings were adversarially verified and **3 confirmed, the rest refuted/uncertain**. Notably the
correctness-critical dimensions — network-propagation math, the untyped-GCN / typed-static encoder wiring,
and the screening prediction/truth alignment — produced **nothing that survived** verification. All 3
confirmed findings were fixed.

One bug was caught during implementation, before the review: `nested_family_factories` originally reused a
single perturbation-encoder instance across configs, so two configs in one `run_screening` would share and
co-train the same encoder parameters — fixed to build a fresh encoder per factory call (a callable, not an
instance).

## First review — findings and resolutions

1. **The H2a screening test was tautological** (test_screening.py, weak-assertion, medium).
   `_nested_comparison.contrast` sets `"better"` to the literal string argument (`TYPED_STATIC`), so the
   test's `summary["h2a"]["better"] == TYPED_STATIC` was true by construction; `delta` was never asserted
   and `supported` was only type-checked. An inverted contrast (`w > b`, or `w - b`) would ship green while
   reporting the wrong winner / wrong-sign delta on the module's primary confirmatory output. **Fixed:** the
   test now cross-checks `delta` and `supported` against the per-config `systema` values already in
   `summary["results"]`, pinning both the sign of the delta and the boolean direction to the actual metrics.

2. **`load_registry` returned `None` on a null `runs:` key** (experiment_registry.py, robustness, low).
   `doc.get("runs", [])` returns `None` when the key is *present but null* (a truncated / externally-authored
   manifest), because the default only applies when the key is absent — crashing `register_run`'s
   `sum(... for r in runs)` and `log_run`'s `for r in runs` with `TypeError: NoneType is not iterable`,
   even though the empty-file and missing-file paths both correctly return `[]`. **Fixed:** `return
   doc.get("runs") or []`, so a null `runs` degrades to empty like the other degenerate paths. New test
   `test_registry_load_tolerates_null_and_missing`.

3. **The driver artifact guard omitted `ID_MAPPING_PATH`** (run_screening.py, integration-config, low).
   `build_hetero_graph()` reads `id_mapping.parquet` unconditionally, but the presence guard listed the
   other three graph-build inputs and not this one, so a missing `id_mapping.parquet` raised a bare
   `FileNotFoundError` instead of the friendly `[screen] required artifacts absent …` message. **Fixed:**
   added `config.ID_MAPPING_PATH` to the `required` list.

## Post-review hardening (from the real-data smoke, not the review)

The bounded real-data smoke (A100, blocked-target-OOD) surfaced that the typed encoder OOMs a single 80 GB
GPU on real dense PPI subgraphs at batch 32 — the graph model's first real training (Module 5's real run was
expr-only). Two responses, both in this changeset: `run_screening` is now **failure-isolating** (a config's
OOM/crash is caught, logged failed in the registry, and the wave continues) with a test
(`test_run_screening_isolates_failed_config`); and the driver sets `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` for CUDA and defaults `--batch-size` to 8. See
`docs/specs/2026-07-16-module7-screening.md` for the full smoke result (honest H2a/H2b negative).

## Second review — xhigh workflow-backed `/code-review` (of committed `6b6021f`)

A deeper pass (`/code-review` at xhigh: 6 finders → 24 candidates → 21 independent verifiers) over the
committed Module 7 diff surfaced **15 verified findings** (4 refuted). None was a crash in the happy path —
the correctness-critical maths (network-propagation diffusion, encoder wiring, prediction/truth alignment)
again held. The findings were triaged into four tiers and fixed across four commits
(`9db57ae` → `32fb473` → `4e25f4b` → `04e6148`):

**Tier 1 — correctness / deliverable** (`9db57ae`):
1. Registry cap counted executions, not distinct configs → re-running the driver exhausted the 32-cap. Now
   counts **distinct `config_id`s** per family; every execution is still logged.
2. `summary.json` emitted bare `NaN`/`Infinity` (invalid JSON) for a diverged metric / `best_val=inf` →
   non-finite floats sanitized to `null` (`allow_nan=False` backstop).
3. Driver returned exit 0 on a wholly-failed wave → returns non-zero when nothing completed.
4. `NetworkPropagationBaseline` (feat-007's 3rd graph reference) had no scoring path → new
   `score_network_propagation` + `run_screening(extra_scorers=…)`, wired into the driver table.

**Tier 2 — forward-looking correctness / audit** (`32fb473`):
5. Checkpoints seed-namespaced (`…/name/{seed}/ckpt`) so multi-seed sweeps don't overwrite `best.pt`.
6. `gpu_hours` recorded (wall-clock proxy, on completed AND failed) — the audit field was dead.
7. `nested_family_configs(names=[])` yields zero configs (`is None` guard), not the defaults.
8. `MAX_COMPARATOR_FAMILIES=2` enforced (report §1291).
9. Docstring clarified: the untyped-GNN is an internal EG-IPG ablation (counts against the 32-cap), not an
   external comparator (those are feat-010).

**Tier 3 — test-strength gaps + a reproducibility fix** (`4e25f4b`):
10. Best-vs-last checkpoint reload was never exercised (1-epoch tests) → a multi-epoch test where best !=
    last, asserting the scored predictions come from the **best** checkpoint.
11. Success-path registry logging was untested → a completed-run test pinning the `log_run` arg wiring.
    Reproducibility subtlety it surfaced: module **weight init** draws from the global torch RNG, which the
    Trainer's seeded generators deliberately don't touch, so `seed` alone didn't fully determine a run. New
    `seeded_init(seed)` context manager (`training/trainer.py`); `screen_config` and `run_train.run` build
    their model inside it.

**Tier 4 — efficiency / DRY** (`04e6148`, all value-preserving, verified on real data):
12. Val fold scored in one loader pass (`collect_predictions` returns the truths it was discarding).
13. `train_mean` + `collect_targets_truth` read `z@B` straight from the zscore CSR (`dataset_delta_z`),
    skipping the per-row `__getitem__` encoder-batch build.
14. `compute_all_metrics` and `run_module6_smoke._score` share one `response_metric_suite` (`metrics.py`).
15. `run_screening` rejects duplicate config names up front (the multi-seed `by_name` collapse).

## Verification

`./init.sh` green at **171 tests** (145 prior + 25 Module 7 across `test_graph_baselines.py` [7] and
`test_screening.py` [18], + the `seeded_init` test in `test_training.py`). Fully synthetic — tiny marts + a
small in-memory PPI graph. The four review-fix tiers were additionally validated on the real
blocked-target-OOD fold (network-propagation scoring, valid `summary.json`, CSR/loader value-parity).
