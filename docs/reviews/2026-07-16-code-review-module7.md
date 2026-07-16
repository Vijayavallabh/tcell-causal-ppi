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

## Findings and resolutions

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

## Verification

`./init.sh` green at **159 tests** (145 prior + 14 Module 7: 7 in `test_graph_baselines.py`, 7 in
`test_screening.py`). Fully synthetic — tiny marts + a small in-memory PPI graph, no real marts required.
