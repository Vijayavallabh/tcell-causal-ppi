#!/bin/bash
# Full real-data pipeline, Modules 1-7 (feat-* implemented so far). Run unattended, persistent across
# logout, from the repo root:
#
#     nohup ./run_full_pipeline.sh > data/logs/pipeline.nohup.log 2>&1 &
#
# Then monitor:  tail -f data/logs/pipeline.nohup.log   (per-module logs in data/logs/m*.log)
# Module 0 (data marts) is DESTRUCTIVE and is NOT run here — it must already be built on disk.
# The M7 typed graph configs (typed_static / condition_gated) are per-subgraph and take HOURS on the
# full 21k-row train fold; expression_only / untyped_gnn / network_propagation finish quickly.
cd "$(dirname "$0")" || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=src
LOG=data/logs; mkdir -p "$LOG"
echo "[pipeline] $(date) START"

# --- Modules 1-4: independent read-only smokes, one GPU each (4 idle A100s at PCI 0,1,2,4) ---
CUDA_VISIBLE_DEVICES=0 uv run python -m tcell_pipeline.run_module1_smoke          > "$LOG/m1.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=1 uv run python -m tcell_pipeline.graph.run_module2_smoke    > "$LOG/m2.log" 2>&1 & p2=$!
CUDA_VISIBLE_DEVICES=2 uv run python -m tcell_pipeline.run_module3_smoke          > "$LOG/m3.log" 2>&1 & p3=$!
CUDA_VISIBLE_DEVICES=4 uv run python -m tcell_pipeline.rationale.run_module4_smoke > "$LOG/m4.log" 2>&1 & p4=$!
wait $p1; echo "[pipeline] M1 exit=$?"
wait $p2; echo "[pipeline] M2 exit=$?"
wait $p3; echo "[pipeline] M3 exit=$?"
wait $p4; echo "[pipeline] M4 exit=$?"

# --- Module 5 Stage-A training (writes stage_a_best.pt) then Module 6 eval (consumes it) — GPU 0 ---
CUDA_VISIBLE_DEVICES=0 uv run python -m tcell_pipeline.training.run_train --expr-only --epochs 5 --device cuda > "$LOG/m5.log" 2>&1
echo "[pipeline] M5 exit=$?"
CUDA_VISIBLE_DEVICES=0 uv run python -m tcell_pipeline.run_module6_smoke --device cuda > "$LOG/m6.log" 2>&1
echo "[pipeline] M6 exit=$?"

# --- Module 7 screening (GPU 1) — the long one; graph configs are hours, failure-isolated ---
CUDA_VISIBLE_DEVICES=1 uv run python -u -m tcell_pipeline.screening.run_screening --epochs 1 --batch-size 8 --device cuda > "$LOG/m7.log" 2>&1
echo "[pipeline] M7 exit=$?"

echo "[pipeline] $(date) DONE — results in data/results/ (screening/summary.json, predictions/, experiment_registry.yaml)"
