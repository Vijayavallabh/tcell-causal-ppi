#!/bin/bash
# Train the fourth difficulty level. data/results/splits_c070 (seq-cosine 0.70, cap 0.05, seed 0) has
# existed since the split sweep and NOTHING has ever been trained against it — found by the 2026-08-11
# repo audit.
#
# WHY THE CHEAP PAIR ONLY. This exists to feed the L4 variance decomposition (NEXT_ACTIONS OPEN E),
# which needs h2a at several difficulty levels, not a full family at each. expression_only and
# typed_static give h2a; the graph-heavy arms cost 4-10x and add nothing to that decomposition.
#
# LEVELS ON DISK AFTER THIS RUNS:
#   0.85/0.05  data/splits          frozen fold, all four arms at n=7
#   0.80/0.10  c080c10_h1 (4 arms n=5) + r2 (seed 1) + r3 (seed 2), cheap pair n=5   <- 3 re-draws
#   0.75/0.15  c075c15_n5           all four arms n=5, seed 0 only
#   0.70/0.05  THIS RUN             cheap pair n=5, seed 0
# That is four levels with one level re-drawn three times. The decomposition still wants re-draws at a
# SECOND level; the cheapest completion is seeds 1 and 2 at 0.75/0.15.
#
#   setsid nohup ./run_c070.sh > data/logs/c070.nohup.log 2>&1 &
#
# RAIL 2: fresh root, data/splits and data/results/screening are read-only.
set -u
cd "$(dirname "$0")" || exit 1
ROOT=data/results/screening_c070
LOG=data/logs/c070
SEEDS="0 1 2 3 4"
export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=8000 OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export SPLITS_ROOT=data/results/splits_c070
export SCREENING_ROOT=$ROOT PREDICTIONS_ROOT=$ROOT/predictions
export REGISTRY_PATH=$ROOT/experiment_registry.yaml

for ro in data/results/screening data/results/screening_lambda0 data/splits; do
  [ "$ROOT" = "$ro" ] && { echo "REFUSING: \$ROOT is read-only root $ro"; exit 2; }
done
[ -f "$SPLITS_ROOT/blocked_target_ood.csv" ] || { echo "REFUSING: $SPLITS_ROOT has no split"; exit 2; }
mkdir -p "$ROOT" "$LOG"

JOBS=()
for arm in expression_only typed_static; do for s in $SEEDS; do JOBS+=("$arm|$s"); done; done
echo "[c070] $(date) START — ${#JOBS[@]} lanes, splits=$SPLITS_ROOT (thr 0.70 / cap 0.05)"

run_card () {
  local gpu=$1; shift
  for job in "$@"; do
    IFS='|' read -r arm s <<< "$job"
    [ -f "$ROOT/$arm/$s.parquet" ] && { echo "[c070] SKIP ${arm}_s${s} (landed)"; continue; }
    echo "[c070] ${arm}_s${s} gpu=$gpu START $(date)"
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
      --only "$arm" --seed "$s" --epochs 20 --batch-size 8 --device cuda \
      --lambda-graph 0 > "$LOG/${arm}_s${s}.log" 2>&1
    echo "[c070] ${arm}_s${s} gpu=$gpu exit=$? $(date)"
  done
}

for i in 0 1 2 3; do
  card=()
  for j in "${!JOBS[@]}"; do [ $((j % 4)) -eq "$i" ] && card+=("${JOBS[$j]}"); done
  run_card "$i" "${card[@]}" &
done
wait

echo "[c070] $(date) DONE. Gate health:"
.venv/bin/python - <<'PY'
import json, pathlib
for h in sorted(pathlib.Path("data/results/screening_c070").glob("*/[0-9]/logs/stage_a_history.json")):
    d = json.loads(h.read_text())
    g = [e["train"]["gate_mean"] for e in d if e.get("train", {}).get("gate_mean") is not None]
    arm, seed = h.parts[-4], h.parts[-3]
    if not g:
        print(f"  {arm}/s{seed}: {len(d)} epochs, no learnable edge gate"); continue
    print(f"  {arm}/s{seed}: {len(d)} epochs, gate {g[0]:.4f} -> {g[-1]:.4f} "
          f"({'ALIVE' if g[-1] > 1e-3 else 'COLLAPSED'})")
PY
echo "[c070] then aggregate: SCREENING_ROOT=$ROOT REGISTRY_PATH=$REGISTRY_PATH SPLITS_ROOT=$SPLITS_ROOT \\"
echo "         PYTHONPATH=src .venv/bin/python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3,4"
