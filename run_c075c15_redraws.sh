#!/bin/bash
# Re-draws at the 0.75/0.15 difficulty level — the piece the L4 variance decomposition is missing.
#
# WHY. Three re-draws already exist at 0.80/0.10, which measures realisation noise at ONE difficulty.
# With only one level re-drawn you cannot separate a difficulty effect from partition noise, because
# every other level has a single realisation and no within-level spread to compare against. These add
# seeds 1 and 2 at 0.75/0.15 (identical threshold and cap, SPLIT_SEED alone differs), giving a second
# level with within-level spread. Measured role agreement against seed 0 is 37% and 52%, so these are
# real re-partitions, not near-copies.
#
# CHEAP PAIR ONLY: h2a (typed_static - expression_only) is what the decomposition needs. The
# graph-heavy arms cost 4-10x and contribute nothing to it.
#
#   setsid nohup ./run_c075c15_redraws.sh > data/logs/c075_redraws.nohup.log 2>&1 &
#
# Co-schedules deliberately: the A100s here sit at 15-45 GiB of 82 and are partly CPU-bound on
# subgraph sampling, so a second job per card uses headroom rather than contending for memory.
# RAIL 2: fresh roots, data/splits and data/results/screening untouched.
set -u
cd "$(dirname "$0")" || exit 1
LOG=data/logs/c075_redraws
SEEDS="0 1 2 3 4"
export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=8000 OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
mkdir -p "$LOG"

JOBS=()
for tag in r2 r3; do
  for arm in expression_only typed_static; do
    for s in $SEEDS; do JOBS+=("$tag|$arm|$s"); done
  done
done
echo "[redraw] $(date) START — ${#JOBS[@]} lanes over 4 cards"

run_card () {
  local gpu=$1; shift
  for job in "$@"; do
    IFS='|' read -r tag arm s <<< "$job"
    local ROOT=data/results/screening_c075c15_$tag
    [ -f "$ROOT/$arm/$s.parquet" ] && { echo "[redraw] SKIP ${tag}_${arm}_s${s}"; continue; }
    mkdir -p "$ROOT"
    echo "[redraw] ${tag}_${arm}_s${s} gpu=$gpu START $(date)"
    SPLITS_ROOT=data/results/splits_c075c15_$tag \
    SCREENING_ROOT=$ROOT PREDICTIONS_ROOT=$ROOT/predictions \
    REGISTRY_PATH=$ROOT/experiment_registry.yaml \
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
      --only "$arm" --seed "$s" --epochs 20 --batch-size 8 --device cuda \
      --lambda-graph 0 > "$LOG/${tag}_${arm}_s${s}.log" 2>&1
    echo "[redraw] ${tag}_${arm}_s${s} gpu=$gpu exit=$? $(date)"
  done
}

for i in 0 1 2 3; do
  card=()
  for j in "${!JOBS[@]}"; do [ $((j % 4)) -eq "$i" ] && card+=("${JOBS[$j]}"); done
  run_card "$i" "${card[@]}" &
done
wait
echo "[redraw] $(date) DONE. Aggregate each root at --seeds 0,1,2,3,4, then run the decomposition:"
echo "  between-level (0.85 / 0.80 / 0.75 / 0.70) vs within-level (re-draws) vs within-re-draw (seed)"
