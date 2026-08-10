#!/bin/bash
# SAME-DIFFICULTY REPLICATE: identical threshold/cap (0.80/0.10) as splits_c080c10, different
# SPLIT_SEED (1). The split statistic already shifts 0.759 -> 0.793 under this reseed, 60% of the
# whole designed difficulty span. This measures whether the CONTRAST moves as much: if h1 on this
# realization differs from the +0.0082 seen on seed 0 by a comparable amount, the between-fold
# differences in Table "folds" are fold noise, not difficulty.
# usage: run_c080c10_r2.sh <gpu> <seed>   (runs expression_only then condition_gated for that seed)
set -u
cd "$(dirname "$0")" || exit 1
ROOT=data/results/screening_c080c10_r2
export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=8000 OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export SPLITS_ROOT=data/results/splits_c080c10_r2
export SCREENING_ROOT=$ROOT PREDICTIONS_ROOT=$ROOT/predictions
export REGISTRY_PATH=$ROOT/experiment_registry.yaml
mkdir -p "$ROOT" data/logs/n5
[ -f "$REGISTRY_PATH" ] || cp -p data/results/screening_c080c10/experiment_registry.yaml "$REGISTRY_PATH"
gpu=$1; seed=$2
for arm in expression_only condition_gated; do
  echo "[r2] ${arm}_s${seed} gpu=$gpu START $(date)"
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
    --only "$arm" --seed "$seed" --epochs 20 --batch-size 8 --device cuda --lambda-graph 0 \
    > "data/logs/n5/r2_${arm}_s${seed}.log" 2>&1
  echo "[r2] ${arm}_s${seed} gpu=$gpu exit=$? $(date)"
done
