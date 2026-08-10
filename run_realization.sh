#!/bin/bash
# Generic same-parameter realization runner: run_realization.sh <gpu> <r-tag> <seed>
# All realizations use IDENTICAL threshold/cap (0.80/0.10) and differ ONLY in SPLIT_SEED, so the
# spread across them is pure partition noise. Runs expression_only (cheap) then condition_gated.
set -u
cd "$(dirname "$0")" || exit 1
gpu=$1; tag=$2; seed=$3
ROOT=data/results/screening_c080c10_${tag}
export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=8000 OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export SPLITS_ROOT=data/results/splits_c080c10_${tag}
export SCREENING_ROOT=$ROOT PREDICTIONS_ROOT=$ROOT/predictions
export REGISTRY_PATH=$ROOT/experiment_registry.yaml
mkdir -p "$ROOT" data/logs/n5
[ -f "$REGISTRY_PATH" ] || cp -p data/results/screening_c080c10/experiment_registry.yaml "$REGISTRY_PATH"
for arm in expression_only condition_gated; do
  echo "[$tag] ${arm}_s${seed} gpu=$gpu START $(date)"
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
    --only "$arm" --seed "$seed" --epochs 20 --batch-size 8 --device cuda --lambda-graph 0 \
    > "data/logs/n5/${tag}_${arm}_s${seed}.log" 2>&1
  echo "[$tag] ${arm}_s${seed} gpu=$gpu exit=$? $(date)"
done
