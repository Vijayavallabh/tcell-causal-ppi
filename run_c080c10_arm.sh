#!/bin/bash
# Generic c080c10 lane: run_c080c10_arm.sh <gpu> <arm> <seed>
# The intermediate split (median train-to-val cosine 0.759, between frozen 0.796 and c075c15 0.741)
# is the third point that turns h2a's non-replication into a dose-response rather than two points.
set -u
cd "$(dirname "$0")" || exit 1
export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=8000 OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export SPLITS_ROOT=data/results/splits_c080c10
export SCREENING_ROOT=data/results/screening_c080c10_h1
export PREDICTIONS_ROOT=$SCREENING_ROOT/predictions
export REGISTRY_PATH=$SCREENING_ROOT/experiment_registry.yaml
gpu=$1; arm=$2; seed=$3
echo "[c080] ${arm}_s${seed} gpu=$gpu START $(date)"
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
  --only "$arm" --seed "$seed" --epochs 20 --batch-size 8 --device cuda --lambda-graph 0 \
  > "data/logs/n5/c080_${arm}_s${seed}.log" 2>&1
echo "[c080] ${arm}_s${seed} gpu=$gpu exit=$? $(date)"
