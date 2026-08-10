#!/bin/bash
# L5 / cause D: is the "no graph" baseline actually graph-free?
# expression_only receives PINNACLE (learned on a PPI network) and three PPI degree scalars, so the
# paper's h1 contrast measures message passing over a graph SUMMARY, not graph vs no graph. This
# builds the missing arm. Channels are ZEROED not removed, so out_dim and parameter count are
# identical across variants and the contrast isolates information rather than capacity.
#
# Comparator is the EXISTING frozen-fold expression_only at n=5 in data/results/screening_lambda0
# (read-only). Settings matched to it exactly: frozen fold (SPLITS_ROOT unset), 20 epochs, batch 8,
# lambda_graph 0, seeds 0-4.  usage: run_feature_ablation.sh <gpu> <variant> <seed>
set -u
cd "$(dirname "$0")" || exit 1
gpu=$1; variant=$2; seed=$3
case "$variant" in
  nograph)     DROP="pinnacle,ppi_degree" ;;   # the true no-graph floor
  nopinnacle)  DROP="pinnacle" ;;              # which channel carries it?
  nodegree)    DROP="ppi_degree" ;;
  *) echo "unknown variant $variant"; exit 2 ;;
esac
ROOT=data/results/ablate_$variant
export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=8000 OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export DROP_TARGET_FEATURES=$DROP
export SCREENING_ROOT=$ROOT PREDICTIONS_ROOT=$ROOT/predictions
export REGISTRY_PATH=$ROOT/experiment_registry.yaml
mkdir -p "$ROOT" data/logs/ablate
[ -f "$REGISTRY_PATH" ] || cp -p data/results/screening_lambda0/experiment_registry.yaml "$REGISTRY_PATH"
echo "[abl] ${variant}_s${seed} gpu=$gpu drop=$DROP START $(date)"
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
  --only expression_only --seed "$seed" --epochs 20 --batch-size 8 --device cuda --lambda-graph 0 \
  > "data/logs/ablate/${variant}_s${seed}.log" 2>&1
echo "[abl] ${variant}_s${seed} gpu=$gpu exit=$? $(date)"
