#!/bin/bash
# One replication training lane: run_replication_lane.sh <gpu> <dataset> <pinnacle-ctx|none> <arm> <seed>
#
# Results land in a FRESH per-dataset root under data/results/replication/. Nothing under
# data/results/screening*, data/splits/ or promoted.json is read or written (rail 2).
set -u
cd "$(dirname "$0")" || exit 1
gpu=$1; DS=$2; CTX=$3; arm=$4; seed=$5

# shellcheck disable=SC1091
source ./run_replication_stage.sh "$DS" "$CTX" >/dev/null || exit 1

export SUBGRAPH_CACHE_SIZE=8000            # the value every landed reference lane used
export SCREENING_ROOT="data/results/replication/$DS"
export PREDICTIONS_ROOT="$SCREENING_ROOT/predictions"
export REGISTRY_PATH="$SCREENING_ROOT/experiment_registry.yaml"

for ro in data/results/screening data/results/screening_lambda0 data/splits; do
  [ "$SCREENING_ROOT" = "$ro" ] && { echo "REFUSING: root is read-only $ro"; exit 2; }
done
mkdir -p "$SCREENING_ROOT" data/logs/repl

tag="${DS}_${arm}_s${seed}"
echo "[repl] $tag gpu=$gpu START $(date)"
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
  --only "$arm" --seed "$seed" --epochs 20 --batch-size 8 --device cuda \
  --lambda-graph 0 > "data/logs/repl/${tag}.log" 2>&1
echo "[repl] $tag gpu=$gpu exit=$? $(date)"
