#!/bin/bash
# Third point on the OOD-difficulty curve. The frozen fold (thr 0.85 / cap 0.05, 5141 sequence
# families) and c075c15 (0.75 / 0.15, 3282 families) both give h1 parity at n=5; c080c10
# (0.80 / 0.10, 4155 families) sits between them and already has expression_only at n=4 from the
# 2026-07-29 campaign, so each condition_gated seed here directly builds the pair.
#
# Cards are refilled as the c075c15 campaign drains. Rail 5 still applies: n>=4 for BOTH arms before
# anything from this split is integrated. Fewer seeds than that = RESULTS_SUMMARY only.
set -u
cd "$(dirname "$0")" || exit 1
SRC=data/results/screening_c080c10          # READ-ONLY source of the 4 landed expression_only lanes
ROOT=data/results/screening_c080c10_h1      # FRESH root
LOG=data/logs/n5
export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=8000 OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export SPLITS_ROOT=data/results/splits_c080c10
export SCREENING_ROOT=$ROOT PREDICTIONS_ROOT=$ROOT/predictions
export REGISTRY_PATH=$ROOT/experiment_registry.yaml
for ro in data/results/screening data/results/screening_lambda0 "$SRC"; do
  [ "$ROOT" = "$ro" ] && { echo "REFUSING: \$ROOT is read-only root $ro"; exit 2; }
done
mkdir -p "$ROOT" "$LOG"
if [ ! -f "$REGISTRY_PATH" ]; then
  find "$SRC" -name '*.parquet' | sort | xargs sha256sum > "$ROOT/src_sha256.before.txt"
  cp -p "$SRC/experiment_registry.yaml" "$REGISTRY_PATH"
  for cfg in condition_gated expression_only untyped_gnn typed_static; do
    mkdir -p "$ROOT/$cfg"; cp -p "$SRC/$cfg"/[0-4].parquet "$ROOT/$cfg/" 2>/dev/null
  done
  echo "[c080] seeded fresh root with $(ls "$ROOT"/*/[0-4].parquet 2>/dev/null | wc -l) landed lanes"
fi
gpu=$1; seed=$2
echo "[c080] condition_gated_s${seed} gpu=$gpu START $(date)"
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
  --only condition_gated --seed "$seed" --epochs 20 --batch-size 8 --device cuda --lambda-graph 0 \
  > "$LOG/c080_condition_gated_s${seed}.log" 2>&1
echo "[c080] condition_gated_s${seed} gpu=$gpu exit=$? $(date)"
find "$SRC" -name '*.parquet' | sort | xargs sha256sum > "$ROOT/src_sha256.after.txt"
diff -q "$ROOT/src_sha256.before.txt" "$ROOT/src_sha256.after.txt" >/dev/null \
  && echo "[c080] source root VERIFIED byte-identical" || echo "[c080] *** SOURCE ROOT CHANGED ***"
