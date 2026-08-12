#!/bin/bash
# Waits for the L4 workers to finish, then aggregates every root and runs the variance decomposition.
#
# WHY THIS EXISTS. The remaining typed_static lanes are ~50 min/epoch and there are eleven of them, so
# completion is a day or more away. Nothing about finishing the analysis should depend on someone being
# at a terminal when the last lane lands. This blocks on the workers, then does the whole tail.
#
#   setsid nohup ./run_l4_finalise.sh > data/logs/l4_finalise.log 2>&1 &
#
# Safe to start immediately and leave: it only waits, aggregates and reports. It writes no results
# root, trains nothing, and never touches data/splits or data/results/screening.
set -u
cd "$(dirname "$0")" || exit 1
OUT=data/results/l4
mkdir -p "$OUT"

echo "[fin] $(date) waiting for run_l4_finish.sh and run_l4_card2.sh to exit"
while pgrep -f 'run_l4_finish\.sh|run_l4_card2\.sh' > /dev/null; do sleep 120; done
echo "[fin] $(date) workers done; aggregating"

# Aggregate each root over exactly the seeds that landed. Aggregating over seeds that do not exist
# would silently shrink n; aggregating over a fixed 0-4 would fail outright on a partial root.
for spec in "screening_c070:data/results/splits_c070" \
            "screening_c075c15_r2:data/results/splits_c075c15_r2" \
            "screening_c075c15_r3:data/results/splits_c075c15_r3"; do
  root="${spec%%:*}"; splits="${spec##*:}"
  R="data/results/$root"
  seeds=$(ls "$R"/typed_static/[0-9].parquet 2>/dev/null | sed 's/.*\///;s/\.parquet//' | sort -u | paste -sd,)
  eo=$(ls "$R"/expression_only/[0-9].parquet 2>/dev/null | wc -l)
  ts=$(ls "$R"/typed_static/[0-9].parquet 2>/dev/null | wc -l)
  echo "[fin] $root: expression_only=$eo typed_static=$ts (aggregating seeds ${seeds:-none})"
  [ -z "$seeds" ] && { echo "[fin]   no typed_static lanes - skipping"; continue; }
  SPLITS_ROOT=$splits SCREENING_ROOT=$R PREDICTIONS_ROOT=$R/predictions \
  REGISTRY_PATH=$R/experiment_registry.yaml PYTHONPATH=src \
    .venv/bin/python -m tcell_pipeline.screening.multiseed --seeds "$seeds" 2>&1 \
    | grep -E "^\[robust\] (h2a|h1_vs|per-config|  [0-9])|INCOMPLETE|UNBALANCED"
done

echo
echo "[fin] ================ VARIANCE DECOMPOSITION ================"
for c in h2a h1_vs_no_graph; do
  echo "[fin] --- $c ---"
  PYTHONPATH=src .venv/bin/python -m tcell_pipeline.screening.variance_decomposition \
    --contrast "$c" --out "$OUT/vardecomp_${c}.json"
  echo
done

echo "[fin] ================ CROSS-DATASET POOLS (unchanged by L4, re-run for one report) ==========="
PYTHONPATH=src .venv/bin/python -m tcell_pipeline.replication.pool --with-reference \
  --out data/results/replication/pooled_with_reference.json 2>&1 | grep -E "h2a|promotion_margin"
PYTHONPATH=src .venv/bin/python -m tcell_pipeline.replication.pool \
  --out data/results/replication/pooled.json 2>&1 | grep -E "h2a|promotion_margin"

echo
echo "[fin] $(date) DONE. Wrote $OUT/vardecomp_*.json"
echo "[fin] Read the CAUTION line: with few levels and few re-draws these ratios are indicative."
