#!/bin/bash
# Runs the rest of the autonomous queue without anyone at a terminal.
#
#   wave 1 (already running)  A1 typed_shared, seeds 0-4
#   wave 2                    A1 typed_permuted, seeds 0-4      -> the tie-breaker for typed_shared
#   wave 3                    A2(a) the injection ladder        -> the empirical detection floor
#   after each wave           aggregate and write the artifact, so a result exists even if the next
#                             wave dies
#
#   setsid nohup ./run_overnight_chain.sh > data/logs/overnight_chain.log 2>&1 &
#   kill with `kill -TERM -<PGID>` (the minus: the whole process GROUP)
#
# WHY IT WAITS ON THE LANES AND NOT ON THE SCRIPT. This script's own command line contains the string
# `run_a1_mechanism.sh`, so a pgrep for the parent would match itself and wait forever. It watches for
# the LANE processes instead, and requires several consecutive quiet checks so the sub-second gap
# between one lane finishing and the next starting cannot be mistaken for the wave being done.
#
# Everything it launches is idempotent and skips landed lanes, so a re-run after a crash resumes.
set -u
cd "$(dirname "$0")" || exit 1
export PYTHONPATH=src
LOG=data/logs
mkdir -p "$LOG"

wait_for_lanes () {  # $1 = pgrep pattern, $2 = human label
  local quiet=0
  echo "[chain] $(date) waiting for: $2"
  while [ $quiet -lt 3 ]; do
    if pgrep -f "$1" > /dev/null; then quiet=0; else quiet=$((quiet + 1)); fi
    sleep 120
  done
  echo "[chain] $(date) $2 is quiet"
}

landed () { ls "$1"/[0-9].parquet 2>/dev/null | wc -l; }

# ---- wave 1: wait for the typed_shared lanes already in flight -----------------------------------
wait_for_lanes "run_screening --only typed_shared" "A1 wave 1 (typed_shared)"
echo "[chain] typed_shared landed: $(landed data/results/screening_a1/typed_shared)/5"

# ---- wave 2: the permuted-relation arm ------------------------------------------------------------
echo "[chain] $(date) === wave 2: typed_permuted ==="
ARMS="typed_permuted" ./run_a1_mechanism.sh >> "$LOG/a1_mechanism.nohup.log" 2>&1
echo "[chain] typed_permuted landed: $(landed data/results/screening_a1/typed_permuted)/5"

# A1's analysis, written whatever landed. merge_registry_n7.py first: a fresh screening root without it
# carries a false "NOT comparable" flag and the verdict is suppressed.
.venv/bin/python merge_registry_n7.py >> "$LOG/a1_merge.log" 2>&1 || \
  echo "[chain] merge_registry_n7 failed — the aggregation below may carry a false NOT-comparable flag"
SCREENING_ROOT=data/results/screening_a1 \
REGISTRY_PATH=data/results/screening_a1/experiment_registry.yaml \
PREDICTIONS_ROOT=data/results/screening_a1/predictions SPLITS_ROOT=data/splits \
  .venv/bin/python -m tcell_pipeline.screening.a1_report \
    --out data/results/screening_a1/a1_mechanism.json 2>&1 | tee "$LOG/a1_report.log" || \
  echo "[chain] a1_report is not present yet — run it by hand once written"

# ---- wave 3: the injection ladder ------------------------------------------------------------------
echo "[chain] $(date) === wave 3: A2(a) injection ladder ==="
while pgrep -f "inject_signal --ladder" > /dev/null; do
  echo "[chain] waiting for the rung roots to finish building"; sleep 120
done
./run_a2_ladder.sh >> "$LOG/a2_ladder.nohup.log" 2>&1
.venv/bin/python -m tcell_pipeline.screening.ladder_report \
  --out data/results/a2_ladder/floor.json 2>&1 | tee "$LOG/ladder_report.log"

echo "[chain] $(date) DONE. Read, in order:"
echo "[chain]   data/results/screening_a1/a1_mechanism.json   D1 and D2, the A1 verdict"
echo "[chain]   data/results/a2_ladder/floor.json             the measured detection floor"
echo "[chain] Then update RESULTS_SUMMARY.md and the paper. Nothing here edits either."
