#!/bin/bash
# Waits for the A2(a) ladder lanes, then writes the floor report. Safe to start immediately and leave:
# it only waits and analyses. It trains nothing and writes no results root.
#
#   setsid nohup ./run_ladder_finalise.sh > data/logs/ladder_finalise.log 2>&1 &
#
# It watches the LANE processes rather than run_a2_ladder.sh, because its own command line would match
# a pgrep for the parent, and it requires several consecutive quiet checks so the sub-second gap between
# one lane ending and the next starting is not mistaken for the run being over.
set -u
cd "$(dirname "$0")" || exit 1
export PYTHONPATH=src

quiet=0
echo "[fin] $(date) waiting for the ladder lanes"
while [ $quiet -lt 3 ]; do
  if pgrep -f "run_screening --only" > /dev/null; then quiet=0; else quiet=$((quiet + 1)); fi
  sleep 180
done
echo "[fin] $(date) lanes are quiet; landed per rung:"
for r in data/results/a2_ladder/*d[0-9][0-9][0-9]; do
  [ -d "$r" ] || continue
  echo "[fin]   $(basename "$r"): expression_only=$(ls "$r"/expression_only/[0-9].parquet 2>/dev/null | wc -l) untyped_gnn=$(ls "$r"/untyped_gnn/[0-9].parquet 2>/dev/null | wc -l)"
done

.venv/bin/python -m tcell_pipeline.screening.ladder_report --out data/results/a2_ladder/floor.json
echo "[fin] $(date) DONE — data/results/a2_ladder/floor.json"
