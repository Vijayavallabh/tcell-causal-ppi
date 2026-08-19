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
  # The bracket is load-bearing. `pgrep -f "run_screening --only"` matched the shell that WROTE
  # this script with a heredoc, because the script's own source text -- including this very line --
  # sits in that shell's command line, and it was still alive. The finaliser then waited on its own
  # creator forever: all 48 lanes landed at 12:18 and it was still in this loop at 12:40.
  # `[r]un_screening` matches the literal "run_screening" but not the text "[r]un_screening".
  if pgrep -f "[r]un_screening --only" > /dev/null; then quiet=0; else quiet=$((quiet + 1)); fi
  sleep 180
done
echo "[fin] $(date) lanes are quiet; landed per rung:"
for r in data/results/a2_ladder/*d[0-9][0-9][0-9]; do
  [ -d "$r" ] || continue
  echo "[fin]   $(basename "$r"): expression_only=$(ls "$r"/expression_only/[0-9].parquet 2>/dev/null | wc -l) untyped_gnn=$(ls "$r"/untyped_gnn/[0-9].parquet 2>/dev/null | wc -l)"
done

.venv/bin/python -m tcell_pipeline.screening.ladder_report --out data/results/a2_ladder/floor.json
echo "[fin] $(date) DONE — data/results/a2_ladder/floor.json"
