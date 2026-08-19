#!/bin/bash
# B1a finaliser: wait for the typed_gcnnorm lanes, then merge the registry and run the report.
#
#   setsid nohup ./run_b1_finalise.sh > data/logs/b1_finalise.log 2>&1 &
#
# NO pgrep ANYWHERE, deliberately. run_ladder_finalise.sh waited on a `pgrep -f` pattern that matched
# the shell which had just written it via heredoc, so it blocked on its own creator and never fired.
# This waits on ARTIFACTS instead - the landed parquets, and the runner's own "DONE." line in its nohup
# log - which cannot match anything but themselves.
set -u
cd "$(dirname "$0")" || exit 1

ROOT=data/results/screening_b1
ARM=typed_gcnnorm
WANT=5
NOHUP=data/logs/b1_gcnnorm.nohup.log
POLL=300
DEADLINE=$(( $(date +%s) + 48*3600 ))

count () { ls "$ROOT/$ARM"/[0-9].parquet 2>/dev/null | wc -l; }

echo "[b1f] waiting for $WANT $ARM lanes under $ROOT (poll ${POLL}s, 48h deadline)"
while :; do
  n=$(count)
  [ "$n" -ge "$WANT" ] && { echo "[b1f] all $n lanes landed $(date)"; break; }
  # The runner announces its own completion; if it is finished with fewer lanes, report what landed
  # rather than blocking to the deadline. Rail 5 makes n<4 preliminary, and the report labels it.
  if grep -q "DONE\." "$NOHUP" 2>/dev/null; then
    echo "[b1f] runner reported DONE with $n/$WANT lanes — reporting what landed $(date)"; break
  fi
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "[b1f] DEADLINE reached at $n/$WANT lanes"; break; }
  sleep $POLL
done

echo "[b1f] $(date) merging registry (MANDATORY before aggregating a fresh root)"
.venv/bin/python merge_registry_n7.py

echo "[b1f] $(date) B1 report"
PYTHONPATH=src .venv/bin/python -m tcell_pipeline.screening.b1_report \
  --root "$ROOT" --out "$ROOT/b1_message_form.json"
echo "[b1f] $(date) DONE"
