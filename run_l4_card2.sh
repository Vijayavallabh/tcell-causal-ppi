#!/bin/bash
# Second worker for the L4 queue, on the card that just freed up.
#
# WHY A SEPARATE WORKER instead of relaunching run_l4_finish.sh with four cards. The co-tenant that
# owned card 2 exited, so a fourth A100 is now idle. Relaunching would kill the in-flight lanes, one of
# which is 12 epochs deep (~9 h of compute). Waiting for it to finish idles the free card for hours.
# This takes work the main scheduler will not reach for a long time instead.
#
# HOW COLLISION IS AVOIDED. run_l4_finish.sh walks its per-card lists in order and c070 sits at the END
# of every one of them, behind the r2/r3 lanes. This worker takes c070 in REVERSE seed order, so the
# two pointers move toward each other and are many lanes apart. Both also re-check for a landed
# parquet immediately before starting, and this worker additionally takes an atomic claim file
# (set -o noclobber) so a lane can only be started once even if the pointers do meet.
# STOP THIS WORKER once the main queue reaches c070; do not let them overlap on the same seed.
#
#   setsid nohup ./run_l4_card2.sh > data/logs/l4_card2.nohup.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
GPU=2
ROOT=data/results/screening_c070
LOG=data/logs/l4_card2
export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=8000 OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
mkdir -p "$LOG" "$ROOT/typed_static"

free=$(CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -c \
  "import torch; f,_=torch.cuda.mem_get_info(0); print(int(f/2**30))" 2>/dev/null)
name=$(CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -c \
  "import torch; print(torch.cuda.get_device_properties(0).name)" 2>/dev/null)
case "$name" in *A100*) ;; *) echo "[c2] card $GPU is '$name', not an A100 — refusing"; exit 2 ;; esac
[ -z "$free" ] || [ "$free" -lt 40 ] && { echo "[c2] card $GPU has ${free:-0} GiB free (<40) — refusing"; exit 2; }
echo "[c2] $(date) START — card $GPU: $name, ${free} GiB free; c070 typed_static in reverse seed order"

for s in 4 3 2 1 0; do
  [ -f "$ROOT/typed_static/$s.parquet" ] && { echo "[c2] SKIP s$s (landed)"; continue; }
  claim="$ROOT/typed_static/.$s.claim"
  if ! ( set -o noclobber; echo "$$ $(date)" > "$claim" ) 2>/dev/null; then
    echo "[c2] SKIP s$s (claimed by another worker)"; continue
  fi
  echo "[c2] c070/typed_static/s$s gpu=$GPU START $(date)"
  SPLITS_ROOT=data/results/splits_c070 \
  SCREENING_ROOT=$ROOT PREDICTIONS_ROOT=$ROOT/predictions \
  REGISTRY_PATH=$ROOT/experiment_registry.yaml \
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
    --only typed_static --seed "$s" --epochs 20 --batch-size 8 --device cuda \
    --lambda-graph 0 > "$LOG/c070_typed_static_s${s}.log" 2>&1
  rc=$?
  echo "[c2] c070/typed_static/s$s gpu=$GPU exit=$rc $(date)"
  # Release the claim on failure so the main scheduler can retry the lane; keep it on success only as
  # a marker (the landed parquet is the real record).
  [ $rc -ne 0 ] && rm -f "$claim"
done
echo "[c2] $(date) DONE"
