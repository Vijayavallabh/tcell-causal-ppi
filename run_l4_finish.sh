#!/bin/bash
# Finish the L4 difficulty lanes: c070 + the two 0.75/0.15 re-draws, one unified queue.
#
# WHY THIS REPLACES run_c070.sh + run_c075c15_redraws.sh RUNNING TOGETHER.
# Those two scripts each dealt their own jobs across four cards, so six typed_static lanes ran
# concurrently on hardware that had three usable cards. Measured result: ~90 min/epoch (against ~50
# when four lanes had four cards) and five OOM kills. Packing did not raise throughput, it turned one
# queue into six slow ones. This runs ONE lane per card and nothing else.
#
# CARD 2 IS EXCLUDED, DELIBERATELY. A co-tenant job on this shared box
# (`launch.py --config configs/pfdv2.yaml --train --gpu 2`) holds 80,948 MiB of that card's 81,920 and
# a CUDA context cannot even be created on it. Verified by mapping CUDA_VISIBLE_DEVICES to physical
# cards through torch's mem_get_info rather than assuming smi ordering:
#     CUDA 0 -> smi 0 (free 76.8 GiB)     CUDA 2 -> smi 2 (BUSY, context creation fails)
#     CUDA 1 -> smi 1 (free 77.5 GiB)     CUDA 3 -> smi 4 (free 75.6 GiB)
# Scheduling onto card 2 is what produced most of the OOMs, not concurrency alone.
#
#   setsid nohup ./run_l4_finish.sh > data/logs/l4_finish.nohup.log 2>&1 &
#   kill with `kill -TERM -<PGID>` (the minus: the whole GROUP, or each card loop starts its next job)
#
# Idempotent: every landed lane is skipped, so this is safe to re-run after any interruption.
# RAIL 2: fresh roots only; data/splits and data/results/screening are never written.
set -u
cd "$(dirname "$0")" || exit 1
LOG=data/logs/l4_finish
CARDS=(0 1 3)                 # NOT 2 - see above
export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=8000 OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
mkdir -p "$LOG"

# root:splits pairs. The cheap pair only: h2a is what the variance decomposition consumes.
declare -A SPLITS=(
  [screening_c070]=data/results/splits_c070
  [screening_c075c15_r2]=data/results/splits_c075c15_r2
  [screening_c075c15_r3]=data/results/splits_c075c15_r3
)

# Build the queue from what is actually MISSING, not from a fixed list, so a re-run after a partial
# failure schedules only the gaps.
# ORDER MATTERS: this queue is ~40 h and may be stopped early, so the most valuable lanes go first.
# The decomposition's binding constraint is within-level spread at a SECOND difficulty level - it
# already has three re-draws at 0.80/0.10 and nothing at any other level. So r2/r3 at 0.75/0.15 come
# first, low seeds first so each cell reaches a usable mean early; c070 is a fourth SINGLE-realisation
# level, which is the least informative addition and goes last.
JOBS=()
for s in 0 1 2 3 4; do
  for root in screening_c075c15_r2 screening_c075c15_r3; do
    for arm in expression_only typed_static; do
      [ -f "data/results/$root/$arm/$s.parquet" ] || JOBS+=("$root|$arm|$s")
    done
  done
done
for s in 0 1 2 3 4; do
  for arm in expression_only typed_static; do
    [ -f "data/results/screening_c070/$arm/$s.parquet" ] || JOBS+=("screening_c070|$arm|$s")
  done
done
[ ${#JOBS[@]} -eq 0 ] && { echo "[l4] nothing missing; exiting"; exit 0; }

# Preflight: refuse to start on a card that cannot hold a lane. typed_static on the reference fold
# peaked near 50 GiB in the n=7 run, so require 40 GiB free rather than merely "some".
USABLE=()
for g in "${CARDS[@]}"; do
  free=$(CUDA_VISIBLE_DEVICES=$g .venv/bin/python -c \
    "import torch; f,_=torch.cuda.mem_get_info(0); print(int(f/2**30))" 2>/dev/null)
  name=$(CUDA_VISIBLE_DEVICES=$g .venv/bin/python -c \
    "import torch; print(torch.cuda.get_device_properties(0).name)" 2>/dev/null)
  case "$name" in *A100*) ;; *) echo "[l4] card $g is '$name', not an A100 - skipping"; continue ;; esac
  if [ -z "$free" ] || [ "$free" -lt 40 ]; then
    echo "[l4] card $g has only ${free:-0} GiB free (<40) - skipping"; continue
  fi
  echo "[l4] card $g OK: $name, ${free} GiB free"; USABLE+=("$g")
done
[ ${#USABLE[@]} -eq 0 ] && { echo "[l4] no usable card; exiting"; exit 2; }

echo "[l4] $(date) START — ${#JOBS[@]} lanes over ${#USABLE[@]} cards (one lane per card)"

run_card () {
  local gpu=$1; shift
  for job in "$@"; do
    IFS='|' read -r root arm s <<< "$job"
    [ -f "data/results/$root/$arm/$s.parquet" ] && { echo "[l4] SKIP $root/$arm/s$s"; continue; }
    mkdir -p "data/results/$root"
    echo "[l4] $root/$arm/s$s gpu=$gpu START $(date)"
    SPLITS_ROOT=${SPLITS[$root]} \
    SCREENING_ROOT=data/results/$root \
    PREDICTIONS_ROOT=data/results/$root/predictions \
    REGISTRY_PATH=data/results/$root/experiment_registry.yaml \
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
      --only "$arm" --seed "$s" --epochs 20 --batch-size 8 --device cuda \
      --lambda-graph 0 > "$LOG/${root}_${arm}_s${s}.log" 2>&1
    echo "[l4] $root/$arm/s$s gpu=$gpu exit=$? $(date)"
  done
}

n=${#USABLE[@]}
for i in "${!USABLE[@]}"; do
  card=()
  for j in "${!JOBS[@]}"; do [ $((j % n)) -eq "$i" ] && card+=("${JOBS[$j]}"); done
  run_card "${USABLE[$i]}" "${card[@]}" &
done
wait

echo "[l4] $(date) DONE. Gate health (typed_static pins its gate to 1 by construction):"
.venv/bin/python - <<'PY'
import json, pathlib
for root in ("screening_c070", "screening_c075c15_r2", "screening_c075c15_r3"):
    for h in sorted(pathlib.Path(f"data/results/{root}").glob("*/[0-9]/logs/stage_a_history.json")):
        d = json.loads(h.read_text())
        g = [e["train"]["gate_mean"] for e in d if e.get("train", {}).get("gate_mean") is not None]
        arm, seed = h.parts[-4], h.parts[-3]
        tag = (f"gate {g[0]:.4f} -> {g[-1]:.4f} " + ("ALIVE" if g[-1] > 1e-3 else "COLLAPSED")) if g \
              else "no learnable edge gate"
        print(f"  {root:22s} {arm:16s} s{seed}  {len(d):2d} epochs  {tag}")
PY
