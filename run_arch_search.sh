#!/bin/bash
# AAAI architecture search, stage 1: per-relation normalisation x STRING confidence threshold.
#
# Scored on a TARGET-GROUPED INNER HOLDOUT of train. VAL IS NEVER TOUCHED here; the winner is confirmed
# on val once, with 5 paired seeds, in a separate run. The sealed challenge split is not opened.
#
# Cells are balanced so each card carries one unpruned (thr=0.0, 8.0M-edge) cell, which dominates cost.
#
#   setsid nohup ./run_arch_search.sh > data/logs/arch.nohup.log 2>&1 &
#
# LAUNCH WITH `setsid`, AND KILL WITH `kill -TERM -<PGID>` (note the minus: that signals the whole
# process GROUP). Killing the workers first leaves each lane's `for` loop free to start the NEXT cell
# before you get to the shell — on 2026-07-22 that orphaned a lane which then squatted 42 GB and OOMed
# another cell, while `nvidia-smi` five seconds after the kill reported the card free because the
# replacement had not allocated yet. Enumerate what you started; do not trust a post-kill snapshot.
set -u
cd "$(dirname "$0")" || exit 1

OUT=data/results/arch_search
LOG=data/logs
EPOCHS=5
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=src
export SUBGRAPH_CACHE_SIZE=9000
export OMP_NUM_THREADS=4
# variable-size subgraphs fragment the caching allocator; without this a cell creeps to OOM hours in
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$LOG"

for g in 0 1 2 4; do
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$g")
  case "$name" in *A100*) ;; *) echo "REFUSING: nvidia-smi $g is '$name', not an A100"; exit 2;; esac
done

echo "[arch] $(date) START — 9 cells, ${EPOCHS} epochs, inner holdout, val untouched"
uv run python -m tcell_pipeline.arch_search --list

lane () {  # $1=gpu  $2...=cell indices, run sequentially on that card
  local gpu=$1; shift
  for c in "$@"; do
    echo "[arch] cell=$c gpu=$gpu START $(date)"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -u -m tcell_pipeline.arch_search \
      --cells "$c" --epochs $EPOCHS --device cuda --out "$OUT" > "$LOG/arch_cell${c}.log" 2>&1
    echo "[arch] cell=$c gpu=$gpu exit=$? $(date)"
  done
}

lane 0 0 5 &   # add/all      + mean/high
lane 1 3 2 &   # mean/all     + add/high
lane 2 6 8 &   # gcn/all      + gcn/high
lane 4 1 4 7 & # the three medium-threshold cells (cheaper)
wait

echo "[arch] $(date) DONE — results:"
uv run python - <<'PY'
import json, pathlib
rows = [json.loads(p.read_text()) for p in sorted(pathlib.Path("data/results/arch_search").glob("*.json"))]
rows.sort(key=lambda r: -r["systema"])
print(f"{'cell':52s} {'systema':>9s} {'ep':>3s} {'gate':>7s} {'edges':>10s} {'h':>5s}")
for r in rows:
    print(f"{r['cell_id']:52s} {r['systema']:9.6f} {r['epochs_run']:3d} "
          f"{r['gate_mean_last']:7.4f} {sum(r['edges'].values()):10,d} {r['hours']:5.2f}")
PY
