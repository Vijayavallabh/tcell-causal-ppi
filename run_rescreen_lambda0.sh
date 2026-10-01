#!/bin/bash
# feat-011 RE-SCREEN (2026-07-21, option C): re-run the condition-gated arm with the edge-gate penalty
# OFF, because that penalty annihilated its gates inside epoch 0 in all 5 seeds of the original campaign.
#
# WHY ONLY ONE ARM. L_graph carries gradient to exactly ONE of the four arms — verified by
# test_graph_penalty_carries_gradient_to_condition_gated_only, which fails if that ever stops being true:
#   expression_only  no graph encoder            -> 0/17  params moved by the penalty
#   typed_static     gates pinned to 1.0         -> 0/165 (constant term, no grad_fn)
#   untyped_gnn      returns edge_gates=None     -> 0/29
#   condition_gated  learnable condition gates   -> 7/165  <- the only arm the defect touched
# So the campaign's other 15 lanes remain valid comparators and only these 5 lanes must re-run. Same
# frozen fold, same frozen basis, same 20-epoch budget, same batch size, same seeds — only lambda_graph
# changes, so the paired-by-seed contrasts against the frozen lanes stay well formed.
#
# NOTHING FROZEN IS TOUCHED. New lanes are written to a SEPARATE root with a COPY of the registry:
# data/results/screening/{condition_gated/*,promoted.json} and the frozen H1 checkpoint that
# promoted.json points at are never opened for writing. Verify with the sha256 manifest this script
# writes before and after (rescreen_frozen_sha256.{before,after}.txt).
#
# The SEALED challenge split is NOT touched — every lane scores val. Module 0 is NOT run. The frozen
# program basis is NOT regenerated.
#
#   setsid nohup ./run_rescreen_lambda0.sh > data/logs/rescreen.nohup.log 2>&1 &
#
# LAUNCH WITH `setsid`, AND KILL WITH `kill -TERM -<PGID>` (note the minus: whole process GROUP).
# Killing the workers first leaves each lane's `for seed in ...` loop free to start the NEXT seed before
# you reach the shell. On 2026-07-22 that orphaned seed 4, which squatted 42 GB and OOMed a later job —
# and `nvidia-smi` five seconds after the kill showed the card free, because the replacement had not
# allocated yet. Enumerate what you started; never trust a post-kill snapshot as proof nothing runs.
set -u
cd "$(dirname "$0")" || exit 1

ROOT=data/results/screening_lambda0          # new lanes land here; the frozen root is read-only
FROZEN=data/results/screening
LOG=data/logs
EPOCHS=20
GPUS=(0 1 2 4)                               # nvidia-smi indices of the four A100s (3 is a T400 4GB)

export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=src
export SUBGRAPH_CACHE_SIZE=9000
export OMP_NUM_THREADS=4                     # NEVER the core count: 64 here produced ~830 threads, load 600
export SCREENING_ROOT=$ROOT
export PREDICTIONS_ROOT=$ROOT/predictions
export REGISTRY_PATH=$ROOT/experiment_registry.yaml

[ "$SCREENING_ROOT" = "$FROZEN" ] && { echo "REFUSING: would overwrite the frozen screening root"; exit 2; }
mkdir -p "$ROOT" "$LOG"

# --- preflight: the cards must really be A100s (torch's cuda:N is NOT nvidia-smi's N) ---------------
for g in "${GPUS[@]}"; do
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$g")
  case "$name" in *A100*) ;; *) echo "REFUSING: nvidia-smi $g is '$name', not an A100"; exit 2;; esac
done

# --- preflight: prove the frozen artifacts are untouched, before and after -------------------------
frozen_manifest () {
  find "$FROZEN" -name '*.parquet' -o -name 'promoted.json' -o -name 'stage_a_best.pt' \
    | sort | xargs sha256sum
}
frozen_manifest > "$ROOT/rescreen_frozen_sha256.before.txt"

# --- seed the new root with the 15 STILL-VALID lanes + the registry they are fenced against ---------
cp "$FROZEN/../experiment_registry.yaml" "$REGISTRY_PATH"
for cfg in expression_only untyped_gnn typed_static; do
  mkdir -p "$ROOT/$cfg"
  cp "$FROZEN/$cfg"/[0-4].parquet "$ROOT/$cfg/" 2>/dev/null
done

echo "[rescreen] $(date) START — condition_gated x seeds 0-4 at lambda_graph=0, ${EPOCHS} epochs"
echo "[rescreen] root=$ROOT registry=$REGISTRY_PATH cache=$SUBGRAPH_CACHE_SIZE OMP=$OMP_NUM_THREADS"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

run_seed () {  # $1=gpu  $2...=seeds to run SEQUENTIALLY on that gpu
  local gpu=$1; shift
  for seed in "$@"; do
    echo "[rescreen] seed=$seed gpu=$gpu START $(date)"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -u -m tcell_pipeline.screening.run_screening \
      --only condition_gated --seed "$seed" --epochs $EPOCHS --batch-size 8 --device cuda \
      --lambda-graph 0 > "$LOG/rescreen_s${seed}.log" 2>&1
    echo "[rescreen] seed=$seed gpu=$gpu exit=$? $(date)"
  done
}

# 5 lanes over 4 cards: the first card takes two, sequentially.
run_seed "${GPUS[0]}" 0 4 &
run_seed "${GPUS[1]}" 1 &
run_seed "${GPUS[2]}" 2 &
run_seed "${GPUS[3]}" 3 &
wait

frozen_manifest > "$ROOT/rescreen_frozen_sha256.after.txt"
if diff -q "$ROOT/rescreen_frozen_sha256.before.txt" "$ROOT/rescreen_frozen_sha256.after.txt" >/dev/null; then
  echo "[rescreen] frozen artifacts VERIFIED byte-identical"
else
  echo "[rescreen] *** FROZEN ARTIFACTS CHANGED — investigate before reading any result ***"
  diff "$ROOT/rescreen_frozen_sha256.before.txt" "$ROOT/rescreen_frozen_sha256.after.txt"
fi

echo "[rescreen] $(date) DONE. Gate means per epoch:"
for s in 0 1 2 3 4; do
  python3 - "$s" <<'PY'
import json, sys, pathlib
s = sys.argv[1]
p = pathlib.Path(f"data/results/screening_lambda0/condition_gated/{s}/logs/stage_a_history.json")
if not p.exists():
    print(f"  seed {s}: NO HISTORY — lane did not complete"); raise SystemExit
h = json.loads(p.read_text())
g = [e["train"]["gate_mean"] for e in h]
print(f"  seed {s}: {len(h)} epochs, gate mean {g[0]:.4f} -> {g[-1]:.4f} "
      f"({'ALIVE' if g[-1] > 1e-3 else 'COLLAPSED'})")
PY
done
echo "[rescreen] next: PYTHONPATH=src SCREENING_ROOT=$ROOT REGISTRY_PATH=$REGISTRY_PATH \\"
echo "             uv run python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3,4"
