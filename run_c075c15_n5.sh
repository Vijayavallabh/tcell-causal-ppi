#!/bin/bash
# Harder blocked-target-OOD split (seq-cosine 0.75 / family cap 0.15) — take the null from n=3 to n=5,
# and add the two arms the 2026-07-29 campaign ran out of budget for.
#
# WHY seed 2 is RE-RUN rather than resurrected. data/results/screening_c075c15/condition_gated/2/ holds a
# stage_a_best.pt from a lane that trained 13 epochs and was then KILLED by the campaign wind-down. Its
# peers (seeds 0,1,3) all stopped by the EARLY_STOP_PATIENCE=10 rule, exactly 10 epochs after their best;
# seed 2's best was epoch 7, so it was killed 5 epochs BEFORE its stopping rule could fire. Its checkpoint
# is therefore an argmin over a truncated epoch range: scoring it would bias condition_gated DOWNWARD,
# which is the direction that manufactures the null this project exists to test honestly. Re-run it.
#
# NOTHING FROZEN IS TOUCHED. New lanes land in a FRESH root seeded with a copy of the 10 landed lanes;
# the source root is sha256-manifested before and after. The sealed challenge split is never opened.
#
#   setsid nohup ./run_c075c15_n5.sh > data/logs/n5.nohup.log 2>&1 &
#
# KILL WITH `kill -TERM -<PGID>` (note the minus: the whole process GROUP). Killing workers alone leaves
# each lane's `for` loop free to launch its next job before you reach the shell.
set -u
cd "$(dirname "$0")" || exit 1

SRC=data/results/screening_c075c15          # READ-ONLY: the 10 lanes that already landed
ROOT=data/results/screening_c075c15_n5      # FRESH root: everything new lands here
SPLITS=data/results/splits_c075c15          # the harder split, frozen 2026-07-29
LOG=data/logs/n5
EPOCHS=20                                   # same cap the landed seeds used; arms early-stop at 11-14
BATCH=8
GPUS=(0 1 2 3)                              # verified below to be the four A100s, not the T400

export PYTHONPATH=src
export SUBGRAPH_CACHE_SIZE=8000             # the value every landed cond_gated seed used
export OMP_NUM_THREADS=4                    # NEVER the core count: 64 here produced ~830 threads, load 600

# MANDATORY on this box (2026-08-03). Without it EVERY cuda lane dies in <60s with
#   NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED at PeerToPeerAccess.cpp:83
# Chain: run_screening.run() does os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
# for --device cuda; expandable segments go through the CUDA driver API; PyTorch's DriverAPI::get() calls
# nvmlInit_v2_(); NVML then fails because the loader binds nvidia-smi/libtorch to a THIRD-PARTY 535.309.01
# driver tree another user left on the system library path
# (/mnt/md0/IITM/ipcv/Rohith/nvidia/NVIDIA-Linux-x86_64-535.309.01/libnvidia-ml.so.535.309.01) while the
# running kernel module is 580.173.02. Unsetting LD_LIBRARY_PATH does NOT help — the stray tree wins
# anyway. Preloading the CORRECT, already-installed 580 library fixes it, and also restores nvidia-smi.
# Setting PYTORCH_CUDA_ALLOC_CONF ourselves is NOT a fix worth preferring: the landed seeds all trained
# WITH expandable segments, so keeping them on is also the comparable choice.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export SPLITS_ROOT=$SPLITS
export SCREENING_ROOT=$ROOT
export PREDICTIONS_ROOT=$ROOT/predictions
export REGISTRY_PATH=$ROOT/experiment_registry.yaml

for ro in data/results/screening data/results/screening_lambda0 "$SRC"; do
  [ "$ROOT" = "$ro" ] && { echo "REFUSING: \$ROOT is the read-only root $ro"; exit 2; }
done
mkdir -p "$ROOT" "$LOG"

# --- preflight: ask torch what each card ACTUALLY is. nvidia-smi is unusable on this box (NVML userspace
# is 535.x against a 580.173.02 kernel module), and `torch`'s cuda:N is not nvidia-smi's N regardless. ---
for g in "${GPUS[@]}"; do
  name=$(CUDA_VISIBLE_DEVICES=$g .venv/bin/python -c \
    "import torch;print(torch.cuda.get_device_properties(0).name)" 2>/dev/null)
  case "$name" in *A100*) echo "[n5] card $g = $name" ;;
    *) echo "REFUSING: CUDA_VISIBLE_DEVICES=$g is '$name', not an A100"; exit 2 ;; esac
done

# --- preflight: prove the source root is untouched, before and after -------------------------------
src_manifest () { find "$SRC" -name '*.parquet' -o -name '*.pt' | sort | xargs sha256sum; }
src_manifest > "$ROOT/src_sha256.before.txt"

# --- seed the fresh root with the 10 landed lanes + the registry they are fenced against ------------
cp -p "$SRC/experiment_registry.yaml" "$REGISTRY_PATH"
for cfg in condition_gated expression_only untyped_gnn typed_static; do
  mkdir -p "$ROOT/$cfg"
  cp -p "$SRC/$cfg"/[0-4].parquet "$ROOT/$cfg/" 2>/dev/null   # -p: keep mtimes, the staleness check reads them
done
echo "[n5] seeded fresh root with: $(ls "$ROOT"/*/[0-4].parquet 2>/dev/null | wc -l) landed lanes"

echo "[n5] $(date) START — root=$ROOT splits=$SPLITS epochs=$EPOCHS batch=$BATCH lambda_graph=0"

run_lane () {  # $1=gpu, then (arm seed) pairs run SEQUENTIALLY on that card
  local gpu=$1; shift
  while [ $# -ge 2 ]; do
    local arm=$1 seed=$2; shift 2
    local tag="${arm}_s${seed}"
    echo "[n5] $tag gpu=$gpu START $(date)"
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
      --only "$arm" --seed "$seed" --epochs $EPOCHS --batch-size $BATCH --device cuda \
      --lambda-graph 0 > "$LOG/${tag}.log" 2>&1
    echo "[n5] $tag gpu=$gpu exit=$? $(date)"
  done
}

# cond_gated seeds 2 and 4 are THE deliverable (h1 at n=5). typed_static seeds 0-3 replicate the one
# contrast that survives correction (h2a: static graph worse) on the harder split. untyped_gnn is filler.
run_lane "${GPUS[0]}" condition_gated 2  typed_static 2 &
run_lane "${GPUS[1]}" condition_gated 4  typed_static 3 &
run_lane "${GPUS[2]}" typed_static 0     untyped_gnn 2  untyped_gnn 4 &
run_lane "${GPUS[3]}" typed_static 1     untyped_gnn 3 &
wait

src_manifest > "$ROOT/src_sha256.after.txt"
if diff -q "$ROOT/src_sha256.before.txt" "$ROOT/src_sha256.after.txt" >/dev/null; then
  echo "[n5] source root VERIFIED byte-identical"
else
  echo "[n5] *** SOURCE ROOT CHANGED — investigate before reading any result ***"
  diff "$ROOT/src_sha256.before.txt" "$ROOT/src_sha256.after.txt"
fi

echo "[n5] $(date) DONE. Gate health per graph lane (collapse <=1e-3 is an undecidable experiment):"
.venv/bin/python - <<'PY'
import json, pathlib
for h in sorted(pathlib.Path("data/results/screening_c075c15_n5").glob("*/[0-4]/logs/stage_a_history.json")):
    d = json.loads(h.read_text())
    arm, seed = h.parts[-4], h.parts[-3]
    # `if not g: continue` is NOT enough: untyped_gnn returns edge_gates=None by design, so its
    # history has the KEY with a None VALUE. That list is truthy, and the f-string then dies with
    # "unsupported format string passed to NoneType.__format__", aborting the loop for every lane
    # after it. On 2026-08-03 that killed this report right after the graph arms had printed.
    g = [e["train"]["gate_mean"] for e in d if e.get("train", {}).get("gate_mean") is not None]
    if not g:
        print(f"  {arm}/s{seed}: {len(d)} epochs, no gate (arm has no learnable edge gate)")
        continue
    print(f"  {arm}/s{seed}: {len(d)} epochs, gate {g[0]:.4f} -> {g[-1]:.4f} "
          f"({'ALIVE' if g[-1] > 1e-3 else 'COLLAPSED'})")
PY
echo "[n5] next: SCREENING_ROOT=$ROOT REGISTRY_PATH=$REGISTRY_PATH PYTHONPATH=src \\"
echo "             .venv/bin/python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3,4"
