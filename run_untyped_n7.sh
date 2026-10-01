#!/bin/bash
# Take the REFERENCE screen's untyped-graph contrast from n=5 to n=7.
#
# WHY. promotion_margin (untyped_gnn - expression_only) on the frozen fold is +0.0045, 95% CI
# [+0.0011,+0.0079], all five seeds positive, Holm 0.042 -- but Bonferroni 0.083, so it fails this
# project's both-corrections rule by that one test. It is also the reference-dataset anchor for the
# 5/5 cross-dataset pattern that flipped candidate cause C from Refuted to Survives, and that pattern
# is currently POST-HOC. Two more paired seeds are the cheapest thing that could move it off post-hoc
# footing on the dataset the paper is actually about. If it then clears both corrections, the paper's
# central claim is confirmatory on the reference screen rather than inferred from a pooled estimate.
#
# BOTH ARMS ARE RUN AT BOTH NEW SEEDS. The contrast is PAIRED per seed, so an untyped_gnn seed with no
# matching expression_only seed adds nothing and would silently drop from the aggregation.
#
#   setsid nohup ./run_untyped_n7.sh > data/logs/untyped_n7.nohup.log 2>&1 &
#   kill with `kill -TERM -<PGID>` (the minus: the whole process GROUP)
#
# RAIL 2. data/results/screening is READ-ONLY. Everything lands in a FRESH root seeded with copies of
# the ten landed lanes; the source is sha256-manifested before and after and the diff is printed.
set -u
cd "$(dirname "$0")" || exit 1

SRC=data/results/screening                 # READ-ONLY reference root
ROOT=data/results/screening_untyped_n7     # FRESH
LOG=data/logs/untyped_n7
EPOCHS=20
BATCH=8
NEW_SEEDS="5 6"
ARMS="expression_only untyped_gnn"

export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=8000 OMP_NUM_THREADS=4
# Mandatory on this box: a stray 535.309.01 driver tree on the library path against a 580.173.02
# kernel module makes every cuda lane die in <60s inside nvmlInit_v2_. Preload the correct library.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export SPLITS_ROOT=data/splits             # the frozen fold, read-only
export SCREENING_ROOT=$ROOT
export PREDICTIONS_ROOT=$ROOT/predictions
export REGISTRY_PATH=$ROOT/experiment_registry.yaml

for ro in data/results/screening data/results/screening_lambda0 data/splits; do
  [ "$ROOT" = "$ro" ] && { echo "REFUSING: \$ROOT is the read-only root $ro"; exit 2; }
done
mkdir -p "$ROOT" "$LOG"

src_manifest () { find "$SRC" -name '*.parquet' -o -name '*.pt' | sort | xargs sha256sum; }
src_manifest > "$ROOT/src_sha256.before.txt"

cp -p "$SRC/experiment_registry.yaml" "$REGISTRY_PATH" 2>/dev/null
for cfg in $ARMS condition_gated typed_static; do
  mkdir -p "$ROOT/$cfg"
  cp -p "$SRC/$cfg"/[0-4].parquet "$ROOT/$cfg/" 2>/dev/null
done
echo "[n7] seeded fresh root with $(ls "$ROOT"/*/[0-4].parquet 2>/dev/null | wc -l) landed lanes"

for g in 0 1 2 3; do
  name=$(CUDA_VISIBLE_DEVICES=$g .venv/bin/python -c \
    "import torch;print(torch.cuda.get_device_properties(0).name)" 2>/dev/null)
  case "$name" in *A100*) echo "[n7] card $g = $name" ;;
    *) echo "REFUSING: CUDA_VISIBLE_DEVICES=$g is '$name', not an A100"; exit 2 ;; esac
done

echo "[n7] $(date) START — root=$ROOT splits=$SPLITS_ROOT seeds='$NEW_SEEDS' arms='$ARMS'"

lane () {  # $1=gpu $2=arm $3=seed
  local tag="${2}_s${3}"
  [ -f "$ROOT/$2/$3.parquet" ] && { echo "[n7] SKIP $tag (landed)"; return; }
  echo "[n7] $tag gpu=$1 START $(date)"
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
    --only "$2" --seed "$3" --epochs $EPOCHS --batch-size $BATCH --device cuda \
    --lambda-graph 0 > "$LOG/${tag}.log" 2>&1
  echo "[n7] $tag gpu=$1 exit=$? $(date)"
}

lane 0 expression_only 5 &
lane 1 untyped_gnn     5 &
lane 2 expression_only 6 &
lane 3 untyped_gnn     6 &
wait

src_manifest > "$ROOT/src_sha256.after.txt"
if diff -q "$ROOT/src_sha256.before.txt" "$ROOT/src_sha256.after.txt" >/dev/null; then
  echo "[n7] source root VERIFIED byte-identical"
else
  echo "[n7] *** SOURCE ROOT CHANGED — investigate before reading any result ***"
  diff "$ROOT/src_sha256.before.txt" "$ROOT/src_sha256.after.txt"
fi

echo "[n7] $(date) DONE. Re-aggregating at n=7:"
SCREENING_ROOT=$ROOT REGISTRY_PATH=$REGISTRY_PATH PYTHONPATH=src \
  .venv/bin/python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3,4,5,6 2>&1 \
  | grep -E "promotion_margin|h1_vs_no_graph|h2a|INCOMPLETE|UNBALANCED"
