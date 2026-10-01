#!/bin/bash
# Balance the n=7 root: typed_static and condition_gated at seeds 5 and 6.
#
# WHY. The n=7 promotion_margin contrast is already clean - both of its arms (untyped_gnn,
# expression_only) have all seven seeds. But typed_static and condition_gated stop at seed 4, so the
# per-config RANKING in that report compares different seed bases and multiseed flags it UNBALANCED.
# The headline number is unaffected; the ranking is. These four lanes make every arm n=7 so the whole
# report is on one basis, and they also take h1 (condition_gated - expression_only) and h2a
# (typed_static - expression_only) to n=7 on the frozen fold, which no other run has done.
#
#   setsid nohup ./run_balance_n7.sh > data/logs/balance_n7.nohup.log 2>&1 &
#
# RAIL 2: writes only into the existing fresh root; data/results/screening and data/splits are read
# only. The source manifest is re-verified at the end.
set -u
cd "$(dirname "$0")" || exit 1
ROOT=data/results/screening_untyped_n7
LOG=data/logs/balance_n7
export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=8000 OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export SPLITS_ROOT=data/splits
export SCREENING_ROOT=$ROOT PREDICTIONS_ROOT=$ROOT/predictions
export REGISTRY_PATH=$ROOT/experiment_registry.yaml
mkdir -p "$LOG"

lane () {
  local tag="${2}_s${3}"
  [ -f "$ROOT/$2/$3.parquet" ] && { echo "[bal] SKIP $tag (landed)"; return; }
  echo "[bal] $tag gpu=$1 START $(date)"
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
    --only "$2" --seed "$3" --epochs 20 --batch-size 8 --device cuda \
    --lambda-graph 0 > "$LOG/${tag}.log" 2>&1
  echo "[bal] $tag gpu=$1 exit=$? $(date)"
}

echo "[bal] $(date) START — bringing typed_static and condition_gated to n=7 in $ROOT"
lane 0 typed_static    5 &
lane 1 typed_static    6 &
lane 2 condition_gated 5 &
lane 3 condition_gated 6 &
wait

echo "[bal] $(date) DONE. Gate health (collapse <=1e-3 is undecidable, not a result):"
.venv/bin/python - <<'PY'
import json, pathlib
for h in sorted(pathlib.Path("data/results/screening_untyped_n7").glob("*/[56]/logs/stage_a_history.json")):
    d = json.loads(h.read_text())
    g = [e["train"]["gate_mean"] for e in d if e.get("train", {}).get("gate_mean") is not None]
    if not g:
        print(f"  {h.parts[-4]}/s{h.parts[-3]}: {len(d)} epochs, no learnable edge gate"); continue
    print(f"  {h.parts[-4]}/s{h.parts[-3]}: {len(d)} epochs, gate {g[0]:.4f} -> {g[-1]:.4f} "
          f"({'ALIVE' if g[-1] > 1e-3 else 'COLLAPSED'})")
PY
echo "[bal] then: .venv/bin/python merge_registry_n7.py && re-aggregate at --seeds 0,1,2,3,4,5,6"
