#!/bin/bash
# The multi-dataset replication campaign. Four datasets, four seeds, arms fixed per prereg Amendment 3.2.
#
#   setsid nohup ./run_replication_campaign.sh > data/logs/repl/campaign.log 2>&1 &
#   kill with `kill -TERM -<PGID>` (the minus matters: the whole process GROUP, or each lane's loop
#   launches its next job before you reach the shell)
#
# ARMS. FrangiehIzar2021_RNA has 3 conditions, so it is the only dataset where the condition gate has
# anything to gate and h1 is testable: it runs all four arms. Every other dataset is single-condition,
# where condition_gated is arithmetically identical to typed_static - running it would produce a number
# that says nothing about gating - so those run three arms and their primary contrast is h2a.
#
# Nothing under data/results/screening*, data/splits/ or promoted.json is read or written.
set -u
cd "$(dirname "$0")" || exit 1
LOG=data/logs/repl
mkdir -p "$LOG"
SEEDS="0 1 2 3"
GPUS=(0 1 2 3)

# dataset:pinnacle-ctx:arms   (ctx per prereg 3.3; no neuron context exists upstream, hence none)
DATASETS=(
  "FrangiehIzar2021_RNA:melanocyte:expression_only typed_static condition_gated untyped_gnn"
  "ReplogleWeissman2022_rpe1:retinal_pigment_epithelial_cell:expression_only typed_static untyped_gnn"
  "ReplogleWeissman2022_K562_essential:none:expression_only typed_static untyped_gnn"
  "TianKampmann2021_CRISPRi:none:expression_only typed_static untyped_gnn"
)

# Build the job list, then deal it round-robin. Dealing by job rather than by dataset keeps all four
# cards busy to the end: dataset sizes differ 10x, so a per-dataset split would idle three cards while
# the largest one finishes alone.
JOBS=()
for entry in "${DATASETS[@]}"; do
  IFS=: read -r ds ctx arms <<< "$entry"
  for arm in $arms; do for s in $SEEDS; do JOBS+=("$ds|$ctx|$arm|$s"); done; done
done
echo "[campaign] $(date) START — ${#JOBS[@]} lanes over ${#GPUS[@]} cards, 20 epochs, lambda_graph=0"

run_card () {
  local gpu=$1; shift
  for job in "$@"; do
    IFS='|' read -r ds ctx arm s <<< "$job"
    # Idempotent: a landed lane is skipped, so the campaign can be killed and relaunched.
    if [ -f "data/results/replication/$ds/$arm/$s.parquet" ]; then
      echo "[campaign] SKIP ${ds}_${arm}_s${s} (already landed)"; continue
    fi
    ./run_replication_lane.sh "$gpu" "$ds" "$ctx" "$arm" "$s"
  done
}

for i in "${!GPUS[@]}"; do
  card_jobs=()
  for j in "${!JOBS[@]}"; do [ $((j % ${#GPUS[@]})) -eq "$i" ] && card_jobs+=("${JOBS[$j]}"); done
  run_card "${GPUS[$i]}" "${card_jobs[@]}" &
done
wait

echo "[campaign] $(date) DONE. Gate health per graph lane (collapse <=1e-3 is undecidable, not a result):"
.venv/bin/python - <<'PY'
import json, pathlib
for h in sorted(pathlib.Path("data/results/replication").glob("*/*/[0-9]/logs/stage_a_history.json")):
    d = json.loads(h.read_text())
    ds, arm, seed = h.parts[-5], h.parts[-4], h.parts[-3]
    # untyped_gnn stores the KEY with a None VALUE, and a list of Nones is truthy - guard on the value.
    g = [e["train"]["gate_mean"] for e in d if e.get("train", {}).get("gate_mean") is not None]
    if not g:
        print(f"  {ds}/{arm}/s{seed}: {len(d)} epochs, no learnable edge gate")
        continue
    print(f"  {ds}/{arm}/s{seed}: {len(d)} epochs, gate {g[0]:.4f} -> {g[-1]:.4f} "
          f"({'ALIVE' if g[-1] > 1e-3 else 'COLLAPSED'})")
PY
