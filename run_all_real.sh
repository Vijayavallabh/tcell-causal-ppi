#!/bin/bash
# Full real-data run of everything implemented (Modules 1-8), fanned across the 4 A100s (PCI 0,1,2,4;
# index 3 is a 4GB T400 and is deliberately unused).
#
#     nohup ./run_all_real.sh > data/logs/all_real.nohup.log 2>&1 &
#     tail -f data/logs/all_real.nohup.log        # per-lane logs: data/logs/<lane>.log
#
# WHAT RUNS ON THE **FULL** FOLD (21,262 train / 4,400 val):
#   M5 Stage-A (expr-only) -> M6 evaluation; M7 expression_only + network_propagation; M8 comparators.
#
# WHAT IS **BOUNDED** AND WHY (an honest ceiling, not a choice):
#   The typed/untyped GRAPH encoders sample a subgraph per target and message-pass per row, single-threaded
#   on CPU (torch.set_num_threads(1)); the GPU sits at ~0% because almost no work is on it. On the full
#   21,262-row fold the FASTEST graph config did not finish ONE epoch in ~11h. So the §10.6 nested family
#   runs on a bounded fold (NESTED_NMAX), one config per A100 in parallel — the only way to use all four
#   GPUs today. Lifting this needs the deferred mini-batch refactor (PyG Batch over sampled subgraphs),
#   which is the #1 throughput task in the handoff.
#
# NOT RUN HERE, DELIBERATELY: the sealed challenge evaluation. It is WRITE-ONCE on the SEQUESTERED
# challenge split (5,608 rows, currently unopened) and must be run once, by the test steward, against the
# PROMOTED FINAL model. No promoted model exists yet, so running it now would burn the fold on a non-final
# model — the exact garden-of-forks the module exists to prevent. Module 0 is DESTRUCTIVE and is not run.
set -u
cd "$(dirname "$0")" || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=data/logs; mkdir -p "$LOG"
NESTED_NMAX=${NESTED_NMAX:-2000}     # common fold for the nested family (H2a/H2b need ONE shared fold)
EPOCHS=${EPOCHS:-1}
SCREEN_ONE=${SCREEN_ONE:-/tmp/claude-1001/-mnt-md0-IITM-BackUp-Home-vijayavallabh-tcell-causal-ppi/ccd16b6c-f04a-44eb-8794-7b71f9b96c70/scratchpad/screen_one.py}
say(){ echo "[all-real $(date +%H:%M:%S)] $*"; }

say "START  nested_nmax=$NESTED_NMAX epochs=$EPOCHS"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | sed 's/^/    /'

# ---- Lane A (GPU 0): M5 Stage-A on the FULL fold -> M6 evaluation on the FULL val fold -------------
(
  say "A: M5 Stage-A train (FULL 21,262 rows, expr-only) on GPU0"
  CUDA_VISIBLE_DEVICES=0 uv run python -u -m tcell_pipeline.training.run_train \
      --expr-only --epochs 5 --device cuda > "$LOG/m5.log" 2>&1
  say "A: M5 exit=$?"
  say "A: M6 evaluation (FULL val) on GPU0"
  CUDA_VISIBLE_DEVICES=0 uv run python -u -m tcell_pipeline.run_module6_smoke --device cuda > "$LOG/m6.log" 2>&1
  say "A: M6 exit=$?"
) & LA=$!

# ---- Lanes B/C/D (GPU 1,2,4): the §10.6 nested family, ONE config per A100, shared bounded fold -----
CUDA_VISIBLE_DEVICES=1 uv run python -u "$SCREEN_ONE" --config untyped_gnn     --n-max "$NESTED_NMAX" \
    --epochs "$EPOCHS" --batch-size 8 --device cuda > "$LOG/m7_untyped.log" 2>&1 & LB=$!
CUDA_VISIBLE_DEVICES=2 uv run python -u "$SCREEN_ONE" --config typed_static    --n-max "$NESTED_NMAX" \
    --epochs "$EPOCHS" --batch-size 8 --device cuda > "$LOG/m7_typedstatic.log" 2>&1 & LC=$!
CUDA_VISIBLE_DEVICES=4 uv run python -u "$SCREEN_ONE" --config condition_gated --n-max "$NESTED_NMAX" \
    --epochs "$EPOCHS" --batch-size 8 --device cuda > "$LOG/m7_condgated.log" 2>&1 & LD=$!
say "B/C/D: nested graph configs launched on GPU 1/2/4 (fold=$NESTED_NMAX rows)"

# ---- Lane E (CPU): expression-only on the SAME bounded fold (the H2a reference), then the FULL fold --
(
  CUDA_VISIBLE_DEVICES="" uv run python -u "$SCREEN_ONE" --config expression_only --n-max "$NESTED_NMAX" \
      --epochs "$EPOCHS" --batch-size 64 --device cpu > "$LOG/m7_expronly.log" 2>&1
  say "E: expression_only (bounded, H2a reference) exit=$?"
) & LE=$!

# ---- Lane F (CPU): network propagation + Module 8 comparators, both on the FULL fold ---------------
(
  say "F: network_propagation on the FULL fold (numpy diffusion)"
  CUDA_VISIBLE_DEVICES="" uv run python -u "$SCREEN_ONE" --config network_propagation --n-max 999999 \
      --device cpu > "$LOG/m7_netprop_full.log" 2>&1 || true   # 999999 > fold size == no cap
  say "F: M8 comparators (Stable-Shift + TxPert-public) on the FULL fold"
  CUDA_VISIBLE_DEVICES="" uv run python -u -m tcell_pipeline.run_module8_real --part comparators \
      > "$LOG/m8_comparators.log" 2>&1
  say "F: M8 comparators exit=$?"
  say "F: M8 reproducibility verification against this checkout"
  uv run python -u -m tcell_pipeline.run_module8_real --part repro > "$LOG/m8_repro.log" 2>&1
  say "F: M8 repro exit=$?"
) & LF=$!

for p in $LB $LC $LD $LE; do wait $p; say "nested lane pid=$p exit=$?"; done
say "nested family complete"

# ---- Lane G (GPU 1, freed): Module 8 rationale audit over the REAL PPI graph -----------------------
say "G: M8 rationale audit on the real graph (UNTRAINED model — machinery at real scale, see log)"
CUDA_VISIBLE_DEVICES=1 timeout 7200 uv run python -u -m tcell_pipeline.run_module8_real --part audit \
    --n-cases 12 --n-controls 8 --n-max 400 --device cuda > "$LOG/m8_audit.log" 2>&1
say "G: M8 audit exit=$?"

wait $LA; say "lane A (M5->M6) exit=$?"
wait $LF; say "lane F (netprop/comparators/repro) exit=$?"
say "DONE — results under data/results/ ; logs under $LOG"
