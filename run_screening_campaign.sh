#!/bin/bash
# feat-011 FULL-FOLD screening campaign: the §10.6 nested confirmatory family trained to a 20-epoch
# budget on ONE shared full fold (21,262 train / 4,400 val), reporting H2a/H2b on the primary endpoint.
# Run unattended from the repo root:
#
#     nohup ./run_screening_campaign.sh > data/logs/campaign.nohup.log 2>&1 &
#
# Monitor:  tail -f data/logs/campaign.nohup.log   (per-lane logs in data/logs/campaign_*.log)
#
# Module 0 (data marts) is DESTRUCTIVE and is NOT run here. The SEALED challenge split is NOT touched:
# every lane scores `val`, and nothing here calls evaluation/sealed_eval.py.
#
# Why these settings (all measured 2026-07-17, real fold, A100 — see the Module 7 spec):
#   --epochs 20   Trainer early-stops on val total with patience=10, but its min-delta is 1e-6 against
#                 ~1e-3/epoch improvements, so it will almost certainly NOT fire: this budget IS the
#                 wall clock. If val is still descending at 20 the run is NOT converged -- say so.
#   --batch-size 8  bs=32 buys ~5% for 3x the memory and has OOMed 80 GB on dense hub subgraphs.
#   SUBGRAPH_CACHE_SIZE=9000  covers a lane's 8,541 unique targets (7,079 train + 1,462 val, disjoint)
#                 at ~4.5 MB each => ~38.6 GB RSS per graph lane, ~116 GB for the three (440 GB free).
#                 Sampling is ~85 of condition_gated's ~183 ms/row because DONOR_INVARIANCE re-forwards
#                 each batch 3x; caching it removes ~1.8x end-to-end.
#   One config per GPU: fanned ~12.5 h vs ~25 h sequential. Lanes share ONE registry, which is safe
#                 only because register_run/log_run now hold an exclusive advisory lock.
# GPU 2 was held by another tenant at launch and the T400 at index 3 is unusable (4 GB), so the three
# A100s at PCI 0/1/4 take the graph lanes and the CPU-only members ride along.
cd "$(dirname "$0")" || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=src
export SUBGRAPH_CACHE_SIZE=9000
LOG=data/logs; mkdir -p "$LOG"
EPOCHS=20
COMMON="--epochs $EPOCHS --batch-size 8"
echo "[campaign] $(date) START — nested family, full fold, ${EPOCHS}-epoch budget"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# --- the two heavy typed lanes, one A100 each (est. 12.5 h / 9.8 h) ---
CUDA_VISIBLE_DEVICES=0 uv run python -u -m tcell_pipeline.screening.run_screening \
  --only condition_gated $COMMON --device cuda > "$LOG/campaign_condition_gated.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 uv run python -u -m tcell_pipeline.screening.run_screening \
  --only typed_static    $COMMON --device cuda > "$LOG/campaign_typed_static.log" 2>&1 & p1=$!

# --- the cheap lanes share the third A100, sequentially (est. 2.3 h + 0.35 h) ---
(
  CUDA_VISIBLE_DEVICES=4 uv run python -u -m tcell_pipeline.screening.run_screening \
    --only untyped_gnn     $COMMON --device cuda > "$LOG/campaign_untyped_gnn.log" 2>&1
  echo "[campaign] untyped_gnn exit=$?"
  CUDA_VISIBLE_DEVICES=4 uv run python -u -m tcell_pipeline.screening.run_screening \
    --only expression_only $COMMON --device cuda > "$LOG/campaign_expression_only.log" 2>&1
  echo "[campaign] expression_only exit=$?"
  # non-neural topology reference: CPU-only, no Trainer, minutes
  uv run python -u -m tcell_pipeline.screening.run_screening \
    --only network_propagation $COMMON --device cpu > "$LOG/campaign_network_prop.log" 2>&1
  echo "[campaign] network_propagation exit=$?"
) & p2=$!

wait $p0; echo "[campaign] condition_gated exit=$?"
wait $p1; echo "[campaign] typed_static exit=$?"
wait $p2

# --- recombine the lanes -> summary.json + H2a/H2b (non-zero if any lane never landed) ---
echo "[campaign] merging lanes"
uv run python -u -m tcell_pipeline.screening.run_screening --merge
echo "[campaign] merge exit=$?"
echo "[campaign] $(date) DONE — data/results/screening/summary.json, experiment_registry.yaml"
