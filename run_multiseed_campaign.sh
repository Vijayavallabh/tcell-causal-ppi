#!/bin/bash
# feat-011 5-seed robustness campaign: retrain the §10.6 NEURAL family on seeds 1-4 (seed 0 is on disk)
# on the SAME frozen full fold (blocked_target_ood, 21,262 train / 4,400 val — REUSED from
# BLOCKED_SPLIT_PATH, never redrawn), one A100 per seed. `--seed s` reseeds init (seeded_init) + data
# order (Trainer seed) but loads the frozen split by name, so the fold is identical across seeds.
#
# Each seed's four --only lanes run SEQUENTIALLY on its GPU. A --only lane writes only its per-(config,
# seed) parquet + predictions + registry row; it does NOT write summary.json, so seed 0's summary.json
# and promoted.json (the frozen H1) are untouched. A lane that OOMs/crashes is logged and the seed's
# remaining lanes continue; the aggregator marks it missing and shrinks n loudly.
#
# network_propagation is seed-independent (deterministic topology diffusion, no trained weights) and is
# NOT re-run — the paired H2a/H2b/promotion contrasts use only the four neural configs.
#
# Module 0 (DESTRUCTIVE) is NOT run. The SEALED challenge split is NOT touched (every lane scores val;
# nothing here calls evaluation/sealed_eval.py).
#
# Budget: 17.73 GPU-h/seed (0.36 expr + 2.31 untyped + 6.89 typed + 8.17 cond, from seed-0 promoted.json)
#         x 4 seeds = 70.9 GPU-h; one-seed-per-GPU is perfectly balanced => ~17.7 h wall on 4 A100s.
#
# Run unattended from the repo root, then aggregate:
#   nohup ./run_multiseed_campaign.sh > data/logs/robust.nohup.log 2>&1 &
#   PYTHONPATH=src uv run python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3,4
cd "$(dirname "$0")" || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=src
export SUBGRAPH_CACHE_SIZE=9000
LOG=data/logs; mkdir -p "$LOG"
COMMON="--epochs 20 --batch-size 8 --device cuda"
CONFIGS="expression_only untyped_gnn typed_static condition_gated"  # cheap -> expensive (early signal)
echo "[robust] $(date) START — seeds 1-4, one A100 each; $COMMON; cache=$SUBGRAPH_CACHE_SIZE"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

run_seed () {  # $1=gpu  $2=seed
  local gpu=$1 seed=$2
  for cfg in $CONFIGS; do
    echo "[robust] seed=$seed gpu=$gpu -> $cfg $(date)"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -u -m tcell_pipeline.screening.run_screening \
      --only "$cfg" --seed "$seed" $COMMON > "$LOG/robust_s${seed}_${cfg}.log" 2>&1
    echo "[robust] seed=$seed $cfg exit=$?"
  done
}

run_seed 0 1 & q1=$!
run_seed 1 2 & q2=$!
run_seed 2 3 & q3=$!
run_seed 4 4 & q4=$!
wait $q1; echo "[robust] seed1 lanes done"
wait $q2; echo "[robust] seed2 lanes done"
wait $q3; echo "[robust] seed3 lanes done"
wait $q4; echo "[robust] seed4 lanes done"

echo "[robust] aggregating seeds 0-4 (paired H2a/H2b + promotion margin, CI on systema)"
PYTHONPATH=src uv run python -u -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3,4
echo "[robust] $(date) DONE — data/results/screening/robustness_5seed.{json,md}; promoted.json untouched"
