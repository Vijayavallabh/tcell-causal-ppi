#!/bin/bash
# AAAI stage-2: improve on untyped_gnn. wgcn (full graph + edge weights), gcn_p (fair GCN @thr0.4),
# gat (learned edge-attention @thr0.4, heads=2). Same inner holdout, seed 0, 5 epochs. VAL UNTOUCHED.
# LAUNCH WITH setsid; kill with `kill -TERM -<PGID>` (whole group).
set -u
cd "$(dirname "$0")" || exit 1
OUT=data/results/arch_search; LOG=data/logs; EPOCHS=5
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=src SUBGRAPH_CACHE_SIZE=9000 OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRAPH_ENCODE_CHUNK=2   # bound in-flight subgraphs: GAT attention OOMs the card otherwise
mkdir -p "$OUT" "$LOG"
for g in 0 2 4; do
  n=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$g"); case "$n" in *A100*) ;; *) echo "REFUSE g$g=$n"; exit 2;; esac
done
echo "[s2] $(date) START — wgcn/gcn_p/gat on inner holdout, ${EPOCHS} ep"
lane () { local gpu=$1 cell=$2 name=$3
  echo "[s2] cell=$cell gpu=$gpu START $(date)"
  CUDA_VISIBLE_DEVICES=$gpu uv run python -u -m tcell_pipeline.arch_search --stage2 --cells "$cell" \
    --epochs $EPOCHS --device cuda --out "$OUT" > "$LOG/arch_s2_${name}.log" 2>&1
  echo "[s2] cell=$cell gpu=$gpu exit=$? $(date)"; }
lane 0 0 wgcn &
lane 2 1 gcnp &
lane 4 2 gat &
wait
echo "[s2] $(date) DONE"
