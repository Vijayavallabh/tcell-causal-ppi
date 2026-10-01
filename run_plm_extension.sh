#!/bin/bash
# L1-PRE step 2: ESM-2 vectors for the replication accessions the frozen store lacks.
# Waits for the ablation to release a card, then extends the store BY COPY.
# The frozen store is a read-only input to every result in the project; it is sha256-checked here.
set -u
cd "$(dirname "$0")" || exit 1
FROZEN=data/intermediate/plm_embeddings.parquet
EXT=data/intermediate/replication/plm_embeddings_extended.parquet
BEFORE=$(sha256sum "$FROZEN" | cut -d' ' -f1)
until [ "$(pgrep -cf 'screening.run_screening')" = "0" ]; do sleep 60; done
echo "[plm-ext] cards free $(date); seeding extended store from frozen"
mkdir -p "$(dirname "$EXT")"
[ -f "$EXT" ] || cp -p "$FROZEN" "$EXT"
export PYTHONPATH=src OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export PLM_UNIPROT_SOURCE=data/intermediate/replication/id_mapping_extended.parquet
export PLM_EMBEDDINGS_PATH=$EXT
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m tcell_pipeline.embeddings_plm
echo "[plm-ext] exit=$? $(date)"
AFTER=$(sha256sum "$FROZEN" | cut -d' ' -f1)
[ "$BEFORE" = "$AFTER" ] && echo "[plm-ext] frozen store VERIFIED byte-identical" \
                         || echo "[plm-ext] *** FROZEN STORE CHANGED - INVESTIGATE ***"
