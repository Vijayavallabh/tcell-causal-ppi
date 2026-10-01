#!/bin/bash
# Per-dataset environment for the replication chain. Sourcing this is MANDATORY before any stage:
# INTERMEDIATE_ROOT gives per-dataset isolation (de_obs, de_var, de_layers, program basis all derive
# from it), but that ALSO redirects the feature stores into an empty sandbox. If PLM/PINNACLE are not
# explicitly pointed back at the extended stores, every target silently gets a zero vector and the
# graph arm trains on nothing - the manufactured-null hazard, arriving through path isolation.
#   usage: source run_replication_stage.sh <dataset> <pinnacle-context-slug|none>
set -u
DS=$1; CTX=${2:-none}
export PYTHONPATH=src OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export INTERMEDIATE_ROOT="data/intermediate/replication/$DS"
export DE_STATS_PATH="data/intermediate/replication/$DS.DE_stats_v2.h5ad"
export ID_MAPPING_PATH="data/intermediate/replication/id_mapping_extended.parquet"
export PLM_EMBEDDINGS_PATH="data/intermediate/replication/plm_embeddings_extended.parquet"
if [ "$CTX" = "none" ]; then
  export PINNACLE_EMBEDDINGS_PATH="data/intermediate/replication/pinnacle_EMPTY.parquet"
else
  export PINNACLE_EMBEDDINGS_PATH="data/intermediate/replication/pinnacle_${CTX}.parquet"
fi
read -r DE_N_OBS DE_N_VARS <<< "$(.venv/bin/python - "$DE_STATS_PATH" <<'PY'
import sys, anndata as ad, warnings; warnings.filterwarnings("ignore")
a = ad.read_h5ad(sys.argv[1], backed="r"); print(a.n_obs, a.n_vars); a.file.close()
PY
)"
export DE_N_OBS DE_N_VARS
# CONDITIONS is the model's condition vocabulary. Its default is the REFERENCE screen's
# Rest,Stim8hr,Stim48hr; a replication dataset's own labels are absent from it and training dies on the
# first batch with a bare KeyError naming the condition. Read them from the DE build's provenance so
# the vocabulary always matches the matrix that was actually built.
CONDITIONS=$(.venv/bin/python -c "import json,sys;print(','.join(json.load(open(sys.argv[1]))['conditions']))" \
  "data/intermediate/replication/$DS.DE_stats_v2.provenance.json")
export CONDITIONS
# SPLITS_ROOT must be redirected too: it is NOT derived from INTERMEDIATE_ROOT, so without this the
# splits stage would write into data/splits/ and destroy the frozen reference fold (rail 2, read-only).
export SPLITS_ROOT="data/intermediate/replication/$DS/splits"
mkdir -p "$INTERMEDIATE_ROOT"
# Stage 2 writes the perturbation table beside the DE matrix; the rest of the pipeline expects it
# inside the root. Link rather than copy so there is one file and no chance of the two diverging.
PT="../$DS.perturbation_condition.parquet"
[ -e "$INTERMEDIATE_ROOT/perturbation_condition.parquet" ] || \
  ln -sf "$PT" "$INTERMEDIATE_ROOT/perturbation_condition.parquet" 2>/dev/null || true
echo "[env] $DS: DE ${DE_N_OBS}x${DE_N_VARS} | root=$INTERMEDIATE_ROOT | pinnacle=$(basename "$PINNACLE_EMBEDDINGS_PATH")"
