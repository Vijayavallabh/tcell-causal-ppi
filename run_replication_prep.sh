#!/bin/bash
# Stages 2-6 for one replication dataset: perturbation table -> de_extraction -> blocked split ->
# program basis. CPU only, no GPU, idempotent (each stage skips if its artifact exists).
#
#   ./run_replication_prep.sh <dataset> <pinnacle-ctx|none> <K>
#
# K is fixed per dataset by prereg Amendment 3.1 (K=128 where train rows >= 256, else the largest
# power of two <= train/2). It is an ARGUMENT rather than a default so that a K deviation can never
# happen silently - the caller has to name it, and it lands in this script's log.
set -u
cd "$(dirname "$0")" || exit 1
DS=$1; CTX=${2:-none}; K=${3:-128}

# shellcheck disable=SC1091
source ./run_replication_stage.sh "$DS" "$CTX" || exit 1
R="$INTERMEDIATE_ROOT"

# Refuse to run against the reference root. INTERMEDIATE_ROOT drives de_obs/de_var/de_layers and the
# program basis; pointed at data/intermediate it would overwrite the frozen reference feature store.
case "$R" in data/intermediate/replication/*) ;; *) echo "REFUSING: root '$R' is not under replication/"; exit 2;; esac

step () { echo "[prep:$DS] === $* ==="; }

step "stage 2 perturbation_condition"
if [ ! -e "data/intermediate/replication/$DS.perturbation_condition.parquet" ]; then
  .venv/bin/python -m tcell_pipeline.replication.perturbation_table --dataset "$DS" || exit 1
  ln -sf "../$DS.perturbation_condition.parquet" "$R/perturbation_condition.parquet"
else echo "  already built"; fi

step "stage 3 de_extraction"
# Gate on de_var.parquet, the LAST artifact stage 3 writes, not on zscore.npz, the first. Keying on
# the first makes a run that died mid-stage look complete, and the next stage fails somewhere else.
if [ ! -f "$R/de_var.parquet" ]; then
  .venv/bin/python -m tcell_pipeline.de_extraction || exit 1
else echo "  already built"; fi

step "stage 3b guide-quality scalars"
if [ ! -f "data/intermediate/replication/$DS.guide_quality.json" ]; then
  .venv/bin/python -m tcell_pipeline.replication.guide_quality --dataset "$DS" || exit 1
else echo "  already built"; fi

step "stage 6 blocked target-OOD split"
if [ ! -f "$SPLITS_ROOT/blocked_target_ood.csv" ]; then
  .venv/bin/python -m tcell_pipeline.splits || exit 1
else echo "  already built"; fi

step "stage 4 program basis K=$K"
if [ ! -f "$R/gene_program_loadings.parquet" ]; then
  PROGRAM_DIM=$K .venv/bin/python -m tcell_pipeline.programs.run_program_basis --K "$K" || exit 1
else echo "  already built"; fi

# Feature-coverage gate. Every route to a manufactured null found so far ends the same way: the graph
# arm trains on zero vectors and reports a number anyway. Check it here, once, rather than trusting
# that the env script did its job.
step "coverage gate"
.venv/bin/python - "$DS" <<'PY' || exit 1
import sys, pandas as pd, numpy as np
from tcell_pipeline import config
pc = pd.read_parquet(config.PERTURBATION_CONDITION_PATH)
n = len(pc)
uni = pc["uniprot_id"].astype("string").notna().sum()
plm = pd.read_parquet(config.PLM_EMBEDDINGS_PATH, columns=["uniprot_id"])["uniprot_id"].astype(str)
have = pc["uniprot_id"].astype(str).isin(set(plm)).sum()
pin_p = config.PINNACLE_EMBEDDINGS_PATH
pin = 0
if pin_p.exists():
    pv = pd.read_parquet(pin_p)
    key = "uniprot_id" if "uniprot_id" in pv.columns else pv.columns[0]
    pin = pc["uniprot_id"].astype(str).isin(set(pv[key].astype(str))).sum()
print(f"  rows={n}  uniprot={uni}/{n} ({uni/n:.1%})  ESM-2={have}/{n} ({have/n:.1%})  "
      f"PINNACLE={pin}/{n} ({pin/n:.1%}, ctx={pin_p.name})")
if have / n < 0.5:
    print(f"  *** ABORT: only {have/n:.1%} of rows have an ESM-2 vector. The graph arm would train on "
          f"zeros and still report a number. Fix the feature store before running any lane.")
    sys.exit(1)
PY
echo "[prep:$DS] READY  root=$R  splits=$SPLITS_ROOT  K=$K  pinnacle=$(basename "$PINNACLE_EMBEDDINGS_PATH")"
