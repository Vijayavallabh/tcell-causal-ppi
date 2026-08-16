#!/bin/bash
# A1 — WHY does edge typing hurt? Pre-registered in docs/replication-prereg.md Amendment 4.
#
# The n=7 family settled WHICH component costs the graph its benefit (edge typing, -0.0120 systema,
# 7/7 seeds, survives Bonferroni AND Holm) and not WHY. Two explanations are confounded inside that
# contrast: the relation PARTITION is the wrong inductive bias, or typed message passing simply carries
# 4x the message parameters over the same edges and the damage is capacity. This runs the two arms that
# separate them, paired per seed against the landed typed_static lanes on the frozen fold:
#
#   typed_shared    one _RelMessage tied across all four relations   -> D1 = typed_shared   - typed_static
#   typed_permuted  per-relation weights, relation labels shuffled   -> D2 = typed_permuted - typed_static
#
#   setsid nohup ./run_a1_mechanism.sh > data/logs/a1_mechanism.nohup.log 2>&1 &
#   kill with `kill -TERM -<PGID>` (the minus: the whole process GROUP)
#
# IDEMPOTENT AND RESTARTABLE. Every lane checks for its landed parquet first and takes an atomic claim
# file, so re-running after a crash resumes rather than repeats, and a second copy of this script cannot
# double-start a lane. ARMS is read from the environment, so the typed_permuted half can be added to the
# queue by re-running once that arm exists without disturbing lanes already in flight.
#
# RAIL 2. data/results/screening is READ-ONLY. Everything lands in a FRESH root seeded with copies of the
# landed reference lanes; the source is sha256-manifested before and after and the diff is printed.
set -u
cd "$(dirname "$0")" || exit 1

SRC=data/results/screening                  # READ-ONLY reference root (the frozen fold's landed lanes)
ROOT=data/results/screening_a1              # FRESH
LOG=data/logs/a1
EPOCHS=20
BATCH=8
SEEDS="${SEEDS:-0 1 2 3 4}"                 # n=5, matching the landed reference family (Amendment 4.4)
ARMS="${ARMS:-typed_shared}"                # typed_permuted joins once it is implemented + amended
CARDS="${CARDS:-0 1 2 3}"

# Training configuration is pinned to what the landed typed_static lanes actually used
# (run_multiseed_campaign.sh): 20 epochs, batch 8, cache 9000, and NO --lambda-graph override, i.e. the
# config default 0.01. The penalty is gradient-free on both arms (the gate is pinned to 1.0, so the term
# is a per-batch constant), so it cannot differentiate them; it is matched for exactness, not for effect.
export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=9000 OMP_NUM_THREADS=4
# Mandatory on this box: a stray 535.309.01 driver tree on the library path against a 580.173.02 kernel
# module makes every cuda lane die in <60s inside nvmlInit_v2_. Preload the correct library.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export SPLITS_ROOT=data/splits              # the frozen fold, read-only
export SCREENING_ROOT=$ROOT
export PREDICTIONS_ROOT=$ROOT/predictions
export REGISTRY_PATH=$ROOT/experiment_registry.yaml

for ro in data/results/screening data/results/screening_lambda0 data/splits; do
  [ "$ROOT" = "$ro" ] && { echo "REFUSING: \$ROOT is the read-only root $ro"; exit 2; }
done
mkdir -p "$ROOT" "$LOG"

src_manifest () { find "$SRC" -name '*.parquet' -o -name '*.pt' | sort | xargs sha256sum; }
src_manifest > "$ROOT/src_sha256.before.txt"

# Seed the fresh root with the landed reference lanes so the paired contrasts aggregate in one place.
cp -p "$SRC/experiment_registry.yaml" "$REGISTRY_PATH" 2>/dev/null
for cfg in expression_only untyped_gnn typed_static condition_gated; do
  mkdir -p "$ROOT/$cfg"
  cp -p "$SRC/$cfg"/[0-4].parquet "$ROOT/$cfg/" 2>/dev/null
done
echo "[a1] seeded fresh root with $(ls "$ROOT"/*/[0-4].parquet 2>/dev/null | wc -l) landed reference lanes"

# nvidia-smi ordering is NOT cuda ordering on this box (its index 3 is a T400). Resolve every card
# through torch, and refuse a card that is not an A100 or is already occupied.
for g in $CARDS; do
  read -r name free <<< "$(CUDA_VISIBLE_DEVICES=$g .venv/bin/python -c \
    "import torch; f,_=torch.cuda.mem_get_info(0); \
     print(torch.cuda.get_device_properties(0).name.replace(' ','_'), int(f/2**30))" 2>/dev/null)"
  case "$name" in *A100*) ;; *) echo "REFUSING: CUDA_VISIBLE_DEVICES=$g is '${name:-unreadable}', not an A100"; exit 2 ;; esac
  [ "${free:-0}" -lt 40 ] && { echo "REFUSING: card $g has ${free:-0} GiB free (<40)"; exit 2; }
  echo "[a1] card $g = $name, ${free} GiB free"
done

echo "[a1] $(date) START — root=$ROOT splits=$SPLITS_ROOT arms='$ARMS' seeds='$SEEDS' cards='$CARDS'"

# One job list, shared by every worker. A worker takes the next job whose parquet is absent and whose
# claim file it can create atomically (set -o noclobber), so five lanes spread over four cards without a
# scheduler and a crashed worker's lane is retried rather than lost.
JOBS=""
for arm in $ARMS; do for s in $SEEDS; do JOBS="$JOBS $arm:$s"; done; done

worker () {  # $1 = gpu
  local gpu=$1 job arm s tag claim rc
  for job in $JOBS; do
    arm="${job%%:*}"; s="${job##*:}"; tag="${arm}_s${s}"
    [ -f "$ROOT/$arm/$s.parquet" ] && continue
    mkdir -p "$ROOT/$arm"
    claim="$ROOT/$arm/.$s.claim"
    ( set -o noclobber; echo "$$ gpu=$gpu $(date)" > "$claim" ) 2>/dev/null || continue
    echo "[a1] $tag gpu=$gpu START $(date)"
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
      --only "$arm" --seed "$s" --epochs $EPOCHS --batch-size $BATCH --device cuda \
      > "$LOG/${tag}.log" 2>&1
    rc=$?
    echo "[a1] $tag gpu=$gpu exit=$rc $(date)"
    # Release the claim on failure so a re-run retries the lane; on success the parquet is the record.
    [ $rc -ne 0 ] && rm -f "$claim"
  done
  echo "[a1] worker gpu=$gpu drained $(date)"
}

for g in $CARDS; do worker "$g" & done
wait

src_manifest > "$ROOT/src_sha256.after.txt"
if diff -q "$ROOT/src_sha256.before.txt" "$ROOT/src_sha256.after.txt" >/dev/null; then
  echo "[a1] source root VERIFIED byte-identical"
else
  echo "[a1] *** SOURCE ROOT CHANGED — investigate before reading any result ***"
  diff "$ROOT/src_sha256.before.txt" "$ROOT/src_sha256.after.txt"
fi

echo "[a1] $(date) DONE. Landed:"
for arm in $ARMS; do
  echo "[a1]   $arm: $(ls "$ROOT/$arm"/[0-9].parquet 2>/dev/null | wc -l)/$(echo $SEEDS | wc -w) lanes"
done
echo "[a1] next: .venv/bin/python merge_registry_n7.py   (MANDATORY before aggregating a fresh root)"
echo "[a1]       PYTHONPATH=src .venv/bin/python -m tcell_pipeline.screening.a1_mechanism"
