#!/bin/bash
# A2(a) — the empirical detection floor. Pre-registered in docs/replication-prereg.md Amendment 6.
#
# Each rung is a copy of the real responses with a KNOWN graph-dependent signal added at a known size:
# for every target, delta times the mean response of its PPI neighbours computed from TRAIN ROWS ONLY.
# The smallest delta whose untyped_gnn - expression_only contrast clears both corrections is the
# pipeline's MEASURED sensitivity, replacing the modelled MDE the paper currently hedges with.
#
#   PYTHONPATH=src .venv/bin/python -m tcell_pipeline.screening.inject_signal --ladder \
#       --out data/intermediate/inject          # BUILD THE RUNGS FIRST (CPU, ~20 min, ~11 GB)
#   setsid nohup ./run_a2_ladder.sh > data/logs/a2_ladder.nohup.log 2>&1 &
#
# CARD SELECTION IS BEST-EFFORT. The preflight SKIPS any card with less than 40 GiB free and runs on
# whatever is left, refusing only if nothing is usable. An earlier version exited when any single card
# failed, and a co-tenant on card 2 at 02:06 on 2026-08-18 cost eleven hours of idle time on the other
# three. On a shared box, per-card availability is the normal case, not an error.
#
# IDEMPOTENT: every lane checks for its parquet and takes an atomic claim, so a re-run resumes.
#
# RAIL 1. The injected matrices leave challenge and calibration rows bit-identical and no sealed
# response enters any statistic; see Amendment 6.3. Nothing here reads data/splits except to partition.
# RAIL 2. data/results/screening is untouched; every rung writes its own FRESH results root.
set -u
cd "$(dirname "$0")" || exit 1

INJECT=data/intermediate/inject
ROOT=data/results/a2_ladder
LOG=data/logs/a2_ladder
SEEDS="${SEEDS:-0 1 2 3}"                   # n=4 per arm per rung (rail 5)
ARMS="${ARMS:-expression_only untyped_gnn}" # Amendment 6.5: the pipeline's best graph detector
CARDS="${CARDS:-0 1 2 3}"
EPOCHS=20
BATCH=8

export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=9000 OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export SPLITS_ROOT=data/splits              # the frozen fold; screening only reads it
mkdir -p "$ROOT" "$LOG"

rungs=$(ls -d "$INJECT"/d[0-9]* "$INJECT"/permuted_d[0-9]* 2>/dev/null | sort)
[ -z "$rungs" ] && { echo "REFUSING: no rungs under $INJECT — build them first (see the header)"; exit 2; }
for r in $rungs; do
  [ -f "$r/injection_provenance.json" ] || { echo "REFUSING: $r has no injection_provenance.json"; exit 2; }
  [ -f "$r/de_layers/zscore.npz" ] || { echo "REFUSING: $r has no injected response layer"; exit 2; }
done
echo "[a2] rungs: $(echo $rungs | tr ' ' '\n' | xargs -n1 basename | tr '\n' ' ')"

# SELECT usable cards; do not refuse the whole run because one is busy. The first version of this
# preflight exited when ANY card failed, and a co-tenant holding card 2 at 02:06 cost eleven hours of
# idle A100 time on the other three. A shared box makes per-card availability the normal case, so the
# run proceeds on whatever is usable and says which cards it dropped.
USABLE=""
for g in $CARDS; do
  read -r name free <<< "$(CUDA_VISIBLE_DEVICES=$g .venv/bin/python -c \
    "import torch; f,_=torch.cuda.mem_get_info(0); \
     print(torch.cuda.get_device_properties(0).name.replace(' ','_'), int(f/2**30))" 2>/dev/null)"
  case "$name" in
    *A100*)
      if [ "${free:-0}" -lt 40 ]; then
        echo "[a2] SKIP card $g: ${free:-0} GiB free (<40), someone else is on it"
      else
        USABLE="$USABLE $g"; echo "[a2] card $g = $name, ${free} GiB free"
      fi ;;
    *) echo "[a2] SKIP card $g: '${name:-unreadable}' is not an A100" ;;
  esac
done
[ -z "$USABLE" ] && { echo "REFUSING: no usable A100 — every card is busy or unreadable"; exit 2; }
CARDS="$USABLE"

echo "[a2] $(date) START — arms='$ARMS' seeds='$SEEDS' cards='$CARDS'"

JOBS=""
for r in $rungs; do for arm in $ARMS; do for s in $SEEDS; do JOBS="$JOBS $r:$arm:$s"; done; done; done

worker () {  # $1 = gpu
  local gpu=$1 job rung arm s tag out claim rc
  for job in $JOBS; do
    rung="${job%%:*}"; arm=$(echo "$job" | cut -d: -f2); s="${job##*:}"
    out="$ROOT/$(basename "$rung")"
    tag="$(basename "$rung")_${arm}_s${s}"
    [ -f "$out/$arm/$s.parquet" ] && continue
    mkdir -p "$out/$arm"
    claim="$out/$arm/.$s.claim"
    ( set -o noclobber; echo "$$ gpu=$gpu $(date)" > "$claim" ) 2>/dev/null || continue
    echo "[a2] $tag gpu=$gpu START $(date)"
    # INTERMEDIATE_ROOT isolation ALSO redirects the feature stores. The rung roots symlink them back
    # to the reference copies, and these two exports pin them again: without both, every target
    # silently gets a ZERO vector while the arm still reports a number — the manufactured-null hazard.
    INTERMEDIATE_ROOT="$rung" \
    PLM_EMBEDDINGS_PATH="$rung/plm_embeddings.parquet" \
    PINNACLE_EMBEDDINGS_PATH="$rung/pinnacle_embeddings.parquet" \
    ID_MAPPING_PATH="$rung/id_mapping.parquet" \
    SCREENING_ROOT="$out" PREDICTIONS_ROOT="$out/predictions" \
    REGISTRY_PATH="$out/experiment_registry.yaml" \
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
      --only "$arm" --seed "$s" --epochs $EPOCHS --batch-size $BATCH --device cuda \
      > "$LOG/${tag}.log" 2>&1
    rc=$?
    echo "[a2] $tag gpu=$gpu exit=$rc $(date)"
    [ $rc -ne 0 ] && rm -f "$claim"
  done
  echo "[a2] worker gpu=$gpu drained $(date)"
}

for g in $CARDS; do worker "$g" & done
wait

echo "[a2] $(date) DONE. Landed per rung:"
for r in $rungs; do
  out="$ROOT/$(basename "$r")"
  echo "[a2]   $(basename "$r"): expression_only=$(ls "$out"/expression_only/[0-9].parquet 2>/dev/null | wc -l) \
untyped_gnn=$(ls "$out"/untyped_gnn/[0-9].parquet 2>/dev/null | wc -l)"
done
echo "[a2] next: PYTHONPATH=src .venv/bin/python -m tcell_pipeline.screening.ladder_report"
