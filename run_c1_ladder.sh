#!/bin/bash
# C1b — the TYPED, GATED arm's own detection floor. Pre-registered as Amendment 9 (2026-08-21),
# which landed BEFORE this script ran, as rail 7 requires.
#
# WHY THIS EXISTS. Amendment 6 measured the floor on `untyped_gnn`, chosen on cost and sensitivity,
# and closed with "this is a bound on the pipeline's sensitivity, not on the typed encoder's
# specifically". The paper's headline null is about `condition_gated`. The a-fortiori argument runs one
# way only: the untyped arm detects at 0.02 response SDs, so a weaker arm needs AT LEAST that. It gives
# no upper bound, and this measures one.
#
#   setsid nohup ./run_c1_ladder.sh > data/logs/c1_ladder.nohup.log 2>&1 &
#
# *** THIS CAMPAIGN HAS RUN AND IS CLOSED (21-26 Aug 2026). 48 lanes, 0 failures, ~280 GPU-h. ***
# RESULT: condition_gated recovers NO injected signal at any size up to 0.40 response SDs, against
# untyped_gnn's measured 0.02 on the identical rungs. The control did not clear, so the reading is a
# floor rather than a vacuum, and all 24 lanes kept live gates (min 0.0240). The mechanism is variance:
# the gated arm's paired SD is 4.0-9.5x the untyped arm's. Full account atop RESULTS_SUMMARY.md;
# artifact data/results/c1_ladder/floor_condition_gated.json.
# The script is IDEMPOTENT and reaps stale claims, so re-running it is a no-op that re-verifies the
# landed lanes rather than retraining them. To add SEEDS instead, set SEEDS="4 5 6 ..." — the post-hoc
# power note says ~13-16 would be needed to match the untyped arm's sensitivity.
#
# *** --lambda-graph 0 IS NOT OPTIONAL AND IS NOT AN ENVIRONMENT VARIABLE. ***
# config.LAMBDA_GRAPH is a plain module constant, so `export LAMBDA_GRAPH=0` silently does NOTHING.
# The override is the CLI flag below, and run_a2_ladder.sh does not pass it — which is exactly why
# this is a separate script rather than ARMS="..." ./run_a2_ladder.sh. At the config default of 0.01,
# run_screening's own --help says it: "the unnormalised per-edge sum is ~103x the response term and
# annihilates the gates inside epoch 0". `condition_gated` is the ONLY arm here whose gate is live, so
# every one of these 24 lanes would have trained with a dead gate, measured a floor for a different
# model than the null describes, and tripped Amendment 3.4's own kill criterion (gate <= 1e-3 =
# UNDECIDABLE). A campaign designed to trip its own kill criterion is not a campaign. Amendment 9.2.
#
# RAIL 1. The injected matrices leave challenge and calibration rows bit-identical; no sealed response
# enters any statistic. Nothing here reads data/splits except to partition.
# RAIL 2. data/results/a2_ladder and every other landed root are READ-ONLY. This writes a FRESH root.
set -u
cd "$(dirname "$0")" || exit 1

INJECT=data/intermediate/inject
ROOT=data/results/c1_ladder                 # FRESH root; Amendment 6's ladder is not touched
LOG=data/logs/c1_ladder
SEEDS="${SEEDS:-0 1 2 3}"                   # n=4 per arm per rung (rail 5)
ARMS="${ARMS:-expression_only condition_gated}"
CARDS="${CARDS:-0 1 2 3}"
EPOCHS=20
BATCH=8

export PYTHONPATH=src SUBGRAPH_CACHE_SIZE=9000 OMP_NUM_THREADS=4
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.173.02
export SPLITS_ROOT=data/splits              # the frozen fold; screening only reads it
mkdir -p "$ROOT" "$LOG"

rungs=$(ls -d "$INJECT"/d[0-9]* "$INJECT"/permuted_d[0-9]* 2>/dev/null | sort)
[ -z "$rungs" ] && { echo "REFUSING: no rungs under $INJECT — Amendment 9 reuses the SIX already built"; exit 2; }
for r in $rungs; do
  [ -f "$r/injection_provenance.json" ] || { echo "REFUSING: $r has no injection_provenance.json"; exit 2; }
  [ -f "$r/de_layers/zscore.npz" ] || { echo "REFUSING: $r has no injected response layer"; exit 2; }
done
echo "[c1] rungs: $(echo $rungs | tr ' ' '\n' | xargs -n1 basename | tr '\n' ' ')"

# SELECT usable cards; do not refuse the whole run because one is busy. An earlier version of the A2
# preflight exited when ANY card failed and a co-tenant on card 2 cost eleven hours of idle A100 time.
USABLE=""
for g in $CARDS; do
  read -r name free <<< "$(CUDA_VISIBLE_DEVICES=$g .venv/bin/python -c \
    "import torch; f,_=torch.cuda.mem_get_info(0); \
     print(torch.cuda.get_device_properties(0).name.replace(' ','_'), int(f/2**30))" 2>/dev/null)"
  case "$name" in
    *A100*)
      if [ "${free:-0}" -lt 40 ]; then
        echo "[c1] SKIP card $g: ${free:-0} GiB free (<40), someone else is on it"
      else
        USABLE="$USABLE $g"; echo "[c1] card $g = $name, ${free} GiB free"
      fi ;;
    *) echo "[c1] SKIP card $g: '${name:-unreadable}' is not an A100" ;;
  esac
done
[ -z "$USABLE" ] && { echo "REFUSING: no usable A100 — every card is busy or unreadable"; exit 2; }
CARDS="$USABLE"

# REAP STALE CLAIMS BEFORE ANYTHING ELSE. The atomic claim is what makes this idempotent, but it is
# only released when a lane exits non-zero INSIDE the worker. A campaign killed from outside - a
# SIGTERM to the process group, a reboot, a co-tenant OOM - leaves a claim with no parquet, and every
# future run then SKIPS that lane silently and reports a short ladder as if it were complete. A claim
# whose parquet is absent is by definition a lane that did not finish, so it is safe to clear here.
stale=0
for c in $(find "$ROOT" -name ".*.claim" 2>/dev/null); do
  d=$(dirname "$c"); s=$(basename "$c" .claim); s=${s#.}
  if [ ! -f "$d/$s.parquet" ]; then rm -f "$c"; stale=$((stale+1)); fi
done
[ $stale -gt 0 ] && echo "[c1] reaped $stale stale claim(s) from an interrupted run"

echo "[c1] $(date) START — arms='$ARMS' seeds='$SEEDS' cards='$CARDS' lambda_graph=0"

# JOB ORDER IS SEED-MAJOR, AND THAT IS A DELIBERATE CHOICE WITH A REASON.
# Amendment 9.9 anticipates this campaign being stopped short: 24 condition_gated lanes at the 7.5-16
# GPU-hours these lanes measured is 180-380 GPU-hours, and the deadline is 8 days out on a shared box
# at load ~92. Ordering rung-major would leave a stopped campaign with three rungs at n=4 and three at
# n=0, which the floor rule cannot read at all: the floor is the smallest rung that clears AND is
# cleared by every LARGER rung, so a missing large rung makes every smaller one unreadable.
# Seed-major means that at every moment the ladder is BALANCED — all six rungs at whatever n has
# landed — so a campaign stopped at n=2 is an underpowered complete ladder rather than a useless
# partial one. The cheap expression_only lanes go FIRST within each seed so that every condition_gated
# lane lands with its pair already present and the report can be run incrementally.
JOBS=""
for s in $SEEDS; do
  for arm in $ARMS; do
    for r in $rungs; do JOBS="$JOBS $r:$arm:$s"; done
  done
done
echo "[c1] $(echo $JOBS | wc -w) lanes queued, seed-major so a stopped campaign is still a balanced ladder"

worker () {  # $1 = gpu
  local gpu=$1 job rung arm s tag out claim rc gmin
  for job in $JOBS; do
    rung="${job%%:*}"; arm=$(echo "$job" | cut -d: -f2); s="${job##*:}"
    out="$ROOT/$(basename "$rung")"
    tag="$(basename "$rung")_${arm}_s${s}"
    [ -f "$out/$arm/$s.parquet" ] && continue
    mkdir -p "$out/$arm"
    claim="$out/$arm/.$s.claim"
    ( set -o noclobber; echo "$$ gpu=$gpu $(date)" > "$claim" ) 2>/dev/null || continue
    echo "[c1] $tag gpu=$gpu START $(date)"
    # INTERMEDIATE_ROOT isolation ALSO redirects the feature stores. The rung roots symlink them back
    # to the reference copies, and these exports pin them again: without both, every target silently
    # gets a ZERO vector while the arm still reports a number — the manufactured-null hazard.
    INTERMEDIATE_ROOT="$rung" \
    PLM_EMBEDDINGS_PATH="$rung/plm_embeddings.parquet" \
    PINNACLE_EMBEDDINGS_PATH="$rung/pinnacle_embeddings.parquet" \
    ID_MAPPING_PATH="$rung/id_mapping.parquet" \
    SCREENING_ROOT="$out" PREDICTIONS_ROOT="$out/predictions" \
    REGISTRY_PATH="$out/experiment_registry.yaml" \
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m tcell_pipeline.screening.run_screening \
      --only "$arm" --seed "$s" --epochs $EPOCHS --batch-size $BATCH --device cuda \
      --lambda-graph 0 \
      > "$LOG/${tag}.log" 2>&1
    rc=$?
    # AMENDMENT 9.2: gate health is a REPORTED quantity, not an assumption. The minimum gate mean over
    # the lane's epochs is echoed here and read again by the finaliser. Below config.GATE_DEAD (1e-3)
    # the lane is UNDECIDABLE under Amendment 3.4 and must shrink n, never be counted as evidence.
    if [ "$arm" = "condition_gated" ]; then
      # FROM THE TRAINER'S OWN HISTORY, not from the lane log. run_screening's lane log carries no
      # per-epoch line at all; the "gate mean" wording belongs to run_rescreen_lambda0.sh, which
      # post-processes this same file. Grepping the log would report nothing for a healthy lane.
      gmin=$(.venv/bin/python -c "
import json,sys,pathlib
p=pathlib.Path('$out/$arm/$s/logs/stage_a_history.json')
if not p.exists(): print('no-history'); raise SystemExit
g=[e['train']['gate_mean'] for e in json.loads(p.read_text())
   if e.get('train',{}).get('gate_mean') is not None]
print(f'{min(g):.6f}' if g else 'ungated')" 2>/dev/null)
      echo "[c1] $tag gate_mean_min=${gmin:-none}"
      case "${gmin:-1}" in
        0|0.000*) echo "[c1] *** $tag GATE COLLAPSED (${gmin}) — UNDECIDABLE, Amendment 3.4 ***" ;;
      esac
    fi
    echo "[c1] $tag gpu=$gpu exit=$rc $(date)"
    [ $rc -ne 0 ] && rm -f "$claim"
  done
  echo "[c1] worker gpu=$gpu drained $(date)"
}

for g in $CARDS; do worker "$g" & done
wait

echo "[c1] $(date) DONE. Landed per rung:"
for r in $rungs; do
  out="$ROOT/$(basename "$r")"
  echo "[c1]   $(basename "$r"): expression_only=$(ls "$out"/expression_only/[0-9].parquet 2>/dev/null | wc -l) \
condition_gated=$(ls "$out"/condition_gated/[0-9].parquet 2>/dev/null | wc -l)"
done
echo "[c1] next: PYTHONPATH=src .venv/bin/python -m tcell_pipeline.screening.ladder_report \\"
echo "[c1]         --root $ROOT --arm condition_gated \\"
echo "[c1]         --reference-root data/results/screening_lambda0 \\"
echo "[c1]         --out $ROOT/floor_condition_gated.json"
