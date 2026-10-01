# Reproducing the paper

This guide regenerates, from a bare machine, every result file read by *The Regularizer That Switched
Off the Experiment: Why Our Protein-Interaction Null Was About the Encoder, Not the Prior* (ICBINB
workshop, NeurIPS 2026). Section 3 maps each part of the paper to its file, section 4 gives the
commands, and section 5 checks your files against the paper.

Conventions: run everything from the repository root with `PYTHONPATH=src`, and keep
`OMP_NUM_THREADS` between 4 and 8 (nested BLAS threads thrash a many-core machine). Every experiment
writes its own result root under `data/results/`. The split in `data/splits/` is an input and is never
rewritten, and the sealed challenge split is never scored.

## 1. Environment

```bash
git clone https://github.com/Vijayavallabh/tcell-causal-ppi && cd tcell-causal-ppi
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126"   # the build for your driver
uv pip install -r requirements.txt
uv tool install awscli
```

**Hardware.** `typed_static` and `condition_gated` lanes peak at 47 to 51 GB of GPU memory, so each needs
its own 80 GB card; `untyped_gnn` needs 2 to 4 GB and `expression_only` about 2.5 GB. A 20-epoch lane on
the reference screen costs about 0.35 GPU-hours for `expression_only`, 2 for `untyped_gnn` and 7 to 17
for the two typed arms (the range is load on a shared machine). Subgraph sampling is CPU-bound at about
one core per lane, so running more lanes than GPUs does not add throughput. Set `SUBGRAPH_CACHE_SIZE`
to the number of targets a lane touches (the launchers use 8000 on the reference fold) when the RAM is
there, about 4.5 MB per target. Statistics, figures and the paper check run on CPU.

**NVML.** If every CUDA lane dies within a minute on a PyTorch assert naming `nvmlInit_v2_`, the NVML
library on the loader path does not match the kernel module; preload the matching one with
`export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.<driver version>`. The `LD_PRELOAD`
lines in the `run_*.sh` launchers are that fix for the machine the paper ran on; edit or delete them.

**Determinism.** GPU training is not bit-deterministic (scatter-add accumulation order), so a retrained
lane reproduces the paper's numbers to within seed noise, not to the last digit. Everything downstream
of the lanes (statistics, pooling, the figure) is deterministic.

## 2. Data

### 2.1 The reference screen

The genome-scale CD4+ T-cell CRISPRi Perturb-seq screen of Zhu et al. (2025)
([dataset card](https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq)). Only
the processed layer, about 100 GiB, is needed; the 1.6 TiB of cell-level files are not.

```bash
mkdir -p data/raw
for f in GWCD4i.DE_stats.h5ad GWCD4i.DE_stats.by_donors.h5mu GWCD4i.DE_stats.by_guide.h5mu \
         GWCD4i.pseudobulk_merged.h5ad; do
  aws s3 cp "s3://genome-scale-tcell-perturb-seq/marson2025_data/$f" "data/raw/$f" --no-sign-request
done
aws s3 sync s3://genome-scale-tcell-perturb-seq/marson2025_data/suppl_tables/ data/raw/suppl_tables/ --no-sign-request
aws s3 sync s3://genome-scale-tcell-perturb-seq/marson2025_data/metadata/ data/raw/metadata/ --no-sign-request
# the other supplementary tables live in the authors' analysis repository
curl -s https://api.github.com/repos/emdann/GWT_perturbseq_analysis_2025/contents/metadata/suppl_tables \
  | python3 -c "import sys, json; [print(x['download_url']) for x in json.load(sys.stdin)]" \
  | while read -r url; do f=data/raw/suppl_tables/$(basename "$url"); [ -f "$f" ] || curl -sL --fail "$url" -o "$f"; done
```

### 2.2 The replication datasets

The harmonized scPerturb files (Peidli et al. 2024), Zenodo record 7041849, CC-BY-4.0, about 15 GB for
the files in the checksum manifest:

```bash
mkdir -p data/raw/scperturb && cd data/raw/scperturb
for f in $(awk '{print $2}' ../../manifests/scperturb_SHA256SUMS.txt); do
  curl -L --fail -o "$f" "https://zenodo.org/records/7041849/files/$f?download=1"
done
sha256sum -c ../../manifests/scperturb_SHA256SUMS.txt && cd ../../..
```

### 2.3 Marts, graph, embeddings and program basis

```bash
PYTHONPATH=src python -m tcell_pipeline.run_module0                 # marts + typed PPI graph; downloads the PPI databases
PYTHONPATH=src python -m tcell_pipeline.embeddings_plm              # ESM-2 650M per target protein (GPU, resumable)
PYTHONPATH=src python -m tcell_pipeline.embeddings_pinnacle         # PINNACLE, CD4 helper T-cell context
PYTHONPATH=src python -m tcell_pipeline.programs.run_program_basis  # sparse PCA, K=128, training fold only
```

`run_module0` rewrites the marts that every result is built on, so run it once. The program basis is
the coordinate system of every result: rerun it only with its defaults, which are seed-deterministic.
The split arrives with the clone; `python -m tcell_pipeline.splits` regenerates its files byte for byte
(its leakage report differs in the seventh digit of two diagnostic cosines), but no step requires it.

## 3. Where each number comes from

`paper/icbinb/verify_numbers.py` reads one result file per table and checks the numbers in the text
around it. Paths are under `data/results/`.

| In the paper | Result file | Section |
|---|---|---|
| Figure 1, the penalty sweep | `q4_lambda_sweep_22ep.json`; the text also cites `q4_lambda_sweep_12ep.json` and the three-epoch `screening_lambda0/lambda_sweep_empirical.json` | 4.1 |
| Table 1, the five-seed family with live gates | `screening_lambda0/robustness_5seed.json` | 4.2 |
| the same family at the default penalty | `screening/robustness_5seed.json` | 4.2 |
| the second metric | `screening_lambda0/second_metric_5seed.json` | 4.3 |
| every n=7 number | `screening_n7_live/robustness_5seed.json` | 4.4 |
| folds, re-draws, variance decomposition, power | `screening_c075c15_n5/`, `screening_c080c10_{h1,r2,r3}/`, `screening_c075c15_r{2,3}/`, `screening_c070/`, `splits_*/manifest.json`, `l4_v2/`, `a2_power_v2/` | 4.5 |
| the eight-dataset replication and its pooling | `replication/<dataset>/robustness_5seed.json`, `replication/pooled*.json`, `camera_ready/pooled_*_hksj.json`, `replication/frangieh_on_target_qc.json` | 4.6 |
| the architecture search | `arch_search/`, `a2_power/arch_search_bound.json` | 4.7 |
| the rationale audit | `rationale_audit_lambda0/audit_report.json`, `a2_power/rationale_bound.json` | 4.8 |
| the graph-free baseline | `feature_ablation_report.json` | 4.9 |
| why typing hurts | `screening_a1/a1_mechanism.json`, `screening_b1/b1_message_form.json` | 4.10 |
| the injected-signal floors | `a2_ladder/floor.json`, `a2_ladder/b3_power.json`, `c1_ladder/floor_condition_gated.json` | 4.11 |
| other metrics and rank intervals | `a3_external_live/rescored.json`, `a3_external_live/k_sweep.json`, `b2_deciles_live/deciles.json` | 4.12 |
| training lengths, live gates, paralog residual | `camera_ready/facts.json` | 4.13 |

The corrections appendix quotes superseded values beside corrected ones. Those come from the same
commands run over the superseded roots (`screening_untyped_n7/`, `a3_external/`, `b2_deciles/`, `l4/`,
`a2_power/`), noted below where it matters.

## 4. Commands

**The lane.** Every screening root is one command repeated over arms and seeds; it trains one arm at
one seed for up to 20 epochs and scores the validation split:

```bash
SPLITS_ROOT=<split root> SCREENING_ROOT=<root> PREDICTIONS_ROOT=<root>/predictions \
REGISTRY_PATH=<root>/experiment_registry.yaml SUBGRAPH_CACHE_SIZE=8000 PYTHONPATH=src \
python -m tcell_pipeline.screening.run_screening --only <arm> --seed <s> --epochs 20 --batch-size 8 \
  --device cuda --lambda-graph 0
```

The arms are `expression_only`, `untyped_gnn`, `typed_static` and `condition_gated`. The `run_*.sh`
launchers are the scripts that filled each root as it was run; their headers say which lanes and why.
`--lambda-graph` is a flag, not an environment variable: the configuration default of 0.01 is the
penalty that switches the graph off, and only the default-penalty campaign in 4.2 used it.

**Aggregating a root.** `multiseed` runs the four pre-registered paired contrasts with Bonferroni and
Holm and writes `robustness_5seed.{json,md}` into the root:

```bash
SCREENING_ROOT=<root> REGISTRY_PATH=<root>/experiment_registry.yaml SPLITS_ROOT=<split root> \
PYTHONPATH=src python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3,4
```

It checks from the run registry that every lane scored the same fold. For a root assembled from lanes
run elsewhere, pass `--no-registry` and it checks each lane's recorded `n_train` and `n_val` instead.
Read `family_size` in the output: it must be 4.

### 4.1 Gate collapse and Figure 1

The precondition every graph result is read against: the collapse factor of the edge gates against
initialization. Near 1e-7 the graph is switched off and a graph contrast is undecidable.

```bash
PYTHONPATH=src python -m tcell_pipeline.probe_graph_gradients --n-max 8 --batch-size 2 --steps 1
```

Figure 1 is one seed of `condition_gated` on the first 512 training and 128 validation rows, 22 epochs,
at nine penalty weights, batch 4. Each weight takes about 0.4 GPU-hours; `--lambdas` splits the weights
across cards, and the parts merge by concatenating their `runs` lists.

```bash
SUBGRAPH_CACHE_SIZE=1000 PYTHONPATH=src python -m tcell_pipeline.screening.lambda_sweep --device cuda \
  --out data/results/q4_lambda_sweep_22ep.json
# the shorter sweeps the text also cites
SUBGRAPH_CACHE_SIZE=1000 PYTHONPATH=src python -m tcell_pipeline.screening.lambda_sweep --device cuda \
  --epochs 12 --out data/results/q4_lambda_sweep_12ep.json
SUBGRAPH_CACHE_SIZE=1000 PYTHONPATH=src python -m tcell_pipeline.screening.lambda_sweep --device cuda \
  --epochs 3 --out data/results/screening_lambda0/lambda_sweep_empirical.json
python paper/icbinb/make_lambda_sweep.py   # paper/figures/lambda_sweep.pdf; needs camera_ready/facts.json (4.13)
```

What reproduces, measured on 2026-10-01 against the published sweep: the penalized runs match the
original first-epoch gates to six digits (0.250814 at weights 0.003 and 0.01) and, at the three weights
replicated in full (1e-5, 0.003, 0.01), end within a factor of 0.97 to 1.26 of the published values with
the same fraction of gates dead. The unpenalized run is chaotic: identical runs already differ by 0.11
after one epoch, and three replicates ended at 0.37, 0.58 and 0.74 against the published 0.76. Its gates
are live in every run, as in the full-data checkpoints (0.27 to 0.78), so the figure's claims stand, but
its value is one run's, and `verify_numbers.py` will flag that number against a fresh sweep.

### 4.2 The five-seed family (Table 1)

The default-penalty campaign, whose gates collapsed (`data/results/screening/`): seed 0 with every arm
plus the network-propagation reference, then seeds 1 to 4. It uses the default registry,
`data/results/experiment_registry.yaml`.

```bash
./run_screening_campaign.sh
./run_multiseed_campaign.sh
PYTHONPATH=src python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3,4
```

Table 1 re-trains only `condition_gated` at `--lambda-graph 0`. The other three arms have no learnable
gate (`typed_static` pins it to 1), so the penalty is a constant for them and their lanes are reused.

```bash
./run_rescreen_lambda0.sh
SCREENING_ROOT=data/results/screening_lambda0 \
REGISTRY_PATH=data/results/screening_lambda0/experiment_registry.yaml \
PYTHONPATH=src python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3,4
```

### 4.3 The second metric

The same paired family re-read on every metric the lanes persisted; no re-scoring.

```bash
PYTHONPATH=src python -m tcell_pipeline.screening.second_metric --root data/results/screening_lambda0 \
  --out data/results/screening_lambda0/second_metric_5seed.json
```

### 4.4 Seven seeds

`run_untyped_n7.sh` adds seeds 5 and 6 of `untyped_gnn` and `expression_only`, and `run_balance_n7.sh`
adds them for the two typed arms, into `screening_untyped_n7/`; seeds 0 to 4 there are copies of the
default-penalty lanes. `merge_registry_n7.py` supplies the reference registry's fold evidence.

```bash
./run_untyped_n7.sh && ./run_balance_n7.sh
python merge_registry_n7.py
```

The paper's n=7 numbers use live gates throughout, so `condition_gated` seeds 0 to 4 come from the
penalty-free re-screen:

```bash
R=data/results/screening_n7_live
for a in expression_only typed_static untyped_gnn; do
  mkdir -p $R/$a && cp -p data/results/screening_untyped_n7/$a/{0..6}.parquet $R/$a/
done
mkdir -p $R/condition_gated
cp -p data/results/screening_lambda0/condition_gated/{0..4}.parquet \
      data/results/screening_untyped_n7/condition_gated/{5,6}.parquet $R/condition_gated/
SCREENING_ROOT=$R PYTHONPATH=src python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3,4,5,6 --no-registry
```

`multiseed` refuses to pool a gated arm whose seeds ran at different penalties, which is how the
superseded n=7 numbers (aggregated over `screening_untyped_n7/`) are now caught.

### 4.5 Folds, re-draws and the variance decomposition

Each fold is a split at a sequence-similarity threshold and family-size cap; a re-draw repeats a
specification with a different partition seed. The seven split roots:

```bash
for spec in "c070 0.70 0.05 0" "c075c15 0.75 0.15 0" "c075c15_r2 0.75 0.15 1" "c075c15_r3 0.75 0.15 2" \
            "c080c10 0.80 0.10 0" "c080c10_r2 0.80 0.10 1" "c080c10_r3 0.80 0.10 2"; do
  set -- $spec
  SEQ_SIM_COSINE_THRESHOLD=$2 GROUP_SIZE_CAP=$3 SPLIT_SEED=$4 SPLITS_ROOT=data/results/splits_$1 \
    PYTHONPATH=src python -m tcell_pipeline.splits
done
```

The screening roots on them, each filled with the lane above at `--lambda-graph 0`:

| Root | Split root | Arms and seeds | Launchers |
|---|---|---|---|
| `screening_c075c15_n5` | `splits_c075c15` | all four arms, 0 to 4 | `run_c075c15_n5.sh` (seeded with the earlier `screening_c075c15` lanes) |
| `screening_c080c10_h1` | `splits_c080c10` | all four arms, 0 to 4 | `run_c080c10_h1.sh`, `run_c080c10_arm.sh` |
| `screening_c080c10_r2`, `_r3` | `splits_c080c10_r2`, `_r3` | `expression_only`, `condition_gated`, 0 to 4 | `run_c080c10_r2.sh`, `run_realization.sh` |
| `screening_c075c15_r2`, `_r3` | `splits_c075c15_r2`, `_r3` | `expression_only`, `typed_static`, 0 to 4 | `run_c075c15_redraws.sh`, `run_l4_finish.sh` |
| `screening_c070` | `splits_c070` | `expression_only`, `typed_static`, 0 to 4 | `run_c070.sh`, `run_l4_finish.sh`, `run_l4_card2.sh` |

Aggregate each root with `multiseed` as above. `run_l4_finalise.sh` aggregates the difficulty roots and
runs the decomposition; the corrected decomposition and power table are:

```bash
PYTHONPATH=src python -m tcell_pipeline.screening.variance_decomposition --contrast h2a --out data/results/l4_v2/vardecomp_h2a.json
PYTHONPATH=src python -m tcell_pipeline.screening.variance_decomposition --contrast h1_vs_no_graph --out data/results/l4_v2/vardecomp_h1_vs_no_graph.json
PYTHONPATH=src python -m tcell_pipeline.screening.power_simulation --out data/results/a2_power_v2/power_simulation.json
```

### 4.6 The replication datasets

The method is fixed in advance by `docs/replication-prereg.md`; `docs/replication-dataset-survey.md`
records the candidate measurements. First the survey, the extended identifier map and embedding stores
(built by copy, so the reference stores are never rewritten), and the per-context PINNACLE embeddings:

```bash
R=data/raw/scperturb
PYTHONPATH=src python -m tcell_pipeline.replication.survey --out data/intermediate/replication/survey_l1.json \
  --glob $R/DatlingerBock2017.h5ad $R/FrangiehIzar2021_RNA.h5ad $R/NormanWeissman2019_filtered.h5ad \
         $R/PapalexiSatija2021_eccite_RNA.h5ad $R/ReplogleWeissman2022_rpe1.h5ad $R/ShifrutMarson2018.h5ad
PYTHONPATH=src python -m tcell_pipeline.replication.survey --glob "$R/Tian*.h5ad" \
  --out data/intermediate/replication/survey_tian.json
PYTHONPATH=src python -m tcell_pipeline.replication.extend_id_mapping   # targets of exactly these ten surveys
./run_plm_extension.sh
PYTHONPATH=src python -m tcell_pipeline.replication.pinnacle_contexts
```

The identifier map is extended from these ten surveys only, so the two K562 screens use the reference
map for their targets, as in the paper; surveying more files would map more of their targets and change
their results.

Then, per dataset: the pre-registered differential-expression matrix (Amendment 2) with its on-target
check, the preparation stages, and the Frangieh on-target figures the paper quotes.

```bash
for d in FrangiehIzar2021_RNA ReplogleWeissman2022_rpe1 ReplogleWeissman2022_K562_essential \
         ReplogleWeissman2022_K562_gwps NormanWeissman2019_filtered TianKampmann2021_CRISPRi \
         TianKampmann2021_CRISPRa; do
  PYTHONPATH=src python -m tcell_pipeline.replication.de_amendment2 --dataset $d
done
PYTHONPATH=src python -m tcell_pipeline.replication.de_amendment2 --dataset FrangiehIzar2021_RNA \
  --qc-only data/results/replication/frangieh_on_target_qc.json

# <dataset> <PINNACLE context or none> <K, prereg Amendment 3.1>
./run_replication_prep.sh FrangiehIzar2021_RNA melanocyte 128
./run_replication_prep.sh ReplogleWeissman2022_rpe1 retinal_pigment_epithelial_cell 128
./run_replication_prep.sh ReplogleWeissman2022_K562_essential none 128
./run_replication_prep.sh ReplogleWeissman2022_K562_gwps none 128
./run_replication_prep.sh NormanWeissman2019_filtered none 16
./run_replication_prep.sh TianKampmann2021_CRISPRi none 32
./run_replication_prep.sh TianKampmann2021_CRISPRa none 16
```

The campaign trains every lane (seeds 0 to 3; `condition_gated` only on Frangieh, the one dataset with
more than one condition) and skips lanes already on disk. `run_replication_stage.sh` must be sourced
before any stage run by hand: without it the feature stores resolve to an empty sandbox and every target
silently gets a zero vector.

```bash
./run_replication_campaign.sh
for entry in FrangiehIzar2021_RNA:melanocyte ReplogleWeissman2022_rpe1:retinal_pigment_epithelial_cell \
             ReplogleWeissman2022_K562_essential:none ReplogleWeissman2022_K562_gwps:none \
             NormanWeissman2019_filtered:none TianKampmann2021_CRISPRi:none TianKampmann2021_CRISPRa:none; do
  ( source ./run_replication_stage.sh ${entry%%:*} ${entry##*:}
    R=data/results/replication/${entry%%:*}
    SCREENING_ROOT=$R REGISTRY_PATH=$R/experiment_registry.yaml \
      python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3 )
done
```

Pooling, with and without the reference screen, and the Hartung-Knapp sensitivity:

```bash
PYTHONPATH=src python -m tcell_pipeline.replication.pool --out data/results/replication/pooled.json
PYTHONPATH=src python -m tcell_pipeline.replication.pool --with-reference --out data/results/replication/pooled_with_reference.json
PYTHONPATH=src python -m tcell_pipeline.replication.pool --with-reference --datasets FrangiehIzar2021_RNA \
  ReplogleWeissman2022_rpe1 ReplogleWeissman2022_K562_essential ReplogleWeissman2022_K562_gwps \
  --out data/results/replication/pooled_k128_subset.json
PYTHONPATH=src python -m tcell_pipeline.replication.pool --out data/results/camera_ready/pooled_hksj.json
PYTHONPATH=src python -m tcell_pipeline.replication.pool --with-reference --out data/results/camera_ready/pooled_with_reference_hksj.json
PYTHONPATH=src python -m tcell_pipeline.replication.pool --with-reference --datasets FrangiehIzar2021_RNA \
  ReplogleWeissman2022_rpe1 ReplogleWeissman2022_K562_essential ReplogleWeissman2022_K562_gwps \
  --out data/results/camera_ready/pooled_k128_subset_hksj.json
```

Fold re-draws are not pooled: they re-partition one dataset, and pooling them would count it several
times.

### 4.7 The architecture search

Graph-encoder variants scored on a target-grouped inner holdout of the training fold, so validation is
not touched by the choice.

```bash
./run_arch_search.sh      # condition_gated: aggregation x edge-confidence threshold, nine cells
SUBGRAPH_CACHE_SIZE=9000 PYTHONPATH=src python -m tcell_pipeline.arch_search --refs --epochs 5 \
  --device cuda --out data/results/arch_search     # the no-graph and untyped references
./run_arch_stage2.sh      # untyped variants: edge weights, pruning, attention
PYTHONPATH=src python -m tcell_pipeline.screening.arch_search_power --out data/results/a2_power/arch_search_bound.json
```

### 4.8 The rationale audit

Fifty cases on the live-gate `condition_gated` checkpoint of seed 3, against matched random controls.
The driver refuses a checkpoint with collapsed gates.

```bash
RATIONALE_AUDIT_ROOT=data/results/rationale_audit_lambda0 PYTHONPATH=src \
python -m tcell_pipeline.run_module8_real --part audit --device cuda \
  --ckpt data/results/screening_lambda0/condition_gated/3/ckpt/stage_a_best.pt
PYTHONPATH=src python -m tcell_pipeline.screening.rationale_audit_bound --out data/results/a2_power/rationale_bound.json
```

### 4.9 The graph-free baseline

`expression_only` with its graph-derived target features zeroed (both, PINNACLE only, degrees only),
seeds 0 to 4, compared with the unablated lanes of `screening_lambda0`:

```bash
for v in nograph nopinnacle nodegree; do for s in 0 1 2 3 4; do ./run_feature_ablation.sh <gpu> $v $s; done; done
python analyze_feature_ablation.py        # writes data/results/feature_ablation_report.json
```

### 4.10 Why typing hurts

Two diagnostic arms bracket the typed encoder (tied message weights; permuted relation labels at
preserved edge counts), then the message form (symmetric degree normalization):

```bash
SEEDS="0 1 2 3 4" ARMS="typed_shared typed_permuted" ./run_a1_mechanism.sh
PYTHONPATH=src python -m tcell_pipeline.screening.a1_report
SEEDS="0 1 2 3 4" ARMS=typed_gcnnorm ROOT=data/results/screening_b1 LOG=data/logs/b1 ./run_a1_mechanism.sh
PYTHONPATH=src python -m tcell_pipeline.screening.b1_report --root data/results/screening_b1 \
  --out data/results/screening_b1/b1_message_form.json
```

### 4.11 The injected-signal floors

Each rung adds a known multiple of each target's neighbors' mean training response (a negative-control
rung gets another target's neighborhood), and the ladder asks which rung the arm recovers.

```bash
PYTHONPATH=src python -m tcell_pipeline.screening.inject_signal --ladder --out data/intermediate/inject
./run_a2_ladder.sh && ./run_ladder_finalise.sh          # untyped_gnn
PYTHONPATH=src python -m tcell_pipeline.screening.ladder_report --out data/results/a2_ladder/floor.json
PYTHONPATH=src python -m tcell_pipeline.screening.b3_power --out data/results/a2_ladder/b3_power.json
./run_c1_ladder.sh                                      # condition_gated, same rungs, --lambda-graph 0
PYTHONPATH=src python -m tcell_pipeline.screening.ladder_report --root data/results/c1_ladder \
  --arm condition_gated --reference-root data/results/screening_lambda0 \
  --out data/results/c1_ladder/floor_condition_gated.json
```

### 4.12 Other metrics and rank intervals

The stored predictions re-scored under other groups' endpoints and on disjoint intervals of the response
ranking; no training. The live-gate predictions replace the gated arm's; the other three are unchanged.

```bash
P=data/results/predictions_live; mkdir -p $P
ln -s ../predictions/{expression_only,typed_static,untyped_gnn} $P/
ln -s ../screening_lambda0/predictions/condition_gated $P/condition_gated
PYTHONPATH=src python -m tcell_pipeline.screening.rescore_external --predictions-root $P --out data/results/a3_external_live/rescored.json
PYTHONPATH=src python -m tcell_pipeline.screening.rescore_external --predictions-root $P --k-sweep --out data/results/a3_external_live/k_sweep.json
PYTHONPATH=src python -m tcell_pipeline.screening.rank_deciles --predictions-root $P --out data/results/b2_deciles_live/deciles.json
```

### 4.13 Training facts

Epochs to the best checkpoint, gate means at it, lane cost, the train-to-validation paralog residual and
the functional-edge median, read from the lanes and the graph:

```bash
PYTHONPATH=src python -m tcell_pipeline.screening.camera_ready_facts --out data/results/camera_ready/facts.json
```

## 5. Checking the paper against your files

```bash
cd paper/icbinb && python verify_numbers.py --check
```

It re-derives every table and anchored number in `main.tex` from the files in section 3 and prints
each disagreement. Against retrained lanes, expect disagreements in the last printed digit of
lane-level numbers; the conclusions are the contrasts, intervals and corrected p-values.
