# tcell-causal-ppi

**Perturbation-Informed Protein–Program Graphs for T Cell Interventions** — AAAI submission (AI track).

> We introduce an intervention-informed heterogeneous graph predictor for biological systems. Given a
> perturbation target, cell context, and typed protein-network neighborhood, it predicts transcriptional
> response programs, estimates calibrated uncertainty, and returns a minimal predictive subgraph whose
> removal measurably degrades the prediction.



## Idea

Learn a **context-conditioned, intervention-informed graph over proteins and transcriptional programs**, where:

- genetic perturbations provide **interventional supervision**,
- physical protein networks provide **uncertain structured priors**, and
- the model returns **calibrated predictions** plus a **faithful protein/program predictive rationale**.

The method sits between two literatures it does not duplicate: perturbation-response models predict
expression but rarely return a faithful typed mechanism, and protein-interaction models predict edges
but rarely learn from cell-state-specific intervention effects. The contribution is a **reusable AI
method** — intervention-gated, evidence-aware heterogeneous graph learning — with a genome-scale
primary-human T cell Perturb-seq dataset as the stress test, not the sole contribution.

**What the paper claims** is methodological: better out-of-distribution intervention prediction,
calibration, and explanation faithfulness than expression-only and static-network baselines. It does
**not** claim to have discovered the T cell regulatory network, that a learned edge is a true direct
biochemical interaction, or that deep learning beats all baselines.

> **Open feasibility risk (2026-07-14 literature refresh).** A July 2026 tabular benchmark that
> includes this exact CD4+ screen reports it as a **near-null-signal regime** — models barely separate
> from the mean baseline. Before H1 is frozen, the plan must confirm on development data that a
> target-specific signal is detectable above the perturbed mean; if it is not, a rigorous **negative
> benchmark** is an accepted outcome, not a delayed superiority claim. See the report's Risk Register
> and Limitations for the full evidence.

## Method — EG-IPG (Evidence-Gated Intervention-Informed Protein–Program Graph)

Formal task, per perturbation example:

```
f_theta( do(g), c, d, q_pre, N_k(g), x0 ) -> ( Y_hat, U_hat, S_hat )
```

| Symbol | Meaning |
|---|---|
| `do(g)` | experimentally assigned perturbation on target gene/protein `g` (denotes the intervention, not do-calculus identification of internal edges) |
| `c`     | culture condition ∈ {Rest, Stim8hr, Stim48hr} |
| `d`     | donor / donor-pair context |
| `q_pre` | prediction-time covariates known before observing the response (guide sequence, design features, control-derived baseline expression) |
| `N_k(g)`| typed protein-network neighborhood of `g` |
| `x0`    | optional baseline expression / pseudobulk context |
| `Y_hat` | predicted response (program-level and/or gene-level DE) |
| `U_hat` | calibrated uncertainty |
| `S_hat` | sparse predictive-rationale subgraph linking target protein to affected programs |

Four modules:

1. **Perturbation & context encoder** — target embedding + condition + donor + prediction-time quality → intervention vector.
2. **Typed graph encoder** — relational GNN / graph transformer with per-relation, condition-gated edges over sampled local neighborhoods.
3. **Program decoder** — predicts program deltas (+ optional gene-level deltas) with an expression-only residual pathway and a learned mixture gate.
4. **Sparse predictive-rationale head** — returns a subgraph trained for sufficiency/necessity, not post-hoc attention.

### Target representations

Three target layers, in order of priority for the first paper:

1. **Program-level response** — latent immune programs learned from DE stats using sparse PCA, NMF,
   ICA, or a shallow VAE fitted inside each training fold. Program counts: 64, 128, 256, 512 compared.
   Curated anchors: Th1, Th2, cytokine modules, activation, interferon, proliferation, exhaustion/aging.
2. **Gene-level DE response** — continuous prediction of `zscore` or `log_fc`; multilabel up/down DE
   calls. Used to decode from programs and check fine-grained accuracy.
3. **Distributional single-cell response** — follow-on only; not a first-submission dependency.

### Graph schema

Heterogeneous graph with node types: `gene/protein`, `transcriptional_program`, `protein_complex`,
`pathway`, optional `condition`, optional `donor`.

Edge types:

| Edge type | Source |
|---|---|
| Physical PPI | BioPlex, HuRI, Krogan, BioGRID physical |
| Co-complex | BioPlex AP-MS, CORUM complex membership |
| Functional association | STRING, pathway co-membership |
| Regulator-to-program | learned from perturbation effects |
| Gene-to-program loadings | from latent program model |
| Condition gates | context-specific modulation of edge weights |

Protein complexes modeled as bipartite complex nodes (simpler to implement, debug, and explain than
hypergraph message passing). PLM (ESM-2 650M, 1280-d) and PINNACLE (128-d) embeddings are precomputed
once into frozen Parquet stores and treated as frozen inputs — no end-to-end PLM fine-tuning in the first
paper. See *Setup → Precompute target embeddings* for generation.

### Loss function

Two frozen stages:

- **Stage A (H1 predictor)**: `L_pred = L_response + L_DE + L_invariance + L_graph`
- **Stage B (secondary heads)**: freeze the H1 checkpoint, then fit `L_calibration` and `L_rationale`
  separately. Any joint fine-tuning that changes H1 predictions is exploratory and cannot replace the
  frozen predictor.

> **As-built (Module 5, feat-008):** built in `src/tcell_pipeline/training/`. `L_response` is Huber at the
> program (`Δz` vs `z@B`) and gene (`Δx` vs z-score) levels; `L_DE` is a focal-BCE up/down head over
> `h_do`; `L_invariance` penalises the variance of `Δz` across the real per-donor control profiles (donor
> resampling — see §Train the H1 predictor); `L_graph` weights its unsourced term by the real per-edge
> source confidence. Stage B is a loss module only (no fit loop yet). Details:
> `docs/specs/2026-07-16-module5-training.md`; review history: `docs/reviews/2026-07-16-code-review-module5.md`.

A trainable target-ID embedding is **prohibited** in H1 (allowed only as a shortcut diagnostic in
negative controls) because it cannot generalize to unseen genes.

### Training splits

Multiple non-overlapping split families:

| Split | Purpose |
|---|---|
| Random within-condition | Sanity check only — not a publishable headline |
| Held-out target genes | Headline target-OOD, blocked by sequence/paralog similarity |
| Held-out protein families/complexes | Hardest biologically blocked generalization test |
| Held-out condition | Exploratory stress test (only 3 contexts) |
| Held-out donor | 4-fold descriptive sensitivity (not population generalization) |
| Low-network-degree | Test genes with sparse or missing PPI priors |
| Off-target / low-quality | Selective prediction and uncertainty on flagged perturbations |
| Joint target-and-context OOD | Unseen target in unseen context — exploratory |

Random target holdout is **diagnostic only**. Headline target-OOD must block family/sequence,
complex/pathway, and close graph-neighborhood similarity. All response-derived transformations
(program bases, scaling, feature selection) are fit inside training folds only.

### Baselines

**Mandatory simple baselines**: no-effect, control mean, per-condition mean, Systema perturbed mean,
Systema matching mean, nearest neighbor, ridge, elastic-net, low-rank matrix factorization, CatBoost,
TabPFN/TabICL (exact-CD4 pipeline), gene-wise majority, gene-embedding kNN, PPI diffusion.

**Core confirmatory comparator set** (frozen at G2): Systema perturbed mean, ridge/low-rank (whichever
wins development), strongest eligible exact-CD4 tabular model, expression-only MLP, typed static graph,
Stable-Shift, TxPert-public (if compatibility gate passes).

> **Comparator availability (2026-07-14 refresh).** Stable-Shift's first-party code was not confirmed
> available — the `Sajib-006/PerturbGraph` repo hosts the related **PerturbGraph** method, not
> Stable-Shift — so a row-compatible reimplementation may be required. TxPert-public reproduces only its
> STRING/GO public subset, not the proprietary-graph paper-best configuration. Both must be named
> exactly in any comparison (see feat-010).

**Conditional field baselines** (included only if adapter/license/exposure validated before G2): GEARS,
CPA, scGPT, scLDM.CD4, CRADLE-VAE, Departures, D-SPIN/RegFormer/GRNFormer.

Policy: never compare only to weak deep-learning baselines. Include a simple linear baseline in every
headline table. Record each baseline's inputs, pretraining exposure, inductive/transductive status,
checkpoint, and tuning budget.

> **As-built (Module 6, feat-006):** the six mandatory simple baselines — no-effect, Systema perturbed
> mean, per-condition mean, ridge, nearest-neighbor, low-rank — are built in
> `src/tcell_pipeline/baselines/simple_baselines.py` behind a common `fit(X, z, conditions) → predict →
> (Δz, Δx)` contract (gene space decoded through the frozen basis, `Δx = Δz @ Bᵀ`), writing the shared
> prediction schema `predictions/<model>/<split>/<seed>.parquet`. Elastic-net and CatBoost remain
> (feat-006 done-criterion); graph baselines are feat-007. Details:
> `docs/specs/2026-07-16-module6-evaluation.md`.

> **As-built (Module 7, feat-007):** the three **graph baselines** are built in
> `src/tcell_pipeline/baselines/graph_baselines.py`: **network propagation** (non-neural, symmetric-
> normalised PPI diffusion of training responses; predict = graph-proximity-weighted mean), the **untyped
> GNN diagnostic** (homogeneous GCN, all PPI edges collapsed to one relation — topology without evidence
> types), and the **typed static graph** (`TypedGraphEncoder` with the condition gate pinned to 1.0 —
> evidence types + topology, no condition gating). The two neural encoders drop into
> `EGIPGModel(graph_encoder=…)` and train through the existing Stage-A `Trainer`; all three emit the common
> prediction schema. Details: `docs/specs/2026-07-16-module7-screening.md`.

### Evaluation metrics

**Prediction**: MAE, RMSE, Pearson/Spearman, Systema perturbation-specific delta correlation, centroid
accuracy, top-k recall, signed-DE macro-F1/AUPRC, program-level cosine similarity.

**Control-reference safeguards**: if using control-relative deltas, use independent control estimates
and average over preregistered split seeds. Shared-control scores are bias diagnostics only.

**Experimental reproducibility references**: guide-split and donor-split agreement (labeled as
references, not upper bounds).

**Uncertainty**: ECE, Brier score, conformal coverage by subgroup (condition, donor, target degree,
expression, tail programs), selective prediction curves, worst-group coverage. Calibrate at the
generalization cluster, not individual rows.

**Predictive-rationale faithfulness**: necessity (removal degrades prediction), sufficiency (selected
subgraph preserves prediction), minimality (smallest sufficient subgraph), stability across seeds and
bootstrap, structural-OOD audit, matched random controls (degree, relation type, connectivity, hop
distance), source ablation.

> **As-built (Module 6, feat-009):** the prediction metrics (MAE/RMSE, Pearson/Spearman, Systema
> perturbation-specific delta, centroid accuracy, top-k recall, sign accuracy, signed-DE
> macro-F1/per-class P·R/AUPRC, program cosine), the **G2-MQ model-blind qualification gate** (§10.1), and
> the control-reference safeguards are built in `src/tcell_pipeline/evaluation/`, with a **second
> independent implementation** (`metrics_ref.py`) that must agree with the primary one on a fixed fixture.
> Metrics are per-row → macro-averaged; degenerate/non-finite rows contribute 0.0 for the higher-is-better
> metrics (so a zero predictor scores worst). Uncertainty metrics (ECE/Brier/conformal) wait on the Stage-B
> calibration fit. Details: `docs/specs/2026-07-16-module6-evaluation.md`.

## Data

### Primary dataset

Genome-scale primary human CD4+ T cell CRISPRi Perturb-seq (Marson / Pritchard / Zhu / Dann): ~22M
cells, 4 donors, three conditions (Rest, Stim8hr, Stim48hr), processed v1.0. Covers systematic
perturbation of all expressed genes. The preprint emphasizes context-specific regulators of T cell
programs, cytokines, polarization, immune traits, and autoimmune disease risk.

- Dataset card: <https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq>
- Preprint: <https://www.biorxiv.org/content/10.64898/2025.12.23.696273v1>
- Analysis repo: <https://github.com/emdann/GWT_perturbseq_analysis_2025>
- S3 bucket listing: <https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/?list-type=2&prefix=marson2025_data/&max-keys=1000>
- Data README: <https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/data_sharing_readme.md>

### Artifact inventory

The public S3 listing exposes 32 objects under `marson2025_data/`. Total HDF5 footprint is ~1,717 GiB.
The **practical entry point is the aggregate DE + pseudobulk layer (~100 GiB)**, not raw 22M-cell
modeling (~1.6 TiB).

| Artifact | File(s) | Approx size | Role |
|---|---|---:|---|
| DE stats | `GWCD4i.DE_stats.h5ad` | 15.6 GiB | **Core supervised target** (33,983 rows = perturbation×condition; 10,282 genes) |
| Pseudobulk | `GWCD4i.pseudobulk_merged.h5ad` | 41.5 GiB | First modeling layer (guide×donor×condition; 18,129 genes) |
| Guide-level DE | `GWCD4i.DE_stats.by_guide.h5mu` | 27.4 GiB | Guide-replicate robustness / uncertainty supervision |
| Donor-pair DE | `GWCD4i.DE_stats.by_donors.h5mu` | 15.7 GiB | Donor-transfer robustness / uncertainty supervision |
| Supplementary tables | `*.suppl_table.csv` | small | DE summary, sample & sgRNA metadata, QC, signatures |
| Cell-level (raw) | 12 × `D*_*.assigned_guide.h5ad` | ~1,617 GiB | **Phase 2 only** — distributional modeling / targeted validation |

### DE stats — key fields

`GWCD4i.DE_stats.h5ad` is the core supervised target. Rows are perturbed gene × culture condition
(`n_obs = 33,983`), columns are measured genes (`n_vars = 10,282`).

Important `.obs` fields:

- `target_contrast_gene_name`, `culture_condition`, `target_contrast` — stable join keys
- `n_cells_target`, `n_up_genes`, `n_down_genes`, `n_total_de_genes`
- `ontarget_effect_size`, `ontarget_significant`, `target_baseMean`
- `neighboring_gene_KD`, `distal_offtarget_flag`, `low_target_gex`
- `n_guides`, `single_guide_estimate`, `n_downstream`
- `guide_correlation_signif`, `guide_correlation_all`
- `donor_correlation_all_mean`, `donor_correlation_hits_mean`

`.layers`:

| Layer | Description |
|---|---|
| `log_fc` | log2 fold change |
| `zscore` | log fold-change divided by standard error |
| `p_value` | raw p-values |
| `adj_p_value` | FDR-adjusted p-values |
| `lfcSE` | log fold-change standard error |
| `baseMean` | normalized mean expression |

Preferred target representations:

- **Continuous**: clipped `zscore` and/or moderated `log_fc`
- **Binary/multilabel**: up/down DE calls from `adj_p_value < 0.1` and sign
- **Program-level**: latent program scores derived from `zscore` or `log_fc`

### Feature availability: `q_pre` vs `q_post`

A critical distinction for leakage prevention:

- **`q_pre`** — prediction-time covariates known *before* observing the response (guide sequence,
  design features, predicted off-target risk, control-derived baseline expression). These are
  eligible H1 features.
- **`q_post`** — response-derived statistics (on-target effect, on-target significance,
  guide/donor response correlations, neighboring-gene knockdown). These may define training-only QC,
  uncertainty supervision, or evaluation strata, but are **prohibited** as prediction inputs.

### Pseudobulk — key fields

`GWCD4i.pseudobulk_merged.h5ad`: rows are guide×donor×condition, `n_vars = 18,129`.

Key `.obs`: `donor_id`, `culture_condition`, `guide_id`, `perturbed_gene_name`, `guide_type`,
`n_cells`, `total_counts`, `log10_n_cells`, `keep_min_cells`, `keep_effective_guides`,
`keep_total_counts`, `keep_for_DE`, `keep_test_genes`.

### Guide-level and donor-pair MuData

- `GWCD4i.DE_stats.by_guide.h5mu` — guide-replicate robustness, aleatoric uncertainty, confidence
  weights for training loss.
- `GWCD4i.DE_stats.by_donors.h5mu` — donor-transfer robustness, evaluate graceful degradation when
  guide or donor evidence is weak.

### Supplementary analysis tables

`data/raw/suppl_tables/` holds 15 tables — 3 describe the screen (S3), 12 are derived analysis
outputs from the GitHub repo. The derived tables are ready-made biological supervision:

| Table(s) | Role in EG-IPG |
|---|---|
| `CD4T_aging_signature`, `Th2_Th1_polarization_signature`, `IL10IL21bulkRNAseq` | **Program anchors** (aging, Th1/Th2, cytokine) for program-level targets |
| `*_regulator_coefficients`, `clustering_downstream_genes` (1.18M rows) | **Regulator→program** edge supervision; downstream genes carry sign coherence |
| `clustering_results_and_annotations` (112 clusters ↔ CORUM/STRING/KEGG/Reactome), `cluster_autoimmune_enrichment` | **Complex/cluster priors** + biological-alignment / autoimmune-trait metrics |
| `guide_kd_efficiency` | Per-guide KD vs NTC — a **`q_post`** control-weighting/QC source, never an H1 input |
| `K562_comparison` | **Cross-cell-type** generalization reference (CD4 vs K562) |

`QC_summaries_per_sample_lane.csv` and `Th1Th2_validation_summary.suppl_table.csv` are named in the
data README but are not published on S3 or GitHub. Inspect what's present with
`examples/inspect_suppl_tables.py` (core 3) and `examples/inspect_analysis_tables.py` (derived);
`examples/dataset_overview.py` prints the full local-vs-expected inventory.

### Protein-network priors

Built as **typed, confidence-aware** sources, not one undifferentiated edge list:

| Source | Contributes | Notes |
|---|---|---|
| **BioPlex 3.0** | AP-MS co-complex interactome (~120k interactions, ~15k proteins) | Broad physical/co-complex prior |
| **HuRI** | Binary protein-protein interactome | Direct pairwise interactions; complements AP-MS |
| **BioGRID** | Curated physical/genetic/chemical interactions and PTMs | Broad evidence coverage |
| **STRING** | Known and predicted functional associations | Typed as functional association, not physical contact |
| **CORUM** | Manually annotated mammalian protein complexes | Convert to hyperedges or bipartite complex nodes |
| **PINNACLE** | Contextualized protein representations (156 cell types, 24 tissues) | Frozen 128-d target embedding; CD4 helper T-cell context used (see *Precompute target embeddings*) |
| **PRING** | Graph-level PPI benchmark | External graph diagnostics and leakage-aware reconstruction |
| **Krogan Lab** | Partner-relevant AP-MS/proximity/proteomics maps | Only under cleared license/terms; include MiST/SAINT scores |

IDs harmonized across Ensembl → HGNC → UniProt → Entrez. Per-edge provenance fields:
`source`, `evidence_type`, `score`, `is_physical`, `is_functional`, `is_complex`,
`is_direct_binary`, `cell_type_or_context`. Avoid collapsing all sources into one adjacency
matrix — typed edge learning is part of the AI contribution.

### Early sanity checks

- Count how many perturbed targets map from Ensembl to HGNC and UniProt
- Count graph degree distribution for perturbed targets by evidence source
- Confirm non-targeting controls and low-quality/low-expression perturbations are labeled correctly
- Confirm that the same split never leaks a held-out gene through a same-complex or same-family
  near-duplicate
- Compare results using `zscore`, `log_fc`, and binary DE calls — a method that only wins on one
  fragile response encoding is risky

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

Install the [Harness Engineering skill](https://walkinglabs.github.io/learn-harness-engineering/en/skills/) — it provides the `harness-creator` skill used to generate this repo's agent scaffolding (AGENTS.md, feature_list.json, progress.md, init.sh, session-handoff.md):

```bash
npx skills add walkinglabs/learn-harness-engineering --skill harness-creator
```

The skill ships templates and Node.js scripts for creating, auditing, and benchmarking agent harnesses. After installing, you can regenerate or validate the harness with:

```bash
node skills/harness-creator/scripts/create-harness.mjs --target .
node skills/harness-creator/scripts/validate-harness.mjs --target .
```

**After each commit**, use the `harness-creator` skill to update `progress.md`, `session-handoff.md`, and `feature_list.json` so the next session can pick up cleanly. Validate the harness to confirm all five subsystems still pass:

```bash
node skills/harness-creator/scripts/validate-harness.mjs --target .
```

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment:

```bash
git clone https://github.com/Vijayavallabh/tcell-causal-ppi
cd tcell-causal-ppi
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv tool install awscli
```

**GPU (embeddings + encoder).** ESM-2 PLM embedding generation and the Module 1 encoder both run far
faster on a GPU. `torch` from the default index is built for the newest CUDA and may not match an older
driver — if `python -c "import torch; print(torch.cuda.is_available())"` prints `False` on a GPU host,
install the torch build matching the driver's CUDA version. On this host (NVIDIA driver 535 / CUDA 12.2)
that is the CUDA-12.6 build (runs on 12.2 via minor-version compatibility):

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126"
```

`embeddings_plm` uses the GPU automatically. `PerturbationEncoder` is device-aware: `enc.to("cuda")`
runs the whole forward on GPU; with no `.to()` it stays on CPU, so tests and GPU-less hosts work unchanged.

### Download data (staged)

**Processed layer (~100 GiB — recommended first).** Pulls the four aggregate HDF5 artifacts (DE,
pseudobulk, guide/donor MuData), excluding only the ~1,617 GiB raw cell-level files. Supplementary
tables and metadata are fetched in the next two steps.

```bash
mkdir -p data/raw logs

nohup bash -c '
set -u

export AWS_RETRY_MODE=adaptive
export AWS_MAX_ATTEMPTS=50

for f in \
  GWCD4i.DE_stats.h5ad \
  GWCD4i.DE_stats.by_donors.h5mu \
  GWCD4i.DE_stats.by_guide.h5mu \
  GWCD4i.pseudobulk_merged.h5ad
do
  echo "[$(date -Is)] Starting: $f"

  while true; do
    rm -f "data/raw/$f"

    if aws s3 cp \
      "s3://genome-scale-tcell-perturb-seq/marson2025_data/$f" \
      "data/raw/$f" \
      --no-sign-request \
      --cli-connect-timeout 60 \
      --cli-read-timeout 900 \
      --only-show-errors
    then
      echo "[$(date -Is)] Completed: $f"
      break
    fi

    echo "[$(date -Is)] Failed: $f; retrying in 5 minutes"
    sleep 300
  done
done

echo "[$(date -Is)] All downloads completed."
' > logs/marson_large_files.log 2>&1 < /dev/null &
```

**Supplementary tables + metadata.** The S3 `suppl_tables/` prefix hosts only 3 of the tables described
in the data README (`DE_stats`, `sample_metadata`, `sgrna_library_metadata`) plus the 12 Croissant
`metadata/*.jsonld`. Sync those:

```bash
aws s3 sync s3://genome-scale-tcell-perturb-seq/marson2025_data/suppl_tables/ data/raw/suppl_tables/ --no-sign-request
aws s3 sync s3://genome-scale-tcell-perturb-seq/marson2025_data/metadata/      data/raw/metadata/      --no-sign-request
aws s3 cp   s3://genome-scale-tcell-perturb-seq/marson2025_data/data_sharing_readme.md data/raw/ --no-sign-request
```

The remaining supplementary tables (guide-KD efficiency, the aging / Th1-Th2 signatures, autoimmune
enrichment, regulator coefficients, downstream-gene clustering, K562 comparison) are **not on S3** —
they live in the GitHub analysis repo. This step downloads every table in `metadata/suppl_tables/` not
already present locally:

```bash
mkdir -p data/raw/suppl_tables

curl -s "https://api.github.com/repos/emdann/GWT_perturbseq_analysis_2025/contents/metadata/suppl_tables" \
| python3 -c "import sys, json; [print(x['download_url']) for x in json.load(sys.stdin)]" \
| while read -r url; do
    name=$(basename "$url")
    dest="data/raw/suppl_tables/$name"
    if [ -f "$dest" ]; then
      echo "[skip present] $name"
    else
      echo "[download] $name"
      curl -sL --retry 5 --fail "$url" -o "$dest" || { echo "FAILED: $name"; rm -f "$dest"; }
    fi
  done
```

> Two tables named in the data README — `QC_summaries_per_sample_lane.csv` and
> `Th1Th2_validation_summary.suppl_table.csv` — are not published on S3 or in the GitHub repo, so they
> cannot be fetched automatically. Run `python examples/dataset_overview.py` to see the current
> local-vs-expected inventory.

**Raw cell-level files (~1,617 GiB — Phase 2, optional).** Fetch a single donor×condition object on
demand, e.g.:

```bash
aws s3 cp \
  s3://genome-scale-tcell-perturb-seq/marson2025_data/D1_Rest.assigned_guide.h5ad \
  ./data/raw/ --no-sign-request
```

### Precompute target embeddings (ESM-2 + PINNACLE)

The Module 1 encoder consumes two **frozen, pluggable** per-protein embedding stores. Both are optional —
when a store's Parquet is absent the encoder falls back to zero vectors, so training and tests run
unchanged — but real embeddings are generated by two standalone, resumable modules. Run after Module 0
has produced `data/intermediate/perturbation_condition.parquet` and `id_mapping.parquet`:

```bash
# ESM-2 650M protein language-model embeddings (1280-d), one per UniProt accession in the mart.
# Fetches sequences from UniProt REST, embeds on GPU when available, resumable (atomic checkpoints).
PYTHONPATH=src python -m tcell_pipeline.embeddings_plm

# PINNACLE cell-type-contextualised embeddings (128-d) for the CD4 helper T-cell context.
# Downloads the ~1.2 GB Figshare resource on first run, maps gene symbols -> UniProt via id_mapping.
PYTHONPATH=src python -m tcell_pipeline.embeddings_pinnacle
```

Outputs `data/intermediate/{plm,pinnacle}_embeddings.parquet` (`uniprot_id`, `embedding` columns), picked
up automatically by `PluggableEmbeddingStore` at the paths in `config.py`. PLM covers all mart proteins;
PINNACLE is contextual (~1,070 of the screen's proteins fall in the CD4 helper context — the rest keep the
zero fallback). Both are gitignored under `data/intermediate/` and fully regenerable. The PINNACLE context
is configurable via `config.PINNACLE_CONTEXT`.

### Verify Module 1 on the real data

`run_module1_smoke.py` is the Module 1 analogue of `run_module0.py`: it drives all 33,983 real
perturbation-condition rows through `PerturbationEncoder`, reporting PLM/PINNACLE coverage, asserting
every `h_do` is finite (the NaN guard), and checking the leakage fence rejects the mart's real `q_post`
columns. Uses the GPU automatically when available. Exits non-zero on any NaN or fence breach.

```bash
python src/tcell_pipeline/run_module1_smoke.py
```

Unit-level checks (real PLM/PINNACLE parquets + real marts, no synthetic fixtures) run under `./init.sh`
alongside the rest of the suite.

### Verify Module 2 (typed graph encoder) on the real data

`run_module2_smoke.py` builds the full HeteroData graph from the real PPI marts, samples the CD3E
neighbourhood, runs real Module 1 `h_do` through the `TypedGraphEncoder`, and checks `h_graph` is
finite, the readout attention sums to 1, and the **same edge is gated differently across culture
conditions** (the condition gate is the module's core claim). GPU-automatic; exits non-zero on failure.

```bash
python src/tcell_pipeline/graph/run_module2_smoke.py
```

### Fit the fold-local program basis (Module 3)

The program decoder predicts deltas in a latent-program space defined by a basis learned from the
**training-fold DE matrix only** (`Z_train ≈ A·Bᵀ`; README §Target representations, §Training splits).
`run_program_basis` loads the blocked split, keeps train-role rows (with an independent fold-leak check
that raises if any selected row's gene is not train-role), fits the basis, and writes
`data/intermediate/{gene_program_loadings,program_response}.parquet`:

```bash
# paper default: sparse PCA at K=128 (MiniBatchSparsePCA, ~5 min measured on the full 21k-row train set)
PYTHONPATH=src python -m tcell_pipeline.programs.run_program_basis

# fast alternatives for iteration: --method {svd,nmf,fastica}, --K {64,128,256,512}
PYTHONPATH=src python -m tcell_pipeline.programs.run_program_basis --method svd
```

Methods compared (§Target representations): sparse PCA (default), NMF, ICA, SVD. The gene axis of `B`
is the full `de_var` order, so it drops straight into the decoder's frozen loading buffer.

As-built: the production `sparse_pca` basis has been fitted on the real train fold (289 s, K=128) →
`B` (10,282×128) / `A` (21,262×128), all finite, fold-locality exact (saved response rows == the 21,262
train rows), ~23% exact-zero loadings with no dead programs; centered reconstruction MAE 0.687 vs a
0.817 predict-zero baseline (sparse coding trades reconstruction for sparsity vs SVD's ~0.61). The
`data/intermediate/*` parquets are gitignored — regenerate with the command above. The 4-method × 4-K
comparison study and the shallow-VAE basis (feat-005 done-criterion) are still future work.

### Verify Module 3 (program decoder) end to end

`run_module3_smoke.py` fits a fast fold-local SVD basis on the real train rows, assembles the full
`EGIPGModel` (Modules 1+2+3) on the real graph, and forwards 4 real perturbations — checking every
output (`delta_z`, `delta_x`, `sigma`, `lambda`) is finite with `lambda ∈ [0,1]` and `sigma > 0`, and
that the expression-only nested variant (`graph_encoder=None`) pins `lambda` to 0. Design + as-built:
`docs/specs/2026-07-15-module3-program-decoder.md`.

```bash
python src/tcell_pipeline/run_module3_smoke.py
```

### Verify Module 4 (sparse predictive-rationale head) end to end

`run_module4_smoke.py` builds the `EGIPGModel` on the real PPI graph, extracts a sparse rationale over a
real perturbation's neighbourhood (`RationaleHead`), and runs the **fixed-model** faithfulness tests
(`FaithfulnessTester`) against size- and relation-matched controls (`MatchedRandomSampler`) — checking
sufficiency < matched-random, necessity > matched-random, a structural-OOD audit, and that the output is
labelled `predictive_rationale`, never `causal`. This is a **predictive rationale, not a causal
mechanism** — Stage B, fitted after the H1 predictor freeze; deletion scores are *fixed-model
perturbation tests*, not interventions. Design + as-built: `docs/specs/2026-07-16-module4-rationale-head.md`.

```bash
python src/tcell_pipeline/rationale/run_module4_smoke.py
```

The synthetic unit checks (`src/tests/test_rationale.py`) run under `./init.sh` with the rest of the suite.

### Train the H1 predictor (Module 5 — Stage A)

Module 5 makes the four model modules trainable (README §Loss function; walkthrough §8). **Stage A**
fits Module 1+2+3 with `StageALoss` — Huber response (program + gene) + a focal-BCE DE up/down head +
**donor-invariance** + an edge-gate sparsity/unsourced regulariser — over AdamW with grad-clipping, early
stopping, and atomic best/last checkpoints to `data/checkpoints/`. Supervision is q_pre-only
(`PerturbationDataset` enforces the leakage fence); `Δz_true = z@B` (the response projected onto the frozen
fold-local loadings) consistently for every row, so train and validation measure the same quantity. The
graph regulariser's unsourced-reliance term reads the **real per-edge source confidence** (threaded from
the graph encoder), down-weighting well-sourced edges. The donor-invariance term is a **real signal**: the
mart's `donor_pc` is only the per-condition mean, but the 4 real donors survive in
`control_donor_profiles.parquet`, so the trainer (in the train step) resamples the encoder over distinct
real donors and penalises the **variance of the prediction `Δz` across them directly** — forcing the
encoder to emit donor-invariant predictions (`--no-donor-invariance` opts out).
**Stage B** (Gaussian-NLL calibration + `RationaleLoss`) are loss modules only — fitted after the H1
freeze; their fit loops are feat-008's last piece. Design + as-built:
`docs/specs/2026-07-16-module5-training.md`.

```bash
# expression-only nested variant, quick real-data smoke (no graph encoder)
PYTHONPATH=src python -m tcell_pipeline.training.run_train --expr-only --n-max 256 --epochs 3

# full M1→M2→M3 Stage A (graph path is CPU-bound per subgraph — cap for a smoke)
PYTHONPATH=src python -m tcell_pipeline.training.run_train --n-max 4 --epochs 1
```

The synthetic unit checks (`src/tests/test_training.py`) run under `./init.sh` with the rest of the suite.

### Evaluate the model + baselines (Module 6)

`run_module6_smoke.py` scores the trained Stage-A model and the six simple baselines on the real
validation fold with the Module 6 metrics, then runs the **G2-MQ** metric-qualification gate, the §10.5
control-reference safeguards, and the common prediction-schema roundtrip. The model forward runs on
`--device cuda`; the baselines + metrics are numpy/sklearn (CPU). Needs the Stage-A checkpoint from the
previous step. Design + as-built: `docs/specs/2026-07-16-module6-evaluation.md`.

```bash
PYTHONPATH=src python -m tcell_pipeline.run_module6_smoke --device cuda
```

On the real val fold (4,400 rows) the G2-MQ `systema` gate passes (every negative below guide-split-half
and oracle), the null-control predictor scores ~0 under independent controls, and **ridge is currently the
strongest baseline, edging a lightly-trained model** — the near-null-signal regime the report anticipates
until the H1 predictor is trained to convergence. The synthetic unit checks (`src/tests/test_metrics.py`,
`src/tests/test_baselines.py`) run under `./init.sh`.

### Screen the nested family (Module 7 — feat-011)

`screening/run_screening.py` trains and scores the §10.6 nested confirmatory family — expression-only,
typed-static, condition-gated — plus the untyped-GNN diagnostic and the non-neural network-propagation
reference on one fold, and reports **H2a** (typed-static > expression-only) and **H2b** (condition-gated >
typed-static) on the `systema` primary endpoint. Every run is logged in the immutable experiment registry
(`data/results/experiment_registry.yaml`) under the report's **32-trial EG-IPG / 16-per-comparator caps**
(counted by distinct config, so dev re-runs don't exhaust the budget); the harness is **failure-isolating**
(a config that OOMs is logged failed and the wave continues). Design + as-built:
`docs/specs/2026-07-16-module7-screening.md`.

```bash
PYTHONPATH=src python -m tcell_pipeline.screening.run_screening --epochs 1 --batch-size 8 --device cuda
```

**Compute reality.** The typed graph encoders sample a ≤512-node subgraph *per row* and message-pass
single-threaded on CPU (GPU util ~0%), so the graph configs are hours-long on the full 21,262-row fold —
the untyped GNN did not finish one epoch in ~11 h. For a tractable comparison, cap the fold (`--n-max
1000`) and/or run the configs in parallel across the four A100s (one per GPU — ~4× faster wall-clock, ~55
min on 1,000 rows). On a 1,000-row / 1-epoch fold H2a is a hair positive (+0.001 systema) and H2b negative
(−0.006) — noise in the near-null-signal regime; the genuine H2a/H2b test needs convergent training, which
needs the deferred graph-encoder mini-batching upgrade (PyG `Batch` over subgraphs). `run_full_pipeline.sh`
runs Modules 1-7 unattended under nohup. Review: `docs/reviews/2026-07-16-code-review-module7.md`.

## Repository / data-mart layout

Downloads stay immutable under `data/raw/`; everything else is derived and reproducible. `data/` is
git-ignored (see `.gitignore`) except small, reproducibility-critical `manifests/` and `splits/`.

Data paths must be configured, not hard-coded. A practical separation is `$PROJECT_ROOT` for code and
compact outputs, `$DATA_ROOT` for immutable aggregate artifacts, and `$SCRATCH_ROOT` for temporary
shards and training caches. Record resolved absolute paths in every run manifest.

```
data/
  raw/                 # downloaded .h5ad / .h5mu / .csv, unchanged
  manifests/           # object list, source URL, size, ETag/checksum, download date  (tracked)
  intermediate/        # extracted DE .obs/.var; sparse zscore/log_fc/adj_p/lfcSE; program matrices
  graphs/              # normalized typed PPI edge tables by source, complex membership, merged hetero-graph
  splits/              # frozen train/val/test split definitions                       (tracked)
  results/             # baseline + model outputs, calibration tables, explanation subgraphs
  checkpoints/         # run-scoped best and last checkpoints; optimizer state is temporary
  raw_cell_cache/      # optional rolling cache for bounded raw-cell shard experiments (disabled by default)
logs/                  # one machine-readable metrics stream + one human-readable log per run
```

### Storage budget (1TB remote volume)

| Class | Soft cap | Policy |
|---|---:|---|
| Immutable aggregate artifacts | 105 GiB | DE, pseudobulk, guide/donor, supplementary tables only |
| Derived marts, programs, graphs, embeddings | 120 GiB | Prefer sparse NPZ/Zarr or Parquet shards; deduplicate |
| Checkpoints and optimizer states | 160 GiB | Keep `best` + `last` per active run; remove superseded |
| Predictions, calibration, explanations, logs, figures | 80 GiB | Store row-level predictions once; derive summaries without copying |
| Environments and package caches | 30 GiB | Pin one environment; no per-run duplication |
| Temporary download/conversion/shuffle | 100 GiB | Never long-lived results here |
| Optional rolling raw-cell cache | 180 GiB max | Disabled by default; evict after verification |

Default program/DE workflow uses ~595 GiB before the optional raw-cell cache. Preserve at least 15%
free space at all times.

### Minimal derived tables

| Table | Grain | Required columns |
|---|---|---|
| `perturbation_condition.parquet` | one row per perturbation-condition | Target/context IDs, `q_pre` design features, `q_post` QC fields (schema-enforced separation) |
| `gene_response_sparse.npz` | perturbation-condition × gene | z-score, log fold-change, adj p-value, standard error |
| `program_response.parquet` | perturbation-condition × program | latent program delta, uncertainty target, DE summary |
| `gene_program_loadings.parquet` | gene × program | loading, sign, rank, program label |
| `protein_edges.parquet` | source protein × target protein × evidence type | source, confidence, physical/functional flag, binary/co-complex flag, context |
| `complex_membership.parquet` | protein × complex | complex ID, source database, confidence/curation status |

### Loading and storage rules

- Use backed/lazy reads for AnnData exploration
- Never densify full DE layers outside a gene/program subset
- Extract matrices into chunked sparse formats or Parquet/NPZ shards keyed by perturbation-condition row
- Keep `target_contrast`, `target_contrast_gene_name`, `culture_condition`, and response-row index as
  stable join keys
- Treat `adj_p_value`, `lfcSE`, guide/donor reproducibility, off-target flags, and low-expression
  flags as first-class modeling columns, not after-the-fact QC notes
- Download with resumable transfer; verify object size plus ETag/checksum before moving to `raw/`
- Write large derived artifacts atomically (temp path → rename after validation) so interrupted SSH
  sessions cannot leave a file that appears complete
- Do not keep a checkpoint for every epoch — retain best validation + latest resumable for active runs;
  archive final inference weights without optimizer state
- Copy back only code, manifests, aggregate metrics, compact predictions, and final figures. Keep large
  source and scratch artifacts remote

## Responsible use

The VCP dataset card states the processed dataset does not contain PII. Still, donor-related metadata
and immune-trait interpretation require careful handling:

- Report uncertainty and population-context limitations for any disease-risk analysis
- Avoid stigmatizing donor groups or making individual-level health claims
- Do not treat learned graph edges as validated biological mechanisms — they are predictive rationales
  unless independently validated by temporal, mediator, combinatorial-intervention, or wet-lab evidence
- Do not redistribute partner protein maps (e.g., Krogan) or derived edge tables until license terms
  are explicitly cleared
- Contributor roles should be recorded using CRediT-compatible taxonomy; authorship and institutional
  affiliation require documented contributions and explicit approval
