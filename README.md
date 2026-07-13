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
hypergraph message passing). PLM and PINNACLE embeddings precomputed once, stored in `float16`, and
treated as frozen inputs — no end-to-end PLM fine-tuning in the first paper.

### Loss function

Two frozen stages:

- **Stage A (H1 predictor)**: `L_pred = L_response + L_DE + L_invariance + L_graph`
- **Stage B (secondary heads)**: freeze the H1 checkpoint, then fit `L_calibration` and `L_rationale`
  separately. Any joint fine-tuning that changes H1 predictions is exploratory and cannot replace the
  frozen predictor.

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

**Conditional field baselines** (included only if adapter/license/exposure validated before G2): GEARS,
CPA, scGPT, scLDM.CD4, CRADLE-VAE, Departures, D-SPIN/RegFormer/GRNFormer.

Policy: never compare only to weak deep-learning baselines. Include a simple linear baseline in every
headline table. Record each baseline's inputs, pretraining exposure, inductive/transductive status,
checkpoint, and tuning budget.

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

### Protein-network priors

Built as **typed, confidence-aware** sources, not one undifferentiated edge list:

| Source | Contributes | Notes |
|---|---|---|
| **BioPlex 3.0** | AP-MS co-complex interactome (~120k interactions, ~15k proteins) | Broad physical/co-complex prior |
| **HuRI** | Binary protein-protein interactome | Direct pairwise interactions; complements AP-MS |
| **BioGRID** | Curated physical/genetic/chemical interactions and PTMs | Broad evidence coverage |
| **STRING** | Known and predicted functional associations | Typed as functional association, not physical contact |
| **CORUM** | Manually annotated mammalian protein complexes | Convert to hyperedges or bipartite complex nodes |
| **PINNACLE** | Contextualized protein representations (156 cell types, 24 tissues) | Context-aware protein embedding initialization |
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

### Download data (staged)

**Processed layer (~100 GiB — recommended first).** Pulls DE, pseudobulk, guide/donor MuData,
supplementary tables, and metadata, while excluding only the ~1,617 GiB raw cell-level files:

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

**Raw cell-level files (~1,617 GiB — Phase 2, optional).** Fetch a single donor×condition object on
demand, e.g.:

```bash
aws s3 cp \
  s3://genome-scale-tcell-perturb-seq/marson2025_data/D1_Rest.assigned_guide.h5ad \
  ./data/raw/ --no-sign-request
```

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
