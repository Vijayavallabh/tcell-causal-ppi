# tcell-causal-ppi

**Perturbation-Informed Causal Protein–Program Graphs for T Cell Interventions** — AAAI submission (AI track).

> We introduce a causal heterogeneous graph learner for interventional prediction in biological
> systems. Given a perturbation target, cell context, and typed protein-network neighborhood, it
> predicts counterfactual transcriptional programs, estimates calibrated uncertainty, and returns a
> minimal causal subgraph whose removal measurably degrades the prediction.


## Status

Planning + reproducibility scaffold. At this stage the repo contains the environment definition and a
staged data download; model code is not committed yet. See the [Roadmap](#roadmap) for what lands next.

## Idea

Learn a **context-conditioned causal graph over proteins and transcriptional programs**, where:

- genetic perturbations provide **interventional supervision**,
- physical protein networks provide a **mechanistic prior**, and
- the model returns **calibrated counterfactual predictions** plus a **faithful protein/program
  subgraph explanation**.

The method sits between two literatures it does not duplicate: perturbation-response models predict
expression but rarely return a faithful typed mechanism, and protein-interaction models predict edges
but rarely learn from cell-state-specific intervention effects. The contribution is a **reusable AI
method** — intervention-gated, evidence-aware heterogeneous graph learning — with a genome-scale
primary-human T cell Perturb-seq dataset as the stress test, not the sole contribution.

**What the paper claims** is methodological: better out-of-distribution intervention prediction,
calibration, and explanation faithfulness than expression-only and static-network baselines. It does
**not** claim to have discovered the T cell regulatory network, that a learned edge is a true direct
biochemical interaction, or that deep learning beats all baselines.

## Method — EG-CProG (Evidence-Gated Causal Protein–Program Network)

Formal task, per perturbation example:

```
f_theta( do(g), c, d, q, N_k(g), x0 ) -> ( Y_hat, U_hat, S_hat )
```

| Symbol | Meaning |
|---|---|
| `do(g)` | intervention on target gene/protein `g`, modeled as an intervention token (not just a gene label) |
| `c`     | culture condition ∈ {Rest, Stim8hr, Stim48hr} |
| `d`     | donor / donor-pair context |
| `q`     | perturbation-quality covariates (on-target strength, off-target flags, guide/donor reproducibility, …) |
| `N_k(g)`| typed protein-network neighborhood of `g` |
| `x0`    | optional baseline expression / pseudobulk context |
| `Y_hat` | predicted counterfactual response (program-level and/or gene-level DE) |
| `U_hat` | calibrated uncertainty |
| `S_hat` | sparse explanatory subgraph linking target protein to affected programs |

Four modules:

1. **Perturbation & context encoder** — target embedding + condition + donor + quality → intervention vector.
2. **Typed graph encoder** — relational GNN / graph transformer with per-relation, condition-gated edges over sampled local neighborhoods.
3. **Program decoder** — predicts program deltas (+ optional gene-level deltas) with an expression-only residual pathway and a learned mixture gate.
4. **Sparse explanation head** — returns a subgraph trained for sufficiency/necessity, not post-hoc attention.

## Data

### Primary dataset

Genome-scale primary human CD4+ T cell CRISPRi Perturb-seq (Marson / Pritchard / Zhu / Dann): ~22M
cells, 4 donors, three conditions (Rest, Stim8hr, Stim48hr), processed v1.0.

- Dataset card: <https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq>
- Preprint: <https://www.biorxiv.org/content/10.64898/2025.12.23.696273v1>
- Analysis repo: <https://github.com/emdann/GWT_perturbseq_analysis_2025>
- Data README: <https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/data_sharing_readme.md>

### Artifact inventory (approx. sizes per the data resource)

The **practical entry point is the DE + pseudobulk layer, not raw 22M-cell modeling** — that processed
subset is ~100 GiB versus ~1.7 TiB for everything.

| Artifact | File(s) | Approx size | Role |
|---|---|---:|---|
| DE stats | `GWCD4i.DE_stats.h5ad` | 15.6 GiB | **Core supervised target** (rows = perturbation×condition; layers: `log_fc`, `zscore`, `p_value`, `adj_p_value`, `lfcSE`, `baseMean`) |
| Pseudobulk | `GWCD4i.pseudobulk_merged.h5ad` | 41.5 GiB | Baseline features; donor/guide variance; quality masks |
| Guide-level DE | `GWCD4i.DE_stats.by_guide.h5mu` | 27.4 GiB | Guide-replicate robustness / aleatoric uncertainty |
| Donor-pair DE | `GWCD4i.DE_stats.by_donors.h5mu` | 15.7 GiB | Donor-transfer robustness / uncertainty |
| Supplementary tables | `*.suppl_table.csv` | small | DE summary, sample & sgRNA metadata, QC, signatures |
| Cell-level (raw) | 12 × `D*_*.assigned_guide.h5ad` | ~1.58 TiB | **Phase 2 only** — distributional modeling / targeted validation |

### Protein-network priors

Built as **typed, confidence-aware** sources, not one undifferentiated edge list:

- **BioPlex** (AP-MS co-complex), **HuRI** (binary), **BioGRID** (curated physical/genetic), **STRING**
  (functional association — typed as such, not physical contact), **CORUM** (complexes → hyperedges /
  bipartite complex nodes).
- **PINNACLE** context-aware protein embeddings for initialization; **PRING** for graph-level diagnostics.
- **Krogan** partner maps only under cleared license/terms.

IDs harmonized across Ensembl ↔ HGNC ↔ UniProt ↔ Entrez, with per-edge provenance
(`source`, `evidence_type`, `score`, `is_physical`, `is_functional`, `is_complex`, `is_direct_binary`).

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

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
supplementary tables, and metadata, while excluding only the ~1.58 TiB raw cell-level files:

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

**Raw cell-level files (~1.58 TiB — Phase 2, optional).** Drop the `--exclude` above, or fetch a single
donor×condition object on demand, e.g.:

```bash
aws s3 cp \
  s3://genome-scale-tcell-perturb-seq/marson2025_data/D1_Rest.assigned_guide.h5ad \
  ./data/raw/ --no-sign-request
```

## Repository / data-mart layout

Downloads stay immutable under `data/raw/`; everything else is derived and reproducible. `data/` is
git-ignored (see `.gitignore`) except small, reproducibility-critical `manifests/` and `splits/`.

```
data/
  raw/            # downloaded .h5ad / .h5mu / .csv, unchanged
  manifests/      # object list, source URL, size, ETag/checksum, download date  (tracked)
  intermediate/   # extracted DE .obs/.var; sparse zscore/log_fc/adj_p/lfcSE; program matrices
  graphs/         # normalized typed PPI edge tables, complex membership, merged hetero-graph
  splits/         # frozen train/val/test split definitions                       (tracked)
  results/        # baseline + model outputs, calibration tables, explanation subgraphs
logs/             # download / training logs
```

Loading rules: use backed/lazy AnnData reads; never densify full DE layers outside a gene/program
subset; treat `adj_p_value`, `lfcSE`, and guide/donor reproducibility as first-class modeling columns.

## Roadmap

Staged to de-risk compute and prove the graph method before any raw-cell work (full detail in the
[report](perturbation_informed_causal_protein_program_graphs_report.md)):

- **Tier 0 — Data audit & linear baselines** (CPU): manifest, ID map, split suite, program basis,
  no-effect / condition-mean / ridge / elastic-net / nearest-neighbor / network-propagation baselines.
- **Tier 1 — Program-level graph model** (1 GPU, 24–48 GB): train EG-CProG on latent programs; ablate
  no-graph / untyped / static-edge / condition-gated variants on held-out gene & condition splits.
- **Tier 2 — Gene-level decoder & optional distributional add-on** (multi-GPU): decode programs to the
  10,282 measured genes; optional cell-level HVG demo.
- **Tier 3 — Full raw-cell model** — explicitly **not** a first-paper dependency; use scLDM.CD4 as the
  cell-level generative baseline.

Evaluation is split-family–driven (held-out genes, held-out complexes/pathways, condition transfer,
donor transfer, low-network-degree, off-target/low-quality stress) and scored on prediction,
calibration, mechanism recovery, and explanation faithfulness — with simple linear baselines in every
headline table.

## Responsible use

The processed dataset is listed by VCP as containing no PII, but donor metadata and immune-trait
interpretation require care: report uncertainty, avoid individual-level health claims, and do not treat
learned graph edges as validated biological mechanisms. Do not redistribute partner protein maps or
derived edge tables until license terms are explicitly cleared.
