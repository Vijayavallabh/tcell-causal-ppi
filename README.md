# tcell-causal-ppi

Do typed protein-interaction graph priors help predict how primary human T cells respond to genetic
perturbations, over an expression-only baseline? **EG-IPG** (Evidence-Gated Intervention-Informed
Protein-Program Graph) is built to give the graph its best chance, then tested under a bias-aware metric
on a leakage-safe target-out-of-distribution split.

**Result: a well-controlled null.** On the genome-scale CD4+ T-cell Perturb-seq screen of Zhu et al.
(2025), with live edge gates and family-wise error control (Bonferroni and Holm), the evidence-gated
graph is at **parity** with no-graph (Δsystema = −0.0009, 95% CI [−0.0072, +0.0054], p = 0.71, n = 5);
a typed *static* graph is reliably **worse**. Write-up: `paper/main.tex`; full account: `RESULTS_SUMMARY.md`.

> A prior comparison was confounded: an unnormalised edge-sparsity regulariser drove the edge gates to
> ~1e-7 in epoch 0, silently switching the graph off. Setting `lambda_graph=0` keeps the gates live (0.57
> to 0.77), and repairing it left the null unchanged. See the paper's Figure 2.

## Requirements

- Python 3.12 with [uv](https://docs.astral.sh/uv/)
- A CUDA GPU for training (an 80 GB A100 for the full screen; CPU is fine for the test suite and smoke runs)
- ~100 GB free disk for the processed data layer

## Quickstart (from a clone)

The leakage-safe splits and small manifests are tracked in git, so a clone plus the environment runs the
test suite without any download:

```bash
uv pip install -r requirements.txt
./init.sh          # compiles the tree and runs the full pytest suite
```

If `torch.cuda.is_available()` is False, install a GPU torch build matched to your driver, for example:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126"
```

## Reproduce from scratch

Data and derived artifacts are gitignored; rebuild them once on a fresh host. `data/splits/` is tracked
and frozen, so it arrives with the clone. Do not regenerate it.

```bash
# 1. Environment (+ AWS CLI for the public download)
uv pip install -r requirements.txt && uv tool install awscli

# 2. Processed data layer, ~100 GiB, from the public S3 bucket (no credentials; the ~1.6 TB raw
#    single-cell files are NOT needed). Full staged script: docs/reproduction.md > "Download data".
aws s3 sync s3://genome-scale-tcell-perturb-seq/marson2025_data/suppl_tables/ \
  data/raw/suppl_tables/ --no-sign-request

# 3. Build the derived marts + typed PPI graph (also downloads the 5 PPI databases; run ONCE)
PYTHONPATH=src python -m tcell_pipeline.run_module0

# 4. Precompute target embeddings (ESM-2 650M + PINNACLE CD4 context; resumable)
PYTHONPATH=src python -m tcell_pipeline.embeddings_plm
PYTHONPATH=src python -m tcell_pipeline.embeddings_pinnacle

# 5. Fit the fold-local program basis (sparse PCA, K=128, ~5 min)
PYTHONPATH=src python -m tcell_pipeline.programs.run_program_basis

# 6. Verify the install
./init.sh
```

## Reproduce the paper's null

The headline is the corrected five-seed comparison at `lambda_graph=0` (live gates). It re-runs only
`condition_gated`; the other three arms are unaffected, so their frozen lanes stay valid comparators, and
nothing frozen is overwritten (fresh root plus a copied registry).

```bash
# 5 lanes across 4 A100s, ~16 to 28 h, into data/results/screening_lambda0/
setsid nohup ./run_rescreen_lambda0.sh > data/logs/rescreen.nohup.log 2>&1 &

# aggregate once every lane has landed: the 4 pre-registered contrasts, Bonferroni AND Holm
SCREENING_ROOT=data/results/screening_lambda0 \
REGISTRY_PATH=data/results/screening_lambda0/experiment_registry.yaml \
PYTHONPATH=src python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3,4
```

The report is `data/results/screening_lambda0/robustness_5seed.{json,md}`; the headline contrast is
`h1_vs_no_graph` (`condition_gated − expression_only`), read from the artifact with its corrected p. You
can confirm the confound cheaply first (a gate mean near 1e-7 means the graph is switched off):

```bash
PYTHONPATH=src python -m tcell_pipeline.probe_graph_gradients --n-max 8 --batch-size 2 --steps 1
```

The sealed challenge split stays sequestered: never run `evaluation/sealed_eval.py`. The full
per-experiment guide (architecture search, rationale audit, running seeds across several single-GPU
machines) is in `docs/reproduction.md`.

## Repository layout

| Path | What |
|---|---|
| `src/tcell_pipeline/` | the pipeline: data marts, typed graph, EG-IPG model, training, screening, evaluation |
| `data/splits/`, `data/manifests/` | tracked, frozen splits and manifests (everything else under `data/` is gitignored) |
| `paper/` | the AAAI submission (gitignored; builds with pdflatex + bibtex) |
| `docs/` | detailed notes and specs; `docs/reproduction.md` is the full setup and reproduction guide |
| `AGENTS.md`, `NEXT_ACTIONS.txt` | coding-agent instructions and the current experiment plan |

## More

- Full setup and reproduction reference: `docs/reproduction.md`
- Result write-up: `paper/main.tex`, `RESULTS_SUMMARY.md`
- Dataset: Zhu et al. 2025, genome-scale CD4+ T-cell Perturb-seq ([card](https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq))
