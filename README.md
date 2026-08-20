# tcell-causal-ppi

Do typed protein-interaction graph priors help predict how primary human T cells respond to genetic
perturbations, over an expression-only baseline? **EG-IPG** (Evidence-Gated Intervention-Informed
Protein-Program Graph) is built to give the graph its best chance, then tested under a bias-aware metric
on a leakage-safe target-out-of-distribution split.

**Result: a well-controlled null, on three folds.** On the genome-scale CD4+ T-cell Perturb-seq screen
of Zhu et al. (2025), with live edge gates and family-wise error control (Bonferroni and Holm), the
evidence-gated graph is at **parity** with no-graph. Across three blocked target-OOD folds at n = 5
paired seeds each, **no contrast survives correction on any fold**:

| Fold (seq-cosine / cap) | h1 (gated − no-graph) | h2a (static − no-graph) |
|---|---|---|
| 0.85 / 0.05 (frozen) | −0.0009 (p = 0.71) | −0.0131 (survives both) |
| 0.80 / 0.10 | +0.0082 (Bonf. 0.101) | −0.0122 (Bonf. 0.920) |
| 0.75 / 0.15 | +0.0005 (p = 0.904) | −0.0012 (p = 0.614) |

**Our own strongest claim did not replicate.** "A typed static graph is reliably worse" was the one
contrast that ever cleared Bonferroni and Holm, and it holds only on the fold it was found on. We report
that rather than the single-fold version. Full account: `RESULTS_SUMMARY.md`.

**And "reliably worse" turned out to be mostly an implementation detail.** Three diagnostic arms took
the typed encoder apart. Tying the message weights across relations moves the primary endpoint by
+0.0004; randomising every relation label at preserved edge counts moves it by +0.0065; neither clears
correction. What does move it is the *aggregation*: weighting each relation's messages by
`1/sqrt(d_i d_j)` instead of summing them recovers **+0.0139 (5/5 seeds, 79% of the gap** to a plain
untyped GCN), because an unnormalised sum lets `functional_assoc` — 86% of all edges, median score
0.228 — dominate every node update by sheer count. The deficit was barely about the typing.

**It buys nothing.** The repaired encoder beats the no-graph baseline by **+0.0008 (p = 0.71)** and
cannot be told apart from plain topology. Fixing the aggregation removes the damage and reveals nothing
underneath, so the null above stands — with one fewer artifact in front of it. That contrast is a family
of one, where Bonferroni and Holm are the identity; the robust parts are the 5/5 sign agreement and the
size of the share, not the p-value.

**Where a PPI prior helps, and where it hurts.** On disjoint intervals of the observed response, the
untyped graph's harm is confined to each perturbation's **twenty most-moved genes** (−0.0424, 0/5 seeds
positive), while every interval from rank 251 to 10,282 is positive and clears both corrections. At
decile resolution the harm disappears entirely (+0.0064, clearing nothing) — a coarser cut of the same
predictions would say the prior never hurts.

> **Two failure modes worth knowing about.** (1) A conventional edge-sparsity regulariser, unnormalised
> over edges, is ~103× the response term at its default weight and drives the edge gates to ~1e-7 in
> epoch 0, silently switching the graph off while training still reports plausible numbers. Setting
> `lambda_graph=0` keeps the gates live (0.57 to 0.77); repairing it left the null unchanged.
> (2) **Re-drawing one split changes the answer.** Running the whole comparison three times on the
> *same* split specification (identical threshold and cap, only the partition seed varied), five
> paired seeds each, gives h1 = +0.0082 (p = 0.025), +0.0026 (p = 0.027) and +0.0021 (p = 0.27).
> The estimate spans fourfold and the qualitative verdict flips: two re-draws give an uncorrected
> interval excluding zero, the third does not. None survives correction. For scale, that re-draw
> noise is the same size as the differences *between* the three folds above — so a single-realization
> split sweep can be measuring its seed more than its knob.

**Papers** (both gitignored, build with pdflatex + bibtex): `paper/icbinb/main.tex` for the ICBINB-BIO
NeurIPS 2026 workshop (the failure-analysis framing; main text 8pp), and `paper/main.tex` for AAAI.

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

> **On a box whose NVML is broken** (a mismatched `libnvidia-ml.so` on the library path), every
> `--device cuda` lane dies within a minute on a PyTorch assert naming `nvmlInit_v2_`. The fix is to
> preload the library matching your kernel module, e.g.
> `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.<your-driver-version>`.
> Check with `ldd $(which nvidia-smi) | grep nvidia-ml`. See `docs/agent-lessons.md`.

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

### Reproduce the other folds, and the split-realization check

Split difficulty is set by two env vars, and the partition seed is a third. Every run needs a **fresh**
`SPLITS_ROOT` and `SCREENING_ROOT`; the frozen roots are read-only inputs.

```bash
# a harder fold: block more paralogs (lower threshold, larger family cap)
SEQ_SIM_COSINE_THRESHOLD=0.75 GROUP_SIZE_CAP=0.15 \
SPLITS_ROOT=data/results/splits_c075c15 \
PYTHONPATH=src python -m tcell_pipeline.splits

# the same specification, re-drawn: identical threshold and cap, different partition
SPLIT_SEED=1 SEQ_SIM_COSINE_THRESHOLD=0.80 GROUP_SIZE_CAP=0.10 \
SPLITS_ROOT=data/results/splits_c080c10_r2 \
PYTHONPATH=src python -m tcell_pipeline.splits
```

Compare `sequence_train_to_challenge_cosine.median` in each `leakage_report.json`. Across three
re-draws of one specification we measured 0.759 / 0.793 / 0.862, so **check the re-draw spread before
interpreting a threshold sweep** — ours was larger than the sweep it was meant to calibrate. The
re-draws also silently changed the validation set from 3,632 to 7,216 targets, and the no-graph
baseline tracked that (0.099 / 0.094 / 0.085) rather than the cosine statistic.

**On a shared box, match the arm to the free memory.** Measured per-lane peaks: `expression_only`
~2.5 GB, `untyped_gnn` ~2–4 GB, `condition_gated` and `typed_static` **47–51 GB**. A card with a
co-tenant can still run the cheap arms productively; launching a graph arm into <47 GB free will fail,
possibly hours in rather than at allocation.

The sealed challenge split stays sequestered: never run `evaluation/sealed_eval.py`. The full
per-experiment guide (architecture search, rationale audit, running seeds across several single-GPU
machines) is in `docs/reproduction.md`.

### Reproduce the multi-dataset replication (eight datasets)

The replication trains the same nested family on public Perturb-seq screens from scPerturb. Every
dataset is isolated by `INTERMEDIATE_ROOT`, which is why `run_replication_stage.sh` must be **sourced**
before any manual stage: it pins the ESM-2 and PINNACLE stores back to the extended copies, redirects
`SPLITS_ROOT` away from the frozen `data/splits/`, and reads the condition vocabulary from the DE
provenance. Without it every target silently gets a zero feature vector and the graph arm trains on
nothing.

```bash
# stages 2-6 for one dataset (CPU, idempotent). K per prereg Amendment 3.1: 128 where the train fold
# has >= 256 rows, else the largest power of two <= train/2 -- a labelled deviation below that.
./run_replication_prep.sh ReplogleWeissman2022_K562_gwps none 128

# all lanes for all datasets across 4 GPUs; skips anything already landed, so it is safe to re-run
setsid nohup ./run_replication_campaign.sh > data/logs/repl/campaign.log 2>&1 &

# per dataset: the pre-registered contrasts under BOTH corrections
SPLITS_ROOT=... SCREENING_ROOT=data/results/replication/<dataset> ... \
PYTHONPATH=src python -m tcell_pipeline.screening.multiseed --seeds 0,1,2,3

# pool across datasets: fixed-effect, random-effects, tau^2, I^2, Cochran's Q
PYTHONPATH=src python -m tcell_pipeline.replication.pool --with-reference
```

Read the random-effects interval, not the fixed-effect one, whenever `I^2` is high — the two disagree
for the untyped-graph contrast (FE excludes zero, RE does not, at `I^2` = 88%) and only one of them is
honest about it.

### Reproduce the difficulty-vs-noise decomposition (L4)

Splits vary along two axes that are easy to confuse: the difficulty setting (threshold/cap) and the
partition seed at a fixed setting. The decomposition separates them, plus training-seed noise.

```bash
# a fourth difficulty level, and re-draws at an existing one
setsid nohup ./run_l4_finish.sh > data/logs/l4_finish.nohup.log 2>&1 &

# waits for the workers, aggregates every root over exactly the seeds that landed, decomposes
setsid nohup ./run_l4_finalise.sh > data/logs/l4_finalise.log 2>&1 &

PYTHONPATH=src python -m tcell_pipeline.screening.variance_decomposition --contrast h2a
```

It returns `None`, never `0.0`, for a component no cell can identify, and prints the degrees of freedom
behind each — with three or four levels these are 2-3 df, so the ratios are indicative and the tool
says so. Coverage is not symmetric across contrasts: the 0.80/0.10 re-draws were run with
`condition_gated`, so they identify h1's within-level variance, and the 0.75/0.15 re-draws identify
h2a's.

## Repository layout

| Path | What |
|---|---|
| `src/tcell_pipeline/` | the pipeline: data marts, typed graph, EG-IPG model, training, screening, evaluation |
| `src/tcell_pipeline/replication/` | scPerturb h5ad → DE-stats adapter, PINNACLE context extractor (for second-dataset work) |
| `data/splits/`, `data/manifests/` | tracked, frozen splits and manifests (everything else under `data/` is gitignored) |
| `paper/`, `paper/icbinb/` | the AAAI and ICBINB-BIO submissions (gitignored; pdflatex + bibtex) |
| `docs/` | notes and specs; `docs/reproduction.md` is the full guide, `docs/agent-lessons.md` the hard-won rules |
| `AGENTS.md`, `NEXT_ACTIONS.txt` | coding-agent instructions and the current experiment plan |
| `session-handoff.md`, `RESULTS_SUMMARY.md` | current state for the next session, and the full result record |

## More

- Full setup and reproduction reference: `docs/reproduction.md`
- Result write-up: `paper/main.tex`, `RESULTS_SUMMARY.md`
- Dataset: Zhu et al. 2025, genome-scale CD4+ T-cell Perturb-seq ([card](https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq))
