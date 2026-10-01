# tcell-causal-ppi

Code to reproduce the results of

> Vijayavallabh Jayamanikandan. **The Regularizer That Switched Off the Experiment: Why Our
> Protein-Interaction Null Was About the Encoder, Not the Prior.** I Can't Believe It's Not Better
> (ICBINB): Failure Modes of AI in Biology, NeurIPS 2026 workshop.
> [OpenReview](https://openreview.net/forum?id=6tm8b2KhUs)

The paper asks whether a typed protein-interaction prior improves prediction of how primary human
CD4+ T cells respond to CRISPRi perturbation, using EG-IPG (Evidence-Gated Intervention-Informed
Protein-Program Graph). A textbook edge-sparsity penalty had switched the graph off within the first
epoch while training still reported a plausible score. With the gates live, the evidence-gated graph is
indistinguishable from an expression-only baseline, and the typed null holds across seven further
Perturb-seq screens; yet a plain untyped graph arm beats the baseline on the same screen. The prior is
not what fails but the encoder built to exploit it.

## Contents

The repository holds the code that regenerates the paper's results and nothing else.

| Path | What |
|---|---|
| `src/tcell_pipeline/` | data marts, typed PPI graph, embeddings, program basis, the model and its training, screening and statistics, replication datasets |
| `run_*.sh`, `merge_registry_n7.py`, `analyze_feature_ablation.py` | the launchers that produced each result, as they were run |
| `data/splits/` | the frozen blocked target-out-of-distribution split every result is scored on |
| `data/manifests/` | the leakage fence's feature-availability manifest and the scPerturb checksums |
| `docs/reproduction.md` | setup, data, and the command behind every result file the paper reads |
| `docs/replication-prereg.md` | the pre-registration, amended only by dated appends made before the run each governs |
| `docs/replication-dataset-survey.md` | the replication-candidate measurements the pre-registration cites |
| `paper/icbinb/` | the paper source, `verify_numbers.py` (checks the paper against your result files) and `make_lambda_sweep.py` (Figure 1) |

Data, checkpoints and result files are not included; the commands in `docs/reproduction.md` regenerate
them.

## Quick start

```bash
git clone https://github.com/Vijayavallabh/tcell-causal-ppi && cd tcell-causal-ppi
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126"   # the build for your driver
uv pip install -r requirements.txt
```

Then follow `docs/reproduction.md`: the data (section 2), the commands (section 4), and the check of
the paper against your files (section 5):

```bash
cd paper/icbinb && python verify_numbers.py --check
```

A typed-graph training lane needs an 80 GB GPU; the guide lists the cost of each experiment.

## License

The code is MIT-licensed (`LICENSE`). No data is redistributed here: the reference screen (Zhu et al.
2025), the scPerturb files (CC-BY-4.0) and the protein-interaction databases each carry their own
terms.

## Citation

```bibtex
@inproceedings{jayamanikandan2026regularizer,
  title     = {The Regularizer That Switched Off the Experiment: Why Our Protein-Interaction Null
               Was About the Encoder, Not the Prior},
  author    = {Jayamanikandan, Vijayavallabh},
  booktitle = {I Can't Believe It's Not Better (ICBINB): Failure Modes of AI in Biology,
               NeurIPS 2026 Workshop},
  year      = {2026}
}
```
