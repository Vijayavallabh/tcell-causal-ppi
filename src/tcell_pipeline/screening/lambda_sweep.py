"""Figure 1's sweep: the condition-gated arm's edge gates against the edge-sparsity weight lambda_graph.

RECONSTRUCTED 2026-10-01. The script that produced data/results/q4_lambda_sweep_22ep.json (2026-07-29)
lived in a session scratchpad and was lost; its logs (data/logs/campaign/q4b_card*.log) record the
setting: one seed, the first 512 training and first 128 validation rows of the frozen blocked-target-OOD
fold, 22 epochs with no early stopping, nine lambda values. This re-implements that loop on the committed
trainer, the same way pilot_lambda_graph.py drives Trainer._epoch, and writes the same JSON schema.
The batch size was not logged. Batch 4 reproduces the artifact's lambda=0.01 epoch-0 gates exactly
(train 0.250814, val 0.021111, as do the code at the sweep's own commit and today's); batch 8 gives
0.408494 / 0.173691. Gated runs are not bit-deterministic on GPU, so later epochs agree to noise
(the 12- and 22-epoch originals already differ in the fourth digit at epoch 1).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<free A100> OMP_NUM_THREADS=4 PYTHONPATH=src \\
        python -m tcell_pipeline.screening.lambda_sweep --device cuda --out data/results/q4_lambda_sweep_22ep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from tcell_pipeline import config
from tcell_pipeline.graph import build_hetero_graph
from tcell_pipeline.screening.screening import CONDITION_GATED, nested_family_factories
from tcell_pipeline.training.dataset import PerturbationDataset
from tcell_pipeline.training.losses import StageALoss
from tcell_pipeline.training.trainer import Trainer, seeded_init

LAMBDAS = (0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def sweep_one(lam: float, train_ds, val_ds, factory, *, epochs: int, batch_size: int, seed: int,
              device: str, scratch: Path) -> dict:
    with seeded_init(seed):
        model = factory()
    dec = model.decoder
    loss = StageALoss(dec.gene_dim, dec.program_dim, h_do_dim=dec.h_do_dim, lambda_graph=float(lam))
    trainer = Trainer(model, train_ds, val_ds, loss=loss, batch_size=batch_size, seed=seed, device=device,
                      ckpt_dir=scratch, log_dir=scratch)
    traj = []
    for epoch in range(epochs):          # every epoch, no early stopping: the figure is a trajectory
        tr = trainer._epoch(trainer.train_loader, train=True)
        va = trainer._epoch(trainer.val_loader, train=False)
        traj.append({"epoch": epoch, "train_gate": tr["gate_mean"], "val_gate": va["gate_mean"],
                     "train_frac_dead": tr["gate_frac_dead"]})
        print(f"[sweep] lambda={lam:g} epoch {epoch}: train_gate={tr['gate_mean']:.6g} "
              f"val_gate={va['gate_mean']:.6g} dead={tr['gate_frac_dead']:.4f}", flush=True)
    last = traj[-1]
    return {"lambda_graph": float(lam), "trajectory": traj, "gate_mean_final_train": last["train_gate"],
            "gate_mean_final_val": last["val_gate"], "gate_frac_dead_final": last["train_frac_dead"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lambdas", default=",".join(str(x) for x in LAMBDAS))
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)   # see the docstring: 4, not the campaign's 8
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, scratch = Path(a.out), Path(a.out).with_suffix(".scratch")
    if out.exists():
        raise SystemExit(f"{out} exists; pass a fresh --out rather than overwrite a result")
    torch.set_num_threads(1)  # many-core box: tiny per-subgraph GNN ops thrash the pool (as run_screening)
    if a.device.startswith("cuda"):
        print(f"[sweep] device={a.device} name={torch.cuda.get_device_name(a.device)}", flush=True)
    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()
    graph, gene_to_idx = build_hetero_graph()
    factory = nested_family_factories(gene_names, graph, gene_to_idx)[CONDITION_GATED]
    train_ds = PerturbationDataset("train", n_max=a.n_train)
    val_ds = PerturbationDataset("val", n_max=a.n_val)
    print(f"[sweep] train={len(train_ds)} val={len(val_ds)} epochs={a.epochs} seed={a.seed}", flush=True)
    runs = []
    for lam in (float(x) for x in a.lambdas.split(",")):
        runs.append(sweep_one(lam, train_ds, val_ds, factory, epochs=a.epochs, batch_size=a.batch_size,
                              seed=a.seed, device=a.device, scratch=scratch))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"runs": runs}, indent=2))   # checkpoint after each lambda
    print(f"[sweep] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
