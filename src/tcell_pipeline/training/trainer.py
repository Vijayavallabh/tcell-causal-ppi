"""Trainer: Stage A optimisation of the EG-IPG H1 predictor (Module 1 + 2 + 3) (§8.1-8.2).

AdamW over the model AND the loss's own parameters (the DE head + shared-component extractor live on
``StageALoss``); the frozen basis B is a decoder buffer, so it never enters the optimiser. Early
stopping on validation total, gradient clipping, atomic best/last checkpoints, and per-epoch loss
components written to the logs dir.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tcell_pipeline import config
from tcell_pipeline.training.dataset import PerturbationDataset
from tcell_pipeline.training.losses import StageALoss


class Trainer:
    def __init__(
        self,
        model,
        train_ds,
        val_ds=None,
        loss: StageALoss | None = None,
        lr: float = config.LR,
        weight_decay: float = config.WEIGHT_DECAY,
        max_epochs: int = config.MAX_EPOCHS,
        patience: int = config.EARLY_STOP_PATIENCE,
        batch_size: int = config.BATCH_SIZE,
        grad_clip: float = config.GRAD_CLIP,
        device: str = "cpu",
        ckpt_dir: Path = config.CHECKPOINTS_ROOT,
        log_dir: Path = config.LOGS_ROOT,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        self.device = device
        self.model = model.to(device)
        self.loss = (loss or StageALoss(model.decoder.gene_dim, model.decoder.program_dim)).to(device)
        self.params = list(self.model.parameters()) + list(self.loss.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=weight_decay)
        self.max_epochs, self.patience, self.grad_clip = max_epochs, patience, grad_clip
        self.ckpt_dir, self.log_dir = Path(ckpt_dir), Path(log_dir)
        collate = PerturbationDataset.collate
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
        self.val_loader = (
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate) if val_ds else None
        )

    def _epoch(self, loader, train: bool) -> dict:
        self.model.train(train)
        self.loss.train(train)
        agg: dict = {}
        n = 0
        with torch.set_grad_enabled(train):
            for batch, targets, conditions, dz_true, dx_true, _ in loader:
                dz_true, dx_true = dz_true.to(self.device), dx_true.to(self.device)
                out = self.model(batch, targets, conditions)
                comps = self.loss(out, dz_true, dx_true, targets, conditions)
                if train:
                    self.opt.zero_grad()
                    comps["total"].backward()
                    torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
                    self.opt.step()
                for k, v in comps.items():
                    agg[k] = agg.get(k, 0.0) + float(v.detach())
                n += 1
        return {k: v / max(n, 1) for k, v in agg.items()}

    def _save_ckpt(self, tag: str, epoch: int, metrics: dict) -> Path:
        config.ensure_dir(self.ckpt_dir)
        path = self.ckpt_dir / f"stage_a_{tag}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(
            {"model": self.model.state_dict(), "loss": self.loss.state_dict(),
             "optimizer": self.opt.state_dict(), "epoch": epoch, "metrics": metrics},
            tmp,
        )
        tmp.replace(path)
        return path

    def _log(self, history: list) -> None:
        config.write_text_atomic(json.dumps(history, indent=2), self.log_dir / "stage_a_history.json")

    def run(self) -> dict:
        best_val, best_path, wait, history = float("inf"), None, 0, []
        for epoch in range(self.max_epochs):
            train_m = self._epoch(self.train_loader, train=True)
            val_m = self._epoch(self.val_loader, train=False) if self.val_loader else train_m
            history.append({"epoch": epoch, "train": train_m, "val": val_m})
            self._log(history)
            self._save_ckpt("last", epoch, val_m)
            if val_m["total"] < best_val - 1e-6:
                best_val, wait = val_m["total"], 0
                best_path = self._save_ckpt("best", epoch, val_m)
            else:
                wait += 1
                if wait >= self.patience:
                    break
        return {"best_ckpt": str(best_path) if best_path else None,
                "best_val": best_val, "epochs_run": len(history), "history": history}
