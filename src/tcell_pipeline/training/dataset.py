"""PerturbationDataset: split-aware (target, condition) -> supervised training example (§8, §9).

Emits only q_pre features (the leakage fence is enforced downstream by PerturbationEncoder, which
raises on any q_post column; ``build_encoder_batch`` never assembles one). Supervision:
  - ``delta_z_true``: the z-score projected onto the frozen loadings, ``z @ B``, for EVERY row. Using one
    consistent definition across splits matters: the earlier design used the sparse-PCA score A from
    program_response for train rows but ``z @ B`` for out-of-fold rows (val/cal/challenge, which have no
    A), so the model was trained against A yet its validation loss measured a different quantity — the
    metric driving early-stopping/best-checkpoint wasn't the trained objective. ``z @ B`` is fold-local
    (B is fit on train rows only) and reproducible from B alone.
  - ``delta_x_true``: the per-gene z-score row from the sparse DE layer.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from tcell_pipeline import config
from tcell_pipeline.encoders.batch import DONOR_COLS, build_encoder_batch
from tcell_pipeline.programs.program_basis import load_program_basis, zscore_path

_OBS_COLS = ["n_guides", "single_guide_estimate"]


def load_donor_pool(path: Path = config.CONTROL_DONOR_PROFILES_PATH) -> tuple[dict, "torch.Tensor"]:
    """Real per-donor control-profile PC vectors, grouped by culture condition (§ donor invariance).

    The training mart's ``donor_pc`` is the per-condition MEAN of these; the pool exposes the individual
    donors so the invariance term can re-run the encoder under each real donor. Returns
    ``(pool: {condition -> (n_donors, 32) float32}, global_mean (32,))``; empty pool if the profile
    parquet is absent (invariance then degrades to a no-op)."""
    if not Path(path).exists():
        return {}, torch.zeros(config.DONOR_PCA_DIMS)
    prof = pd.read_parquet(path)
    pool = {str(cond): torch.tensor(g[DONOR_COLS].to_numpy("float32"))
            for cond, g in prof.groupby("culture_condition")}
    return pool, torch.tensor(prof[DONOR_COLS].to_numpy("float32")).mean(0)


def sample_donor_variants(donor_pool: dict, fallback: "torch.Tensor", conditions: list[str],
                          samples: int, generator) -> list:
    """``samples`` donor-swapped ``donor_pc`` tensors (each (B, 32)), one distinct real donor per row for
    its condition (drawn without replacement while the pool lasts). A condition absent from the pool
    reuses ``fallback`` for every draw → that row contributes zero invariance variance."""
    variants: list[list] = [[] for _ in range(samples)]
    for cond in conditions:
        pool = donor_pool.get(str(cond))
        if pool is None or pool.shape[0] == 0:
            for k in range(samples):
                variants[k].append(fallback)
            continue
        n = pool.shape[0]
        perm = torch.randperm(n, generator=generator)
        for k in range(samples):
            variants[k].append(pool[int(perm[k % n])])
    return [torch.stack(v) for v in variants]


class PerturbationDataset(Dataset):
    def __init__(
        self,
        role: str = "train",
        n_max: int | None = None,
        split_path: Path = config.BLOCKED_SPLIT_PATH,
        pc_path: Path = config.PERTURBATION_CONDITION_PATH,
        obs_path: Path = config.DE_OBS_PATH,
        var_path: Path = config.DE_VAR_PATH,
        basis_path: Path = config.PROGRAM_LOADINGS_PATH,
        zscore_npz: Path | None = None,
        donor_profiles_path: Path = config.CONTROL_DONOR_PROFILES_PATH,
    ) -> None:
        split = pd.read_csv(split_path)
        genes = set(split.loc[split["role"] == role, "hgnc_symbol"])
        pc = pd.read_parquet(pc_path)
        obs = pd.read_parquet(obs_path, columns=_OBS_COLS)
        if len(obs) != len(pc):  # the positional mask below relies on de_obs being row-aligned to pc
            raise ValueError(f"de_obs ({len(obs)}) and perturbation_condition ({len(pc)}) row counts differ")
        keep = pc["hgnc_symbol"].isin(genes).to_numpy()
        self.pc = pc.loc[keep].reset_index(drop=True)
        self.obs = obs.loc[keep].reset_index(drop=True)  # de_obs is row-aligned to perturbation_condition
        if n_max is not None:
            self.pc, self.obs = self.pc.iloc[:n_max], self.obs.iloc[:n_max]
        self.row_index = self.pc["row_index"].to_numpy()

        gene_names = pd.read_parquet(var_path, columns=["gene_name"])["gene_name"].tolist()
        B, _ = load_program_basis(basis_path, gene_order=gene_names)
        self.B = torch.from_numpy(B)  # (G, K), frozen fold-local loadings; delta_z_true = z @ B for all rows
        self._zscore = sp.load_npz(zscore_npz or zscore_path()).tocsr()
        self.donor_pool, self.donor_mean = load_donor_pool(donor_profiles_path)

    def __len__(self) -> int:
        return len(self.pc)

    def __getitem__(self, i: int):
        ri = int(self.row_index[i])
        batch = build_encoder_batch(self.pc.iloc[[i]], self.obs.iloc[[i]])
        dx = torch.from_numpy(self._zscore[ri].toarray().reshape(-1).astype("float32"))  # (G,)
        dz = dx @ self.B                                                                  # (K,) z@B, consistent across splits
        target = str(self.pc["hgnc_symbol"].iloc[i])
        condition = str(self.pc["culture_condition"].iloc[i])
        return batch, target, condition, dz, dx, ri

    @staticmethod
    def collate(items):
        """Merge per-sample encoder batches into one batched dict + parallel lists/tensors."""
        batches, targets, conditions, dzs, dxs, ris = zip(*items)
        batch: dict = {}
        for k, v0 in batches[0].items():
            vals = [b[k] for b in batches]
            batch[k] = torch.cat(vals) if torch.is_tensor(v0) else [x for v in vals for x in v]
        return batch, list(targets), list(conditions), torch.stack(dzs), torch.stack(dxs), list(ris)
