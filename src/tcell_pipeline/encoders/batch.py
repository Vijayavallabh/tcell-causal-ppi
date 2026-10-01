"""Shared mart -> PerturbationEncoder batch builder for the Module 1/2/3 real-data smokes.

Single source of truth for the encoder's loader contract: change the batch schema here, not in three
drifting smoke copies. Reads a slice of the real marts (perturbation_condition + de_obs) into the
dict PerturbationEncoder.forward expects.
"""
from __future__ import annotations

import pandas as pd
import torch

from tcell_pipeline import config

DONOR_COLS = [f"{config.DONOR_PC_PREFIX}{i:02d}" for i in range(config.DONOR_PCA_DIMS)]


def build_encoder_batch(pc: pd.DataFrame, obs: pd.DataFrame) -> dict:
    """Assemble the PerturbationEncoder loader-contract batch dict from aligned real-mart slices."""
    return {
        "uniprot_id": [None if pd.isna(x) else str(x) for x in pc["uniprot_id"]],
        "ppi_degree_physical": torch.tensor(pc["ppi_degree_physical"].to_numpy()),
        "ppi_degree_functional": torch.tensor(pc["ppi_degree_functional"].to_numpy()),
        "ppi_degree_complex": torch.tensor(pc["ppi_degree_complex"].to_numpy()),
        "control_baseline_expr": torch.tensor(pc["control_baseline_expr"].to_numpy()),
        "culture_condition": pc["culture_condition"].tolist(),
        "donor_pc": torch.tensor(pc[DONOR_COLS].to_numpy(dtype="float32")),
        "n_guides": torch.tensor(obs["n_guides"].to_numpy()),
        "single_guide_estimate": torch.tensor(obs["single_guide_estimate"].to_numpy(dtype=bool)),
    }
