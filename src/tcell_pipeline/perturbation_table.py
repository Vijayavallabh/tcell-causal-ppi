"""Build perturbation_condition.parquet: one row per (target gene, culture condition).

Grain matches DE_stats obs order, so ``row_index`` (0..N-1) indexes the extracted DE
layers directly. q_post columns are carried through but tagged prohibited downstream;
unmapped targets are kept (left join), never dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tcell_pipeline import config

JOIN_KEYS = ["target_contrast_gene_name", "culture_condition", "target_contrast"]
PPI_DEGREE_COLS = ["ppi_degree_physical", "ppi_degree_functional", "ppi_degree_complex"]


def build_perturbation_table(de_obs: pd.DataFrame, id_mapping: pd.DataFrame) -> pd.DataFrame:
    present_qpost = [c for c in config.Q_POST_COLS if c in de_obs.columns]
    cols = [c for c in JOIN_KEYS if c in de_obs.columns] + present_qpost
    table = de_obs[cols].copy().reset_index(drop=True)
    table.insert(0, "row_index", np.arange(len(table), dtype=np.int64))

    table["ensembl_id"] = table["target_contrast"].astype(str)
    id_cols = id_mapping[["ensembl_id", "hgnc_symbol", "uniprot_id", "entrez_id", "mapping_status"]]
    table = table.merge(id_cols, on="ensembl_id", how="left")  # keep unmapped rows

    for c in PPI_DEGREE_COLS:
        table[c] = 0
    table["control_baseline_expr"] = np.nan
    return table


def run() -> pd.DataFrame:
    print("[perturbation_table] loading de_obs + id_mapping ...")
    de_obs = pd.read_parquet(config.DE_OBS_PATH)
    id_mapping = pd.read_parquet(config.ID_MAPPING_PATH)
    table = build_perturbation_table(de_obs, id_mapping)
    config.write_parquet_atomic(table, config.PERTURBATION_CONDITION_PATH)
    n_unmapped = int(table["uniprot_id"].isna().sum())
    print(f"[perturbation_table] wrote {len(table)} rows ({n_unmapped} without UniProt) "
          f"-> {config.PERTURBATION_CONDITION_PATH}")
    return table


if __name__ == "__main__":
    run()
