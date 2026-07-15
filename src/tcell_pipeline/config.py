"""Path constants, dataset dimensions, and atomic-write helpers for Module 0.

All roots are overridable via environment variables so nothing is hardcoded to an
absolute path. Defaults live under ``<repo>/data`` following AGENTS.md.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"

DATA_ROOT: Path = Path(os.environ.get("DATA_ROOT", DATA_DIR / "raw"))
INTERMEDIATE_ROOT: Path = Path(os.environ.get("INTERMEDIATE_ROOT", DATA_DIR / "intermediate"))
GRAPH_ROOT: Path = Path(os.environ.get("GRAPH_ROOT", DATA_DIR / "graphs"))
MANIFEST_ROOT: Path = Path(os.environ.get("MANIFEST_ROOT", DATA_DIR / "manifests"))
PPI_CACHE_ROOT: Path = Path(os.environ.get("PPI_CACHE_ROOT", DATA_ROOT / "ppi"))

DE_STATS_PATH: Path = DATA_ROOT / "GWCD4i.DE_stats.h5ad"
PSEUDOBULK_PATH: Path = DATA_ROOT / "GWCD4i.pseudobulk_merged.h5ad"

# --- Derived artifact locations ---
ID_MAPPING_PATH: Path = INTERMEDIATE_ROOT / "id_mapping.parquet"
AMBIGUITY_REPORT_PATH: Path = INTERMEDIATE_ROOT / "ambiguity_report.txt"
DE_OBS_PATH: Path = INTERMEDIATE_ROOT / "de_obs.parquet"
DE_VAR_PATH: Path = INTERMEDIATE_ROOT / "de_var.parquet"
DE_LAYERS_DIR: Path = INTERMEDIATE_ROOT / "de_layers"
PERTURBATION_CONDITION_PATH: Path = INTERMEDIATE_ROOT / "perturbation_condition.parquet"
CONTROL_BASELINE_PATH: Path = INTERMEDIATE_ROOT / "control_baseline_expr.parquet"
CONTROL_DONOR_PROFILES_PATH: Path = INTERMEDIATE_ROOT / "control_donor_profiles.parquet"
PROTEIN_EDGES_PATH: Path = GRAPH_ROOT / "protein_edges.parquet"
COMPLEX_MEMBERSHIP_PATH: Path = GRAPH_ROOT / "complex_membership.parquet"
FEATURE_AVAILABILITY_PATH: Path = MANIFEST_ROOT / "feature_availability.yaml"

# --- DE_stats geometry (verified in examples/inspect_de_stats.py) ---
DE_N_OBS: int = 33983
DE_N_VARS: int = 10282
DE_LAYERS: tuple[str, ...] = ("log_fc", "zscore", "p_value", "adj_p_value", "baseMean", "lfcSE")
# zscore / log_fc are clipped to this range then stored sparse; the rest stay dense.
CLIPPED_SPARSE_LAYERS: tuple[str, ...] = ("zscore", "log_fc")
# p-values are stored as -log10(p) in float32: raw float32 underflows to 0.0 below ~1e-45,
# zeroing exactly the strongest hits. baseMean/lfcSE keep their raw float32 values.
NEGLOG10_LAYERS: tuple[str, ...] = ("p_value", "adj_p_value")
RAW_DENSE_LAYERS: tuple[str, ...] = ("baseMean", "lfcSE")
CLIP_LIMIT: float = 10.0
P_VALUE_FLOOR: float = 1e-300
DE_CHUNK_ROWS: int = 1000

# Pseudobulk geometry (examples/inspect_pseudobulk.py) — CSR, the only NTC-control source.
PSEUDOBULK_N_OBS: int = 278684
PSEUDOBULK_N_VARS: int = 18129
DONOR_PCA_DIMS: int = 32
DONOR_PC_PREFIX: str = "donor_pc_"

PPI_SOURCES: tuple[str, ...] = ("bioplex", "huri", "biogrid", "string", "corum")

# q_post = response-derived; PROHIBITED as H1 prediction-time input (leakage fence).
Q_POST_COLS: list[str] = [
    "ontarget_effect_size", "ontarget_significant", "neighboring_gene_KD",
    "distal_offtarget_flag", "low_target_gex", "n_up_genes", "n_down_genes",
    "n_total_de_genes", "n_downstream", "guide_correlation_signif",
    "guide_correlation_all", "donor_correlation_all_mean", "donor_correlation_hits_mean",
]

# q_pre = known before the perturbation response is observed; eligible H1 inputs.
# donor PCA columns (donor_pc_00 ...) are also q_pre, matched by prefix at tagging time.
Q_PRE_COLS: list[str] = [
    "culture_condition", "target_contrast", "target_contrast_gene_name",
    "ensembl_id", "hgnc_symbol", "uniprot_id", "entrez_id",
    "ppi_degree_physical", "ppi_degree_functional", "ppi_degree_complex",
    "control_baseline_expr",
]

# Bookkeeping / provenance columns that are deliberately neither q_pre nor q_post.
# Anything landing in metadata that is NOT here gets a REVIEW warning (leakage-fence tripwire).
KNOWN_METADATA_COLS: list[str] = ["row_index", "mapping_status"]

# --- Module 1 (Perturbation & Context Encoder) ---
PLM_EMBED_DIM: int = 1280        # protein language model (e.g. ESM-2 650M) per-protein vector
PINNACLE_EMBED_DIM: int = 512    # PINNACLE cell-type-contextualised protein embedding
GUIDE_SEQ_EMBED_DIM: int = 64    # placeholder guide-sequence embedding (zeros until available)
H_DO_DIM: int = 256              # fused perturbation-condition embedding h_do
CONDITIONS: list[str] = ["Rest", "Stim8hr", "Stim48hr"]
PLM_EMBEDDINGS_PATH: Path = INTERMEDIATE_ROOT / "plm_embeddings.parquet"
PINNACLE_EMBEDDINGS_PATH: Path = INTERMEDIATE_ROOT / "pinnacle_embeddings.parquet"


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _tmp_path(final: Path, tmp_suffix: str = ".tmp") -> Path:
    return final.with_name(final.name + tmp_suffix)


def write_parquet_atomic(df: pd.DataFrame, final: Path) -> None:
    ensure_dir(final.parent)
    tmp = _tmp_path(final)
    df.to_parquet(tmp, index=False)
    tmp.replace(final)


def write_text_atomic(text: str, final: Path) -> None:
    ensure_dir(final.parent)
    tmp = _tmp_path(final)
    tmp.write_text(text)
    tmp.replace(final)


def save_npy_atomic(final: Path, writer: Callable[[Path], None]) -> None:
    """Fill a dense .npy via ``writer(tmp)`` (e.g. np.save or a memmap fill), then rename."""
    ensure_dir(final.parent)
    tmp = _tmp_path(final)
    writer(tmp)
    tmp.replace(final)


def save_npz_atomic(final: Path, csr) -> None:
    import scipy.sparse as sp

    ensure_dir(final.parent)
    tmp = final.with_suffix(".tmp.npz")  # keep .npz so save_npz does not append another
    sp.save_npz(tmp, csr)
    tmp.replace(final)


def open_dense_memmap(tmp: Path, shape: tuple[int, int]) -> np.memmap:
    """Create a writable float32 .npy on disk without holding the full array in RAM."""
    return np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=shape)
