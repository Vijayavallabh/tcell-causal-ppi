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
PLM_EMBED_DIM: int = 1280        # ESM-2 650M (t33) per-protein vector, mean-pooled over residues
PINNACLE_EMBED_DIM: int = 128    # PINNACLE cell-type-contextualised protein embedding (real dim)
GUIDE_SEQ_EMBED_DIM: int = 64    # placeholder guide-sequence embedding (zeros until available)
H_DO_DIM: int = 256              # fused perturbation-condition embedding h_do
CONDITIONS: list[str] = ["Rest", "Stim8hr", "Stim48hr"]
PLM_EMBEDDINGS_PATH: Path = INTERMEDIATE_ROOT / "plm_embeddings.parquet"
PINNACLE_EMBEDDINGS_PATH: Path = INTERMEDIATE_ROOT / "pinnacle_embeddings.parquet"
# PINNACLE (Li et al. 2024) contextual protein embeddings — Figshare article 22708126.
# The screen is CD4+ T cells, so we take the CD4 helper T-cell context.
PINNACLE_RAW_DIR: Path = DATA_ROOT / "pinnacle" / "pinnacle_embeds"
PINNACLE_FIGSHARE_URL: str = "https://ndownloader.figshare.com/files/48005749"
PINNACLE_CONTEXT: str = "cd4-positive helper t cell"

# --- Module 2 (Typed Graph Encoder) ---
GRAPH_HOPS: int = 2               # neighbourhood radius sampled around each perturbation target
NEIGHBORHOOD_CAP: int = 512      # max protein nodes per sampled subgraph
GRAPH_HIDDEN_DIM: int = 256      # unified node hidden dim in message passing (== H_DO_DIM)
GRAPH_LAYERS: int = 3            # relational message-passing layers
GRAPH_N_HEADS: int = 4           # cross-attention heads in the graph readout
EDGE_DROPOUT: float = 0.1        # DropEdge probability during training
EDGE_FEATURE_DIM: int = 8        # source_onehot(5)+score(1)+is_direct_binary(1)+n_supporting(1)
COMPLEX_EMBED_DIM: int = 256     # learned protein-complex node embedding dim
CONDITION_EMBED_DIM: int = 64    # culture-condition embedding feeding the edge gate (Module 1 pattern)
# protein node feature vector = frozen PLM + PINNACLE + 3 PPI degrees + control baseline expr
PROTEIN_FEATURE_DIM: int = PLM_EMBED_DIM + PINNACLE_EMBED_DIM + 4


# --- feat-003 (Leakage-Safe Train/Val/Test Splits) ---
SPLITS_ROOT: Path = Path(os.environ.get("SPLITS_ROOT", DATA_DIR / "splits"))
SPLIT_ROLES: tuple[str, ...] = ("train", "val", "calibration", "challenge")  # challenge == sequestered test
SPLIT_FRACTIONS: dict[str, float] = {"train": 0.60, "val": 0.15, "calibration": 0.10, "challenge": 0.15}
SPLIT_SEED: int = 0
# Centered ESM-2 cosine, representative (non-chaining) clustering: measured to give a 3.1% largest
# family on the real marts. A tuning knob the leakage report calibrates (see docs/specs feat-003).
SEQ_SIM_COSINE_THRESHOLD: float = 0.85
GROUP_SIZE_CAP: float = 0.05          # max family-group size as a fraction of target genes
BLOCKED_SPLIT_PATH: Path = SPLITS_ROOT / "blocked_target_ood.csv"
RANDOM_SPLIT_PATH: Path = SPLITS_ROOT / "random.csv"
SPLIT_MANIFEST_PATH: Path = SPLITS_ROOT / "manifest.json"
SPLIT_LEAKAGE_REPORT_PATH: Path = SPLITS_ROOT / "leakage_report.json"


# --- Module 3 (Program Decoder) ---
# The decoder's gene axis is derived from the loaded basis B (B.shape[0]), NOT a config constant, so
# it always matches the fold-local loadings rather than a value that could silently drift from them.
PROGRAM_DIM: int = 128            # K latent programs (compared at 64/128/256/512 in §6.5)
PROGRAM_METHOD: str = "sparse_pca"  # fold-local basis: sparse_pca | nmf | fastica | svd (§6.1)
PROGRAM_LOADINGS_PATH: Path = INTERMEDIATE_ROOT / "gene_program_loadings.parquet"  # B: gene_name + program_k
PROGRAM_RESPONSE_PATH: Path = INTERMEDIATE_ROOT / "program_response.parquet"       # A: row_index + program_k
PROGRAM_COL_PREFIX: str = "program_"


# --- Module 4 (Sparse Predictive-Rationale Head; Stage B, fitted AFTER the H1 predictor freeze) ---
# This head produces a PREDICTIVE rationale (which evidence edges the frozen model leans on), NOT a
# causal mechanism — deletion scores are fixed-model perturbation tests, not interventions.
RATIONALE_TOP_K: int = 15         # edges kept in the sparse rationale S (top-k by importance)
RATIONALE_TAU: float = 0.5        # necessity distance margin / contrastive margin (delta_nec)
LAMBDA_SPARSE: float = 0.01       # |S| sparsity penalty
LAMBDA_SUFF: float = 1.0          # sufficiency: ||dz_S - dz_full||^2
LAMBDA_NEC: float = 1.0           # necessity: hinge on ||dz_\S - dz_full||
LAMBDA_CONTRAST: float = 0.5      # rationale beats matched-random reconstruction
N_MATCHED_CONTROLS: int = 100     # matched-random controls per rationale (size + relation matched)


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
