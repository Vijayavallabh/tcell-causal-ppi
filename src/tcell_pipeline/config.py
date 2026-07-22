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
# Subgraphs are mini-batched through message passing, so peak memory scales with the batch — and a hub's
# 512-node neighbourhood carries tens of thousands of edges (the typed encoder OOMs 80 GB at batch 32).
# The encoders therefore message-pass at most this many subgraphs at once and stitch the results, so a
# caller's batch_size (evaluation defaults to BATCH_SIZE=64) sets the optimisation batch, not the memory
# ceiling. Costs ~8% throughput vs one big batch (bs=32 measured 15.2 vs 16.3 rows/s at bs=8) for a 3x
# smaller footprint. 0/None disables chunking.
GRAPH_ENCODE_CHUNK: int = int(os.environ.get("GRAPH_ENCODE_CHUNK", 8))
# Sampling a target's subgraph costs ~28 ms and depends on neither the donor, the model weights, nor
# train/eval mode — but DONOR_INVARIANCE re-forwards each batch 1+DONOR_INVARIANCE_SAMPLES times per
# step, so it is paid 3x per row per step, and targets recur ~3x per epoch besides (7,079 unique over
# 21,262 train rows). Memoising per target removes ~85 of condition_gated's ~183 ms/row. Real
# subgraphs average 4.6 MB, so the default holds the batch (killing the within-step 3x for ~0.3 GB);
# raise it toward the fold's unique-target count to also catch cross-row and cross-epoch reuse, at
# ~4.6 MB each (~32 GB for a whole fold). 0 disables the cache.
SUBGRAPH_CACHE_SIZE: int = int(os.environ.get("SUBGRAPH_CACHE_SIZE", 64))
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


# --- Module 5 (Loss + Training; Stage A trains M1+M2+M3, Stage B fits the calibration head and the
#     rationale head over the FROZEN Stage-A predictor — feat-008) ---
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-5
MAX_EPOCHS: int = 100
EARLY_STOP_PATIENCE: int = 10
BATCH_SIZE: int = 64
GRAD_CLIP: float = 1.0
HUBER_DELTA: float = 1.0          # L_response/L_gene Huber transition point
FOCAL_GAMMA: float = 2.0          # focal down-weighting of easy (abundant non-DE) genes in L_DE
LAMBDA_DE: float = 0.1            # weight on the DE up/down classification head
LAMBDA_INV: float = 0.1           # weight on donor-invariance of the shared program component
LAMBDA_GRAPH: float = 0.01        # weight on the edge-gate sparsity + unsourced-reliance penalty
GATE_DEAD: float = 1e-3           # an edge gate below this contributes nothing in float32 — the pilot's
#                                   collapse criterion; shared by the trainer's gate logging and the
#                                   rationale audit's collapsed-gate guard (one threshold, not two)
LAMBDA_GENE: float = 0.5          # weight on gene-level (delta_x) reconstruction vs program-level
# DE up/down call from the per-gene z-score: |z| >~ 1.64 is the two-sided 10% tail, the proxy this
# dataset carries for adj_p < 0.1 (the exact adj_p layer is not part of the __getitem__ contract).
DE_CALL_ZSCORE: float = 1.645
# Donor invariance (L_invariance, §8.2 item 4): the mart's donor_pc is the mean of the real per-donor
# control profiles; DONOR_INVARIANCE re-runs the encoder under the actual per-donor vectors
# (control_donor_profiles, 4 real donors) and penalises the variance of the shared program component
# across them — a real, dense donor-generalisation signal. SAMPLES = distinct real donors drawn per step
# (>=2 for a non-zero variance). Off -> the term is a no-op.
DONOR_INVARIANCE: bool = True
DONOR_INVARIANCE_SAMPLES: int = 2
CHECKPOINTS_ROOT: Path = Path(os.environ.get("CHECKPOINTS_ROOT", DATA_DIR / "checkpoints"))
LOGS_ROOT: Path = Path(os.environ.get("LOGS_ROOT", DATA_DIR / "logs"))
# Stage-B outputs: the freeze-gate report over the calibration + rationale fits (feat-008). The fitted
# heads themselves are seed-namespaced under CHECKPOINTS_ROOT/stage_b/<seed>/ (stage_b.stage_b_ckpt_dir).
STAGE_B_ROOT: Path = Path(os.environ.get("STAGE_B_ROOT", DATA_DIR / "results" / "stage_b"))


# --- Module 6 (Evaluation Metrics + Simple Baselines; feat-009 + feat-006) ---
METRICS_TOP_K: int = 20           # top-k strongest up/down genes for recall (§10.4)
METRICS_SIGN_TOP_N: int = 50      # strongest effects scored for sign accuracy (§10.4)
# Common prediction store: predictions/<model>/<split>/<seed>.parquet (feat-006/009 output schema).
PREDICTIONS_ROOT: Path = Path(os.environ.get("PREDICTIONS_ROOT", DATA_DIR / "results" / "predictions"))


# --- Module 7 (Graph Baselines + Screening Harness; feat-007 + feat-011) ---
# Per-config screening results (metrics table + summary) and the immutable run registry (report §protocol).
SCREENING_ROOT: Path = Path(os.environ.get("SCREENING_ROOT", DATA_DIR / "results" / "screening"))
REGISTRY_PATH: Path = Path(os.environ.get("REGISTRY_PATH", DATA_DIR / "results" / "experiment_registry.yaml"))
# Trial caps enforced by the registry (report §protocol / line 1187): at most 32 one-seed configs across
# the ENTIRE EG-IPG family, at most 16 per close trainable comparator family.
MAX_EGIPG_TRIALS: int = 32
MAX_COMPARATOR_TRIALS: int = 16
MAX_COMPARATOR_FAMILIES: int = 2  # "no more than two close trainable comparator families" (report §1291)
N_SCREENING_SEEDS: int = 1        # one seed for architecture/hyperparameter screening (report §857)
N_FINAL_SEEDS: int = 5            # paired development seeds for the promoted final configurations


# --- Module 8 (External Comparators + Rationale Audit + Sealed Eval + Reproducibility;
#     feat-010 + feat-012 + feat-013) ---
COMPARATORS_ROOT: Path = Path(os.environ.get("COMPARATORS_ROOT", DATA_DIR / "results" / "comparators"))
RATIONALE_AUDIT_ROOT: Path = Path(os.environ.get("RATIONALE_AUDIT_ROOT", DATA_DIR / "results" / "rationale_audit"))
SEALED_ROOT: Path = Path(os.environ.get("SEALED_ROOT", DATA_DIR / "results" / "sealed"))
REPRODUCIBILITY_ROOT: Path = Path(os.environ.get("REPRODUCIBILITY_ROOT", DATA_DIR / "results" / "reproducibility"))
# H1 predictive-superiority margin: EG-IPG must beat the strongest baseline by MORE than this on the
# sequestered challenge split, with 95% confidence, to confirm H1 (report §protocol, §10.7 hypothesis rule).
DELTA_PRED: float = 0.05
N_BOOTSTRAP: int = 10000           # paired-row bootstrap resamples for the sealed-eval confidence intervals
N_RATIONALE_AUDIT_CASES: int = 50  # stratified rationale-audit case budget (feat-012)


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
