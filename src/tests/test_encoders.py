import pandas as pd
import pytest
import torch

from tcell_pipeline import config
from tcell_pipeline.encoders import (
    ContextEncoder,
    PerturbationEncoder,
    PluggableEmbeddingStore,
    QualityEncoder,
    TargetEncoder,
)

DE_OBS_PATH = config.INTERMEDIATE_ROOT / "de_obs.parquet"
DONOR_COLS = [f"{config.DONOR_PC_PREFIX}{i:02d}" for i in range(config.DONOR_PCA_DIMS)]


def _real_batch(idx) -> dict:
    """Build a PerturbationEncoder batch from the real Module 0 marts at row indices ``idx``.

    Skips when the (gitignored) marts aren't on disk -- same convention as the
    feature_availability drift-guard test, so a dataless checkout still runs green.
    perturbation_condition and de_obs are row-aligned 1:1 (row_index == positional),
    so .iloc[idx] pulls the same perturbation from each. n_guides/single_guide_estimate
    only live in de_obs; everything else comes from perturbation_condition.
    """
    pc_path = config.PERTURBATION_CONDITION_PATH
    if not pc_path.exists() or not DE_OBS_PATH.exists():
        pytest.skip("Module 0 marts not generated (real data absent)")
    idx = list(idx)
    pc = pd.read_parquet(pc_path).iloc[idx].reset_index(drop=True)
    obs = pd.read_parquet(DE_OBS_PATH, columns=["n_guides", "single_guide_estimate"]).iloc[idx]
    return {
        "uniprot_id": [None if pd.isna(x) else x for x in pc["uniprot_id"]],
        "ppi_degree_physical": torch.tensor(pc["ppi_degree_physical"].to_numpy()),
        "ppi_degree_functional": torch.tensor(pc["ppi_degree_functional"].to_numpy()),
        "ppi_degree_complex": torch.tensor(pc["ppi_degree_complex"].to_numpy()),
        "control_baseline_expr": torch.tensor(pc["control_baseline_expr"].to_numpy()),
        "culture_condition": pc["culture_condition"].tolist(),
        "donor_pc": torch.tensor(pc[DONOR_COLS].to_numpy(dtype="float32")),
        "n_guides": torch.tensor(obs["n_guides"].to_numpy()),
        "single_guide_estimate": torch.tensor(obs["single_guide_estimate"].to_numpy(dtype=bool)),
    }


def real_batch(n: int = 8) -> dict:
    return _real_batch(range(n))


# --- store tests: exercised against the real ESM-2 PLM embeddings parquet ---

def _real_plm_ids() -> list[str]:
    if not config.PLM_EMBEDDINGS_PATH.exists():
        pytest.skip("PLM embeddings not generated (run tcell_pipeline.embeddings_plm)")
    return [str(u) for u in pd.read_parquet(config.PLM_EMBEDDINGS_PATH, columns=["uniprot_id"])["uniprot_id"]]


def test_real_plm_present_loaded_and_missing_id_zero():
    ids = _real_plm_ids()
    store = PluggableEmbeddingStore(config.PLM_EMBEDDINGS_PATH, config.PLM_EMBED_DIM)
    out = store.lookup([ids[0], "XXXXXXNOTAPROTEIN"])  # real id -> real vector; bogus -> zeros
    assert out.shape == (2, config.PLM_EMBED_DIM)
    assert torch.isfinite(out).all()
    assert out[0].abs().sum().item() > 0.0     # real ESM-2 vector is non-trivial
    assert out[1].abs().sum().item() == 0.0    # absent protein falls back to zeros


def test_real_plm_store_dim_mismatch_raises():
    # feed the REAL 1280-d parquet but declare the wrong dim -> the store must reject it,
    # the same guard that catches a corrupt/mismatched embeddings file at load time.
    _real_plm_ids()
    store = PluggableEmbeddingStore(config.PLM_EMBEDDINGS_PATH, config.PLM_EMBED_DIM + 1)
    with pytest.raises(ValueError):
        store.lookup([_real_plm_ids()[0]])


def test_real_pinnacle_context_embeddings_loaded():
    # the CD4 helper T-cell PINNACLE context: a covered protein loads a real 128-d vector;
    # a protein outside the context (or bogus) falls back to zeros.
    if not config.PINNACLE_EMBEDDINGS_PATH.exists():
        pytest.skip("PINNACLE embeddings not generated (run tcell_pipeline.embeddings_pinnacle)")
    ids = [str(u) for u in pd.read_parquet(config.PINNACLE_EMBEDDINGS_PATH, columns=["uniprot_id"])["uniprot_id"]]
    store = PluggableEmbeddingStore(config.PINNACLE_EMBEDDINGS_PATH, config.PINNACLE_EMBED_DIM)
    out = store.lookup([ids[0], "XXXXXXNOTAPROTEIN"])
    assert out.shape == (2, config.PINNACLE_EMBED_DIM)
    assert torch.isfinite(out).all()
    assert out[0].abs().sum().item() > 0.0
    assert out[1].abs().sum().item() == 0.0


# --- structural tests: no batch / no data needed ---

def test_output_dims():
    enc = PerturbationEncoder()
    assert enc.target.out_dim == config.PLM_EMBED_DIM + config.PINNACLE_EMBED_DIM + 4 == 1412
    assert enc.context.out_dim == 96
    assert enc.quality.out_dim == 2 + config.GUIDE_SEQ_EMBED_DIM == 66


def test_condition_embeddings_distinct():
    w = ContextEncoder().condition.weight.detach()
    assert w.shape == (len(config.CONDITIONS), 64)
    assert not torch.allclose(w[0], w[1])
    assert not torch.allclose(w[1], w[2])


def test_no_trainable_gene_id_embedding():
    enc = PerturbationEncoder()
    embeds = [m for m in enc.modules() if isinstance(m, torch.nn.Embedding)]
    # the only embedding is the 3-way culture condition; no per-gene identity table exists.
    assert len(embeds) == 1
    assert embeds[0].num_embeddings == len(config.CONDITIONS)


# --- forward-pass tests: exercised on the real Module 0 marts ---

def test_forward_shape_on_real_batch():
    b = real_batch(4)
    assert TargetEncoder()(b).shape == (4, 1412)
    assert ContextEncoder()(b).shape == (4, 96)
    assert QualityEncoder()(b).shape == (4, 66)
    assert PerturbationEncoder()(b).shape == (4, config.H_DO_DIM)


def test_no_nans_on_real_batch():
    torch.manual_seed(0)
    # head(8) includes the row-0 NaN control_baseline_expr, so this already crosses the guard.
    h = PerturbationEncoder()(real_batch(8))
    assert torch.isfinite(h).all()


def test_real_nan_rows_do_not_poison_output():
    # the real screen has NaN control_baseline_expr (~5% of rows) and occasional NaN n_guides;
    # one NaN scalar must not turn the whole LayerNorm'd h_do into NaN. Find the real NaN rows
    # in the current marts rather than hardcoding positions (survives mart regeneration).
    if not config.PERTURBATION_CONDITION_PATH.exists() or not DE_OBS_PATH.exists():
        pytest.skip("Module 0 marts not generated (real data absent)")
    ctrl = pd.read_parquet(config.PERTURBATION_CONDITION_PATH, columns=["control_baseline_expr"])
    ng = pd.read_parquet(DE_OBS_PATH, columns=["n_guides"])
    rows = []
    if ctrl["control_baseline_expr"].isna().any():
        rows.append(int(ctrl["control_baseline_expr"].isna().to_numpy().argmax()))
    if ng["n_guides"].isna().any():
        rows.append(int(ng["n_guides"].isna().to_numpy().argmax()))
    if not rows:
        pytest.skip("no NaN scalar rows in current marts")
    h = PerturbationEncoder()(_real_batch(rows + [0, 1]))  # pad to a small batch
    assert torch.isfinite(h).all()


def test_qpost_cols_rejected():
    enc = PerturbationEncoder()
    b = real_batch(4)
    b["n_up_genes"] = torch.tensor([1, 2, 3, 4])  # a q_post (response-derived) column
    with pytest.raises(ValueError):
        enc(b)


def test_encoder_runs_on_gpu_when_available():
    # device-aware: moving the encoder to CUDA runs the whole forward on GPU (CPU-built
    # embedding/scalar tensors are moved to the module's device inside forward).
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    enc = PerturbationEncoder().to("cuda")
    h = enc(real_batch(4))
    assert h.is_cuda and h.shape == (4, config.H_DO_DIM) and torch.isfinite(h).all()
