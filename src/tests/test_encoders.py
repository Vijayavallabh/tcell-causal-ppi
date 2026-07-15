import numpy as np
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


def make_batch(b: int = 4) -> dict:
    return {
        "uniprot_id": ["P0DP23", "Q8N726", None, "P42771"][:b],
        "ppi_degree_physical": torch.tensor([3, 0, 10, 1][:b]),
        "ppi_degree_functional": torch.tensor([5, 0, 2, 1][:b]),
        "ppi_degree_complex": torch.tensor([1, 0, 0, 2][:b]),
        "control_baseline_expr": torch.tensor([0.5, 0.0, 1.2, 0.3][:b]),
        "culture_condition": ["Rest", "Stim8hr", "Stim48hr", "Rest"][:b],
        "donor_pc": torch.randn(b, config.DONOR_PCA_DIMS),
        "n_guides": torch.tensor([2, 1, 4, 3][:b]),
        "single_guide_estimate": torch.tensor([False, True, False, False][:b]),
    }


def test_missing_embeddings_zero_fallback(tmp_path):
    store = PluggableEmbeddingStore(tmp_path / "absent.parquet", config.PLM_EMBED_DIM)
    out = store.lookup(["P12345", None])
    assert out.shape == (2, config.PLM_EMBED_DIM)
    assert torch.count_nonzero(out) == 0


def test_present_embeddings_loaded_and_missing_id_zero(tmp_path):
    p = tmp_path / "plm.parquet"
    pd.DataFrame(
        {
            "uniprot_id": ["P1", "P2"],
            "embedding": [
                np.ones(config.PLM_EMBED_DIM, dtype=np.float32),
                np.full(config.PLM_EMBED_DIM, 2.0, dtype=np.float32),
            ],
        }
    ).to_parquet(p)
    store = PluggableEmbeddingStore(p, config.PLM_EMBED_DIM)
    out = store.lookup(["P1", "PX"])  # PX absent -> zeros
    assert out[0].sum().item() == float(config.PLM_EMBED_DIM)
    assert out[1].abs().sum().item() == 0.0


def test_wrong_embedding_dim_raises(tmp_path):
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"uniprot_id": ["P1"], "embedding": [np.ones(7, dtype=np.float32)]}).to_parquet(p)
    with pytest.raises(ValueError):
        PluggableEmbeddingStore(p, config.PLM_EMBED_DIM).lookup(["P1"])


def test_output_dims():
    enc = PerturbationEncoder()
    assert enc.target.out_dim == config.PLM_EMBED_DIM + config.PINNACLE_EMBED_DIM + 4 == 1796
    assert enc.context.out_dim == 96
    assert enc.quality.out_dim == 2 + config.GUIDE_SEQ_EMBED_DIM == 66
    h = enc(make_batch(4))
    assert h.shape == (4, config.H_DO_DIM)


def test_subencoder_dims():
    b = make_batch(4)
    assert TargetEncoder()(b).shape == (4, 1796)
    assert ContextEncoder()(b).shape == (4, 96)
    assert QualityEncoder()(b).shape == (4, 66)


def test_condition_embeddings_distinct():
    w = ContextEncoder().condition.weight.detach()
    assert w.shape == (len(config.CONDITIONS), 64)
    assert not torch.allclose(w[0], w[1])
    assert not torch.allclose(w[1], w[2])


def test_no_nans_batch_of_four():
    torch.manual_seed(0)
    h = PerturbationEncoder()(make_batch(4))
    assert torch.isfinite(h).all()


def test_no_trainable_gene_id_embedding():
    enc = PerturbationEncoder()
    embeds = [m for m in enc.modules() if isinstance(m, torch.nn.Embedding)]
    # the only embedding is the 3-way culture condition; no per-gene identity table exists.
    assert len(embeds) == 1
    assert embeds[0].num_embeddings == len(config.CONDITIONS)


def test_missing_scalars_do_not_poison_output():
    # real data has NaN control_baseline_expr (~5% of rows) and occasional NaN n_guides;
    # one NaN scalar must not turn the whole LayerNorm'd h_do into NaN.
    torch.manual_seed(0)
    b = make_batch(4)
    b["control_baseline_expr"] = torch.tensor([float("nan"), 0.0, 1.2, 0.3])
    b["n_guides"] = torch.tensor([float("nan"), 1.0, 4.0, 3.0])
    h = PerturbationEncoder()(b)
    assert torch.isfinite(h).all()


def test_qpost_cols_rejected():
    enc = PerturbationEncoder()
    b = make_batch(4)
    b["n_up_genes"] = torch.tensor([1, 2, 3, 4])  # a q_post (response-derived) column
    with pytest.raises(ValueError):
        enc(b)
