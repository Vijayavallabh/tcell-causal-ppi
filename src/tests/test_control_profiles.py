import numpy as np
import pandas as pd

from tcell_pipeline import config
from tcell_pipeline import control_profiles as cp


def test_ntc_mask_handles_nan_and_spares_kntc1():
    guide_type = pd.Series(["non-targeting", "targeting", "targeting", "targeting"])
    gene = pd.Series(["NTC-1", "KNTC1", "A1BG", float("nan")])  # NaN must not raise
    m = cp.ntc_mask(guide_type, gene)
    assert m.tolist() == [True, False, False, False]


def test_ntc_mask_word_boundary_and_targeting_name():
    guide_type = pd.Series(["targeting", "targeting"])
    gene = pd.Series(["NTC", "KNTC1"])  # bare NTC matches; KNTC1 does not
    assert cp.ntc_mask(guide_type, gene).tolist() == [True, False]


def test_accumulate_ntc_group_means_and_full_rank_embedding():
    rng = np.random.default_rng(1)
    n_groups, n_genes = 4, 40
    codes = np.array([0, 1, 2, 3] * 25, dtype=np.int64)  # 100 rows, 25 per group
    rows = rng.random((100, n_genes)).astype(np.float32)
    blocks = [rows[:60], rows[60:]]  # streamed in two chunks

    gm, emb = cp.accumulate_ntc(iter(blocks), codes, n_groups, n_genes, config.DONOR_PCA_DIMS)

    assert gm.shape == (n_groups, n_genes)
    expected0 = rows[codes == 0].mean(axis=0)
    assert np.allclose(gm[0], expected0, atol=1e-5)  # vectorized group mean is correct
    assert emb.shape == (n_groups, config.DONOR_PCA_DIMS)  # padded to the fixed schema width
    assert not np.allclose(emb, 0.0)  # PCA actually fit (not the zero-pad degeneracy)
