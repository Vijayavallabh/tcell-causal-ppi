import h5py
import numpy as np
import scipy.sparse as sp

from tcell_pipeline import de_extraction


def test_clip_layer_range_and_dtype():
    block = np.array([[-50.0, 0.0, 3.0], [12.0, -9.0, 100.0]])
    out = de_extraction.clip_layer(block)
    assert out.dtype == np.float32
    assert out.min() >= -10.0 and out.max() <= 10.0
    assert out[0, 2] == 3.0 and out[1, 0] == 10.0 and out[0, 0] == -10.0


def test_neglog10_layer_preserves_small_pvalues():
    p = np.array([[1e-100, 1.0, 1e-300], [0.5, 1e-45, np.nan]])
    out = de_extraction.neglog10_layer(p)
    assert out.dtype == np.float32
    assert abs(out[0, 0] - 100.0) < 1e-3       # 1e-100 would underflow to 0 as raw float32
    assert out[0, 1] == 0.0                      # -log10(1) == 0
    assert out[0, 2] > 200.0                     # floored, not inf
    assert np.isnan(out[1, 2])                    # NaN p-value preserved


def _make_h5(path, arr):
    with h5py.File(path, "w") as f:
        g = f.create_group("layers")
        g.create_dataset("zscore", data=arr)
        g.create_dataset("p_value", data=np.abs(arr) / (np.abs(arr).max() + 1))
        g.create_dataset("lfcSE", data=np.abs(arr))


def test_sparse_neglog10_and_raw_roundtrip(tmp_path):
    arr = np.linspace(-20, 20, 20 * 6).reshape(20, 6)
    h5 = tmp_path / "de.h5"
    _make_h5(h5, arr)
    with h5py.File(h5, "r") as f:
        de_extraction._extract_sparse_layer(f["layers"]["zscore"], tmp_path / "zscore.npz", chunk=7)
        de_extraction._extract_dense_layer(
            f["layers"]["p_value"], tmp_path / "neglog10_p_value.npy", 7, de_extraction.neglog10_layer)
        de_extraction._extract_dense_layer(
            f["layers"]["lfcSE"], tmp_path / "lfcSE.npy", 7, de_extraction._to_float32)
        p_src = f["layers"]["p_value"][:]

    z = sp.load_npz(tmp_path / "zscore.npz").toarray()
    assert z.shape == arr.shape and z.dtype == np.float32
    assert np.allclose(z, np.clip(arr, -10, 10).astype(np.float32))

    nl = np.load(tmp_path / "neglog10_p_value.npy")
    assert nl.dtype == np.float32
    assert np.allclose(nl, de_extraction.neglog10_layer(p_src), equal_nan=True)

    d = np.load(tmp_path / "lfcSE.npy")
    assert d.dtype == np.float32 and np.allclose(d, np.abs(arr).astype(np.float32))
