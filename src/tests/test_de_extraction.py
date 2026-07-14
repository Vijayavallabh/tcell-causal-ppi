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


def _make_h5(path, arr):
    with h5py.File(path, "w") as f:
        g = f.create_group("layers")
        g.create_dataset("zscore", data=arr)
        g.create_dataset("adj_p_value", data=np.abs(arr))


def test_sparse_and_dense_roundtrip(tmp_path):
    arr = np.linspace(-20, 20, 20 * 6).reshape(20, 6)
    h5 = tmp_path / "de.h5"
    _make_h5(h5, arr)
    with h5py.File(h5, "r") as f:
        de_extraction._extract_sparse_layer(f["layers"]["zscore"], tmp_path / "zscore.npz", chunk=7)
        de_extraction._extract_dense_layer(f["layers"]["adj_p_value"], tmp_path / "adj.npy", chunk=7)

    z = sp.load_npz(tmp_path / "zscore.npz").toarray()
    assert z.shape == arr.shape and z.dtype == np.float32
    assert np.allclose(z, np.clip(arr, -10, 10).astype(np.float32))

    d = np.load(tmp_path / "adj.npy")
    assert d.shape == arr.shape and d.dtype == np.float32
    assert np.allclose(d, np.abs(arr).astype(np.float32))
