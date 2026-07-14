"""Extract the 6 dense DE_stats layers to compact on-disk arrays, plus obs/var tables.

Each layer is read from HDF5 in row chunks so the full 33983x10282 float64 matrix is
never materialised in RAM. zscore/log_fc are clipped to +/-10 and stored as float32
CSR (.npz); p-value layers are stored as -log10(p) float32 (raw float32 underflows the
strongest hits to 0); baseMean/lfcSE stay raw float32. Geometry and the q_post schema
are asserted before writing so a drifted re-download fails loudly, not silently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp

from tcell_pipeline import config

Transform = Callable[[np.ndarray], np.ndarray]


def clip_layer(block: np.ndarray, limit: float = config.CLIP_LIMIT) -> np.ndarray:
    """Clip to [-limit, limit] and downcast to float32 (used for zscore/log_fc)."""
    return np.clip(block, -limit, limit).astype(np.float32, copy=False)


def neglog10_layer(block: np.ndarray, floor: float = config.P_VALUE_FLOOR) -> np.ndarray:
    """-log10(p) as float32, flooring p to avoid inf; NaN p-values are preserved as NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (-np.log10(np.clip(block.astype(np.float64, copy=False), floor, None))).astype(np.float32)


def _to_float32(block: np.ndarray) -> np.ndarray:
    return np.asarray(block, dtype=np.float32)


def _extract_sparse_layer(dset: h5py.Dataset, final: Path, chunk: int) -> None:
    n = dset.shape[0]
    blocks: list[sp.csr_matrix] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        blocks.append(sp.csr_matrix(clip_layer(dset[start:stop])))
    matrix = sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix((0, dset.shape[1]))
    config.save_npz_atomic(final, matrix)


def _extract_dense_layer(dset: h5py.Dataset, final: Path, chunk: int, transform: Transform) -> None:
    n, g = dset.shape

    def _writer(tmp: Path) -> None:
        mm = config.open_dense_memmap(tmp, (n, g))
        try:
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                mm[start:stop] = transform(dset[start:stop])
            mm.flush()
        finally:
            del mm

    config.save_npy_atomic(final, _writer)


def run() -> None:
    chunk = config.DE_CHUNK_ROWS
    config.ensure_dir(config.DE_LAYERS_DIR)
    print(f"[de_extraction] streaming layers from {config.DE_STATS_PATH} (chunk={chunk} rows)")
    with h5py.File(config.DE_STATS_PATH, "r") as f:
        layers = f["layers"]
        missing = [n for n in config.DE_LAYERS if n not in layers]
        assert not missing, f"DE layers missing: {missing}"
        first = layers[config.DE_LAYERS[0]]
        assert tuple(first.shape) == (config.DE_N_OBS, config.DE_N_VARS), \
            f"DE geometry drifted: {first.shape} != {(config.DE_N_OBS, config.DE_N_VARS)}"
        for name in config.CLIPPED_SPARSE_LAYERS:
            out = config.DE_LAYERS_DIR / f"{name}.npz"
            print(f"[de_extraction]   {name}: clip +/-{config.CLIP_LIMIT:.0f} -> sparse csr {out.name}")
            _extract_sparse_layer(layers[name], out, chunk)
        for name in config.NEGLOG10_LAYERS:
            out = config.DE_LAYERS_DIR / f"neglog10_{name}.npy"
            print(f"[de_extraction]   {name}: -log10(p) float32 -> {out.name}")
            _extract_dense_layer(layers[name], out, chunk, neglog10_layer)
        for name in config.RAW_DENSE_LAYERS:
            out = config.DE_LAYERS_DIR / f"{name}.npy"
            print(f"[de_extraction]   {name}: dense float32 -> {out.name}")
            _extract_dense_layer(layers[name], out, chunk, _to_float32)

    a = ad.read_h5ad(config.DE_STATS_PATH, backed="r")
    obs = a.obs.reset_index().rename(columns={"index": "obs_name"})
    var = a.var.reset_index().rename(columns={"index": "var_name"})
    missing_q = [c for c in config.Q_POST_COLS if c not in obs.columns]
    assert not missing_q, f"q_post schema drifted; missing from DE obs: {missing_q}"
    config.write_parquet_atomic(obs, config.DE_OBS_PATH)
    config.write_parquet_atomic(var, config.DE_VAR_PATH)
    print(f"[de_extraction] wrote de_obs.parquet ({len(obs)} rows), de_var.parquet ({len(var)} rows)")


if __name__ == "__main__":
    run()
