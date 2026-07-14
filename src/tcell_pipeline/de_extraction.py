"""Extract the 6 dense DE_stats layers to compact on-disk arrays, plus obs/var tables.

Each layer is read from HDF5 in row chunks so the full 33983x10282 float64 matrix is
never materialised in RAM. zscore/log_fc are clipped to +/-10 and stored as float32
CSR (.npz); the remaining layers are streamed into float32 dense .npy via a memmap.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp

from tcell_pipeline import config


def clip_layer(block: np.ndarray, limit: float = config.CLIP_LIMIT) -> np.ndarray:
    """Clip to [-limit, limit] and downcast to float32 (used for zscore/log_fc)."""
    return np.clip(block, -limit, limit).astype(np.float32, copy=False)


def _extract_sparse_layer(dset: h5py.Dataset, final: Path, chunk: int) -> None:
    n = dset.shape[0]
    blocks: list[sp.csr_matrix] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        blocks.append(sp.csr_matrix(clip_layer(dset[start:stop])))
    matrix = sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix((0, dset.shape[1]))
    config.save_npz_atomic(final, matrix)


def _extract_dense_layer(dset: h5py.Dataset, final: Path, chunk: int) -> None:
    n, g = dset.shape
    tmp = final.with_name(final.name + ".tmp")
    config.ensure_dir(final.parent)
    mm = config.open_dense_memmap(tmp, (n, g))
    try:
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            mm[start:stop] = dset[start:stop].astype(np.float32, copy=False)
        mm.flush()
    finally:
        del mm
    tmp.replace(final)


def run() -> None:
    chunk = config.DE_CHUNK_ROWS
    config.ensure_dir(config.DE_LAYERS_DIR)
    print(f"[de_extraction] streaming layers from {config.DE_STATS_PATH} (chunk={chunk} rows)")
    with h5py.File(config.DE_STATS_PATH, "r") as f:
        for name in config.CLIPPED_SPARSE_LAYERS:
            out = config.DE_LAYERS_DIR / f"{name}.npz"
            print(f"[de_extraction]   {name}: clip +/-{config.CLIP_LIMIT:.0f} -> sparse csr {out.name}")
            _extract_sparse_layer(f["layers"][name], out, chunk)
        for name in config.DENSE_LAYERS:
            out = config.DE_LAYERS_DIR / f"{name}.npy"
            print(f"[de_extraction]   {name}: dense float32 -> {out.name}")
            _extract_dense_layer(f["layers"][name], out, chunk)

    a = ad.read_h5ad(config.DE_STATS_PATH, backed="r")
    obs = a.obs.reset_index().rename(columns={"index": "obs_name"})
    var = a.var.reset_index().rename(columns={"index": "var_name"})
    config.write_parquet_atomic(obs, config.DE_OBS_PATH)
    config.write_parquet_atomic(var, config.DE_VAR_PATH)
    print(f"[de_extraction] wrote de_obs.parquet ({len(obs)} rows), de_var.parquet ({len(var)} rows)")


if __name__ == "__main__":
    run()
