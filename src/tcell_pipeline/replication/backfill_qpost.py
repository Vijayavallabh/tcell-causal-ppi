"""Stage 3a: add the q_post schema columns to a DE matrix built before the builder emitted them.

WHY THIS EXISTS. `de_extraction` asserts every `config.Q_POST_COLS` column is present in the DE obs
table. The Amendment-2 builder emits them now, but matrices built before that change do not have them,
and every one of those dies at stage 3 with a bare AssertionError listing thirteen column names. That
has now happened twice, to seven datasets, and was hand-patched both times; this makes it a stage.

WHY NaN AND NOT ZERO. These are the reference screen's post-hoc QC annotations, which the replication
pipeline does not compute. NaN says "not measured here". Zero would say "measured, and negative",
which is a different and false claim. Nothing downstream reads them - they exist so the schema assert
that protects the REFERENCE path stays meaningful rather than being weakened for the replication path.

Idempotent: a matrix that already has all thirteen is left untouched, so this is safe to run in the
prep chain on every dataset every time.

    PYTHONPATH=src python -m tcell_pipeline.replication.backfill_qpost --dataset <name>
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np


def _n_obs(obs) -> int:
    """Row count of an anndata obs group.

    NOT len(obs[some_column]). A categorical column is stored as a GROUP holding `categories` and
    `codes`, so len() on it returns 2 - the number of members - and silently yields two-row columns
    that corrupt the file. This cost a repair pass on three matrices; take the index, which anndata
    always writes, and fall back to the first column that is a real dataset.
    """
    idx = obs.attrs.get("_index", "_index")
    idx = idx.decode() if isinstance(idx, bytes) else str(idx)
    if idx in obs and isinstance(obs[idx], h5py.Dataset):
        return obs[idx].shape[0]
    for k in obs:
        if isinstance(obs[k], h5py.Dataset):
            return obs[k].shape[0]
        if isinstance(obs[k], h5py.Group) and "codes" in obs[k]:
            return obs[k]["codes"].shape[0]
    raise RuntimeError("cannot determine n_obs")


def repair(path: str) -> list[str]:
    """Delete any Q_POST_COLS whose length disagrees with the obs index, so backfill can redo them."""
    from tcell_pipeline import config

    bad = []
    with h5py.File(path, "a") as f:
        obs = f["obs"]
        n = _n_obs(obs)
        for c in config.Q_POST_COLS:
            if c in obs and isinstance(obs[c], h5py.Dataset) and obs[c].shape[0] != n:
                del obs[c]
                bad.append(c)
        if bad:
            order = [o.decode() if isinstance(o, bytes) else str(o)
                     for o in obs.attrs.get("column-order", [])]
            obs.attrs["column-order"] = np.array([o for o in order if o not in bad],
                                                 dtype=h5py.special_dtype(vlen=str))
    return bad


def backfill(path: str) -> list[str]:
    """Add any missing Q_POST_COLS to `path`'s obs as all-NaN. Returns the columns added."""
    from tcell_pipeline import config

    with h5py.File(path, "a") as f:
        obs = f["obs"]
        order = [o.decode() if isinstance(o, bytes) else str(o)
                 for o in obs.attrs.get("column-order", [])]
        missing = [c for c in config.Q_POST_COLS if c not in obs]
        if not missing:
            return []
        n = _n_obs(obs)
        for c in missing:
            obs.create_dataset(c, data=np.full(n, np.nan, dtype=np.float64))
            order.append(c)
        obs.attrs["column-order"] = np.array(order, dtype=h5py.special_dtype(vlen=str))
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--path", default=None, help="override the DE matrix path")
    a = ap.parse_args()
    p = a.path or f"data/intermediate/replication/{a.dataset}.DE_stats_v2.h5ad"
    fixed = repair(p)
    if fixed:
        print(f"[qpost] {a.dataset}: removed {len(fixed)} malformed columns before backfilling")
    added = backfill(p)
    print(f"[qpost] {a.dataset}: {'+' + str(len(added)) + ' columns' if added else 'already complete'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
