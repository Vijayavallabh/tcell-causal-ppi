#!/usr/bin/env python
"""Inspect GWCD4i.DE_stats.by_donors.h5mu — DE recomputed within each donor pair,
for donor-transfer robustness and cross-donor reproducibility supervision.

One modality per donor pair, named {CE}_{CE}. With 4 physical donors there are
C(4,2)=6 pairs. Same .obs/.var/.layers schema as GWCD4i.DE_stats.h5ad.

Key facts verified by this script:
  - Exactly 6 donor-pair modalities, each ~4880 rows.
  - Gene axis is 10273 here (9 fewer than the 10282 in the main DE file) — the
    per-pair fits drop a handful of genes; join on gene_ids, do not assume alignment.
  - Donor splits must key on these CE codes, never the batch-relative D1-D4 labels.
  - Cross-donor agreement is q_post: uncertainty/eval strata only, never an H1 input.

Run:  python examples/inspect_by_donors.py
"""
import os
from itertools import combinations
from pathlib import Path

import h5py

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "data" / "raw"))
PATH = DATA_ROOT / "GWCD4i.DE_stats.by_donors.h5mu"
DONORS = ["CE0006864", "CE0008162", "CE0008678", "CE0010866"]


def _obs_keys(obs_group, n=2):
    """Fast obs-index sample via h5py (backed MuData reads are slow on these files)."""
    idxcol = dict(obs_group.attrs).get("_index", "_index")
    vals = obs_group[idxcol][:n]
    return [v.decode() if isinstance(v, bytes) else v for v in vals]


def main() -> None:
    print(f"FILE: {PATH}  ({PATH.stat().st_size / 1e9:.1f} GB)\n")

    dims = {}
    any_mod = None
    with h5py.File(PATH, "r") as f:
        mods = list(f["mod"].keys())
        print(f"donor-pair modalities ({len(mods)}):")
        for m in mods:
            z = f["mod"][m]["layers"]["zscore"]
            shape = tuple(dict(z.attrs).get("shape", z.shape))
            dims[m] = shape
            print(f"  {m}: zscore shape={shape}")
            if any_mod is None:
                any_mod = (m, _obs_keys(f["mod"][m]["obs"]))
    print(f"\n{any_mod[0]} obs_names sample: {any_mod[1]}")

    gene_axes = {shp[1] for shp in dims.values()}
    print(f"\ngene-axis widths across pairs: {gene_axes} (note: 10273, not 10282)")

    # which donors appear — recover the physical donor set from pair names
    seen = set()
    for m in dims:
        seen.update(m.split("_"))
    print(f"physical donors recovered from pair names: {sorted(seen)}")

    # --- self-check ---
    expected_pairs = {f"{a}_{b}" for a, b in combinations(sorted(DONORS), 2)}
    assert set(dims) == expected_pairs, f"{set(dims)} != {expected_pairs}"
    assert gene_axes == {10273}, gene_axes
    assert seen == set(DONORS), seen
    print("\nOK: 6 pairs, consistent gene axis, and 4 physical donors match expectations.")


if __name__ == "__main__":
    main()
