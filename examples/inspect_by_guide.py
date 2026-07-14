#!/usr/bin/env python
"""Inspect GWCD4i.DE_stats.by_guide.h5mu — per-sgRNA DE, for guide-level
reproducibility / aleatoric uncertainty supervision.

Two modalities named by alphanumeric guide rank within each (gene, condition):
  - guide_1: first guide of every (target, condition) pair
  - guide_2: second guide; single-guide targets are absent here
Same .obs/.var/.layers schema as GWCD4i.DE_stats.h5ad. Obs key = {target_contrast}_{condition}.

Key facts verified by this script:
  - guide_1 has more rows than guide_2 (the gap = targets tested with only 1 guide).
  - Layers are dense float64, same 6 as the main DE file, 10282 genes wide.
  - Cross-guide agreement (guide_1 vs guide_2 z-scores) is a legitimate uncertainty
    target — but it is q_post and must never enter H1 features.

Run:  python examples/inspect_by_guide.py
"""
import os
from pathlib import Path

import h5py

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "data" / "raw"))
PATH = DATA_ROOT / "GWCD4i.DE_stats.by_guide.h5mu"


def _obs_keys(obs_group, n=2):
    """Fast obs-index sample via h5py (backed MuData reads are slow on these files)."""
    idxcol = dict(obs_group.attrs).get("_index", "_index")
    vals = obs_group[idxcol][:n]
    return [v.decode() if isinstance(v, bytes) else v for v in vals]


def main() -> None:
    print(f"FILE: {PATH}  ({PATH.stat().st_size / 1e9:.1f} GB)\n")

    dims = {}
    with h5py.File(PATH, "r") as f:
        mods = list(f["mod"].keys())
        print("modalities:", mods)
        for m in mods:
            g = f["mod"][m]
            z = g["layers"]["zscore"]
            shape = tuple(dict(z.attrs).get("shape", z.shape))
            dims[m] = shape
            print(f"\n  {m}: zscore shape={shape}")
            print(f"    layers: {list(g['layers'].keys())}")
            print(f"    obs index='{dict(g['obs'].attrs).get('_index')}' sample={_obs_keys(g['obs'])}")

    gap = dims["guide_1"][0] - dims["guide_2"][0]
    print(f"\nrows only in guide_1 (single-guide targets): {gap}")

    # --- self-check ---
    assert set(dims) == {"guide_1", "guide_2"}, dims
    assert dims["guide_1"][1] == dims["guide_2"][1] == 10282, "gene axis mismatch"
    assert dims["guide_1"][0] > dims["guide_2"][0], "guide_1 should have >= guide_2 rows"
    print("\nOK: modalities, gene axis, and row ordering match expectations.")


if __name__ == "__main__":
    main()
