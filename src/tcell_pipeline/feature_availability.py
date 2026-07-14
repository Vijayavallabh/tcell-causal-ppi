"""Emit feature_availability.yaml tagging every perturbation_condition column.

Every column is exactly one of q_pre (prediction-time eligible), q_post (response-derived,
prohibited as H1 input), or metadata. q_post always wins over q_pre so the leakage fence
holds, guaranteeing q_pre and q_post are disjoint.
"""
from __future__ import annotations

import yaml

from tcell_pipeline import config


def classify_columns(columns: list[str]) -> dict[str, list[str]]:
    q_post, q_pre, metadata = [], [], []
    for c in columns:
        if c in config.Q_POST_COLS:
            q_post.append(c)
        elif c in config.Q_PRE_COLS or c.startswith(config.DONOR_PC_PREFIX):
            q_pre.append(c)
        else:
            metadata.append(c)
    return {"q_pre": q_pre, "q_post": q_post, "metadata": metadata}


def run() -> dict[str, list[str]]:
    import pandas as pd

    print(f"[feature_availability] reading columns from {config.PERTURBATION_CONDITION_PATH}")
    cols = list(pd.read_parquet(config.PERTURBATION_CONDITION_PATH).columns)
    manifest = classify_columns(cols)
    if manifest["metadata"]:
        # metadata is the permissive fall-through; list it so a response-derived column that
        # is not in Q_POST_COLS cannot slip past the leakage fence unnoticed.
        print(f"[feature_availability] REVIEW metadata (neither q_pre nor q_post): {manifest['metadata']}")
    config.ensure_dir(config.FEATURE_AVAILABILITY_PATH.parent)
    tmp = config.FEATURE_AVAILABILITY_PATH.with_name(config.FEATURE_AVAILABILITY_PATH.name + ".tmp")
    tmp.write_text(yaml.safe_dump(manifest, sort_keys=False))
    tmp.replace(config.FEATURE_AVAILABILITY_PATH)
    print(f"[feature_availability] q_pre={len(manifest['q_pre'])} "
          f"q_post={len(manifest['q_post'])} metadata={len(manifest['metadata'])} "
          f"-> {config.FEATURE_AVAILABILITY_PATH}")
    return manifest


if __name__ == "__main__":
    run()
