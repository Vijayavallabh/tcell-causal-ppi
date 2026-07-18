"""Promotion (feat-011): name the frozen H1 the downstream campaigns consume.

Screening produces one scored row per nested-family member; promotion picks the FINAL model and its
runner-up from them and records what was chosen, on what basis, and by how much. feat-010 (external
comparators), feat-012 (the rationale audit on the frozen H1) and feat-013 (the sealed opening) all
need "the promoted model" to exist as a loadable checkpoint — without this record there is nothing to
point them at.

Ranked on ``systema_pert_specific_delta`` (``PRIMARY_METRIC``), the locked primary endpoint — NOT on
``best_val``, which is not comparable across the family at all: the graph members' totals carry
sparsity/unsourced regulariser terms that expression_only has no equivalent of (a real run measured
typed_static val 435.4 against expression_only's 3.47).

This is deliberately NOT the report's five-seed promotion (``config.N_FINAL_SEEDS``): that re-runs the
winner under five paired seeds and is a separate compute campaign. What this writes is the single-seed
screening promotion, and ``promoted.json`` says so in ``basis``.
"""
from __future__ import annotations

import json
from pathlib import Path

from tcell_pipeline import config
from tcell_pipeline.screening.screening import NETWORK_PROP, PRIMARY_METRIC

# Non-neural topology-diffusion reference: it has no trained weights, so nothing downstream could load
# it as the frozen H1 no matter how it scores. It stays in the screening TABLE and out of promotion.
_NOT_PROMOTABLE = (NETWORK_PROP,)


def _checkpoint_for(screening_root: Path, name: str, seed: int) -> Path:
    """Where screen_config parks a run's best-validation weights (its ckpt_dir default)."""
    return Path(screening_root) / name / str(seed) / "ckpt" / "stage_a_best.pt"


def promote(names: list[str], seed: int = 0, screening_root: Path = config.SCREENING_ROOT,
            noise_margin: float = 0.0, registry_path: Path | None = None,
            pin: str | None = None) -> dict:
    """Rank the completed promotable members by the primary endpoint; record final + runner-up.

    ``noise_margin`` is the delta below which the gap between final and runner-up is not a result. The
    capped-fold campaign reported H2a=+0.0010 that was pure noise, so a promotion decided by a margin
    that size is a coin toss: it is still made (something has to be frozen) but flagged
    ``margin_within_noise`` so it cannot be read as a clean win.

    ``pin`` freezes a SPECIFIC config as the final H1 regardless of its screening rank — for a negative
    fold where the PI keeps the pre-registered confirmatory H1 rather than the argmax winner (only the
    typed+gated model can support the feat-012 rationale audit). The choice is recorded honestly:
    ``pinned``, its 1-based ``pinned_rank``, the true ``screening_winner``, and a ``margin`` to the
    runner-up that is NEGATIVE when the pinned H1 was out-scored. Pinning a config with no completed
    result raises — the PI asked for a specific H1 and must be told it isn't there, not silently handed
    the winner.

    ``registry_path`` (optional) rejects a stale parquet left by a prior run: a config whose latest
    registry run is not ``completed`` is skipped even if a parquet sits at its path, so a half-finished
    campaign cannot promote last time's winner. Raises FileNotFoundError if the chosen H1 has no
    weights on disk — promotion hands downstream a path to LOAD, and failing here beats failing hours
    into feat-012/013.
    """
    import math

    import pandas as pd
    from tcell_pipeline.screening.screening import _config_statuses, _finite_or_none, _is_stale
    statuses = _config_statuses(registry_path, seed) if registry_path is not None else None
    ranking = []
    for name in names:
        if name in _NOT_PROMOTABLE:
            continue
        if _is_stale(name, statuses):
            continue                                   # stale parquet from a prior run — not this result
        path = Path(screening_root) / name / f"{seed}.parquet"
        if not path.exists():
            continue                                   # a lane that never landed cannot be promoted
        row = pd.read_parquet(path).iloc[0].to_dict()
        if row.get("status") != "completed" or PRIMARY_METRIC not in row:
            continue
        metric = float(row[PRIMARY_METRIC])
        if not math.isfinite(metric):
            continue                                   # a NaN/Inf primary endpoint cannot be the H1
        ranking.append({"name": name, "seed": seed, PRIMARY_METRIC: metric,
                        "gpu_hours": row.get("gpu_hours"),
                        "checkpoint": str(_checkpoint_for(screening_root, name, seed))})
    if not ranking:
        raise ValueError(f"nothing promotable among {names} at {screening_root} (seed {seed})")

    # sort by name as well as score: exact ties are reachable (the family sits within noise and the
    # score survives a parquet round-trip), and the choice must not ride on dict/filesystem order
    ranking.sort(key=lambda r: (-r[PRIMARY_METRIC], r["name"]))
    names_ranked = [r["name"] for r in ranking]

    if pin is not None:
        if pin not in names_ranked:
            raise ValueError(f"pinned config {pin!r} has no completed result among {names_ranked} — cannot "
                             f"freeze it as the H1")
        final = ranking[names_ranked.index(pin)]
        # runner-up is the best-scoring OTHER member: on a pinned negative fold this surfaces the model
        # that out-scored the frozen H1 (a negative margin), which is the honest comparison
        others = [r for r in ranking if r["name"] != pin]
        runner_up = others[0] if others else None
    else:
        final, runner_up = ranking[0], (ranking[1] if len(ranking) > 1 else None)

    ckpt = Path(final["checkpoint"])
    if not ckpt.exists():
        raise FileNotFoundError(
            f"promoted config {final['name']!r} has no {ckpt.name} at {ckpt} — a result row is not a "
            f"model; feat-010/012/013 need weights to load")

    margin = (final[PRIMARY_METRIC] - runner_up[PRIMARY_METRIC]) if runner_up else None
    out = {
        "final": final,
        "runner_up": runner_up,
        "ranking": ranking,
        "margin": margin,
        # abs(): a pinned H1 can be BEHIND its runner-up (negative margin); "within noise" is about the
        # gap's size, not its sign
        "margin_within_noise": bool(margin is not None and abs(margin) <= noise_margin),
        "pinned": pin,
        "pinned_rank": (names_ranked.index(pin) + 1) if pin is not None else None,
        "screening_winner": names_ranked[0],
        "tie": bool(runner_up is not None and final[PRIMARY_METRIC] == runner_up[PRIMARY_METRIC]),
        "basis": (f"single-seed screening on {PRIMARY_METRIC} (seed {seed}); NOT the report's "
                  f"{config.N_FINAL_SEEDS}-seed promotion, which is a separate campaign"),
    }
    # sanitize non-finite floats -> None before dumping (allow_nan=False rejects them): ranking metrics
    # are already finite by the filter above, but gpu_hours/margin come straight off disk — a backstop,
    # mirroring the merge/summary path so promoted.json is always valid JSON
    config.write_text_atomic(json.dumps(_finite_or_none(out), indent=2, default=float, allow_nan=False),
                             Path(screening_root) / "promoted.json")
    return out
