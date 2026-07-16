"""Experiment registry (feat-011): an immutable YAML manifest giving every screening run an ID, hypothesis,
inputs, split, seed, and registered budget — and enforcing the trial caps the protocol freezes at G2
(report §protocol / line 1187: "at most 32 registered one-seed configurations across the entire EG-IPG
family; at most 16 for each of no more than two close trainable comparator families").

``register_run`` reserves an ID (and refuses to exceed a family's cap); ``log_run`` records the outcome of
a reserved run — status, metrics, checkpoint, GPU hours — including FAILED runs, so the registry is a
complete audit trail rather than a record of successes only.
ponytail: single-process sequential read-modify-write, no file lock; add advisory locking before running
the four real GPU lanes concurrently against one manifest.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from tcell_pipeline import config

_EGIPG_FAMILY = "egipg"


def _cap_for(family: str) -> int:
    return config.MAX_EGIPG_TRIALS if family == _EGIPG_FAMILY else config.MAX_COMPARATOR_TRIALS


def load_registry(path: Path = config.REGISTRY_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    doc = yaml.safe_load(path.read_text()) or {}
    return doc.get("runs") or []  # a present-but-null `runs:` (truncated/external file) degrades to empty


def _save(runs: list[dict], path: Path) -> None:
    config.write_text_atomic(yaml.safe_dump({"runs": runs}, sort_keys=False), Path(path))


def register_run(config_id: str, hypothesis: str, inputs, split: str, seed: int, budget,
                 family: str = _EGIPG_FAMILY, path: Path = config.REGISTRY_PATH) -> str:
    """Reserve an immutable run ID for ``config_id``. The cap counts DISTINCT ``config_id``s per family (the
    report's "32 one-seed configurations", not executions): re-registering an already-seen config — a dev
    re-run or a retry after failure — always succeeds and never consumes a fresh slot, so repeatedly running
    the driver can't silently exhaust the 32 EG-IPG / 16 per-comparator budget. Only a NEW config beyond the
    cap raises ``ValueError`` — a hard ceiling on the frozen search surface. Every execution is still logged
    (a new run ID appended), so the audit trail stays complete."""
    runs = load_registry(path)
    cap = _cap_for(family)
    seen = {r["config_id"] for r in runs if r.get("family") == family}
    if config_id not in seen and len(seen) >= cap:
        raise ValueError(f"{family} family trial cap reached ({len(seen)}/{cap} distinct configs); "
                         f"cannot register new config {config_id!r}")
    if family != _EGIPG_FAMILY:  # "no more than two close trainable comparator families" (report §1291)
        comp_families = {r["family"] for r in runs if r.get("family") != _EGIPG_FAMILY}
        if family not in comp_families and len(comp_families) >= config.MAX_COMPARATOR_FAMILIES:
            raise ValueError(f"comparator-family cap reached ({len(comp_families)}/"
                             f"{config.MAX_COMPARATOR_FAMILIES}); cannot register new family {family!r}")
    run_id = f"run-{len(runs) + 1:04d}"
    runs.append({
        "run_id": run_id, "config_id": config_id, "family": family, "hypothesis": hypothesis,
        "inputs": inputs, "split": split, "seed": int(seed), "budget": budget,
        "status": "registered", "metrics": {}, "checkpoint": None, "gpu_hours": None,
    })
    _save(runs, path)
    return run_id


def log_run(run_id: str, status: str, metrics: dict | None = None, checkpoint: str | None = None,
            gpu_hours: float | None = None, path: Path = config.REGISTRY_PATH) -> dict:
    """Record the outcome of a reserved run (``completed`` / ``failed`` / any status). Raises if the ID was
    never registered — an outcome can only attach to a reserved run."""
    runs = load_registry(path)
    for r in runs:
        if r["run_id"] == run_id:
            r["status"] = status
            r["metrics"] = metrics or {}
            r["checkpoint"] = checkpoint
            r["gpu_hours"] = gpu_hours
            _save(runs, path)
            return r
    raise KeyError(f"run_id {run_id!r} not found in registry {path}")
