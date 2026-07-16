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
    """Reserve an immutable run ID for ``config_id``. Raises ``ValueError`` once the family's registered
    trial count would exceed its cap (32 EG-IPG / 16 per comparator family) — the budget is a hard ceiling,
    not advisory, so screening cannot silently expand its search surface."""
    runs = load_registry(path)
    cap = _cap_for(family)
    used = sum(1 for r in runs if r.get("family") == family)
    if used >= cap:
        raise ValueError(f"{family} family trial cap reached ({used}/{cap}); no further runs may register")
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
