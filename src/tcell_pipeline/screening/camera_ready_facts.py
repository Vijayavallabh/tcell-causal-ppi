"""Small measurements the ICBINB paper quotes that no results JSON held, re-derived for the camera-ready.

Before 2026-10-01 these lived in training logs, code comments or prose, and four of them were wrong in
the accepted paper: the typed arms' training length, the live gate range, the "no close paralog"
guarantee and the functional-edge median. Persisting them lets ``paper/icbinb/verify_numbers.py`` gate
them like every other number. Read-only apart from the one JSON it writes.

    PYTHONPATH=src python -m tcell_pipeline.screening.camera_ready_facts \
        --out data/results/camera_ready/facts.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tcell_pipeline import config

# Table 1's five-seed family: the gated arm from the live-gate (lambda_graph=0) root, the rest from the
# reference root, exactly as screening_lambda0/robustness_5seed.json pairs them.
FAMILY_LANES = {"expression_only": "data/results/screening", "typed_static": "data/results/screening",
                "untyped_gnn": "data/results/screening", "condition_gated": "data/results/screening_lambda0"}
SEEDS = range(5)


def lane_training(root: str, arm: str, seed: int) -> dict:
    """Epochs run, the epoch early stopping kept (argmin validation total loss, the checkpoint that was
    scored), and the edge-gate mean at that epoch and at the last one. ``None`` for arms with no gate."""
    hist = json.loads((Path(root) / arm / str(seed) / "logs" / "stage_a_history.json").read_text())
    best = int(np.argmin([r["val"]["total"] for r in hist]))
    gate = lambda r: r["val"].get("gate_mean")                      # noqa: E731
    return {"epochs_run": len(hist), "best_epoch": best,
            "gate_at_best": gate(hist[best]), "gate_last": gate(hist[-1])}


def training_table() -> dict:
    out = {arm: {s: lane_training(root, arm, s) for s in SEEDS} for arm, root in FAMILY_LANES.items()}
    for arm, lanes in out.items():
        pq = [pd.read_parquet(Path(FAMILY_LANES[arm]) / arm / f"{s}.parquet").iloc[0] for s in SEEDS]
        for s, row in zip(SEEDS, pq):
            lanes[s]["gpu_hours"] = float(row["gpu_hours"])
    return out


def val_paralog_residual() -> dict:
    """The share of VALIDATION targets with a training target at centered ESM-2 cosine >= threshold, by
    the split's own audit function. The frozen audit only measured train-to-challenge, so the paper's
    "no training target is a close paralog of a validation target" was never checked. The same call on
    the challenge roles must reproduce the frozen audit, which is the control that this is the audit."""
    from tcell_pipeline.splits import _sequence_residual
    pc = pd.read_parquet(config.PERTURBATION_CONDITION_PATH, columns=["hgnc_symbol", "uniprot_id"])
    genes = sorted(pc["hgnc_symbol"].dropna().unique())
    gene_uni = dict(zip(pc["hgnc_symbol"], pc["uniprot_id"].astype("string")))
    emb = pd.read_parquet(config.PLM_EMBEDDINGS_PATH)
    u2v = {str(u): np.asarray(v, dtype=np.float32) for u, v in zip(emb["uniprot_id"], emb["embedding"])}
    vec = {g: u2v[gene_uni[g]] for g in genes if isinstance(gene_uni.get(g), str) and gene_uni[g] in u2v}
    role = dict(pd.read_csv(config.BLOCKED_SPLIT_PATH).values)
    th = config.SEQ_SIM_COSINE_THRESHOLD
    frozen = json.loads(Path(config.SPLIT_LEAKAGE_REPORT_PATH).read_text())
    control = _sequence_residual(role, genes, vec, th)
    as_chal = {g: {"val": "challenge", "train": "train"}.get(r, "other") for g, r in role.items()}
    val = _sequence_residual(as_chal, genes, vec, th)
    naive = _sequence_residual({g: "challenge" if i % 20 == 0 else "train" for i, g in enumerate(genes)},
                               genes, vec, th)
    n_val = sum(1 for g in genes if role.get(g) == "val" and g in vec)
    return {"threshold": th, "n_val_targets": n_val,
            "val_frac_ge_threshold": val["frac_ge_threshold"],
            "val_n_ge_threshold": int(round(val["frac_ge_threshold"] * n_val)),
            "val_max_cosine": val["max"],
            "control_challenge_frac": control["frac_ge_threshold"],
            "frozen_audit_challenge_frac": frozen["sequence_train_to_challenge_cosine"]["frac_ge_threshold"],
            "naive_every_20th_frac": naive["frac_ge_threshold"]}


def functional_edge_median() -> dict:
    """Median STRING score over the functional_assoc relation (the STRING edges; 6,857,702 of them)."""
    e = pd.read_parquet(config.PROTEIN_EDGES_PATH, columns=["source", "score"])
    s = e.loc[e["source"] == "string", "score"]
    return {"n_edges": int(len(s)), "n_all_edges": int(len(e)), "median_score": float(s.median())}


def run(out: Path) -> dict:
    facts = {"training": training_table(), "val_paralog": val_paralog_residual(),
             "functional_edges": functional_edge_median()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, indent=2))
    return facts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/results/camera_ready/facts.json")
    f = run(Path(ap.parse_args().out))
    for arm, lanes in f["training"].items():
        print(f"{arm:16s} " + "  ".join(
            f"s{s}: {v['epochs_run']}ep best {v['best_epoch']} gate {v['gate_at_best']}"
            f" {v['gpu_hours']:.2f}h" for s, v in lanes.items()))
    print(json.dumps({k: v for k, v in f.items() if k != "training"}, indent=1))
