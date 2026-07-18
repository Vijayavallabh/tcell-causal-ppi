"""Module 7 screening driver: run the §10.6 nested family (+ the untyped-graph diagnostic) on the real
blocked-target-OOD development split and report H2a / H2b on the primary endpoint.

    PYTHONPATH=src python -m tcell_pipeline.screening.run_screening --epochs 2 --device cuda

Builds the real PPI graph + train/val PerturbationDatasets once, screens each family member through the
Stage-A Trainer, writes predictions (common output schema), a per-config metrics table, a summary JSON, and
registers every run in the experiment registry. The graph message passing is CPU-bound per subgraph; the
encoders run on ``--device``.

Memory: the typed encoder's per-edge signed messages are heavy on real DENSE PPI subgraphs (a hub's
512-node neighbourhood carries tens of thousands of STRING functional edges) — it OOMs a single 80 GB A100
at batch 32. So ``--device cuda`` enables ``expandable_segments`` and the default batch is small; a config
that still OOMs is isolated by ``run_screening`` (logged failed, the wave continues) rather than aborting
the run. ``--device cpu`` (1 TB RAM) is the report's home for graph message passing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.graph import build_hetero_graph  # noqa: E402
from tcell_pipeline.programs.program_basis import zscore_path  # noqa: E402
from tcell_pipeline.screening.screening import (  # noqa: E402
    CONDITION_GATED,
    EXPRESSION_ONLY,
    NETWORK_PROP,
    TYPED_STATIC,
    UNTYPED_GNN,
    merge_lane_results,
    nested_family_configs,
    run_screening,
    score_network_propagation,
)
from tcell_pipeline.training.dataset import PerturbationDataset  # noqa: E402

_WAVE = [EXPRESSION_ONLY, UNTYPED_GNN, TYPED_STATIC, CONDITION_GATED]  # report §screening first wave
_TABLE = _WAVE + [NETWORK_PROP]  # + the non-neural topology-diffusion reference (feat-007)
_COLS = ["name", "primary", "pearson", "systema", "centroid", "prog_cos", "mae", "rmse", "topk", "sign"]


def _print_table(by_name: dict, names: list[str]) -> None:
    print("\n" + " ".join(f"{c:>16}" if c == "name" else f"{c:>9}" for c in _COLS))
    for n in names:
        r = by_name.get(n, {"name": n, "status": "missing"})
        if r.get("status") == "completed":
            print(" ".join(f"{r['name']:>16}" if c == "name" else f"{r[c]:>9.4f}" for c in _COLS))
        else:
            print(f"{n:>16}  {r.get('status', '?').upper()}: {r.get('error', '')}")


def _print_contrasts(summary: dict) -> None:
    for hyp in ("h2a", "h2b"):
        if hyp in summary:
            c = summary[hyp]
            print(f"[screen] {hyp.upper()}: {c['better']} vs {c['worse']} Δsystema={c['delta']:+.4f} "
                  f"supported={c['supported']}")
        else:
            print(f"[screen] {hyp.upper()}: NOT COMPUTED — a member is missing or failed")


def promote_final(seed: int = config.SPLIT_SEED, noise_margin: float = 0.0, pin: str | None = None) -> int:
    """Name the frozen H1 (+ runner-up) from the screened rows — what feat-010/012/013 consume."""
    from tcell_pipeline.screening.promotion import promote
    p = promote(_TABLE, seed=seed, noise_margin=noise_margin, registry_path=config.REGISTRY_PATH, pin=pin)
    print("[promote] ranking on systema_pert_specific_delta (the locked primary endpoint):")
    for i, r in enumerate(p["ranking"], 1):
        mark = "  <- PINNED as H1" if r["name"] == p.get("pinned") else ""
        print(f"    {i}. {r['name']:>16}  systema={r['systema']:+.4f}{mark}")
    print(f"[promote] FINAL     : {p['final']['name']}  -> {p['final']['checkpoint']}")
    print(f"[promote] RUNNER-UP : {p['runner_up']['name'] if p['runner_up'] else '(none)'}")
    if p.get("pinned") is not None:
        print(f"[promote] PINNED    : {p['pinned']} frozen as the pre-registered H1, ranked "
              f"{p['pinned_rank']}/{len(p['ranking'])}; screening winner was {p['screening_winner']}")
    if p["margin"] is not None:
        note = ("  ** WITHIN NOISE **" if p["margin_within_noise"]
                else ("  ** the frozen H1 is BEHIND the runner-up **" if p["margin"] < 0 else ""))
        print(f"[promote] margin    : {p['margin']:+.4f} (final − runner-up){note}")
    if p["tie"]:
        print("[promote] ** EXACT TIE on the primary endpoint — broken by name, not by evidence **")
    print(f"[promote] basis     : {p['basis']}")
    return 0


def merge(seed: int = config.SPLIT_SEED) -> int:
    """Recombine the fan-out lanes' per-config rows into summary.json + H2a/H2b (see --only)."""
    summary = merge_lane_results(_TABLE, seed=seed, registry_path=config.REGISTRY_PATH)
    by_name = {r["name"]: r for r in summary["results"]}
    _print_table(by_name, _TABLE)
    _print_contrasts(summary)
    print(f"[screen] summary -> {summary['summary_path']}")
    done = [r for r in summary["results"] if r.get("status") == "completed"]
    if len(done) < len(_TABLE):
        missing = [r["name"] for r in summary["results"] if r.get("status") != "completed"]
        print(f"[screen] INCOMPLETE: {len(done)}/{len(_TABLE)} configs have results; missing {missing}")
        return 1
    print(f"[screen] OK ({len(done)}/{len(_TABLE)} completed)")
    return 0


def run(epochs: int = 2, batch_size: int = 8, seed: int = config.SPLIT_SEED,
        n_max: int | None = None, device: str = "cpu", only: str | None = None) -> int:
    torch.set_num_threads(1)
    if device.startswith("cuda"):
        # reduce allocator fragmentation for the dense-subgraph typed encoder (set before the first CUDA
        # allocation, i.e. before any .to(device)); does NOT lift the fundamental per-batch memory need
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    required = [config.BLOCKED_SPLIT_PATH, config.PERTURBATION_CONDITION_PATH, config.DE_OBS_PATH,
                config.DE_VAR_PATH, config.PROGRAM_LOADINGS_PATH, zscore_path(),
                config.PROTEIN_EDGES_PATH, config.COMPLEX_MEMBERSHIP_PATH, config.ID_MAPPING_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(f"[screen] required artifacts absent: {missing} — run run_module0.py, splits, "
              f"run_program_basis, and the PPI graph build first")
        return 1

    if only is not None and only not in _TABLE:
        print(f"[screen] --only {only!r} is not one of {_TABLE}")
        return 1
    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()
    graph, gene_to_idx = build_hetero_graph()
    train_ds = PerturbationDataset("train", n_max=n_max)
    val_ds = PerturbationDataset("val", n_max=n_max)
    wave = _WAVE if only is None else [n for n in _WAVE if n == only]
    print(f"[screen] {len(train_ds)} train / {len(val_ds)} val; wave={wave}; epochs={epochs}; "
          f"device={device}; cache={config.SUBGRAPH_CACHE_SIZE}")

    configs = nested_family_configs(gene_names, graph, gene_to_idx, epochs, names=wave,
                                    batch_size=batch_size, seed=seed)

    def netprop(train_ds, val_ds, train_mean, *, predictions_root, screening_root, split):
        return score_network_propagation(train_ds, val_ds, train_mean, graph=graph, gene_to_idx=gene_to_idx,
                                         basis=train_ds.B.numpy(), seed=seed, batch_size=batch_size,
                                         predictions_root=predictions_root, screening_root=screening_root,
                                         split=split)
    netprop.screen_name = NETWORK_PROP

    # network propagation is cheap + CPU-only: it rides along with the full wave, and gets its own
    # lane only when asked for by name
    extra = [netprop] if only in (None, NETWORK_PROP) else []
    summary = run_screening(configs, train_ds, val_ds, device=device, registry_path=config.REGISTRY_PATH,
                            extra_scorers=extra, write_summary=(only is None))

    by_name = {r["name"]: r for r in summary["results"]}
    _print_table(by_name, _TABLE if only is None else [only])
    if only is None:
        _print_contrasts(summary)
        print(f"[screen] summary -> {summary['summary_path']}")
    else:  # H2a/H2b span configs, so a lane cannot form them; --merge does, once every lane has landed
        print(f"[screen] lane {only!r} done; run --merge once every lane has landed for H2a/H2b")

    completed = [r for r in summary["results"] if r.get("status") == "completed"]
    if not completed:  # non-zero exit so CI/cron doesn't read a wholly-failed wave as success
        print("[screen] FAILED: no config completed — no predictions or H2a/H2b produced")
        return 1
    print(f"[screen] OK ({len(completed)}/{len(summary['results'])} completed)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="small by default: the typed encoder OOMs 80GB on dense real subgraphs at batch 32 "
                         "(≤8 on one GPU, or --device cpu); a config that still OOMs is isolated, not fatal")
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--n-max", type=int, default=None, help="cap examples per split (quick runs)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (encoders; graph message passing is CPU-bound)")
    ap.add_argument("--only", default=None, metavar="NAME",
                    help=f"screen ONE config ({'|'.join(_TABLE)}) so the wave can fan one process per GPU. "
                         f"Writes that config's row + predictions + registry entry but NOT summary.json "
                         f"(a lane's summary would claim the whole wave); finish with --merge")
    ap.add_argument("--merge", action="store_true",
                    help="recombine the lanes' rows -> summary.json + H2a/H2b; non-zero if any is missing")
    ap.add_argument("--promote", action="store_true",
                    help="rank the screened rows on the primary endpoint -> promoted.json (the frozen H1 "
                         "+ runner-up that feat-010/012/013 consume)")
    ap.add_argument("--noise-margin", type=float, default=0.0,
                    help="a final-vs-runner-up gap <= this (abs) is flagged a coin toss, not a win (--promote)")
    ap.add_argument("--pin", default=None, metavar="NAME",
                    help="freeze NAME as the H1 regardless of its screening rank (--promote) — for a negative "
                         "fold where the pre-registered confirmatory H1 is kept over the argmax winner")
    a = ap.parse_args()
    if a.merge:
        sys.exit(merge(a.seed))
    if a.promote:
        sys.exit(promote_final(a.seed, a.noise_margin, a.pin))
    sys.exit(run(a.epochs, a.batch_size, a.seed, a.n_max, a.device, a.only))
