"""Stage-B orchestrator: fit the calibration head + the rationale head over a FROZEN Stage-A model,
then put both through the near-null-signal freeze gate (feat-008 §a-§c).

    PYTHONPATH=src python -m tcell_pipeline.training.run_stage_b --ckpt data/checkpoints/stage_a_best.pt

Both heads are fitted on TRAIN and gated on VAL. Nothing here can promote a fit on its own numbers: the
gate compares every headline against its own control (a per-program constant sigma and a row-permuted
sigma for calibration; matched-random edge sets and the zero-init head for the rationale), corrects the
whole run's contrast family with BOTH Bonferroni and Holm, and the process EXIT CODE is the gate's
decision — 0 only on a freeze, 1 on a refusal, 2 when nothing was decidable. A run that prints
"undecidable" must not exit 0.

An expression-only model has no rationale to fit; its rationale contrasts are simply absent, which the
gate reads as UNDECIDABLE (never as a pass).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path for direct runs

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.encoders.batch import build_encoder_batch  # noqa: E402
from tcell_pipeline.graph import build_hetero_graph  # noqa: E402
from tcell_pipeline.graph.typed_graph_encoder import TypedGraphEncoder  # noqa: E402
from tcell_pipeline.model import EGIPGModel  # noqa: E402
from tcell_pipeline.programs.program_basis import zscore_path  # noqa: E402
from tcell_pipeline.rationale.rationale_fit import fit_rationale_head, rationale_contrasts  # noqa: E402
from tcell_pipeline.training.dataset import PerturbationDataset  # noqa: E402
from tcell_pipeline.training.freeze_gate import evaluate_gate, exit_code, fmt, render  # noqa: E402
from tcell_pipeline.training.stage_b import (  # noqa: E402
    calibration_contrasts,
    fit_calibration,
    frozen_caches,
    stage_b_ckpt_dir,
)
from tcell_pipeline.training.trainer import seeded_init  # noqa: E402

CALIBRATION_CONTRASTS = ("vs_constant_sigma", "vs_permuted_sigma")
RATIONALE_CONTRASTS = ("sufficiency_vs_random", "necessity_vs_random", "sufficiency_vs_untrained")
# The overall gate REQUIRES all five by name. Defaulting `required` to "whatever was supplied" would let
# a run that never produced the rationale contrasts (expression-only model, or a fit that wrote no
# checkpoint) pass on the calibration alone — absence of evidence reading as a pass. The per-artifact
# decisions below still report each family separately, so nothing is lost by being strict here.
REQUIRED_CONTRASTS = CALIBRATION_CONTRASTS + RATIONALE_CONTRASTS


def overall_gate(contrasts: dict, *, alpha: float = 0.05) -> dict:
    """The run-level gate: ALL FIVE contrasts required by name.

    A named function rather than an inline call so the `required=` argument is reachable from a test.
    Letting `required` default to "whatever contrasts were supplied" would let a run that never produced
    the rationale contrasts (expression-only model, or a fit that wrote no checkpoint) pass on the
    calibration alone — absence of evidence reading as a pass, and the process exiting 0."""
    return evaluate_gate(contrasts, required=REQUIRED_CONTRASTS, alpha=alpha)


def _load_model(ckpt: Path, expr_only: bool, seed: int, device: str):
    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()
    with seeded_init(seed):
        graph_enc = None if expr_only else TypedGraphEncoder(*build_hetero_graph())
        model = EGIPGModel.from_saved_basis(gene_names, graph_encoder=graph_enc)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device)


def _cases(model, dataset, n_cases: int, seed: int, device: str) -> list:
    """Stratified ``(gene, condition, h_do, row_id)`` cases, reusing the AUDIT's own case selection so a
    fitted head and the feat-012 audit that scores it are drawn from the same strata. Targets absent
    from the PPI graph carry no rationale and are dropped BEFORE selection, so they cannot burn slots."""
    from tcell_pipeline.rationale.rationale_audit import _select_cases, _strata  # same stratification
    covered = [s for s in _strata(dataset, model.graph_encoder.gene_to_idx) if s["covered"]]
    out = []
    with torch.no_grad():
        for case in _select_cases(covered, n_cases, seed):
            i = case["row"]
            batch = build_encoder_batch(dataset.pc.iloc[[i]], dataset.obs.iloc[[i]])
            h_do = model.perturbation_encoder(batch)[0].to(device)
            out.append((case["gene"], case["condition"], h_do, int(dataset.row_index[i])))
    return out


def run(ckpt: Path, *, epochs: int = config.MAX_EPOCHS, n_max: int | None = None,
        n_cases: int = config.N_RATIONALE_AUDIT_CASES, n_controls: int = config.N_MATCHED_CONTROLS,
        n_controls_fit: int = 8,
        seed: int = config.SPLIT_SEED, device: str = "cpu", expr_only: bool = False,
        fit_role: str = "calibration", gate_role: str = "val", ckpt_dir: Path | None = None,
        out_dir: Path = config.STAGE_B_ROOT) -> dict | None:
    """Fit both Stage-B heads on ``fit_role`` and gate them on ``gate_role``.

    ``fit_role`` defaults to the split's own dedicated CALIBRATION partition (2,713 rows) rather than
    train: Stage A was optimised on train, so its train residuals are optimistically small and a sigma
    head fitted there would be miscalibrated for out-of-fold rows — which is the entire point of a
    calibration head. ``--fit-role train`` restores the train-fitted variant. Either way the gate split
    is untouched by the fit."""
    torch.set_num_threads(1)  # many-core box: the tiny per-subgraph GNN ops thrash the default pool
    required = [Path(ckpt), config.BLOCKED_SPLIT_PATH, config.PERTURBATION_CONDITION_PATH,
                config.DE_OBS_PATH, config.DE_VAR_PATH, config.PROGRAM_LOADINGS_PATH, zscore_path()]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(f"[stage-b] required artifacts absent: {missing}")
        return None

    # seed-namespaced by default: both Stage-B heads write fixed filenames, so a shared directory makes
    # a multi-seed sweep overwrite its own earlier artifacts silently
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else stage_b_ckpt_dir(seed)
    t0 = time.perf_counter()
    model = _load_model(Path(ckpt), expr_only, seed, device)
    train_ds, val_ds = PerturbationDataset(fit_role, n_max=n_max), PerturbationDataset(gate_role, n_max=n_max)
    print(f"[stage-b] {len(train_ds)} {fit_role} (fit) / {len(val_ds)} {gate_role} (gate) rows; "
          f"expr_only={expr_only}; device={device}; setup {time.perf_counter() - t0:.1f}s")

    # ONE frozen backbone pass, shared by the fit and its contrasts. The probe measured the duplicate
    # pass at 334s of the calibration phase's 663s — half the cost, for identical constants.
    t0 = time.perf_counter()
    caches = frozen_caches(model, train_ds, val_ds, device=device)
    print(f"[stage-b] frozen backbone pass cached [{time.perf_counter() - t0:.1f}s]")
    t0 = time.perf_counter()
    cal = fit_calibration(model, train_ds, val_ds, caches=caches, max_epochs=epochs, device=device,
                          ckpt_dir=ckpt_dir)
    t_cal = time.perf_counter() - t0
    # fmt(): a `None:.4f` here would crash on exactly the degenerate fit (0 epochs) worth reporting
    print(f"[stage-b] calibration: {cal['epochs_run']} epochs, train_nll={fmt(cal['train_nll'], '.4f')}, "
          f"val_nll={fmt(cal['val_nll'], '.4f')} [{t_cal:.1f}s]")
    t0 = time.perf_counter()
    cal_gate_inputs = calibration_contrasts(model, train_ds, val_ds, caches=caches, device=device, seed=seed)
    print(f"[stage-b] calibration contrasts on {cal_gate_inputs['n_val']} {gate_role} rows; "
          f"decoder lambda mean={cal_gate_inputs['mean_lambda']:.4f} "
          f"[{cal_gate_inputs['min_lambda']:.4f}, {cal_gate_inputs['max_lambda']:.4f}] "
          f"[{time.perf_counter() - t0:.1f}s]")
    contrasts = dict(cal_gate_inputs["contrasts"])

    rat = rat_gate_inputs = None
    if model.graph_encoder is not None:
        t0 = time.perf_counter()
        fit_cases = _cases(model, train_ds, n_cases, seed, device)
        rat = fit_rationale_head(model.graph_encoder, model.decoder, fit_cases, max_epochs=epochs,
                                 n_controls=n_controls_fit, seed=seed, device=device, ckpt_dir=ckpt_dir)
        print(f"[stage-b] rationale: {rat['epochs_run']} epochs over {rat['n_cases']} {fit_role} cases "
              f"[{time.perf_counter() - t0:.1f}s]")
        if rat["best_ckpt"] is None:
            # no epoch improved -> no artifact was written -> there is nothing to gate. Absent contrasts
            # read as UNDECIDABLE downstream, which is the truth here, not a refusal and never a pass.
            print("[stage-b] rationale fit wrote no checkpoint — its contrasts are UNDECIDABLE, not a pass")
        else:
            # GATED on val cases the head never saw: a head scored on its own fit set can only flatter itself
            t0 = time.perf_counter()
            rat_gate_inputs = rationale_contrasts(model.graph_encoder, model.decoder,
                                                  _fitted_head(rat, device),
                                                  _cases(model, val_ds, n_cases, seed, device),
                                                  n_controls=n_controls, seed=seed, device=device)
            contrasts.update(rat_gate_inputs["contrasts"])
            print(f"[stage-b] rationale contrasts on {rat_gate_inputs['n_cases']} {gate_role} cases "
                  f"x {n_controls} matched-random controls; "
                  f"{rat_gate_inputs['n_informative']} informative (deletions moved dz) "
                  f"[{time.perf_counter() - t0:.1f}s]")
    else:
        print("[stage-b] expression-only model: no rationale to fit — its contrasts are UNDECIDABLE, not a pass")

    # one family for the correction (everything tested in this run), one decision per artifact
    overall = overall_gate(contrasts)
    report = {
        "decision": overall["decision"], "alpha": overall["alpha"], "contrasts": overall["contrasts"],
        "calibration": {"fit": _slim(cal), "gate": evaluate_gate(contrasts, required=CALIBRATION_CONTRASTS)["decision"],
                        "constant_sigma": cal_gate_inputs["constant_sigma"], "n_val": cal_gate_inputs["n_val"],
                        # lambda ~ 0 means the frozen decoder ignores its graph — the diagnostic that
                        # explains a rationale whose deletions move nothing
                        "mean_lambda": cal_gate_inputs["mean_lambda"],
                        "lambda_range": [cal_gate_inputs["min_lambda"], cal_gate_inputs["max_lambda"]]},
        "rationale": {"fit": _slim(rat), "n_cases": None if rat is None else rat["n_cases"],
                      "gate": evaluate_gate(contrasts, required=RATIONALE_CONTRASTS)["decision"],
                      "n_informative": None if rat_gate_inputs is None else rat_gate_inputs["n_informative"],
                      "per_case": None if rat_gate_inputs is None else rat_gate_inputs["per_case"]},
        "expr_only": expr_only, "ckpt": str(ckpt), "seed": seed, "ckpt_dir": str(ckpt_dir),
        "fit_role": fit_role, "gate_role": gate_role, "n_fit_rows": len(train_ds), "n_gate_rows": len(val_ds),
    }
    config.ensure_dir(Path(out_dir))
    config.write_text_atomic(json.dumps(report, indent=2, default=float), Path(out_dir) / "stage_b_gate.json")
    config.write_text_atomic(render(overall), Path(out_dir) / "stage_b_gate.md")
    print(render(overall))
    return report


def _fitted_head(rat: dict, device: str):
    """Reload the head from the checkpoint the fit wrote, so the gate scores the artifact that would
    actually be frozen — not an in-memory object that could have drifted from it."""
    from tcell_pipeline.rationale.rationale_head import RationaleHead
    head = RationaleHead()
    head.load_state_dict(torch.load(rat["best_ckpt"], map_location=device, weights_only=True)["head"])
    return head.to(device).eval()


def _slim(res: dict | None) -> dict | None:
    return None if res is None else {k: v for k, v in res.items() if k != "history"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.CHECKPOINTS_ROOT / "stage_a_best.pt"))
    ap.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    ap.add_argument("--n-max", type=int, default=None, help="cap examples per split (quick runs)")
    ap.add_argument("--n-cases", type=int, default=config.N_RATIONALE_AUDIT_CASES)
    ap.add_argument("--n-controls", type=int, default=config.N_MATCHED_CONTROLS,
                    help="matched-random controls for the GATE (the headline comparison)")
    ap.add_argument("--n-controls-fit", type=int, default=8,
                    help="matched-random controls inside the rationale LOSS: a separate, cheaper set "
                         "held fixed per case so the objective stays stationary")
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--expr-only", action="store_true")
    ap.add_argument("--fit-role", default="calibration", help="split role both Stage-B heads are FIT on")
    ap.add_argument("--gate-role", default="val", help="split role the freeze gate evaluates on")
    ap.add_argument("--out-dir", default=str(config.STAGE_B_ROOT))
    ap.add_argument("--ckpt-dir", default=None,
                    help="where the Stage-B heads are written (default: seed-namespaced under CHECKPOINTS_ROOT)")
    a = ap.parse_args(argv)
    report = run(Path(a.ckpt), epochs=a.epochs, n_max=a.n_max, n_cases=a.n_cases, n_controls=a.n_controls,
                 n_controls_fit=a.n_controls_fit,
                 seed=a.seed, device=a.device, expr_only=a.expr_only, fit_role=a.fit_role,
                 gate_role=a.gate_role, ckpt_dir=a.ckpt_dir, out_dir=Path(a.out_dir))
    # nothing ran -> nothing was shown -> undecidable, never 0
    return exit_code(report if report else {})


if __name__ == "__main__":
    sys.exit(main())
