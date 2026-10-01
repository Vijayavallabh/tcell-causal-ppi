"""B2: WHERE in the gene ranking does the graph help, and where does it hurt?

No training. A3 established that the same contrast has opposite signs at opposite ends of the DE
ranking - corrected-significant NEGATIVE on each perturbation's top-20 genes (GEARS' endpoint) and
corrected-significant POSITIVE over all 10,282 (TxPert's) - and its cumulative k-sweep placed the
crossover between the 250th and 500th gene. The paper states that a crossover exists. It does not say
what happens on either side of it, and a cumulative sweep cannot say: every k contains all smaller k, so
a sign change in the running average only bounds where the underlying per-gene effect turned.

This module decomposes the same quantity into DISJOINT rank bins, which is the object the cumulative
sweep is an integral of. "The contrast flips somewhere between 250 and 500" becomes "the graph hurts on
the genes a perturbation moves most and helps on the ones it moves least", which is a statement about
what a PPI prior is actually good for.

    PYTHONPATH=src python -m tcell_pipeline.screening.rank_deciles \
        --out data/results/b2_deciles/deciles.json

Governed by docs/replication-prereg.md Amendment 8.

WHAT A BIN'S LEVEL DOES NOT MEAN. Genes are binned by |observed response|, and the observation is also
the y-variable of the correlation inside the bin. Two consequences, both stated in Amendment 8.2:
range restriction inside a narrow bin attenuates Pearson, and selecting on a noisy statistic biases the
level of any correlation computed within the selection. So the LEVEL of a bin's correlation is not
comparable across bins. The CONTRAST within a bin is unaffected by both, because the two arms are scored
on the identical gene set of the identical rows - the selection is made once, from the observation, and
neither arm's prediction enters it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tcell_pipeline import config
from tcell_pipeline.evaluation.output_schema import prediction_path, read_predictions
from tcell_pipeline.screening.multiseed import CONTRASTS, FAMILY, apply_family_wise, paired_delta_summary
from tcell_pipeline.screening.rescore_external import _align
from tcell_pipeline.screening.screening import collect_targets_truth
from tcell_pipeline.training.dataset import PerturbationDataset

SEEDS = (0, 1, 2, 3, 4)

# The DISJOINT counterpart of A3's cumulative K_SWEEP: the same boundaries, so a bin here is exactly the
# increment between two consecutive points of that sweep and the two analyses decompose each other.
HEAD_EDGES = (0, 20, 50, 100, 250, 500, 1000, 2500, 5000, 10282)


def _bins(n_genes: int, scheme: str) -> list[tuple[int, int, str]]:
    """(lo, hi, label) rank intervals, half-open, ranked by DESCENDING |observed response|."""
    if scheme == "deciles":
        cuts = np.linspace(0, n_genes, 11).round().astype(int)
        return [(int(a), int(b), f"d{i + 1}") for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])) if b > a]
    # `< n_genes` then always close at n_genes: a panel SMALLER than the last edge would otherwise
    # end at the last edge below it and silently drop every gene past it (caught by the bin test).
    edges = [e for e in HEAD_EDGES if e < n_genes] + [n_genes]
    return [(a, b, f"{a + 1}-{b}") for a, b in zip(edges[:-1], edges[1:]) if b > a]


def score_bins(predictions_root: Path, scheme: str = "deciles", seeds=SEEDS, *,
               n_max: int | None = None) -> tuple[dict, list]:
    """``{bin: {arm: {seed: mean rowwise Pearson within that rank bin}}}``.

    The ranking is recomputed from the ALIGNED truth for each arm and seed rather than once globally,
    because ``_align`` may drop rows; ranking on a row set that does not match the scored row set would
    silently score one arm's genes against another's ranking."""
    from tcell_pipeline.evaluation.metrics import _rowwise_pearson

    val = PerturbationDataset("val", n_max=n_max)
    truth = collect_targets_truth(val)
    n_genes = truth["delta_x"].shape[1]
    bins = _bins(n_genes, scheme)
    print(f"[b2] val fold: {len(truth['row_index'])} rows x {n_genes} genes; {len(bins)} {scheme} bins")

    out: dict = {}
    for arm in FAMILY:
        for seed in seeds:
            path = prediction_path(arm, "val", seed, predictions_root)
            if not Path(path).exists():
                print(f"[b2] {arm} s{seed}: MISSING {path}")
                continue
            pred = read_predictions(path)
            dx_hat, dx_true = _align(pred, truth["row_index"], truth["delta_x"])
            idx_full = np.argsort(-np.abs(dx_true), axis=1)     # rank from the OBSERVATION, not the arm
            for lo, hi, label in bins:
                idx = idx_full[:, lo:hi]
                r = _rowwise_pearson(np.take_along_axis(dx_hat, idx, 1),
                                     np.take_along_axis(dx_true, idx, 1))
                out.setdefault(label, {}).setdefault(arm, {})[seed] = float(np.nanmean(r))
    return out, bins


def contrast_bins(scores: dict, bins: list, seeds=SEEDS, alpha: float = 0.05) -> dict:
    """Every pre-registered contrast in every bin, corrected as ONE family over all cells.

    Amendment 8.3. Correcting within a bin and then reporting whichever bin was kind is the
    look-elsewhere effect, and this analysis exists precisely to look in several places at once.
    ``apply_family_wise`` sizes the family from what it is handed, so handing it every cell is the
    implementation."""
    cells = {f"{label}/{key}": paired_delta_summary(scores.get(label, {}).get(better, {}),
                                                    scores.get(label, {}).get(worse, {}),
                                                    alpha=alpha, seeds=seeds)
             for _, _, label in bins for key, better, worse in CONTRASTS}
    for (_, _, label) in bins:
        for key, better, worse in CONTRASTS:
            c = cells[f"{label}/{key}"]
            c["bin"], c["contrast"], c["better"], c["worse"] = label, key, better, worse
    apply_family_wise(cells, alpha)
    return cells


def _crossover(cells: dict, bins: list, key: str) -> dict:
    """The first bin (walking from the most-moved genes down) where the contrast changes sign, and
    whether the bins on either side of it clear correction. This is what the cumulative sweep could
    only bound."""
    seq = [(label, cells[f"{label}/{key}"]) for _, _, label in bins
           if cells[f"{label}/{key}"]["mean"] is not None]
    if not seq:
        return {}
    signs = [(label, float(np.sign(c["mean"])), bool(c["survives_family_wise"])) for label, c in seq]
    flip = next((i for i in range(1, len(signs)) if signs[i][1] and signs[i][1] != signs[i - 1][1]), None)
    return {"contrast": key,
            "sequence": [{"bin": l, "mean": c["mean"], "survives": bool(c["survives_family_wise"])}
                         for l, c in seq],
            "first_sign_change": None if flip is None else
            {"from_bin": signs[flip - 1][0], "to_bin": signs[flip][0],
             "from_sign": signs[flip - 1][1], "to_sign": signs[flip][1],
             "both_survive": signs[flip - 1][2] and signs[flip][2]}}


def run(predictions_root: Path = config.PREDICTIONS_ROOT, out: Path | None = None, *,
        n_max: int | None = None, alpha: float = 0.05) -> dict:
    report = {"predictions_root": str(predictions_root), "seeds": list(SEEDS), "alpha": alpha,
              "schemes": {}}
    for scheme in ("deciles", "head"):
        scores, bins = score_bins(predictions_root, scheme, n_max=n_max)
        if not scores:
            print(f"[b2] no predictions found under {predictions_root}")
            continue
        cells = contrast_bins(scores, bins, alpha=alpha)
        m = next((c.get("family_size") for c in cells.values() if c.get("family_size")), 0)
        print(f"\n{'=' * 108}\nB2 — the pre-registered contrasts within DISJOINT {scheme} of "
              f"|observed response|, ranked most-moved first")
        print(f"     family = all {m} cells of this binning (Amendment 8.3). A bin's LEVEL is not "
              f"comparable across bins; its CONTRAST is.\n{'=' * 108}")
        print(f"{'bin':>10} {'contrast':>17} {'n':>2} {'mean':>11} {'95% CI':>24} {'bonf':>8} "
              f"{'holm':>8} {'SURVIVES':>9}")
        for _, _, label in bins:
            for key, _, _ in CONTRASTS:
                c = cells[f"{label}/{key}"]
                ci = "     -" if c["ci_low"] is None else f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
                mean = "    -" if c["mean"] is None else f"{c['mean']:+.4f}"
                bonf = "  -" if c["p_bonferroni"] is None else f"{c['p_bonferroni']:.4f}"
                holm = "  -" if c["p_holm"] is None else f"{c['p_holm']:.4f}"
                sv = "  -" if c["survives_family_wise"] is None else ("yes" if c["survives_family_wise"] else "no")
                print(f"{label:>10} {key:>17} {c['n']:>2} {mean:>11} {ci:>24} {bonf:>8} {holm:>8} {sv:>9}")
        crossings = {key: _crossover(cells, bins, key) for key, _, _ in CONTRASTS}
        for key, cr in crossings.items():
            fs = cr.get("first_sign_change")
            if fs:
                print(f"[b2] {key}: first sign change between bin {fs['from_bin']} and {fs['to_bin']}"
                      + ("  (BOTH clear correction)" if fs["both_survive"] else "  (not both corrected-significant)"))
            elif cr:
                print(f"[b2] {key}: no sign change across the ranking")
        report["schemes"][scheme] = {"per_seed": scores, "cells": cells, "family_size": m,
                                     "bins": [list(b) for b in bins], "crossings": crossings}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, default=float))
        print(f"\n[b2] wrote {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-root", default=str(config.PREDICTIONS_ROOT))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-max", type=int, default=None)
    a = ap.parse_args()
    run(Path(a.predictions_root), Path(a.out) if a.out else None, n_max=a.n_max)
