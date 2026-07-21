"""feat-005 method x K basis study driver — scores candidate bases, writes NOTHING to data/intermediate.

    PYTHONPATH=src uv run python -m tcell_pipeline.programs.run_basis_study --only sparse_pca:128
    PYTHONPATH=src uv run python -m tcell_pipeline.programs.run_basis_study            # full sweep
    PYTHONPATH=src uv run python -m tcell_pipeline.programs.run_basis_study --table    # assemble only

SAFETY: the frozen production basis (data/intermediate/gene_program_loadings.parquet) is the
coordinate system for every result in this project. This driver fits every candidate IN MEMORY and
writes only under data/results/basis_study/. It never calls save_program_basis/save_program_response
and never touches config.PROGRAM_LOADINGS_PATH — test_basis_study.py pins that.

Cells are written one JSON at a time and skipped if present, so a multi-hour sweep on a shared box
survives an interrupt without losing completed work.
"""
from __future__ import annotations

import os
import sys as _sys

# Whether this assignment can still bind is decided BEFORE it runs: libgomp and OpenBLAS read their
# thread counts in load-time initialisers, so once numpy is in sys.modules the pools are already sized
# and setdefault below is decoration. Running as `-m tcell_pipeline.programs.run_basis_study` imports
# the parent package first, and that pulls in numpy — so on this path the cap NEVER took effect while
# the driver printed "OMP=4" as though it had. That is the recorded shared-box regression: ~830 threads,
# load 600, a 4-minute fit taking 87. Record the truth here and report it; do not fix it by capping
# threads inside the package __init__, which would pin every trainer and screening run too.
_NUMPY_ALREADY_LOADED = "numpy" in _sys.modules
_THREAD_CAP_PRESET = "OMP_NUM_THREADS" in os.environ

os.environ.setdefault("OMP_NUM_THREADS", "4")  # shared 64-core box
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])


def thread_cap_effective() -> bool:
    """Did the BLAS/OpenMP cap actually bind for this process?

    True only if it was already in the environment when Python started (the launch command set it), or
    numpy had not yet been imported when this module was loaded. Anything else means the printed
    ``OMP=`` value is a wish, not a fact."""
    return bool(_THREAD_CAP_PRESET or not _NUMPY_ALREADY_LOADED)


def warn_if_thread_cap_ineffective() -> None:
    if not thread_cap_effective():
        print(f"[basis-study] WARNING: OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']} was set AFTER "
              f"numpy loaded, so the BLAS thread pools are already sized to all "
              f"{os.cpu_count()} cores and this cap has NO EFFECT. Re-launch with it in the command "
              f"(OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 python -m ...) — the "
              f"per-cell time budget below assumes the cap holds.", flush=True)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path for direct runs

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.programs.basis_study import (  # noqa: E402
    fit_vae_basis,
    matched_stability,
    multiplicity_adjust,
    paired_recon_contrast,
    recon_metrics,
    sparsity_metrics,
)
from tcell_pipeline.programs.program_basis import (  # noqa: E402
    fit_program_basis,
    load_zscore_rows,
    train_row_indices,
)

# Local, not a config constant: config.py is shared with the other live sessions and this study is
# self-contained. Everything this driver writes lives under here and nowhere else.
OUT_DIR = config.DATA_DIR / "results" / "basis_study"
METHODS = ("sparse_pca", "nmf", "fastica", "svd")
REF_METHOD, REF_K = "sparse_pca", 128  # the FROZEN production cell — reference for every contrast
KS = (64, 128, 256, 512)

# ---- pre-stated stability protocol (fixed before any cell was run; see the notes doc) ----
N_RESAMPLE = 3          # -> 3 pairwise comparisons per cell
RESAMPLE_FRAC = 0.8     # of the TRAIN rows, sampled WITHOUT replacement
RESAMPLE_SEED = 20260720
FIT_SEED = config.SPLIT_SEED  # held FIXED across resamples: isolates data-sensitivity from init noise
MAX_ITER = 100
VAE_EPOCHS = 20


def load_train_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Raw z-scores for TRAIN rows only, and their gene-wise mean. Fold-locality is enforced here."""
    split = pd.read_csv(config.BLOCKED_SPLIT_PATH)
    pc = pd.read_parquet(config.PERTURBATION_CONDITION_PATH, columns=["row_index", "hgnc_symbol"])
    rows = train_row_indices(split, pc)
    train_genes = set(split.loc[split["role"] == "train", "hgnc_symbol"])
    leaked = set(pc.loc[pc["row_index"].isin(rows), "hgnc_symbol"]) - train_genes
    if leaked:  # raise, not assert: must survive `python -O`
        raise RuntimeError(f"fold leak: {len(leaked)} non-train genes, e.g. {sorted(leaked)[:5]}")
    Z = load_zscore_rows(rows)
    return Z, Z.mean(0, dtype=np.float64).astype(np.float32)


def _fit(Z: np.ndarray, method: str, K: int) -> tuple[np.ndarray, np.ndarray, dict, float]:
    info: dict = {}
    t0 = time.time()
    if method == "vae":
        B, A, vinfo = fit_vae_basis(Z - Z.mean(0, dtype=np.float64).astype(np.float32),
                                    K=K, seed=FIT_SEED, epochs=VAE_EPOCHS)
        # A fixed epoch budget is not a convergence criterion — do not claim one.
        info = {"n_iter": vinfo["epochs_run"], "max_iter": VAE_EPOCHS, "converged": None,
                "final_loss": vinfo["final_loss"]}
    else:
        B, A = fit_program_basis(Z, method=method, K=K, seed=FIT_SEED,
                                 max_iter=cell_max_iter(method, K), info=info)
    return B, A, info, time.time() - t0


def _native_target(Z: np.ndarray, mu: np.ndarray, method: str) -> tuple[np.ndarray, str]:
    """The target each method actually models. NMF sees only the positive part, so scoring it on the
    signed centred target measures a different question — report both rather than rank it blindly."""
    if method == "nmf":
        return np.maximum(Z, 0.0), "positive_part"
    if method == "svd":
        return Z, "raw_uncentred"  # TruncatedSVD does not centre
    return Z - mu, "centred"       # sparse_pca / fastica / vae centre internally


def _heldout_projection(Zc_ho: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Least-squares scores for unseen rows onto span(B) — method-agnostic, needs no fitted encoder."""
    G = np.asarray(B, dtype=np.float64)
    return (Zc_ho.astype(np.float64) @ G @ np.linalg.pinv(G.T @ G)).astype(np.float32)


def stability_and_heldout(Z: np.ndarray, method: str, K: int) -> tuple[dict, dict]:
    """Refit on N_RESAMPLE subsamples of the train rows; match components by Hungarian on |cosine|.

    Each resample also yields an honest out-of-sample reconstruction: fit on the 80%, score the
    held-out 20% (still TRAIN rows — the fold fence is never widened), centred by the FIT subset's
    mean so the held-out rows contribute nothing to the transform.
    """
    n = Z.shape[0]
    n_fit = int(round(RESAMPLE_FRAC * n))
    bases, ho_mae, ho_base, dead = [], [], [], 0
    nat_mae, nat_base, nat_name = [], [], "centred"
    for r in range(N_RESAMPLE):
        idx = np.random.default_rng(RESAMPLE_SEED + r).permutation(n)
        fit_idx, ho_idx = np.sort(idx[:n_fit]), np.sort(idx[n_fit:])
        Zf = Z[fit_idx]
        B, _, info, _ = _fit(Zf, method, K)
        bases.append(B)
        dead += sparsity_metrics(B)["n_dead"]
        mu_f = Zf.mean(0, dtype=np.float64).astype(np.float32)
        Zc_ho = Z[ho_idx] - mu_f
        m = recon_metrics(Zc_ho, _heldout_projection(Zc_ho, B), B)
        ho_mae.append(m["recon_mae"])
        ho_base.append(m["zero_baseline_mae"])
        # ...and on the target the method ACTUALLY models. The in-sample path has carried
        # `recon_native` since the start for exactly this reason, but held-out did not: NMF sees only
        # the positive part, so scoring it on the signed centred target measures a different question
        # — and the study's headline is a method x K ranking read off the held-out column.
        tgt_ho, nat_name = _native_target(Z[ho_idx], mu_f, method)
        m_nat = (m if nat_name == "centred"
                 else recon_metrics(tgt_ho, _heldout_projection(tgt_ho, B), B))
        nat_mae.append(m_nat["recon_mae"])
        nat_base.append(m_nat["zero_baseline_mae"])

    pairs = [matched_stability(bases[i], bases[j]) for i, j in combinations(range(N_RESAMPLE), 2)]
    vals = [p["mean_abs_cosine"] for p in pairs if p["mean_abs_cosine"] is not None]
    stab = {
        # No live pair anywhere => UNDECIDABLE. None is not 0.0: a degenerate fit is not "unstable".
        "mean_abs_cosine": float(np.mean(vals)) if vals else None,
        "pair_values": vals,
        "n_undecidable_pairs": len(pairs) - len(vals),
        "n_dead_across_resamples": int(dead),
    }
    base = float(np.mean(ho_base))
    nbase = float(np.mean(nat_base))
    ho = {"recon_mae": float(np.mean(ho_mae)), "zero_baseline_mae": base,
          "explained_frac": None if base == 0 else 1.0 - float(np.mean(ho_mae)) / base,
          "per_resample_mae": ho_mae,
          # Cross-method ranking is only admissible on a COMMON target; `native` is the one this method
          # was fitted for. Where they differ (nmf, svd) the two must not be compared with each other.
          "native": {"target": nat_name, "recon_mae": float(np.mean(nat_mae)),
                     "zero_baseline_mae": nbase,
                     "explained_frac": None if nbase == 0 else 1.0 - float(np.mean(nat_mae)) / nbase},
          "comparable_across_methods": nat_name == "centred"}
    return stab, ho


def run_cell(Z: np.ndarray, mu: np.ndarray, method: str, K: int, with_stability: bool = True) -> dict:
    B, A, info, secs = _fit(Z, method, K)
    Zc = Z - mu
    rec = recon_metrics(Zc, A, B)
    tgt, tgt_name = _native_target(Z, mu, method)
    nat = recon_metrics(tgt, A, B) if tgt_name != "centred" else rec

    row_mae = rec.pop("row_mae")
    nat.pop("row_mae", None)
    np.save(OUT_DIR / "row_mae" / f"{method}_K{K}.npy", row_mae.astype(np.float32))

    cell = {
        "method": method, "K": K, "fit_seconds": secs,
        "n_train_rows": int(Z.shape[0]), "n_genes": int(Z.shape[1]),
        "convergence": info, "recon": rec,
        "recon_native": {"target": tgt_name, **{k: v for k, v in nat.items()}},
        "sparsity": sparsity_metrics(B),
        "has_stability": with_stability,  # completeness marker: see is_cached()
    }
    if with_stability:
        cell["stability"], cell["heldout"] = stability_and_heldout(Z, method, K)
    return cell


def build_contrasts(row_mae_dir: Path, zero_baseline: float | None = None) -> pd.DataFrame:
    """Paired candidate-vs-frozen reconstruction contrasts over the train rows, with BOTH corrections.

    The reference is the frozen production cell (sparse_pca K=128). It is excluded from its own
    family: a self-contrast has identically-zero differences (undecidable) and would still consume a
    Bonferroni/Holm slot, weakening every real comparison in exchange for a test that says nothing.
    """
    ref_stem = f"{REF_METHOD}_K{REF_K}"
    ref = np.load(row_mae_dir / f"{ref_stem}.npy").astype(np.float64)
    rows = []
    for p in sorted(row_mae_dir.glob("*.npy")):
        if p.stem == ref_stem:
            continue
        method, k = p.stem.rsplit("_K", 1)
        c = paired_recon_contrast(ref, np.load(p).astype(np.float64))
        # Effect size, not just significance: at n=21k the p-value underflows for differences of
        # ~0.1% of baseline, so the conclusion rests on this column.
        delta = None if zero_baseline in (None, 0) else -c["mean_diff"] / zero_baseline
        rows.append({"method": method, "K": int(k), **c, "delta_explained_frac": delta})
    adj = multiplicity_adjust([r["p_raw"] for r in rows])
    for r, b, h in zip(rows, adj["bonferroni"], adj["holm"]):
        r["p_bonferroni"], r["p_holm"], r["family_size"] = b, h, adj["family_size"]
    return pd.DataFrame(rows).sort_values(["method", "K"])


# Single-fit seconds at K=64 with the measured K-scaling exponent: cost ~ base * (K/64)**exp.
# Refitted 2026-07-21 from the completed sweep (all from the same run, so like-for-like; the §7
# probe timings ran under load ~57 and are 2-3x slower for fastica). Used ONLY to order the sweep
# and decide whether a cell can afford its resamples — never a result.
#   svd        8.7 -> 21.4 over K=64..512  -> 0.43  sub-linear (fixed data passes dominate)
#   sparse_pca 184 -> 1212 over K=64..256  -> 1.36  SUPER-linear and still accelerating: K=512
#                                                   exceeded 5400 s for a single fit, i.e. even 1.36
#                                                   UNDERSTATES it. Treat K>=512 as unbounded here.
#   fastica    226 -> 180 over K=64..512   -> 0.00  flat (~90% is a K-INDEPENDENT whitening SVD);
#                                                   measured -0.11, clamped to 0 (cost cannot fall)
#   nmf         86 -> 887 over K=64..512   -> 1.12  ~linear, O(N*G*K) per iteration
#   vae        64.9 at K=128 only          -> 1.0   assumed; single point
_COST_K64 = {"svd": 8.7, "nmf": 86.4, "sparse_pca": 183.7, "fastica": 225.7, "vae": 32.5}
_COST_EXP = {"svd": 0.43, "nmf": 1.12, "sparse_pca": 1.36, "fastica": 0.0, "vae": 1.0}


def est_fit_seconds(method: str, K: int) -> float:
    return _COST_K64.get(method, 1e9) * (K / 64.0) ** _COST_EXP.get(method, 1.0)
_SCORING_OVERHEAD_S = 60.0  # measured: full svd:64 cell was 57.4s on an 8.7s fit
_CELL_CAP_S = 5400.0        # 90 min per cell (approved 2026-07-20)
_BUDGET_S = 10 * 3600.0     # 10 h total (approved 2026-07-20)


CONVERGENCE_BUDGET_ITER = 500
_NEEDS_BUDGET = ("nmf", "fastica")  # the two that were measured hitting their cap


def cell_max_iter(method: str, K: int) -> int:
    """The iterative methods get a real convergence budget only at the frozen K (decision 2026-07-20).

    At max_iter=100 both nmf (100/100 at K=64) and fastica (100/100 at K=128) hit their cap, so
    their cells are UNDECIDABLE rather than compared; at 500 everywhere nmf alone costs ~6 h.
    Spending it at K=128 buys one decidable point per method at the K that matches the frozen basis.

    Consequence to report with the table: the **K=128 column is the fair, fully-decidable
    comparison** (sparse_pca converges in 2 iters, svd has no cap, nmf and fastica get 500). Other K
    values are cost-limited and may come back capped — those cells are not ranked.
    """
    return CONVERGENCE_BUDGET_ITER if (method in _NEEDS_BUDGET and K == REF_K) else MAX_ITER


def sweep_order(cells: list) -> list:
    """Cheapest first. Under a total budget the expensive tail is what gets dropped, so one 10-hour
    fastica cell must not starve fifteen cells that cost minutes."""
    return sorted(cells, key=lambda c: est_fit_seconds(*c))


def wants_stability(method: str, K: int, cap_seconds: float) -> bool:
    """Can this cell afford its 3 resample fits inside the per-cell cap?

    If not, running the full path just burns the whole cap and writes nothing. One fit still fits,
    so the cell degrades to recon+sparsity with stability explicitly not computed. Uses the measured
    cost model, which is approximate — hence a cell can still time out, which is handled separately.
    """
    est_fit = est_fit_seconds(method, K)
    return (1 + N_RESAMPLE * RESAMPLE_FRAC) * est_fit + _SCORING_OVERHEAD_S < cap_seconds


def write_not_measured(path: Path, method: str, K: int, reason: str, seconds: float,
                       row_mae_dir: Path | None = None) -> None:
    """Record a cell the bounded sweep could not complete, shaped like a real cell with None metrics.

    A missing row reads as 'not run'; a blank metric reads as 'ran and found nothing'. Neither is
    true of a capped cell, so the reason is carried explicitly all the way into the table.

    Two things this must NOT do, both found by review after the sweep had already run:

    * **Never overwrite a measured cell.** A completed cell can be re-queued (``is_cached`` returns
      False when only stability is missing) and then time out on the retry — and this function would
      replace a >90-minute fit with an all-None stub, unrecoverably. A stub is strictly less
      informative than the measurement it would destroy, so an existing measured cell wins.
    * **Never leave its ``row_mae`` array behind.** ``run_cell`` saves that array BEFORE the expensive
      stability step, so a cell that dies in stability leaves a live ``.npy`` that ``build_contrasts``
      globs — publishing a mean_diff, CI and Bonferroni/Holm p-values, and consuming a slot in the
      multiplicity family, for a cell the neighbouring table reports as never measured.
    """
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
        if existing.get("recon", {}).get("recon_mae") is not None:
            print(f"[basis-study] {method}:{K} {reason}, but a MEASURED cell is already on disk — "
                  f"keeping it (a stub would destroy {existing.get('fit_seconds', 0.0):.0f}s of fit)",
                  flush=True)
            return
    if row_mae_dir is not None:
        orphan = Path(row_mae_dir) / f"{method}_K{K}.npy"
        orphan.unlink(missing_ok=True)   # else build_contrasts publishes a p-value for a stub cell
    path.write_text(json.dumps({
        "method": method, "K": K, "not_measured": reason, "fit_seconds": seconds,
        "convergence": {"n_iter": None, "max_iter": cell_max_iter(method, K), "converged": None},
        "recon": {"recon_mae": None, "zero_baseline_mae": None, "explained_frac": None},
        "recon_native": {"target": None, "recon_mae": None},
        "sparsity": {"zero_frac": None, "n_dead": None},
        "has_stability": False,
    }, indent=2))


def merge_stability(cell: dict, stab: dict, heldout: dict) -> dict:
    """Attach stability/held-out to a cell whose fit+score is already on disk.

    Backfill only: the fit is deterministic (fixed seed), so re-running it would reproduce identical
    numbers at the cost of another 96 min for K=512. Refuses a not_measured stub — that has no fit
    at all, and attaching stability would fabricate a half-real cell reading as measured.
    """
    if cell.get("not_measured"):
        raise ValueError(f"cannot backfill stability into a not_measured cell: {cell['not_measured']}")
    return {**cell, "stability": stab, "heldout": heldout, "has_stability": True}


def is_cached(path: Path, want_stability: bool) -> bool:
    """True only if the cell on disk is at least as complete as the one being asked for.

    A --no-stability timing probe writes the same path as a full cell. Skipping on mere existence
    would let a probe masquerade as a finished cell and publish a blank stability column.
    """
    if not path.exists():
        return False
    cell = json.loads(path.read_text())
    if cell.get("not_measured"):
        return False  # raising the cap on a later run must RETRY the cell, not skip it as done
    return bool(cell.get("has_stability")) or not want_stability


def run_sweep(cell_cap: float = _CELL_CAP_S, budget: float = _BUDGET_S) -> None:
    """Bounded sweep: cheapest cell first, hard per-cell cap, hard total budget (approved plan).

    Each cell runs as a SUBPROCESS so the cap is enforced by the OS. An in-process alarm would sit
    unserviced inside a long BLAS call, which is precisely where a runaway cell spends its time.
    Anything not completed is written as an explicit not_measured cell, never left absent.
    """
    import subprocess

    t0 = time.time()
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    plan = sweep_order([(m, k) for m in METHODS for k in KS] + [("vae", REF_K)])
    print(f"[sweep] {len(plan)} cells, cheapest first, cap {cell_cap / 60:.0f} min/cell, "
          f"budget {budget / 3600:.1f} h\n[sweep] order: {[f'{m}:{k}' for m, k in plan]}")

    for method, K in plan:
        out = OUT_DIR / "cells" / f"{method}_K{K}.json"
        if is_cached(out, want_stability=True):
            print(f"[sweep] skip {method}:{K} (complete)")
            continue
        left = budget - (time.time() - t0)
        if left <= 60:
            print(f"[sweep] BUDGET EXHAUSTED before {method}:{K}")
            write_not_measured(out, method, K, "budget_exhausted", 0.0,
                               row_mae_dir=OUT_DIR / "row_mae")
            continue
        cap = min(cell_cap, left)
        stab = wants_stability(method, K, cap)
        cmd = [sys.executable, "-m", "tcell_pipeline.programs.run_basis_study", "--only", f"{method}:{K}"]
        if not stab:
            cmd.append("--no-stability")
            print(f"[sweep] {method}:{K} cannot afford resamples in {cap / 60:.0f} min — "
                  f"running fit+score only, stability NOT computed")
        c0 = time.time()
        try:
            subprocess.run(cmd, env=env, timeout=cap, check=False,
                           cwd=str(Path(__file__).resolve().parents[2]))
        except subprocess.TimeoutExpired:
            write_not_measured(out, method, K, "timeout", cap,
                               row_mae_dir=OUT_DIR / "row_mae")
            print(f"[sweep] {method}:{K} TIMEOUT at {cap / 60:.0f} min -> not_measured")
            continue
        if not out.exists():  # died for some other reason — still must not vanish from the table
            write_not_measured(out, method, K, "failed", time.time() - c0,
                               row_mae_dir=OUT_DIR / "row_mae")
            print(f"[sweep] {method}:{K} FAILED -> not_measured")
        print(f"[sweep] {method}:{K} done in {(time.time() - c0) / 60:.1f} min "
              f"({(budget - (time.time() - t0)) / 3600:.1f} h budget left)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="single cell as method:K, e.g. sparse_pca:128 (for timing)")
    ap.add_argument("--no-stability", action="store_true", help="fit+score only; skip the resamples")
    ap.add_argument("--table", action="store_true", help="assemble the table from existing cells only")
    ap.add_argument("--force", action="store_true", help="recompute cells that already have JSON")
    ap.add_argument("--sweep", action="store_true", help="bounded full sweep (per-cell cap + total budget)")
    ap.add_argument("--stability-only", help="backfill stability+heldout for an existing cell, as method:K")
    ap.add_argument("--cell-cap-min", type=float, default=_CELL_CAP_S / 60)
    ap.add_argument("--budget-hours", type=float, default=_BUDGET_S / 3600)
    a = ap.parse_args(argv)

    (OUT_DIR / "cells").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "row_mae").mkdir(parents=True, exist_ok=True)

    if a.stability_only:
        m, k = a.stability_only.split(":"); k = int(k)
        out = OUT_DIR / "cells" / f"{m}_K{k}.json"
        Z, _ = load_train_matrix()
        print(f"[backfill] {m}:{k} — reusing the fit on disk, computing {N_RESAMPLE} resamples")
        t0 = time.time()
        stab, ho = stability_and_heldout(Z, m, k)
        out.write_text(json.dumps(merge_stability(json.loads(out.read_text()), stab, ho), indent=2))
        print(f"[backfill] {m}:{k} done in {(time.time() - t0) / 60:.1f} min  "
              f"stability={stab['mean_abs_cosine']}  heldout_mae={ho['recon_mae']:.6f}")
    elif a.sweep:
        run_sweep(a.cell_cap_min * 60, a.budget_hours * 3600)
    elif not a.table:
        cells = [tuple(a.only.split(":"))] if a.only else \
            [(m, k) for m in METHODS for k in KS] + [("vae", 128)]
        cells = [(m, int(k)) for m, k in cells]
        Z, mu = load_train_matrix()
        eff = "" if thread_cap_effective() else " (NOT IN EFFECT — see warning above)"
        warn_if_thread_cap_ineffective()
        print(f"[basis-study] {Z.shape[0]} train rows x {Z.shape[1]} genes, "
              f"OMP={os.environ['OMP_NUM_THREADS']}{eff}")
        for method, K in cells:
            out = OUT_DIR / "cells" / f"{method}_K{K}.json"
            if is_cached(out, want_stability=not a.no_stability) and not a.force:
                print(f"[basis-study] skip {method} K={K} (cached)")
                continue
            t0 = time.time()
            cell = run_cell(Z, mu, method, K, with_stability=not a.no_stability)
            out.write_text(json.dumps(cell, indent=2))
            print(f"[basis-study] {method:10s} K={K:<4d} {time.time() - t0:8.1f}s total  "
                  f"recon={cell['recon']['recon_mae']:.4f}  zero={cell['sparsity']['zero_frac']:.3%}  "
                  f"conv={cell['convergence'].get('converged')}")

    rows = [json.loads(p.read_text()) for p in sorted((OUT_DIR / "cells").glob("*.json"))]
    if rows:
        tbl = pd.DataFrame([{
            "method": c["method"], "K": c["K"], "fit_seconds": round(c["fit_seconds"], 1),
            "not_measured": c.get("not_measured"),
            "converged": c["convergence"].get("converged"), "n_iter": c["convergence"].get("n_iter"),
            "recon_mae": c["recon"]["recon_mae"], "zero_baseline_mae": c["recon"]["zero_baseline_mae"],
            "explained_frac": c["recon"]["explained_frac"],
            "native_target": c["recon_native"]["target"], "native_recon_mae": c["recon_native"]["recon_mae"],
            "heldout_recon_mae": c.get("heldout", {}).get("recon_mae"),
            "heldout_explained_frac": c.get("heldout", {}).get("explained_frac"),
            "zero_frac": c["sparsity"]["zero_frac"], "n_dead": c["sparsity"]["n_dead"],
            # Without this column a blank stability reads identically for "never resampled" and
            # "resampled but every pair degenerate". Absence of evidence is not a result.
            "stability_computed": bool(c.get("has_stability")),
            "stability_matched_abs_cosine": c.get("stability", {}).get("mean_abs_cosine"),
            "stability_undecidable_pairs": c.get("stability", {}).get("n_undecidable_pairs"),
        } for c in rows]).sort_values(["method", "K"])
        tbl.to_csv(OUT_DIR / "method_k_table.csv", index=False)
        print(f"\n{tbl.to_string(index=False)}\n[basis-study] wrote {OUT_DIR / 'method_k_table.csv'}")
        if (OUT_DIR / "row_mae" / f"{REF_METHOD}_K{REF_K}.npy").exists():
            con = build_contrasts(OUT_DIR / "row_mae",
                                  zero_baseline=next((c["recon"]["zero_baseline_mae"] for c in rows
                                                      if c["method"] == REF_METHOD and c["K"] == REF_K), None))
            con.to_csv(OUT_DIR / "contrasts_vs_frozen.csv", index=False)
            print(f"\n{con.to_string(index=False)}\n[basis-study] wrote contrasts_vs_frozen.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
