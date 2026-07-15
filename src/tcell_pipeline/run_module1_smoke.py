"""Module 1 real-data smoke — drive the ENTIRE real mart through the PerturbationEncoder.

The Module 1 analogue of ``run_module0.py``: not a pipeline that writes artifacts, but an
end-to-end verification that every real perturbation-condition row encodes cleanly. Each of the
33,983 real rows is built into the Module 3 loader-contract batch dict and pushed through the
encoder, reporting embedding coverage, finiteness (the NaN guard), per-condition behavior, h_do
statistics, and the leakage fence on the mart's real q_post columns.

Run after Module 0 (``run_module0.py``) and the embedding generators (``embeddings_plm`` /
``embeddings_pinnacle``) have populated ``data/intermediate/``:

    python src/tcell_pipeline/run_module1_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put src/ on path for direct runs

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.encoders import PerturbationEncoder, build_encoder_batch  # noqa: E402

DE_OBS_PATH = config.INTERMEDIATE_ROOT / "de_obs.parquet"


def run(batch_size: int = 512) -> bool:
    """Drive the whole mart through the encoder; return True iff every h_do is finite."""
    if not config.PERTURBATION_CONDITION_PATH.exists() or not DE_OBS_PATH.exists():
        print("[module1-smoke] marts absent — run run_module0.py first "
              f"(need {config.PERTURBATION_CONDITION_PATH} and {DE_OBS_PATH})")
        return False

    pc_all = pd.read_parquet(config.PERTURBATION_CONDITION_PATH)
    obs_all = pd.read_parquet(DE_OBS_PATH, columns=["n_guides", "single_guide_estimate"])
    n = len(pc_all)
    print(f"Real mart rows: {n}   (perturbation x condition)")

    # embedding coverage in the REAL data (real vector vs zero fallback)
    uid = pc_all["uniprot_id"].astype("string")
    for name, path, ctx in [("PLM", config.PLM_EMBEDDINGS_PATH, ""),
                            ("PINNACLE", config.PINNACLE_EMBEDDINGS_PATH, f"  ({config.PINNACLE_CONTEXT})")]:
        if path.exists():
            ids = set(pd.read_parquet(path, columns=["uniprot_id"])["uniprot_id"].astype(str))
            print(f"Rows with real {name} vector: {int(uid.isin(ids).sum()):6d} / {n}{ctx}")
        else:
            print(f"Rows with real {name} vector:      0 / {n}  (parquet absent -> zero fallback){ctx}")
    print(f"Rows null uniprot / NaN baseline / NaN n_guides: "
          f"{int(uid.isna().sum())} / {int(pc_all['control_baseline_expr'].isna().sum())} / "
          f"{int(obs_all['n_guides'].isna().sum())}  (NaN guard must hold)")
    print("Condition counts:", pc_all["culture_condition"].value_counts().to_dict())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = PerturbationEncoder().eval().to(device)
    print(f"\nEncoder: {sum(p.numel() for p in enc.parameters())} params on {device}, "
          f"out_dim={config.H_DO_DIM}, target.out_dim={enc.target.out_dim}")

    all_finite = True
    sums = sq = None
    hmin, hmax = np.inf, -np.inf
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            h = enc(build_encoder_batch(pc_all.iloc[i:i + batch_size].reset_index(drop=True),
                                 obs_all.iloc[i:i + batch_size]))
            all_finite &= bool(torch.isfinite(h).all())
            hn = h.cpu().numpy()
            sums = hn.sum(0) if sums is None else sums + hn.sum(0)
            sq = (hn ** 2).sum(0) if sq is None else sq + (hn ** 2).sum(0)
            hmin, hmax = min(hmin, float(hn.min())), max(hmax, float(hn.max()))
    dt = time.time() - t0
    mean, std = sums / n, np.sqrt(np.maximum(sq / n - (sums / n) ** 2, 0))
    print(f"\nDrove all {n} rows in {dt:.2f}s ({n / dt:.0f} rows/s)")
    print(f"ALL h_do FINITE across the entire real dataset: {all_finite}")
    print(f"h_do stats: mean|.|={np.abs(mean).mean():.4f}  mean std={std.mean():.4f} "
          f" min={hmin:.3f} max={hmax:.3f}")

    # leakage fence on REAL q_post columns present in the mart
    qpost = [c for c in config.Q_POST_COLS if c in pc_all.columns]
    print(f"\nLeakage fence: mart has {len(qpost)} real q_post columns, e.g. {qpost[:3]}")
    bad = build_encoder_batch(pc_all.iloc[:4].reset_index(drop=True), obs_all.iloc[:4])
    bad[qpost[0]] = torch.tensor(pc_all[qpost[0]].iloc[:4].to_numpy())
    try:
        enc(bad)
        print("  FAIL: encoder accepted a real q_post column!")
        all_finite = False
    except ValueError as e:
        print(f"  OK: encoder rejected real q_post input -> {e}")

    print(f"\n=== Module 1 real-data smoke {'PASSED' if all_finite else 'FAILED'} ===")
    return all_finite


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
