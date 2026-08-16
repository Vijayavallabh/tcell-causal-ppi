"""A2(a): build response matrices with a KNOWN graph-dependent signal injected, at a ladder of sizes.

A2(b) says what this pipeline should be able to detect, from the variance it measured. This is the
empirical half: put a graph signal of known size into the real responses and see which rung the pipeline
actually recovers. The smallest delta that clears both corrections is the MEASURED sensitivity, and a
high floor is a result too - a limitation with a number instead of a hedge.

WHAT IS INJECTED. For each perturbation target we compute the mean response of its PPI neighbours over
TRAIN-FOLD ROWS ONLY, scale that matrix so its spread matches the TRAIN response's, and add ``delta``
times it to every train and validation row of that target. A model that can read the graph can predict a
held-out target from its neighbours; a model that cannot, cannot. That is the whole design.

LEAKAGE IS THE ENTIRE RISK, and it has exactly one shape: if a validation target's injected component
depended on any validation response, the graph arm would be detecting leakage rather than structure and
the ladder would report a sensitivity the pipeline does not have. Two guards, both structural:

  1. the per-target mean is computed over TRAIN rows only, so no validation response enters any
     injected value anywhere in the matrix;
  2. a target is never its own neighbour, so even within the train fold a row's injection is not a copy
     of itself.

``test_inject_signal.py`` asserts both by perturbing a validation response and requiring the injection to
be bit-identical, and that test was watched to FAIL against a deliberately leaky variant before it was
trusted. A test that has never failed is not evidence; this project has shipped two of those.

THE SCALING CONSTANT IS PART OF THE GUARANTEE. It was first computed over train AND validation rows,
which is a leak: tampering with one validation response moved the constant and rescaled every injected
value by a factor of 46. The leakage test caught it before any lane ran. "Train-only" has to mean every
number that reaches the output, constants included, not just the per-target means.

RAIL 1, THE SEALED SPLIT. Rows whose target is neither train nor validation - the challenge and
calibration roles - get an injection of exactly ZERO and are copied through bit-identically. No sealed
response is read into any statistic.

    PYTHONPATH=src python -m tcell_pipeline.screening.inject_signal --delta 0.10 \
        --out data/intermediate/inject/d010
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tcell_pipeline import config

_PP_RELATIONS = ("physical_ppi", "co_complex", "functional_assoc")
LADDER = (0.02, 0.05, 0.10, 0.20, 0.40)
# Files a screening lane reads out of INTERMEDIATE_ROOT. Everything except the response layer is
# symlinked from the reference root, so an injected root cannot silently differ in features, split,
# basis or metadata - only in the thing being injected.
LINKED = ("de_obs.parquet", "de_var.parquet", "perturbation_condition.parquet",
          "control_baseline_expr.parquet", "control_donor_profiles.parquet",
          "gene_program_loadings.parquet", "program_response.parquet", "id_mapping.parquet",
          "plm_embeddings.parquet", "pinnacle_embeddings.parquet")
COPIED_LAYERS = ("log_fc.npz", "baseMean.npy", "lfcSE.npy",
                 "neglog10_p_value.npy", "neglog10_adj_p_value.npy")


def role_map(split_path: Path = config.BLOCKED_SPLIT_PATH) -> dict[str, str]:
    s = pd.read_csv(split_path)
    return dict(zip(s["hgnc_symbol"].astype(str), s["role"].astype(str)))


def neighbour_operator(targets: list[str], gene_to_idx: dict, graph, *, hops: int = 1,
                       weighted: bool = True) -> sp.csr_matrix:
    """Row-normalised PPI adjacency restricted to the target genes, with the diagonal removed.

    ``hops=1`` is direct PPI neighbours. The diagonal is cleared AFTER any multi-hop expansion, so a
    two-hop walk cannot return to the target and hand it its own response back. Rows with no neighbour
    stay all-zero and inject nothing, which is the correct behaviour for an isolated target rather than
    something to paper over."""
    from tcell_pipeline.graph import PROTEIN
    pos = {g: i for i, g in enumerate(targets)}
    score_col = len(config.PPI_SOURCES)
    # gene -> graph node -> target position, as an array lookup: the graph has ~8M edges and a Python
    # loop with dict lookups over them costs tens of seconds for no reason.
    n_nodes = (max(gene_to_idx.values()) + 1) if gene_to_idx else 0
    node_to_pos = np.full(max(n_nodes, 1), -1, dtype=np.int64)
    for g, i in gene_to_idx.items():
        p = pos.get(g)
        if p is not None:
            node_to_pos[i] = p
    rows, cols, vals = [], [], []
    for rel in _PP_RELATIONS:
        store = graph[PROTEIN, rel, PROTEIN]
        ei = store.edge_index
        if ei.numel() == 0:
            continue
        src, dst = ei[0].numpy(), ei[1].numpy()
        w = (store.edge_attr[:, score_col].numpy().astype(np.float64) if weighted
             else np.ones(len(src)))
        ia, ib = node_to_pos[src], node_to_pos[dst]
        keep = (ia >= 0) & (ib >= 0)
        ia, ib, w = ia[keep], ib[keep], w[keep]
        rows.append(np.concatenate([ia, ib]))
        cols.append(np.concatenate([ib, ia]))
        vals.append(np.concatenate([w, w]))
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(cols) if len(cols) else np.zeros(0, dtype=np.int64)
    vals = np.concatenate(vals) if len(vals) else np.zeros(0)
    n = len(targets)
    a = (sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr() if len(rows)
         else sp.csr_matrix((n, n)))
    for _ in range(hops - 1):
        a = (a @ a).tocsr()
    a.setdiag(0.0)                      # AFTER the expansion: no path may return to its own target
    a.eliminate_zeros()
    deg = np.asarray(a.sum(1)).reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(deg > 0, 1.0 / deg, 0.0)
    return sp.diags(scale) @ a


def injection_matrix(Z, row_target, role_of_row, A, targets, *, permute_seed: int | None = None,
                     hops: int = 1) -> dict:
    """The pure core: responses in, scaled injection matrix out. Kept free of file loading so the
    leakage guarantee can be tested on a six-gene fixture instead of a 350-million-entry matrix.

    ``permute_seed`` is the NEGATIVE CONTROL: each target is given some OTHER target's neighbour mean,
    so the injected component has the same size and distribution and no relationship to the graph. A
    ladder that recovers this is not measuring graph structure."""
    is_train = role_of_row == "train"
    usable = is_train | (role_of_row == "val")   # rail 1: challenge/calibration rows are never touched
    pos = {g: i for i, g in enumerate(targets)}

    # R: per-target mean response over TRAIN rows only. A target with no train row keeps a zero row, so
    # it contributes nothing to any neighbour's injection, and NO validation response is ever read.
    R = np.zeros((len(targets), Z.shape[1]), dtype=np.float64)
    counts = np.zeros(len(targets))
    train_rows = np.flatnonzero(is_train)
    for start in range(0, len(train_rows), 2048):           # chunked: the dense slice is 10k wide
        chunk = train_rows[start:start + 2048]
        dense = Z[chunk].toarray()
        for j, r in enumerate(chunk):
            i = pos.get(row_target[r])
            if i is not None:
                R[i] += dense[j]
                counts[i] += 1
    nz = counts > 0
    R[nz] /= counts[nz, None]

    M_target = A @ R                                        # (targets, G) neighbour means
    if permute_seed is not None:
        rng = np.random.default_rng(permute_seed)
        perm = rng.permutation(len(targets))
        for _ in range(64):                                 # no target keeps its own neighbourhood
            if not np.any(perm == np.arange(len(targets))):
                break
            perm = rng.permutation(len(targets))
        M_target = M_target[perm]

    M = np.zeros(Z.shape, dtype=np.float32)
    idx = np.array([pos.get(t, -1) for t in row_target])
    take = usable & (idx >= 0)
    M[take] = np.asarray(M_target)[idx[take]].astype(np.float32)

    # Scale from TRAIN ROWS ONLY. This was originally computed over train+val, which is a real leak and
    # the leakage test caught it: a tampered validation response moved sd_response, moved the scale, and
    # moved every injected value by a factor of 46. The per-target means were clean; the global constant
    # was not. "Train-only" has to mean every number that reaches the output, including the constants.
    train_idx = np.flatnonzero(is_train)
    sd_response = float(Z[train_idx].toarray().std()) if train_idx.size else 0.0
    sd_inject = float(M[is_train].std()) if train_idx.size else 0.0
    scale = (sd_response / sd_inject) if sd_inject > 0 else 0.0
    M *= scale
    return {"M": M, "usable": usable, "targets": targets, "sd_response": sd_response,
            "sd_inject_raw": sd_inject, "scale": scale, "hops": hops,
            "n_targets": len(targets), "n_targets_with_train_rows": int(nz.sum()),
            "n_targets_with_neighbours": int((np.asarray(A.sum(1)).reshape(-1) > 0).sum()),
            "permuted": permute_seed is not None}


def build_injection(*, hops: int = 1, permute_seed: int | None = None) -> dict:
    """Load the reference response, split and graph, then call ``injection_matrix``."""
    from tcell_pipeline.graph import build_hetero_graph

    obs = pd.read_parquet(config.DE_OBS_PATH, columns=["target_contrast_gene_name"])
    row_target = obs["target_contrast_gene_name"].astype(str).to_numpy()
    roles = role_map()
    role_of_row = np.array([roles.get(t, "absent") for t in row_target])
    usable = (role_of_row == "train") | (role_of_row == "val")
    targets = sorted(set(row_target[usable]))

    Z = sp.load_npz(config.DE_LAYERS_DIR / "zscore.npz").tocsr()
    graph, gene_to_idx = build_hetero_graph()
    A = neighbour_operator(targets, gene_to_idx, graph, hops=hops)
    return injection_matrix(Z, row_target, role_of_row, A, targets,
                            permute_seed=permute_seed, hops=hops)


def write_rung(delta: float, out_root: Path, inj: dict | None = None, *, hops: int = 1,
               permute_seed: int | None = None) -> dict:
    """Materialise one rung as a complete INTERMEDIATE_ROOT: injected response, everything else linked."""
    inj = inj or build_injection(hops=hops, permute_seed=permute_seed)
    out_root = Path(out_root)
    (out_root / "de_layers").mkdir(parents=True, exist_ok=True)
    ref = config.INTERMEDIATE_ROOT

    Z = sp.load_npz(config.DE_LAYERS_DIR / "zscore.npz").tocsr()
    Zi = sp.csr_matrix(Z.toarray() + float(delta) * inj["M"])
    sp.save_npz(out_root / "de_layers" / "zscore.npz", Zi)

    for name in LINKED:
        src, dst = (ref / name).resolve(), out_root / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
    for name in COPIED_LAYERS:                              # other layers nothing here reads, but the
        src = (config.DE_LAYERS_DIR / name).resolve()       # root should still look complete
        dst = out_root / "de_layers" / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    prov = {"delta": float(delta), "hops": inj["hops"], "permuted": inj["permuted"],
            "scale": inj["scale"], "sd_response": inj["sd_response"],
            "sd_inject_raw": inj["sd_inject_raw"], "n_targets": inj["n_targets"],
            "n_targets_with_train_rows": inj["n_targets_with_train_rows"],
            "n_targets_with_neighbours": inj["n_targets_with_neighbours"],
            "rows_injected": int(inj["usable"].sum()), "rows_total": int(Z.shape[0]),
            "source_root": str(ref), "source_zscore_sha_note":
                "response layer REPLACED; every other artifact is a symlink to the reference root"}
    (out_root / "injection_provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"[inject] delta={delta} -> {out_root}  ({prov['rows_injected']}/{prov['rows_total']} rows "
          f"injected, scale {inj['scale']:.4g})")
    return prov


def build_ladder(root: Path, ladder=LADDER, *, hops: int = 1, control_delta: float = 0.40,
                 control_seed: int = 17, force: bool = False) -> dict:
    """Every rung of the pre-registered ladder plus the permuted control, from ONE injection matrix.

    The unscaled injection is identical across the real rungs -- only ``delta`` differs -- so building
    it once turns five graph builds and five passes over 21k train rows into one. The control needs its
    own matrix because the permutation happens before scaling."""
    root = Path(root)
    out = {}
    inj = build_injection(hops=hops)
    for d in ladder:
        rung = root / f"d{int(round(d * 1000)):03d}"
        if rung.exists() and force:
            shutil.rmtree(rung)
        if (rung / "injection_provenance.json").exists():
            print(f"[inject] SKIP {rung} (already built)")
            out[str(d)] = json.loads((rung / "injection_provenance.json").read_text())
            continue
        out[str(d)] = write_rung(d, rung, inj)
    ctrl = root / f"permuted_d{int(round(control_delta * 1000)):03d}"
    if ctrl.exists() and force:
        shutil.rmtree(ctrl)
    if (ctrl / "injection_provenance.json").exists():
        print(f"[inject] SKIP {ctrl} (already built)")
        out["permuted"] = json.loads((ctrl / "injection_provenance.json").read_text())
    else:
        out["permuted"] = write_rung(control_delta, ctrl,
                                     build_injection(hops=hops, permute_seed=control_seed))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--permute-seed", type=int, default=None,
                    help="negative control: give each target another target's neighbour mean")
    ap.add_argument("--ladder", action="store_true",
                    help="build every pre-registered rung plus the permuted control under --out")
    ap.add_argument("--force", action="store_true", help="overwrite existing rung roots")
    a = ap.parse_args()
    if a.ladder:
        build_ladder(Path(a.out), hops=a.hops, force=a.force)
    else:
        if a.delta is None:
            ap.error("--delta is required unless --ladder is given")
        out = Path(a.out)
        if out.exists() and a.force:
            shutil.rmtree(out)
        write_rung(a.delta, out, hops=a.hops, permute_seed=a.permute_seed)
