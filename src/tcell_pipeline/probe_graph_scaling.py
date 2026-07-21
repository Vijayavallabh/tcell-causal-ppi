"""Phase-2 diagnostic (READ-ONLY): is the typed encoder's UNNORMALISED sum-aggregation drowning its
residual stream?

Motivating observation (not yet a claim): the one graph arm that uses degree-normalised aggregation
(``untyped_gnn``, GCNConv, symmetric norm baked in) is the best-scoring arm (+0.0951), while both arms
built on ``_RelMessage(MessagePassing(aggr="add"))`` — condition_gated (+0.0834) and typed_static
(+0.0786) — are the worst two. On real subgraphs a hub node aggregates thousands of neighbours, so
``agg = sum_u alpha * m_u`` can be orders of magnitude larger than the node's own state ``h``. In

    _FFN.forward(h, agg) -> LayerNorm(h + net(agg))

that would make the LayerNorm input entirely ``net(agg)``, so the residual carries ~nothing and the
layer discards the node's own representation at every hop.

This probe MEASURES that ratio; it does not assume it. Reports per layer and per node type:
``||h||``, ``||net(agg)||``, their ratio, and the subgraph's degree distribution. A ratio near 1 refutes
the hypothesis and this line of work stops.

    OMP_NUM_THREADS=4 PYTHONPATH=src uv run python -m tcell_pipeline.probe_graph_scaling --n-max 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.graph import build_hetero_graph  # noqa: E402
from tcell_pipeline.graph.graph_builder import COMPLEX, PROTEIN  # noqa: E402
from tcell_pipeline.graph.typed_graph_encoder import (  # noqa: E402
    _MEMBERSHIP,
    _PP_RELATIONS,
    _FFN,
    _store_key,
)
from tcell_pipeline.screening.screening import (  # noqa: E402
    CONDITION_GATED,
    TYPED_STATIC,
    UNTYPED_GNN,
    nested_family_factories,
)
from tcell_pipeline.training.dataset import PerturbationDataset  # noqa: E402
from tcell_pipeline.training.trainer import seeded_init  # noqa: E402

_RELS = (*_PP_RELATIONS, _MEMBERSHIP)


def _attach(model) -> tuple[list, list]:
    """Forward-hook every _FFN so we capture (h, agg) as the real forward runs. Returns (records, handles)."""
    rec, handles = [], []

    def make(name):
        def hook(mod, inp, out):
            h, agg = inp[0], inp[1]
            with torch.no_grad():
                fa = mod.net(agg)
                rec.append({
                    "site": name,
                    "n_nodes": int(h.shape[0]),
                    "h_rms": float(h.pow(2).mean().sqrt()),
                    "agg_rms": float(agg.pow(2).mean().sqrt()),
                    "net_agg_rms": float(fa.pow(2).mean().sqrt()),
                    "ratio_netagg_over_h": float(fa.pow(2).mean().sqrt() / h.pow(2).mean().sqrt().clamp_min(1e-12)),
                    "out_rms": float(out.pow(2).mean().sqrt()),
                })
        return hook

    for i, layer in enumerate(model.graph_encoder.layers):
        for kind in ("ffn_protein", "ffn_complex"):
            mod = getattr(layer, kind, None)
            if isinstance(mod, _FFN):
                handles.append(mod.register_forward_hook(make(f"layers.{i}.{kind}")))
    return rec, handles


def _degrees(graph, gene_to_idx, targets) -> dict:
    """Edge counts and in-degree per relation for the batch's real subgraphs."""
    from tcell_pipeline.graph.neighborhood_sampler import sample_subgraph
    out = {}
    for t in targets:
        if t not in gene_to_idx:
            continue
        sub = sample_subgraph(graph, t, gene_to_idx=gene_to_idx)
        n_p = int(sub[PROTEIN].num_nodes)
        n_c = int(sub[COMPLEX].num_nodes)
        per = {}
        for r in _RELS:
            ei = sub[_store_key(r)].edge_index
            e = int(ei.shape[1])
            # in-degree is per DESTINATION node, and _store_key sends complex_membership to
            # (PROTEIN, membership, COMPLEX) — so its ei[1] holds COMPLEX indices, not protein ones.
            # Binning those against n_p averages over the wrong population (true [2,1,1] over 3 complexes
            # reads as 0.4 when padded to 10 protein bins). max_in_degree is immune; the mean is not.
            n_dst = n_c if r == _MEMBERSHIP else n_p
            deg = torch.bincount(ei[1], minlength=n_dst).float() if e else torch.zeros(max(n_dst, 1))
            per[r] = {"edges": e, "n_dst_nodes": n_dst,
                      "mean_in_degree": float(deg.mean()), "max_in_degree": float(deg.max())}
        out[t] = {"n_protein_nodes": n_p, "n_complex_nodes": int(sub[COMPLEX].num_nodes),
                  "total_edges": sum(v["edges"] for v in per.values()), "per_relation": per}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-max", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)  # many-core box: tiny per-subgraph GNN ops thrash the pool (README §perf)

    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()
    graph, gene_to_idx = build_hetero_graph()
    ds = PerturbationDataset("train", n_max=a.n_max)
    factories = nested_family_factories(gene_names, graph, gene_to_idx)
    batch, targets, conditions, dz, dx, _ = PerturbationDataset.collate(
        [ds[i] for i in range(min(a.batch_size, len(ds)))])
    print(f"[scale] batch targets={list(targets)}")

    print("\n=== subgraph size / degree (real sampled neighbourhoods) ===")
    deg = _degrees(graph, gene_to_idx, targets)
    for t, d in deg.items():
        print(f"  {t}: {d['n_protein_nodes']} protein + {d['n_complex_nodes']} complex nodes, "
              f"{d['total_edges']:,} edges")
        for r, v in d["per_relation"].items():
            print(f"      {r:<20} edges={v['edges']:>7,}  mean_in_deg={v['mean_in_degree']:>8.1f}  "
                  f"max_in_deg={v['max_in_degree']:>8.0f}")

    res = {"degrees": deg, "arms": {}}
    for arm in (CONDITION_GATED, TYPED_STATIC):
        print(f"\n=== {arm}: residual stream at init — is LayerNorm's input all aggregation? ===")
        with seeded_init(a.seed):  # restores the caller's RNG, unlike a bare torch.manual_seed
            model = factories[arm]()
        model.eval()  # EDGE_DROPOUT would otherwise drop ~10% of edges and add run-to-run noise to agg
        rec, handles = _attach(model)
        with torch.no_grad():
            model(batch, targets, conditions)
        for h in handles:
            h.remove()
        print(f"  {'site':<24} {'RMS h':>12} {'RMS agg':>14} {'RMS net(agg)':>14} {'net(agg)/h':>12}")
        for r in rec:
            print(f"  {r['site']:<24} {r['h_rms']:>12.4e} {r['agg_rms']:>14.4e} "
                  f"{r['net_agg_rms']:>14.4e} {r['ratio_netagg_over_h']:>12.2f}")
        res["arms"][arm] = rec
        worst = max((r["ratio_netagg_over_h"] for r in rec), default=0.0)
        print(f"  -> max net(agg)/h ratio = {worst:.2f}   "
              f"({'residual is DROWNED' if worst > 10 else 'residual survives' if worst < 3 else 'borderline'})")

    print(f"\n=== {UNTYPED_GNN} (GCNConv, degree-normalised) — the contrast ===")
    with seeded_init(a.seed):
        um = factories[UNTYPED_GNN]()
    um.eval()
    acts = []
    hs = [c.register_forward_hook(lambda m, i, o: acts.append(float(o.pow(2).mean().sqrt())))
          for c in um.graph_encoder.convs]
    with torch.no_grad():
        um(batch, targets, conditions)
    for h in hs:
        h.remove()
    print(f"  per-conv output RMS: {['%.4e' % v for v in acts]}")
    res["untyped_conv_rms"] = acts

    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, default=float))
        print(f"\n[scale] -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
