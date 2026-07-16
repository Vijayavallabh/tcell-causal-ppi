"""Module 4 real-data smoke: extract a predictive rationale on a real neighbourhood, test it.

Builds the real typed graph, runs Module 1 -> h_do for a real perturbation, encodes its neighbourhood
(Module 2), then extracts the sparse rationale (Module 4) and runs the fixed-model faithfulness tests
against matched-random controls. The program basis B is random here (rationale faithfulness is a
RELATIVE dz comparison, so it does not depend on the basis being the fitted one) — this smoke exercises
the real graph + gating + rationale path, not the program semantics.

    python src/tcell_pipeline/rationale/run_module4_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path for direct runs

import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)  # CPU-only smoke on a many-core box: avoid thread-pool thrash on tiny GNN ops

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.encoders import build_encoder_batch  # noqa: E402
from tcell_pipeline.graph import build_hetero_graph, sample_subgraph  # noqa: E402
from tcell_pipeline.graph.typed_graph_encoder import TypedGraphEncoder  # noqa: E402
from tcell_pipeline.model import EGIPGModel  # noqa: E402
from tcell_pipeline.rationale import (  # noqa: E402
    RATIONALE_LABEL,
    FaithfulnessTester,
    MatchedRandomSampler,
    RationaleHead,
    complement,
)


def run() -> bool:
    required = (config.PROTEIN_EDGES_PATH, config.PERTURBATION_CONDITION_PATH, config.DE_OBS_PATH,
                config.DE_VAR_PATH)
    if not all(p.exists() for p in required):
        print("[module4-smoke] marts absent — run run_module0.py first")
        return False

    torch.manual_seed(0)
    pc_all = pd.read_parquet(config.PERTURBATION_CONDITION_PATH)
    obs_all = pd.read_parquet(config.DE_OBS_PATH, columns=["n_guides", "single_guide_estimate"])
    n_genes = len(pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"]))

    graph, gene_to_idx = build_hetero_graph()
    genc = TypedGraphEncoder(graph, gene_to_idx)
    model = EGIPGModel(torch.randn(n_genes, config.PROGRAM_DIM), graph_encoder=genc).eval()

    in_graph = pc_all["hgnc_symbol"].isin(gene_to_idx).to_numpy()
    picked = list(pc_all.index[in_graph][:1])
    pc, obs = pc_all.loc[picked].reset_index(drop=True), obs_all.loc[picked]
    gene, cond = pc["hgnc_symbol"].iloc[0], pc["culture_condition"].iloc[0]

    with torch.no_grad():
        h_do = model.perturbation_encoder(build_encoder_batch(pc, obs))[0]
        sub = sample_subgraph(graph, gene, gene_to_idx=gene_to_idx)
        r = genc.encode_subgraph(sub, cond, h_do)

    head = RationaleHead().eval()
    with torch.no_grad():
        rat = head(r["gates"], r["node_states"], None, sub)
    mask = rat["selection_mask"]
    n_edges = sum(int(g.numel()) for g in r["gates"].values())

    tester = FaithfulnessTester(genc, model.decoder)
    dz_full = tester._dz(sub, cond, h_do)
    suff_dist = lambda m: float((tester._dz(sub, cond, h_do, m) - dz_full).norm())
    nec_dist = lambda m: float((tester._dz(sub, cond, h_do, complement(m)) - dz_full).norm())

    suff, nec = suff_dist(mask), nec_dist(mask)
    controls = MatchedRandomSampler(n_controls=min(config.N_MATCHED_CONTROLS, 30)).sample(mask)
    rand_suff = sum(suff_dist(c) for c in controls) / len(controls)
    rand_nec = sum(nec_dist(c) for c in controls) / len(controls)
    audit = tester.structural_ood_audit(sub, mask)

    label_ok = rat["label"] == RATIONALE_LABEL and "causal" not in rat["label"]
    finite = all(v == v for v in (suff, nec, rand_suff, rand_nec))  # NaN != NaN
    suff_ok, nec_ok = suff < rand_suff, nec > rand_nec

    print(f"[module4-smoke] target {gene!r} under {cond!r}: {n_edges} edges, |S|={len(rat['selected'])}")
    print(f"  sufficiency  {suff:.4f} < matched-random {rand_suff:.4f}  = {suff_ok}")
    print(f"  necessity    {nec:.4f} > matched-random {rand_nec:.4f}  = {nec_ok}")
    print(f"  structural OOD: components {audit['before']['component_count']}->{audit['after']['component_count']}, "
          f"deleted-fraction {audit['after']['sparsity']:.3f}, "
          f"hop {audit['before']['hop_distance']}->{audit['after']['hop_distance']}")
    print(f"  output labelled {rat['label']!r} (not causal) = {label_ok}")

    ok = label_ok and finite and suff_ok and nec_ok
    print(f"\n=== Module 4 real-data smoke {'PASSED' if ok else 'FAILED'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
