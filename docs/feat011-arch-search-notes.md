# feat-011 architecture search — the ablation behind the model choice (2026-07-22)

Goal: improve `condition_gated` / `typed_static` past `untyped_gnn` for the AAAI submission, and select
the graph architecture the confirmatory 5-seed val campaign then tests. **Every number here is on a
target-grouped INNER holdout of train (17,009 train / 4,253 holdout, 0 gene overlap), seed 0, 5 epochs,
`lambda_graph=0` (live gates). VAL WAS NEVER OPENED. These are SELECTION leads at n=1, not confirmatory
results** — the winner is confirmed only by the paired 5-seed val campaign (see
`feat011-rescreen-notes.md`), because 14 architectures scored on one fold is a multiple-comparisons
hazard and the "best" of 14 is upward-biased by selection.

Driver: `src/tcell_pipeline/arch_search.py`. Artifacts: `data/results/arch_search/*.json` (14 cells).

## The full table (re-derived from the JSON artifacts)

| model | systema | edges | note |
|---|---|---|---|
| **condition_gated · add · thr0.0** | **0.091377** | 8.03M | typed, gated, live gates — **WINNER** |
| condition_gated · mean · thr0.0 | 0.088419 | 8.03M | per-relation mean-normalised |
| condition_gated · gcn · thr0.0 | 0.088133 | 8.03M | symmetric degree-normalised |
| condition_gated · add · thr0.4 | 0.088129 | 2.10M | |
| untyped_gnn_p · gcn · thr0.4 | 0.087485 | 2.10M | stage-2, fair GCN baseline @0.4 |
| **untyped_gnn · gcn · thr0.0** | **0.087369** | 8.03M | the anchor to beat |
| untyped_wgcn · thr0.0 | 0.086865 | 8.03M | stage-2, GCN + STRING edge weights |
| untyped_gat · thr0.4 | 0.086862 | 2.10M | stage-2, GATv2 edge-attention (heads=2) |
| expression_only | 0.086238 | 8.03M | no-graph baseline |
| condition_gated · gcn · thr0.4 | 0.084850 | 2.10M | |
| condition_gated · mean · thr0.4 | 0.084513 | 2.10M | |
| condition_gated · add · thr0.7 | 0.082817 | 1.41M | |
| condition_gated · mean · thr0.7 | 0.082746 | 1.41M | |
| condition_gated · gcn · thr0.7 | 0.082489 | 1.41M | |

## What the search decided

**1. The current typed/gated architecture is the winner, and every architectural lever tried made it
worse or neutral.**
- Normalisation HURT the typed encoder: add (0.0914) > mean (0.0884) ≈ gcn (0.0881). The unnormalised
  sum the encoder already uses is best. (This *refuted* the leading hypothesis — that the typed
  encoder's disadvantage was a degree-confounded `add` aggregation.)
- Pruning HURT, monotonically: thr0.0 (0.0914) > thr0.4 (0.0881) > thr0.7 (0.0828). The 86%
  low-confidence STRING functional edges HELP; removing them loses signal.

**2. The typed/gated model already beats untyped_gnn and no-graph — the first time in the project with
LIVE gates** (every prior campaign measured dead gates). Descriptive inner-holdout contrasts (n=1,
selection fold):
- condition_gated − untyped_gnn = **+0.004007**
- condition_gated − expression_only = **+0.005139**
- untyped_gnn − expression_only = +0.001132 (consistent with the campaign's +0.0045)

**3. Stage-2 upgrades to the untyped baseline did NOT help** (each vs its matched baseline):
- untyped_wgcn (GCN + STRING edge weights) − untyped_gnn = **−0.000504**
- untyped_gat (GATv2 learned edge-attention) − untyped_gnn_p (GCN, same thr0.4 graph) = **−0.000623**

Edge-weighting and learned attention both cost a little. So the answer to "improve the *untyped* graph
architecturally" is a clean negative — and it does not matter, because the typed/gated model is already
ahead of the best untyped one.

## The model to publish

`condition_gated` (typed relations, signed messages, per-edge condition gate, unnormalised aggregation,
full graph) with the `lambda_graph=0` fix. It wins on systema AND keeps the edge gates the rationale
audit (feat-012) needs — an untyped GNN cannot support that story. The 5-seed val campaign confirms it.

## Engineering notes

- New: `AugmentedUntypedEncoder` (`baselines/graph_baselines.py`), conv ∈ {gcn, wgcn, gat}. The
  `UntypedGraphEncoder` message-pass loop was refactored into an overridable `_message_pass` hook; the
  `gcn` path is bit-identical to the original (pinned by
  `test_plain_gcn_augmented_matches_the_original_untyped_encoder`).
- New levers on `TypedGraphEncoder`: `norm ∈ {add, mean, gcn}`, `rel_scale`; and
  `build_hetero_graph(functional_min_score=…)` prunes only functional edges, keeping `gene_to_idx`
  stable across thresholds so cells are comparable.
- The GAT edge-score pathway is verified by GRADIENT flow, not init-output sensitivity: GATv2 feeds the
  score through attention logits that node features dominate at random init, so its init output barely
  moves even though `d out / d score ≠ 0` and it is fully trainable
  (`test_gat_and_wgcn_route_gradient_from_edge_scores_that_plain_gcn_ignores`).
- MEMORY: multi-head GAT attention over the full graph's 74k–86k-edge hub subgraphs OOMs an 80 GB card
  in the training loop (donor-invariance re-forwards multiply it). Measured worst-case-hub
  single-forward+backward: heads=4/full = OOM, heads=2/full = 23 GB, heads=2/thr0.4 = 12 GB. So GAT ran
  at heads=2 on the thr0.4 graph, with a matched GCN baseline on the SAME graph so any gain would be
  attributable to attention rather than the graph change. `GRAPH_ENCODE_CHUNK=2` bounds in-flight
  subgraphs.
- Tests: 547 total (+6 this line of work), all red-first, GCN-equivalence and gradient-flow pinned.
