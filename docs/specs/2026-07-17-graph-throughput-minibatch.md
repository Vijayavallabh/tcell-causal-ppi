# Graph throughput refactor — CSR neighbour index + PyG mini-batching (2026-07-17)

Lifts the compute ceiling that blocked the feat-010 / feat-011 / feat-012 / feat-013 campaigns, all of
which need a graph model trained to convergence on the full 21,262-row fold.

## The task named the fix, not the bottleneck

The deferred task (carried in `session-handoff.md` since 2026-07-16) was written as *"mini-batch the graph
encoders (PyG `Batch` over sampled subgraphs) so message passing runs on many at once"*, on the reasoning
that the GPU sat at 0–4% while the typed encoders held 23–29 GB.

Measuring the real 25,440-node graph on an A100 **before** writing any code said otherwise:

| per row, A100 | share |
| --- | --- |
| `sample_subgraph` | 581 ms — **95%** |
| message passing (`encode_subgraph`) | 34 ms — **5%** |

Message passing was *already* 53× faster on GPU than CPU (1809 → 34 ms/row). Mini-batching it would have
optimised the 5%: Amdahl caps that at **1.05×**. The GPU was idle because almost nothing was on it — the
process was pinned in a single-threaded CPU sampler.

`cProfile` inside the sampler located it exactly:

| `sample_subgraph` internals | tottime |
| --- | --- |
| `torch.isin` | **59%** |
| `_induce` (full-graph remap gather) | **28%** |
| `_grow` body | 13% |
| `torch.argsort` | **0.03%** |

**Root cause:** `_grow` and `_induce` each answered *"which edges touch this node set?"* with a boolean
scan over the entire edge table — `torch.isin(ei[a], frontier)` across 6.86M `functional_assoc` edges,
×2 directions ×3 relations ×2 hops, then `_induce` re-scanning all ~8M edges ×3 relations. ~8M edges swept
per row to find a few thousand. The ranking loop that *looked* expensive was free.

## What changed

1. **`neighborhood_sampler._NeighborIndex`** — a per-relation CSR (`indptr`, edge-id ordering) built
   **once** per graph and cached on the graph object (0.85 s, ~130 MB; two int64 orderings of the 6.86M
   functional_assoc edges dominate). `incident(rel, key, nodes)` returns the incident edge ids in
   O(sum of the node set's degree). It deliberately returns ids **grouped by node, not sorted**: callers
   that need original edge order sort only their surviving subset (~30k) rather than every candidate
   (~160k). Those caller-side sorts are LOAD-BEARING, not leftovers — see the review section below.
   **581 → 22 ms/row.**
2. **PyG mini-batching** of the now-dominant message passing, in `TypedGraphEncoder.forward` and
   `UntypedGraphEncoder.forward`: the batch's subgraphs go through one `Batch.from_data_list` (concatenated
   on CPU, one host→device copy), so one set of relational kernels replaces a per-row Python loop. Edges
   never cross samples because the batch offsets node ids, so `_GraphLayer` needed no change at all.

### The three hard parts

- **The condition gate is per sample, but a batch mixes conditions.** `_gate` scatters `h_cond` per edge
  via the batch vector (`p_batch[edge_index[0]]`), so every edge is gated by *its own* sample's condition.
  The scatter lives *inside* `_gate` rather than at the call site, so an override that ignores the condition
  (`StaticTypedGraphEncoder`) never pays for an `(E, D)` index_select it discards — and it still overrides
  **only** `_gate`.
- **The readout must not leak across samples.** Protein+complex states are concatenated and **stably**
  sorted by sample id — stable, so proteins-then-complexes order within a sample matches the single-subgraph
  `cat([h_p, h_c])` — then `to_dense_batch` + a `key_padding_mask` gives per-sample attention that still
  sums to 1. Every sample has ≥1 protein node (the seed is always selected), so no all-pad row / NaN softmax.
- **Complex nodes batch cleanly**: `orig_idx` is *not* an `*index` attribute, so PyG does not offset it and
  it still addresses the global complex table. Samples with zero complexes are handled (`bat[COMPLEX].batch`
  correctly skips them; the all-empty case is guarded).

## Measured (A100, real graph, forward+backward, mixed conditions)

| | before (`97f8451`) | after |
| --- | --- | --- |
| ms/row (bs=8) | 667 | **61** |
| rows/s | 1.50 | **16.29** |
| GPU utilisation | median **1%**, p90 10% | median **46%**, p90 94% |
| peak memory | 9.0 GB | 10.0 GB |
| 21,262-row epoch | **3.94 h** | **0.36 h** |

**10.9× end-to-end.** `bs=32` is not worth it: 15.2 vs 16.3 rows/s for 31.5 vs 10.0 GB — **use bs=8**.
The BEFORE run reproduces the reported pathology independently (median 1% util). GPU 0 hosts an unrelated
tenant's ~670 MB process that idles at ~1%, which is exactly the BEFORE median — i.e. the old encoder
contributed ≈0% and the AFTER 46% is this work.

## Correctness: equivalence is the gate

Both halves are pinned by exact-equivalence tests, because a divergence would silently change which
subgraph the model sees — the science, not just the speed.

- `test_sampler_matches_full_scan_reference` — sweeps **every** fixture gene × hops × caps; the CSR sampler
  is **bit-identical** to the frozen full-scan oracle. On the real graph the sampled subgraphs are unchanged:
  min 465 / max 512 nodes, mean 30,751 PP edges.
- `test_batched_forward_matches_per_sample_loop` (+ absent/isolated targets, + CUDA) — batched forward equals
  the per-sample loop **edge for edge**, with `encode_subgraph` (the unbatched path Module 4 uses) as the
  oracle, so the two share no batching code.
- `test_batched_gate_uses_each_samples_own_condition`, `test_batched_static_encoder_pins_gate_to_one`,
  `test_untyped_batched_forward_matches_per_sample_loop`.

**The oracles are deliberately self-contained.** An early version imported `_PRIORITY_BONUS` /
`_PP_RELATIONS` / `_SCORE_COL` from the module under test; mutating a shared constant then moved *both*
sides and the test passed on a broken sampler. Inlined. This is the same failure mode the Module 8 pass-3
review named: *a fix that only satisfies its own regression test is not a fix.*

**Teeth verified by injection** — each of these failed the suite: selection priority reordered, traversal
direction dropped, cap off-by-one, induced-edge order reversed, condition broadcast from sample 0,
readout `node_batch` dropped, unstable readout sort, gate split reversed across samples.

DropEdge is train-only and random, so it has no per-sample equivalent; equivalence is asserted in `eval()`.
That is the honest limit of the claim.

## Contracts preserved (verified, not assumed)

- `forward -> (h_graph (B,256), edge_gates, edge_confidences)` with `edge_gates[rel]` / `edge_confidences[rel]`
  **lists over the batch**, one per-edge tensor per sample, aligned per edge — what `RationaleHead` and
  `StageALoss._graph` consume.
- `encode_subgraph(sub, condition, h_do, keep_mask=None)` still works **unbatched** (Module 4 faithfulness +
  `rationale_audit` call it per case) and is the batched path's test oracle.
- Unknown target → zero `h_graph` + empty per-relation tensors; OOV condition → `ValueError`.
- `StaticTypedGraphEncoder` overrides **only** `_gate`. No public signature changed; no new dependency.
  `GraphReadout.forward` gained an optional `node_batch=None` (additive, backwards compatible).

## Not done, deliberately

**Saturation is not reached (median 46%, not ~100%) and the reason is known:** the batch is still sampled
row-by-row on CPU (~22 ms/row, ~36% of a step), so the GPU idles between batches. The next ceiling is
batch-aware sampling, a per-target subgraph cache (11,526 unique targets over 33,983 rows ≈ 2.9 rows/target,
so a cache is worth ~2.9× within an epoch and ~all of it across epochs, at ~17–33 GB/process), or sampling
in DataLoader workers. Left undone: 0.36 h/epoch is tractable, and the cheapest of these costs real memory.
The `ponytail:` markers in both encoders now name this floor rather than the old one.

## Verification

`./init.sh` green at **242** tests (224 + 18 new). Real-data smoke, all three §10.6 nested members,
600 rows / 1 epoch — **machinery at real scale, not science** (1 epoch on 600 rows is noise):
untyped_gnn systema=0.0534, typed_static systema=0.0531, condition_gated systema=0.0432.

The sealed challenge split (5,608 rows) was **not** touched and remains unopened.


## xhigh `/code-review` (2026-07-17, 37 agents) — 15 defects, all resolved

The review confirmed the refactor was functionally correct as shipped (the sampler matched the oracle on
every probed combination) and then found the things the equivalence tests could not see.

**The one real regression — `h_graph` dtype (`typed_graph_encoder.py`, `graph_baselines.py`).**
`torch.zeros(...)` hardcodes float32; the `torch.stack(h_graphs)` it replaced inherited whatever dtype the
readout produced. Under `torch.autocast` the bf16/fp16 readout was silently cast back up into an fp32
`h_graph`, so the graph pathway left the autocast region at a different dtype from every other pathway.
Fixed by allocating with `pooled.dtype`; the all-absent early return (where no readout runs) mirrors
`h_do.dtype`, the tensor `ProgramDecoder` concatenates it with. Note the reviewer's `.half()` framing is
not reachable — `proj(x)` raises first, because the graph's node features stay float32 — but the autocast
path is real, and is what the regression test uses.

**The test blindness — this was the important finding.** `_grow`'s order-restoring `torch.sort` is the
single line the bit-identity claim rests on, and it *reads* redundant (`_build_csr` already stable-sorts;
`incident()` documents that sorting is deferred). It is not redundant: `_PRIORITY_BONUS` adds 1e6 to a
float32 score whose spacing at 1e6 is 0.0625, so scores quantise into ties in bulk (148 physical_ppi edges
→ 17 distinct keys on the fixture) and the tie-break order decides which neighbours survive the cap. The
reviewer deleted the line and **all 237 tests passed** — while the sampler diverged from the oracle on 36
of 600 (gene, cap, hops) combos and on 35 of the fixture's 60 genes. The four hand-picked probe genes
(G000, G001, G030, G059) were exactly the ones that cannot fail. Worse, the module docstring asserted the
*opposite* of `incident()`'s real contract, actively inviting the deletion. Fixed by sweeping **every**
gene (the suite now catches that mutation), correcting the docstring, and labelling both sorts load-bearing.
The lesson generalises past this commit: **hand-picked probe cases are a way of choosing what the test
cannot see.** Mutation-testing the diff is what exposed it — the earlier round mutated 6 things but never
the one line the central claim depended on.

**A vacuous test.** `test_batched_static_encoder_pins_gate_to_one` asserted `allclose(g, ones_like(g))`,
which is **True on a zero-length tensor**. A forward that regressed to returning empty gates for every
sample passed green — and empty gates silently zero `StageALoss._graph`'s sparsity + unsourced terms and
empty the Module-4 rationale, so the H2b comparison would have run against a model whose regulariser never
applied. Now pinned to the sub-graph's real edge counts.

**Cache staleness (one root cause behind six findings).** The CSR index is cached on the graph via
`setattr` with no invalidation; the full scan it replaced re-read `edge_index` every call. Verified by the
reviewer: sample → append an edge → sample again, and the new neighbour is **never** returned, silently —
which would hand an edge-ablation or rewired-network control the topology it thought it had removed. Fixed
with a fingerprint (`data_ptr`, shape, `_version` per relation) that rebuilds when the graph's edges change.

**Also fixed:** `torch.split` handed out *views* into the batch-wide gate tensor where the per-sample loop
gave independent storage (an in-place consumer would raise; holding one sample's gates pinned the whole
batch) → `.clone()`. Positional indexing narrowed the accepted input from any iterable to a sequence →
`list()`. `_gather`'s `torch.arange` ignored the graph's device (latent; the old scan raised too).
Hot-path waste: `ea[eids][:, _SCORE_COL]` materialised every edge-feature column → `ea[eids, _SCORE_COL]`;
`_induce`'s `src_remap[...] >= 0` conjunct is always true by construction → removed; 4n distinct empty
tensors per forward → one shared blank. Those last three took sampling 26 → 22 ms/row and the end-to-end
figure from 9.4× to **10.9×**.

**Also fixed on a second pass (the two originally waved through):**

- **Unguarded precondition (#7).** `incident()` requires a duplicate-free node set — a repeated node
  re-emits its whole CSR row, so its edges come back once per repeat and the sub-graph carries
  duplicates the boolean scan never produced. Both callers dedupe by construction, but the contract was
  docstring-only. Now enforced: the check is ≤512 nodes against a ~160k-edge gather, i.e. free, and the
  failure it prevents is silent corruption of the sampled neighbourhood.
- **Eval peak memory (#10) — this one was reachable, and waving it through was wrong.** The per-row loop
  bounded eval peak to ONE subgraph *regardless of batch size*; the batched version holds the whole
  batch. That is not hypothetical: `screening.collect_predictions` defaults to `batch_size=BATCH_SIZE=64`
  under `no_grad`, and the typed encoder already OOMs 80 GB at batch 32 on real dense subgraphs. The
  original goal had asked for "a working batch size / chunking path" and none was built. Now both
  encoders message-pass at most `config.GRAPH_ENCODE_CHUNK` (default 8) subgraphs at once and stitch the
  results, so the caller's batch size sets the *optimisation* batch and not the memory ceiling.

  | batch 64, `no_grad` (the real eval path) | peak | time |
  | --- | --- | --- |
  | chunking off | 12.53 GB | 2.9 s |
  | chunk = 8 | **2.01 GB** | **2.5 s** |

  6.2× less memory and slightly *faster*. Chunking is invisible to results — every chunk size (1, 2, 3,
  8, and off) agrees with the unchunked batch edge for edge, pinned by
  `test_encode_chunking_does_not_change_results`, and `test_chunking_bounds_message_passing_width`
  asserts no more than `chunk` subgraphs are ever in flight. **Honest limit: chunking does NOT reduce
  TRAINING memory** — autograd retains every chunk's activations until backward, so a batch-64 training
  step still peaks at 55.9 GB. Batch size remains the training knob; chunking bounds evaluation.

**Accepted, not fixed:** eval-only DropEdge equivalence; non-saturation (sampling is still row-by-row on
CPU); the duplicated `_store_key` (dependency direction prevents reuse); ~130 MB index memory.
