# Module 4 — Sparse Predictive-Rationale Head (feat-008 remainder) (design + as-built)

Date: 2026-07-16 · Depends on: feat-016 (Module 2 → `edge_gates`, node states), feat-008 (Module 3 →
`EGIPGModel`/`ProgramDecoder` → `Δz`). Consumers: feat-011 (screening), feat-012 (predictive-rationale
audit).

Design source: `EG_IPG_architecture_walkthrough.md` §7 (Module 4) and
`perturbation_informed_causal_protein_program_graphs_report.md` §Module 4 (lines 703-718) + the
graph-explanation-audit discussion (line 499). Where the walkthrough and the report disagree, the
walkthrough wins (the more-recent authoritative plan).

## Purpose

Given the frozen typed graph encoder, return a **sparse predictive rationale** `S` — the evidence edges
the model leans on to predict a perturbation's program delta — and *test its faithfulness*. This is a
**predictive rationale, NOT a causal mechanism** (report line 718): the deletion scores are *fixed-model
perturbation tests* (report line 499), not interventions. **Stage B**: the head is fitted AFTER the H1
predictor is frozen; nothing here changes H1 predictions.

## Scope (walkthrough §7-justified)

- **In:** edge-importance scoring + top-k selection (§7.1-7.2), the sparsity/sufficiency/necessity/
  contrastive rationale loss (§7.3 / report), the fixed-model faithfulness tester (sufficiency,
  necessity, structural-OOD audit; §7.3-7.4), and the matched-random control sampler (report §Module 4
  contrastive term). Plus the Module-2 enabler `TypedGraphEncoder.encode_subgraph` (final node states +
  a per-edge gate mask for the deletion re-runs).
- **Out (deferred):** the Stage-B *training loop* (the loss module exists; no fit loop), the
  stability-across-splits loss term, and the full frozen-subset faithfulness audit on a *trained* H1
  checkpoint (necessity/sufficiency/minimality/stability vs ≥100 matched controls + structural-OOD
  retraining/mask audit) — that is feat-012 / Phase 4, and cannot run until H1 is trained and frozen.

## Architecture — `src/tcell_pipeline/rationale/`

### RationaleHead — `rationale_head.py` (`nn.Module`)

Per edge `(u, v)` with layer-independent condition gate `ᾱ` (from Module 2's `edge_gates`) and 8-d edge
feature `f_e`:

1. learned relevance `s = σ(Linear([h_u‖h_v‖f_e]))` — `h_u, h_v` are the **final-layer** node states
   (protein for PP relations, complex for `complex_membership`).
2. importance `imp = ᾱ · s ∈ [0,1]` (both factors in `[0,1]`).
3. selection: **top-k** edges by importance across all four relations → `selection_mask` (dict
   rel→bool) + `selected` (list of `(rel, edge_idx, imp)`, highest first).

The scorer `Linear` is **zero-initialised**, so an untrained head has `s == 0.5` everywhere and importance
ranks purely by the frozen gate (faithful by construction); training moves `s` to refine that. Output is
labelled `predictive_rationale`, **never** `causal`. `edge_gates=None` (the expression-only nested member,
no graph) → an empty rationale.

### RationaleLoss — `rationale_loss.py` (`nn.Module`)

    L = λ_sp · |S|
      + λ_suff · ‖Δz_S     − Δz_full‖²                 (rationale reproduces the prediction)
      + λ_nec  · relu(δ_nec − ‖Δz_\S − Δz_full‖)²       (removing it changes the prediction)
      + λ_ct   · relu(margin + ‖Δz_S − Δz_full‖ − mean‖Δz_rand − Δz_full‖)   (beats matched-random)

A pure function of pre-computed program deltas + the head's importance. `|S|` is the summed importance
mass (a soft-L0 surrogate) so the sparsity term flows gradient to the scorer; passing the deltas computed
with the head's continuous importance as soft gate weights makes the whole objective differentiable to the
head. `δ_nec` / `margin` default to `config.RATIONALE_TAU`.

### FaithfulnessTester — `faithfulness.py` (eval utility, NOT an `nn.Module`)

Wraps the frozen graph encoder + decoder.

- `delta_z(sub, cond, h_do, keep_mask=None)` — re-encode + decode → `Δz`, **forcing the encoder/decoder
  into eval** so DropEdge is off (the "fixed-model" contract; `@torch.no_grad` suppresses gradients but
  NOT dropout). The prior train/eval state is restored, so the tester never mutates the caller's model.
- `sufficiency(sub, cond, h_do, mask, dz_full=None) = ‖Δz(keep only S) − Δz_full‖` — small ⇒ S suffices.
- `necessity(sub, cond, h_do, mask, dz_full=None) = ‖Δz(remove S) − Δz_full‖` — large ⇒ S is needed.
  (`dz_full` is mask-invariant; pass it in to skip the recompute across matched-random controls.)
- `structural_ood_audit(sub, mask)` — degree distribution / connected-component count / deleted-fraction
  `sparsity` / hop-distance (eccentricity from an anchor node), **before vs after** deleting S, all
  **protein-protein-scoped** so the sparsity signal is consistent with the connectivity signals. Catches a
  "faithful" rationale that merely fragments the graph out of distribution (report line 499, GInX-Eval).

The gate mask zeroes a relation's condition gate on the dropped edges via
`TypedGraphEncoder.encode_subgraph(keep_mask=…)`; the gate multiplies every message, so a zero weight drops
the edge at all layers.

### MatchedRandomSampler — `matched_random.py`

`n_controls` random selection masks, each matching `selection_mask`'s **per-relation edge count** (→ size
+ relation-type composition + sparsity). Feeds the contrastive loss term and the sufficiency<random /
necessity>random comparison. `ponytail:` the report's fuller degree/connectivity/target-hop matching is a
refinement for the final rationale-quality analysis; deferred until that analysis is run.

### Module-2 enabler — `graph/typed_graph_encoder.py`

`encode_subgraph(sub, condition, h_do, keep_mask=None)` runs message passing on an already-sampled
subgraph and returns `{h_graph, gates, node_states={protein, complex}, attn}`. `keep_mask` (dict
rel→per-edge weight, bool or float) scales that relation's gate, so the deletion tests re-run the frozen
encoder with the rationale kept / removed, and a soft (continuous) weight makes the loss differentiable to
the head. `encode_one` is unchanged — it now delegates to `encode_subgraph`, keeping its `(h_graph, gates,
attn)` 3-tuple contract and all existing callers intact.

## Fixed-model contract (why this is not causal)

Deletion re-runs use the FROZEN encoder+decoder in eval. A large necessity drop can reflect *graph
corruption* (structural OOD) rather than faithful attribution (report line 499), so necessity/sufficiency
are always reported **with** the structural-OOD audit and **against** matched-random controls, and the
returned subgraph is called a *predictive rationale*, never a causal subgraph or discovered pathway.

## Config additions (`config.py`)

`RATIONALE_TOP_K = 15`, `RATIONALE_TAU = 0.5` (necessity/contrastive margin), `LAMBDA_SPARSE = 0.01`,
`LAMBDA_SUFF = 1.0`, `LAMBDA_NEC = 1.0`, `LAMBDA_CONTRAST = 0.5`, `N_MATCHED_CONTROLS = 100`.

## Public interface

- `from tcell_pipeline.rationale import RationaleHead, RationaleLoss, FaithfulnessTester,
  MatchedRandomSampler, RATIONALE_LABEL, complement, edge_index_of, edge_attr_of`
- Real-data smoke: `python src/tcell_pipeline/rationale/run_module4_smoke.py`

## Verification (synthetic tests + real-data smoke)

- `src/tests/test_rationale.py` (10 synthetic tests, dataless — dense random HeteroData): `imp ∈ [0,1]`,
  top-k sorted, sufficiency < matched-random, necessity > matched-random, matched-random size+relation
  match, `structural_ood_audit` (deleted-fraction vs an independent removed-PP count + component-count
  monotonicity), loss components computable + gradients, expression-only → empty rationale, label
  `predictive_rationale` not `causal`, and **faithfulness determinism under active DropEdge** (the
  eval-forcing regression check).
- Real-data smoke `run_module4_smoke.py`: real PPI graph, a real perturbation's neighbourhood — rationale
  extraction + fixed-model faithfulness (sufficiency < matched-random, necessity > matched-random) +
  structural-OOD audit, output labelled `predictive_rationale`. PASSED.
- `./init.sh` green at **79 tests** (69 prior + 10 new).
- **Perf:** this 64-core box thrashes torch's thread pool on the tiny per-subgraph GNN ops (2.5 s → 20 ms
  per encode); the CPU-only Module-4 tests + smoke pin `torch.set_num_threads(1)`.

## Post-review hardening (xhigh `/code-review`)

An xhigh workflow review of this diff produced 13 verified findings (3 refuted); all confirmed defects
were resolved, folded into the description above: **FaithfulnessTester forces eval on every deletion
re-run** (`@torch.no_grad` doesn't disable DropEdge — the fixed-model scores were stochastic on a
train-mode encoder; + a determinism regression test); `structural_ood_audit` sparsity is **PP-scoped** to
match its connectivity metrics; the tautological structural-audit test was replaced with an
independent-count + component-monotonicity check; `sufficiency`/`necessity` take an optional cached
`dz_full` via a public `delta_z()`; `_select` uses `torch.topk`; `_PP_RELATIONS` is imported (not a 4th
copy); `_pp_edges` is vectorised; the smoke uses the public API. Kept as spec-mandated: `RationaleHead`'s
`edge_attrs` param and `subgraph_edges` output (both in the §7 signature).

## Non-goals / ceiling markers

- **No training loop** (Stage B fit) here by design — the report freezes H1 first, then fits
  `L_rationale`; the loss + faithfulness machinery are in place for that loop, but the loop itself is
  feat-008's remaining work. The **Stage A** training loop that produces the frozen H1 checkpoint this
  head sits on top of is Module 5 — `docs/specs/2026-07-16-module5-training.md`.
- **MatchedRandomSampler matches per-relation count only** — full degree/connectivity/hop matching is the
  feat-012 audit's job.
- **structural_ood_audit anchors hop-distance at node 0** (`ponytail:` the audit signature carries no
  seed; before/after share the anchor, so the delta reflects deletion) — a per-target-seed eccentricity is
  a refinement for the feat-012 audit.
