# Session Handoff

## Current Objective

- Goal: Build the EG-IPG model for T cell perturbation response prediction
- Current status: **Module 0 + Module 1 encoder (feat-014) + real PLM/PINNACLE embeddings (feat-015) +
  Module 2 typed graph encoder (feat-016) + leakage-safe splits (feat-003) done.** feat-001, feat-002,
  feat-003, feat-004, feat-014, feat-015, feat-016 done. Next: feat-005 (programs) / feat-006 (baselines).
- Branch / commit: main. **feat-016 (Module 2) at `100a505`; feat-003 (leakage-safe splits) at
  `35e3999`; the xhigh `/code-review` fixes (Tier 1 + Tier 2 + all of Tier 3) at `7760624`.** Committed
  range this session: `100a505..7760624 (+ this handoff-sync commit)`. The review-fix commit touches `splits.py`, `graph_builder.py`,
  `typed_graph_encoder.py`, `test_graph.py`, `test_splits.py`, `config.py`, regenerated
  `data/splits/{leakage_report,manifest}.json` + `random.csv` (blocked CSV byte-identical),
  `feature_list.json` addenda, and new `docs/reviews/2026-07-15-code-review-*.md`.
  The two planning docs (report + walkthrough) got as-built notes but are gitignored (local-only).
  Latest committed is always `git log -1` on main.

## Completed This Session (post code-review fixes — feat-016 + feat-003)

Applied the verified findings from the xhigh `/code-review`
(`docs/reviews/2026-07-15-code-review-feat-016-feat-003.md`). **Committed at `7760624`.**

- **Tier 1 (feat-003 leakage-safety) — split CSVs byte-identical (sha256 unchanged), audit corrected:**
  - `splits.py` audit now publishes **cap-induced family splits** via an uncapped pre-cap component
    pass (`_precap_labels`). The old post-cap "no family group spans >1 role" assertion was blind to
    families the 5% cap *must* break: real data has `cap_induced_family_splits=1` (one giant
    single-linkage family), `family_challenge_sharing_train_frac=0.41` (an upper bound inflated by
    single-linkage chaining — the pairwise residual, 26.4%, is the true leakage).
  - `_sequence_residual` now centers train+challenge in **one global frame** (was per-subset means →
    mismatched frames understating similarity). Corrected effectiveness **53.8%→26.4% = 51% reduction**
    (was 53.5→28.1=47). `manifest.json` gains `sequence_block_active`, `n_genes_with_embedding`.
  - `run()` **fails closed** when PLM embeddings are absent (was silent fail-open publishing a
    sequence-leaky split as safe); override with `SPLITS_ALLOW_NO_SEQUENCE=1`.
- **Tier 2 (feat-016 active bugs):** `graph_builder` degree columns reordered to
  `[physical, functional, complex]` to match Module 1's `TARGET_SCALAR_KEYS`; `typed_graph_encoder`
  `encode_one` now moves `h_do` to the module device (was a device-mismatch crash on the public entry
  point); `test_graph.py` signed-message test seeded + the false `|out| < 1.0` bound dropped (relu is
  unbounded → ~13% flake).
- **Tier 3 (all addressed):**
  - Cheap defenses: `graph_builder` `nan_to_num` on edge features (symmetric with node features),
    `.dropna()` on gene symbols, fail-fast on unknown PPI source; dead config constants
    `N_RELATION_TYPES` / `RELATION_TYPES` / `SPLIT_AUDIT_HOPS` removed.
  - `#10` OOV `culture_condition` now raises a legible `ValueError` via `_condition_index` (still
    fail-fast — closed 3-value vocab, invalid input); the "never crash a batch" docstring narrowed to
    unknown *genes* only.
  - `#11` diagnostic random split uses cumulative-boundary allocation (last boundary == n, no truncated
    tail). `random.csv` regenerated; **`blocked_target_ood.csv` still byte-identical** to 35e3999 and the
    effectiveness numbers are unchanged (gene-level baseline doesn't shift at N=11525).
  - `#12` returned `edge_gates[rel]` is now length **E** (one per original edge) for *all* relations,
    aligned to the sub-graph `edge_index` (was 2E-doubled for PP — the mirror carried an identical gate).
    Full gate→(u,v) identity-forwarding API still **deferred to Module 4** (its consumer isn't built).
- **Regenerated** `data/splits/`; **57 pytest green** (+3 regression checks: OOV raises, edge_gates
  length == E, random split covers all items at small N).

## Completed This Session (feat-003 — leakage-safe splits)

Design brainstormed against the experiment-plan report; spec in
`docs/specs/2026-07-15-feat-003-leakage-safe-splits.md`. **The approved CC-over-3-axes design was
revised after empirical measurement proved it collapses** (naive connected-components → giant
components on every axis: physical 95%, complex 23%, ESM cos≥0.95 92%, Louvain 42%).

- [x] `src/tcell_pipeline/splits.py`: hard block = sequence/paralog family via **representative
  (non-chaining, CD-HIT-style) clustering on centered ESM-2 embeddings** (cos≥0.85 → 3.1% largest
  family) + CORUM co-membership, under a 5%-of-genes **capped union-find** (3986 giant merges refused).
  Physical-PPI neighbourhood is **audit-only** (95% one component — can't be a hard block; report G1 +
  Phase-1 6/9 want its distribution *published*, not zeroed).
- [x] **4-role** partition (train/val/calibration/challenge ~60/15/10/15; realized 62.5/13/7.9/16.6),
  assigned by whole family group, seeded, deficit-greedy. Random diagnostic split (row-level).
- [x] Frozen + hashed to **`data/splits/`** (git-tracked): `blocked_target_ood.csv`, `random.csv`,
  `manifest.json`, `leakage_report.json` (machine-readable: hard-asserts no family group split across
  roles; publishes per-axis train→challenge residual + fail-closed audit).
- [x] **Effectiveness validated** (numbers corrected in the post-review pass below): challenge genes
  with a ≥0.85 train paralog cut **53.8% (random) → 26.4% (blocked) = 51% reduction** (the ~26% floor is
  irreducible given dense ESM geometry). 8 synthetic tests (`test_splits.py`). `./init.sh`: **54 passed**.

## Completed This Session (feat-016 — Module 2 typed graph encoder)

New package `src/tcell_pipeline/graph/` (PyG torch_geometric 2.8), a component of feat-008 built ahead
(depends only on Module 0 outputs + Module 1's h_do):

- [x] `graph_builder.build_hetero_graph()` -> `(HeteroData, gene_to_idx)`: **25440** protein nodes keyed
  by upper-case HGNC, each carrying the **same frozen 1412-d descriptor as Module 1's TargetEncoder**
  (PLM 1280 + PINNACLE 128 + 3 graph-derived degrees + control_baseline_expr, zero-fallback);
  **5628** complex nodes (index-only, the learned `nn.Embedding` lives in the encoder). 4 relations
  split by the `is_*` flags + bipartite membership, each with an 8-d edge feature
  (source one-hot(5)|score|is_direct_binary|n_supporting). Real edge counts: physical_ppi 1123205,
  co_complex 48389, functional_assoc 6857702, complex_membership 18932.
- [x] `neighborhood_sampler.sample_subgraph()`: grows physical/co-complex first then score-fills, caps
  at **512** proteins, pulls in member complexes, returns an induced HeteroData preserving `orig_idx`.
- [x] `typed_graph_encoder.TypedGraphEncoder(nn.Module)`: 3-layer per-relation custom PyG
  `MessagePassing` (RGCNConv/GATConv can't express this) with **signed message**
  `tanh(W_sign h_u)*relu(W_mag h_u)` and **condition gate** `sigmoid(w_gate[h_cond(64)||f_e(8)])`
  computed once per relation (layer-independent) and returned as `edge_gates` for Module 4; residual
  FFN+LayerNorm per node type; DropEdge 0.1. `graph_readout.GraphReadout`: 4-head cross-attention
  (q=h_do, K=V=node states) -> **h_graph R^256**, attention sums to 1. `forward(target_genes,
  conditions, h_do)` loops per-target subgraphs; targets absent from the PPI graph -> zero h_graph.
- [x] **CPU and CUDA** (device-aware; sampled subgraphs moved to the module device). config: GRAPH_HOPS,
  NEIGHBORHOOD_CAP, GRAPH_HIDDEN_DIM, GRAPH_LAYERS, GRAPH_N_HEADS, EDGE_DROPOUT, EDGE_FEATURE_DIM,
  N_RELATION_TYPES, COMPLEX_EMBED_DIM, CONDITION_EMBED_DIM, RELATION_TYPES, PROTEIN_FEATURE_DIM.
- [x] Verified: **8** synthetic tests (`test_graph.py`) + `graph/run_module2_smoke.py` real-data smoke
  (full graph in ~18s, CD3E neighbourhood 512 proteins/740 complexes, real Module 1 h_do -> Module 2
  h_graph (4,256) finite on GPU, gates differ by condition, attention sums to 1). `./init.sh`: **46** passed.

## Completed Prior Session (feat-015 embeddings + GPU-native Module 1 encoder + real-data smoke)

The feat-014 encoder left both target-embedding stores at zero-fallback (no parquet on disk). This
session generated the real embeddings so the PerturbationEncoder runs on real target vectors:

- [x] `embeddings_plm.py`: real **ESM-2 650M** (1280-d, mean-pooled over final-layer residues, BOS/EOS/pad
  excluded). Sequences from the UniProt REST accessions endpoint (cached to `uniprot_sequences.parquet`);
  resumable (skip embedded, atomic checkpoint). **Device-aware** — ran on an A100 -> **11419/11419 mart
  proteins embedded** (100% PLM coverage), all finite.
- [x] `embeddings_pinnacle.py`: real **PINNACLE** (Li et al. 2024, Figshare article 22708126) contextual
  embeddings. Real dim is **128** — config's 512 was a placeholder, **corrected to 128**. Took the
  `cd4-positive helper t cell` context (the CD4+ screen's cell type; `config.PINNACLE_CONTEXT`); gene-symbol
  -> UniProt via id_mapping -> **1119 embeddings, 1070/11419 mart proteins covered** (contextual embeddings
  only span in-network proteins; the rest keep the zero fallback).
- [x] Live encoder dims now derive to target.out_dim **1412** (1280+128+4), fusion `Linear(1574->256)`,
  **404,960** trainable params (was 1796 / 503,264 under the 512 placeholder).
- [x] Tests rewritten to **real data/embeddings — no synthetic parquets** (10 tests in `test_encoders.py`):
  real PLM present-loaded + absent-id zero-fallback + dim-mismatch guard, real PINNACLE CD4-context load,
  forward/NaN tests on the real perturbation_condition + de_obs marts.
- [x] **GPU enabled**: host has 5x A100 80GB but the CUDA-12.2 driver can't run the default cu13x torch;
  swapped to `torch==2.13.0+cu126` (minor-version compat). requirements.txt documents the cu126 install.
- [x] Embedding artifacts are gitignored under `data/intermediate/`; regenerate via
  `python -m tcell_pipeline.embeddings_{plm,pinnacle}`.
- [x] **Module 1 encoder made device-aware (GPU-native)**: `ContextEncoder`/`PerturbationEncoder` forward
  move constructed tensors to the module's device, so `PerturbationEncoder().to('cuda')` runs the whole
  forward on GPU (TargetEncoder/QualityEncoder build CPU tensors that forward relocates). Tests default to
  CPU (portable); `test_encoder_runs_on_gpu_when_available` runs only when CUDA is present. Suite now **38**.
- [x] **`run_module1_smoke.py`** (NEW): full-mart real-data verification — drives all 33,983 rows through
  the encoder on GPU (~2s), asserts every h_do finite, checks the leakage fence rejects the mart's real
  q_post columns. Exits non-zero on any NaN/fence breach. The Module 1 analogue of `run_module0.py`.

Prior session (feat-014): `src/tcell_pipeline/encoders/` package — five nn.Modules fused into `h_do` R^256,
q_pre inputs only, no trainable gene-ID embedding, no free donor-ID embedding, leakage fence at the boundary,
NaN guard. Earlier: ~100 GB download, `examples/`, README, Module 0 + code-review fixes, UniProt/HuRI/CORUM.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Compile + tests | `./init.sh` | Pass | **57 passed** on torch cu126 (54 + 3 post-review regression checks); compileall clean |
| feat-003 split tests | `pytest src/tests/test_splits.py` | Pass | 8 passed; grouping/cap/no-split/determinism/audit-fail-closed/fractions |
| feat-003 real split | `python -m tcell_pipeline.splits` | Pass | 11525 genes → 5141 family groups (largest 5%); wrote data/splits/*; sequence leakage **26.4% (blocked) vs 53.8% (random) = 51% cut** (corrected global-frame residual); split CSVs byte-identical to the 35e3999 freeze |
| Module 2 graph tests | `pytest src/tests/test_graph.py` | Pass | 8 passed; synthetic graph (structure, 2-hop cap, condition gate differs, signed msg, forward finite, edge_gates, zero/absent target, attn sums to 1) |
| Module 2 real-data smoke | `python src/tcell_pipeline/graph/run_module2_smoke.py` | Pass | full 25440-node graph ~18s; CD3E nbhd 512 proteins; Module 1 h_do -> h_graph (4,256) finite on GPU; gates differ by condition; attn sums to 1 |
| Encoder tests (real data) | `pytest src/tests/test_encoders.py` | Pass | 10 passed; real PLM+PINNACLE parquets, real marts — no synthetic parquets |
| PLM generation (GPU) | `python -m tcell_pipeline.embeddings_plm` | Pass | 11419/11419 proteins, 1280-d, finite; A100, 100% util |
| PINNACLE ingestion | `python -m tcell_pipeline.embeddings_pinnacle` | Pass | 1119 embeddings (128-d), 1070/11419 mart coverage (CD4 helper context) |
| Encoder real-data e2e | head of perturbation_condition/de_obs -> PerturbationEncoder | Pass | h_do (8,256) finite; real PLM+PINNACLE vectors flow through |
| Module 1 full-mart smoke | `python src/tcell_pipeline/run_module1_smoke.py` | Pass | on GPU (cuda), 33,983 rows in ~2s; all finite; PLM 33796, PINNACLE 3135 coverage; q_post rejected |
| Module 0 full run (prior) | `python src/tcell_pipeline/run_module0.py` | Pass | all 7 steps on real data; 7.98M edges; leakage fence disjoint |

## Files Added (this session, feat-003 — leakage-safe splits)

- `src/tcell_pipeline/splits.py` (NEW), `src/tests/test_splits.py` (NEW, 8 tests)
- `docs/specs/2026-07-15-feat-003-leakage-safe-splits.md` (NEW): design doc (report-derived + empirical)
- `data/splits/{blocked_target_ood,random}.csv`, `{manifest,leakage_report}.json` (NEW, git-tracked frozen artifacts)
- `src/tcell_pipeline/config.py` — feat-003 constants (SPLITS_ROOT, SPLIT_ROLES/FRACTIONS/SEED, SEQ_SIM_COSINE_THRESHOLD, GROUP_SIZE_CAP, artifact paths)
- `feature_list.json` (feat-003 → done), `progress.md`, `session-handoff.md`

## Files Added (this session, feat-016 — Module 2)

- `src/tcell_pipeline/graph/{__init__,graph_builder,neighborhood_sampler,typed_graph_encoder,graph_readout,run_module2_smoke}.py` (NEW)
- `src/tests/test_graph.py` (NEW): 8 synthetic Module 2 tests
- `src/tcell_pipeline/config.py` — Module 2 constants (GRAPH_*, EDGE_*, N_RELATION_TYPES, COMPLEX/CONDITION_EMBED_DIM, RELATION_TYPES, PROTEIN_FEATURE_DIM)
- `feature_list.json` (feat-016 added, done), `progress.md`, `session-handoff.md`

## Files Changed (prior session, feat-015)

- `src/tcell_pipeline/embeddings_plm.py` (NEW): ESM-2 650M generator (resumable, GPU-aware)
- `src/tcell_pipeline/embeddings_pinnacle.py` (NEW): PINNACLE CD4-context -> UniProt mapper (Figshare download)
- `src/tcell_pipeline/config.py` — PINNACLE_EMBED_DIM 512->128; +PINNACLE_RAW_DIR/FIGSHARE_URL/CONTEXT
- `src/tests/test_encoders.py` — rewritten to real PLM+PINNACLE data (no synthetic parquets); 1796->1412
- `requirements.txt` — +fair-esm, +pyyaml (was undeclared), +cu126 torch install note
- `README.md` — GPU/cu126 setup note + "Precompute target embeddings" step; PINNACLE 128-d detail
- `src/tcell_pipeline/encoders/embedding_store.py` — docstring refresh (embeddings now generated)
- `src/tcell_pipeline/encoders/{context,perturbation}_encoder.py` — device-aware forward (runs on GPU when .to('cuda'))
- `src/tests/test_encoders.py` — +test_encoder_runs_on_gpu_when_available (skips without CUDA)
- `src/tcell_pipeline/run_module1_smoke.py` (NEW): full-mart real-data smoke, GPU-native (Module 1 analogue of run_module0.py)
- `feature_list.json` (feat-015 added, done), `progress.md`, `session-handoff.md`
- Prior session (feat-014): `src/tcell_pipeline/encoders/` package + config Module 1 constants + test_encoders.py

## Decisions Made

- Module 1 batch contract: `PerturbationEncoder.forward` takes a dict with keys `uniprot_id` (list),
  `ppi_degree_physical/functional/complex`, `control_baseline_expr`, `culture_condition` (str names or
  long indices), `donor_pc` (a single (B,32) tensor — loader stacks donor_pc_00..31), `n_guides`,
  `single_guide_estimate`. Any q_post key raises. The Module 3 data loader builds this dict.
- **Embeddings are real (feat-015)**: PLM = ESM-2 650M (1280-d, mean-pooled), 100% mart coverage;
  PINNACLE = real published 128-d contextual vectors (`cd4-positive helper t cell`, config.PINNACLE_CONTEXT),
  1070/11419 coverage. Frozen + pluggable; artifacts gitignored, regenerate via the two embeddings_* modules.
- **GPU**: use `torch==2.13.0+cu126` on this host (CUDA-12.2 driver can't run the default cu13x wheel); the
  5x A100s are otherwise invisible to torch. Embedding generation AND the encoder run on GPU — the encoder
  is device-aware (`PerturbationEncoder().to('cuda')`); TargetEncoder/QualityEncoder build CPU tensors that
  forward moves to the fusion's device. Tests default to CPU (portable); the GPU test runs only when CUDA is present.
- UniProt: reviewed-canonical pick; flag only equal-evidence ties; gene is the perturbation unit
- CORUM host has a broken TLS chain -> per-source verify skip for `corum` only
- Data scope: aggregate layer only; donor key = physical CE codes; controls from pseudobulk
- Near-null-signal regime (2026-07-14): confirm above-mean signal before freezing H1; negative result is valid
- Stable-Shift (feat-010): first-party code unconfirmed; plan a row-compatible reimplementation

## Blockers / Risks

- `data/raw` ~101 GB near the 105 GiB soft cap; feat-015 added PINNACLE raw (~1.3 GB) + PLM embeddings
  parquet (~58 MB) + uniprot sequence cache — watch disk before feat-005
- Near-null-signal regime: H1 superiority not guaranteed on this CD4+ screen
- (Resolved this session: HuRI + CORUM downloads; id_mapping UniProt/Entrez online pass)

## Next Session Startup

1. Read `AGENTS.md`.
2. Read `feature_list.json` and `progress.md`.
3. Review this handoff.
4. Run `./init.sh` before editing.

## Recommended Next Step

- Post-review fixes are committed (`7760624`); working tree is clean. Start **feat-005 (latent program
  extraction)** and/or **feat-006 (simple baselines)** — both
  depend only on feat-003 (done) and consume the frozen `data/splits/`. Fold-local fits (programs,
  scaling) must use the **train** role only; the loader reads `load_split()` and filters by role. Before
  freezing H1, run the near-null-signal check on development data.
- feat-003 calibration knob left open (`docs/specs` + leakage_report.json): the centered-cosine threshold
  (0.85) and 5% cap can be tuned on the published paralog-similarity distribution; the sequence residual
  is 26.4% (vs 53.8% random) — tighten via pairwise must-links or curated families if a downstream result
  needs it. The report also now surfaces `cap_induced_family_splits` (the 5%-cap must break any family
  bigger than one role's budget) — tightening the cap trades partition balance against that residual.
