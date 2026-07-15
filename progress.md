# Session Progress Log

## Current State

**Last Updated:** 2026-07-15 (Module 2 typed graph encoder done; feat-016)
**Active Feature:** feat-016 Typed Graph Encoder (Module 2) — **DONE**. Next: feat-003 (leakage-safe splits)

## Status

### What's Done

- [x] feat-001 Environment & Data Download — **DONE**
  - Env imports OK (anndata 0.13.1, mudata 0.3.10, h5py 3.16.0)
  - Aggregate layer downloaded to `data/raw/` (~101 GB): 4 HDF5 + 15 suppl tables + 12 jsonld
  - Cell-level files intentionally excluded (storage-blocked)
- [x] feat-002 Data Inspection & ID Mapping — **DONE**
  - `examples/` inspectors; `src/tcell_pipeline/id_mapping.py`
  - Ran online on real DE: 12311 unique Ensembl (11526 targets / 10282 measured / 9497 both), all HGNC
    resolved, UniProt/Entrez filled via mygene.info
  - One-to-many UniProt DISAMBIGUATED (reviewed-canonical strategy, see Decisions): 33 multi-accession
    genes -> 23 resolved to a confident canonical, 10 genuine multi-product loci flagged
    `uniprot_ambiguous` with all candidates kept in `uniprot_alternatives`
- [x] feat-004 PPI Graph Construction — **DONE**
  - `ppi_graph.py` typed-edge harmonizer + `complex_membership.py` CORUM bipartite membership
  - All 5 sources fetched + merged on real data -> `data/graphs/protein_edges.parquet`:
    **7,980,907 edges** (bioplex 118162, huri 52256, biogrid 1218142, string 13715404, corum 77696)
  - `complex_membership.parquet`: 18,932 memberships / 5,628 complexes (CORUM 5.3)
  - `ppi_degree_*` computed from the graph into perturbation_condition

- [x] **Module 0 data pipeline** (`src/tcell_pipeline/`, 9 modules + `run_module0.py`) ran end-to-end on
  real data. Derived marts written under `data/intermediate/`, `data/graphs/`, `data/manifests/`
  (all gitignored): id_mapping, DE layers (zscore/log_fc sparse NPZ; neglog10_p_value/neglog10_adj_p_value/
  baseMean/lfcSE dense NPY), de_obs/de_var, protein_edges, complex_membership, perturbation_condition
  (33983 rows; 187 without UniProt; 32425/33983 with a control baseline), control baseline + donor
  profiles, feature_availability.yaml (**q_pre=43 / q_post=13 / metadata=2, leakage fence disjoint**).
- [x] **This session's fixes** (commits e453964..eab027e):
  - UniProt disambiguation via reviewed-canonical strategy (`choose_uniprot`); `uniprot_ambiguous` flags
    only equal-evidence ties (10 loci: CDKN2A p16/p14ARF, GNAS, MOCS2, TMPO...)
  - HuRI download: apex host `interactome-atlas.org` (cert invalid for the `www.` subdomain)
  - CORUM download: migrated to CORUM 5.3 fastapi endpoint (old `coreComplexes.txt.zip` path gone);
    handles new `subunits_gene_name` schema; per-source TLS-verify skip for the broken helmholtz cert chain
  - feature_availability: `KNOWN_METADATA_COLS` allowlist so the leakage-fence REVIEW warning fires only
    on genuinely-unexpected metadata (row_index/mapping_status no longer cry wolf)
  - `mygene` added to requirements.txt
- [x] 23 pytest tests in `src/tests/` (synthetic fixtures; added test_control_profiles.py,
  test_complex_membership.py); `init.sh` green (compileall + pytest)

- [x] **Module 1 Perturbation & Context Encoder** (feat-014, `src/tcell_pipeline/encoders/`) — **DONE**
  - `PluggableEmbeddingStore` (frozen PLM 1280 / PINNACLE 512 by UniProt; zero-fallback when the
    parquet is absent, in-memory cache, NOT an nn.Module), `TargetEncoder` (no trainable gene-ID
    embedding; h_target R^1796), `ContextEncoder` (trainable condition Embedding(3,64) + donor PCs
    through Linear(32,32), no free donor-ID embedding), `QualityEncoder` (n_guides +
    single_guide_estimate + zeros(64) guide-seq placeholder; h_quality R^66), `PerturbationEncoder`
    (fusion Linear(1958->256)+LayerNorm -> h_do R^256; rejects q_post cols at the boundary).
  - 503,264 trainable params, CPU-only, batch-first. Config: PLM_EMBED_DIM/PINNACLE_EMBED_DIM/
    GUIDE_SEQ_EMBED_DIM/H_DO_DIM/CONDITIONS/PLM_EMBEDDINGS_PATH/PINNACLE_EMBEDDINGS_PATH.
  - NaN guard (`as_float_vector` nan_to_num): missing control_baseline_expr (1558/33983) and
    n_guides can't poison the LayerNorm'd h_do. Upgrade path = fold-fit imputation in Module 3 loader.
  - 10 tests (test_encoders.py) + real-data smoke on perturbation_condition/de_obs -> finite (4,256).

- [x] **Module 1 real embedding ingestion** (feat-015, `embeddings_plm.py` + `embeddings_pinnacle.py`) — **DONE**
  - `embeddings_plm.py`: real **ESM-2 650M** (1280-d, mean-pooled), UniProt-REST sequences, resumable,
    **device-aware (GPU)**. Ran on an A100 -> **11419/11419 mart proteins embedded** (100% PLM coverage), finite.
  - `embeddings_pinnacle.py`: real **PINNACLE** (Figshare 22708126) `cd4-positive helper t cell` context.
    Real dim is **128** (config placeholder was 512 -> **corrected to 128**). Gene-symbol->UniProt via id_mapping;
    **1119 embeddings, 1070/11419 mart proteins covered** (contextual — rest keep zero fallback).
  - Live encoder dims now: target.out_dim **1412** (1280+128+4), fusion Linear(1574->256), **404,960** params.
  - Tests rewritten to **real data/embeddings — no synthetic parquets** (still 10 in test_encoders.py).
  - **GPU enabled**: swapped torch cu130->**cu126** (host driver is CUDA 12.2; cu13x can't see the 5x A100s).
    requirements.txt: +fair-esm, +cu126 install note.

- [x] **Module 2 Typed Graph Encoder** (feat-016, `src/tcell_pipeline/graph/`) — **DONE**
  - `graph_builder.build_hetero_graph` -> PyG HeteroData + gene_to_idx: 25440 protein nodes
    (frozen 1412-d TargetEncoder descriptor, graph-derived degrees, zero-fallback) + 5628 complex
    nodes (index-only, learned embedding in the encoder); 4 relations (physical_ppi 1123205 /
    co_complex 48389 / functional_assoc 6857702 / complex_membership 18932) with 8-d edge features.
  - `neighborhood_sampler.sample_subgraph`: physical/co-complex-first then score-fill, cap 512
    proteins + member complexes, induced HeteroData preserving orig_idx.
  - `typed_graph_encoder.TypedGraphEncoder`: 3-layer per-relation MessagePassing with signed
    message `tanh(W_sign h_u)*relu(W_mag h_u)`, condition gate `sigmoid(w_gate[h_cond||f_e])`
    computed once/relation and returned as `edge_gates` for Module 4, residual FFN+LayerNorm,
    DropEdge 0.1. `graph_readout.GraphReadout`: 4-head cross-attention (q=h_do) -> h_graph R^256.
  - CPU **and** CUDA (device-aware). 8 synthetic tests (`test_graph.py`) + real-data smoke
    (`graph/run_module2_smoke.py`): full graph in ~18s, CD3E neighbourhood, Module 1 h_do ->
    Module 2 h_graph (4,256) finite on GPU, gates differ by condition, attention sums to 1.

### What's In Progress

- (none — feat-014 + feat-015 + feat-016 closed)

### What's Next

1. feat-003 Leakage-Safe Train/Val/Test Splits — gene-family / protein-complex / graph-neighborhood
   blocking, hash + freeze split files. Inputs now all present: id_mapping, protein_edges,
   complex_membership, perturbation_condition.
2. Optional: near-null-signal check on development data before freezing H1 (2026-07-14 finding).

## Blockers / Risks

- [ ] `data/raw` ~101 GB near the 105 GiB soft cap; derived marts now also on disk (protein_edges ~35 MB,
  DE layers, control profiles) — watch disk before feat-005 program bases land
- [ ] **Near-null-signal regime (2026-07-14 finding):** this CD4+ screen may be near-null-signal (models
  barely beat the mean). Confirm a detectable above-mean signal before freezing H1; a rigorous negative
  benchmark is a valid outcome.
- [x] RESOLVED: HuRI + CORUM downloads (both source URLs migrated this session)
- [x] RESOLVED: id_mapping UniProt/Entrez (online mygene pass done; only 6 no-hit genes remain, HGNC-resolved)

## Decisions Made

- **UniProt disambiguation**: pick the gene's reviewed human canonical (UniProt REST gene_exact+reviewed)
  by annotation-score then lexical; flag only equal-evidence ties; keep the gene as the perturbation unit
  (CRISPRi knocks down the whole locus) — no forced single-protein pick, alternatives preserved
- **CORUM host**: broken TLS chain (certifi also fails) -> per-source verify skip for `corum` only
- Data scope: aggregate layer only; cell-level (~1.6 TiB) excluded
- Donor key = physical CE codes; independent NTC controls come from pseudobulk (DE has none)
- Distributional metrics: do not use Wasserstein/Energy distance as a sole headline metric
- Stable-Shift (feat-010): first-party code unconfirmed; plan a row-compatible reimplementation
- **Embeddings (feat-015)**: PLM = real ESM-2 650M (1280-d, mean-pooled); PINNACLE = real published
  128-d contextual vectors (NOT the 512 placeholder), `cd4-positive helper t cell` context to match the
  CD4+ screen (configurable via config.PINNACLE_CONTEXT). Frozen features; artifacts gitignored + regenerable.
- **GPU**: host has 5x A100 80GB but the CUDA-12.2 driver can't run the default cu13x torch; use the
  cu126 build (`torch==2.13.0+cu126`, minor-version compat). Embedding generation runs on GPU.
- **Module 2 (feat-016)**: protein node features reuse Module 1's frozen 1412-d TargetEncoder
  descriptor (degrees recomputed from the graph so all 25440 nodes have them); complex nodes are
  index-only (learned embedding lives in the encoder). Custom PyG `MessagePassing` per relation —
  RGCNConv/GATConv can't express the signed `tanh*relu` message or the condition gate. Condition
  gate `alpha` depends only on `h_cond` + edge features (not `h_u`), so it's computed once and
  reused across layers and returned as `edge_gates`. ponytail: per-target subgraph loop in forward
  (upgrade to PyG mini-batching if Module 3 graph-encode throughput demands it).

## Files Added This Session (feat-016 — Module 2 typed graph encoder)

- `src/tcell_pipeline/graph/__init__.py` (NEW)
- `src/tcell_pipeline/graph/graph_builder.py` (NEW): `build_hetero_graph()` -> HeteroData + gene_to_idx
- `src/tcell_pipeline/graph/neighborhood_sampler.py` (NEW): `sample_subgraph()`
- `src/tcell_pipeline/graph/typed_graph_encoder.py` (NEW): `TypedGraphEncoder` + signed message + condition gate
- `src/tcell_pipeline/graph/graph_readout.py` (NEW): `GraphReadout` cross-attention
- `src/tcell_pipeline/graph/run_module2_smoke.py` (NEW): real-data smoke (build graph, CD3E, encode)
- `src/tests/test_graph.py` (NEW): 8 synthetic Module 2 tests
- `src/tcell_pipeline/config.py` — Module 2 constants (GRAPH_*, EDGE_*, N_RELATION_TYPES, PROTEIN_FEATURE_DIM, ...)
- `feature_list.json` — feat-016 added, status done; `progress.md`, `session-handoff.md` — state sync

## Files Modified Prior Session (feat-015 — real embeddings + GPU)

- `src/tcell_pipeline/embeddings_plm.py` (NEW): ESM-2 650M generator (resumable, GPU-aware)
- `src/tcell_pipeline/embeddings_pinnacle.py` (NEW): PINNACLE CD4-context -> UniProt mapper (Figshare download)
- `src/tcell_pipeline/config.py` — PINNACLE_EMBED_DIM 512->128; +PINNACLE_RAW_DIR/FIGSHARE_URL/CONTEXT
- `src/tcell_pipeline/run_module1_smoke.py` (NEW): full-mart real-data smoke (33,983 rows, GPU-native, Module 1 analogue of run_module0.py)
- `src/tcell_pipeline/encoders/{context,perturbation}_encoder.py` — device-aware forward (encoder runs on GPU via .to('cuda'))
- `src/tests/test_encoders.py` — rewritten to real PLM+PINNACLE data (no synthetic parquets); dim literals 1796->1412
- `requirements.txt` — +fair-esm, +cu126 torch install note
- `feature_list.json` — feat-015 added, status done
- `progress.md`, `session-handoff.md` — state sync

### Prior session (feat-014 — Module 1 encoder)

- `src/tcell_pipeline/encoders/` (NEW package): `_tensor.py`, `embedding_store.py`, `target_encoder.py`,
  `context_encoder.py`, `quality_encoder.py`, `perturbation_encoder.py`, `__init__.py`
- `src/tcell_pipeline/config.py` — Module 1 constants; `src/tests/test_encoders.py` (10 tests); feat-014 done
- Post-review leakage-fence hardening (xhigh /code-review, 3 CONFIRMED/PLAUSIBLE latent findings):
  `feature_availability.py` — `_is_donor_pc` tightens the bare `donor_pc_` prefix to digits-only, and
  `_assert_disjoint_fence()` makes `classify_columns` REFUSE at runtime when a name is in both
  Q_PRE_COLS and Q_POST_COLS (previously the output-level disjointness test was a tautology that
  couldn't catch it). `test_feature_availability.py` — config disjointness pin, behavioral raise-on-
  overlap test, donor-prefix test, committed-manifest drift guard. pytest now **37 total**.

Prior session (Module 0 fixes, commits e453964..1732def): id_mapping.py reviewed-canonical UniProt
disambiguation; ppi_graph.py HuRI apex URL + CORUM 5.3 fastapi + `_corum_gene_col` + TLS skip;
complex_membership.py CORUM schema; feature_availability.py + config.py KNOWN_METADATA_COLS allowlist;
requirements.txt `mygene`; AGENTS.md session-handoff.md first-class artifact.

## Notes for Next Session

- `examples/` scripts double as data-understanding docs
- Read the experiment plan report for detailed feature specs (2026-07-14 literature refresh)
- Before feat-011 screening / freezing H1, run the near-null-signal check
- Module 0 marts are on disk but gitignored; rerun `python src/tcell_pipeline/run_module0.py` to regenerate
