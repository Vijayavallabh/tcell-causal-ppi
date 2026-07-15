# feat-003 — Leakage-Safe Train/Val/Test Splits (design)

Date: 2026-07-15 · Depends on: feat-002 (id mapping), feat-004 (PPI graph + complexes),
feat-015 (ESM-2 embeddings). Consumers: feat-005 (programs), feat-006 (baselines).

Design source: `perturbation_informed_causal_protein_program_graphs_report.md` §Training Splits,
§Phase 1 (Data, Leakage, and Split Lock), §Experimental Units and Partitions. Where the report
and the feature name disagree, the report wins (it is the more-recent authoritative plan).

## Purpose

Partition the perturbation targets so that no held-out challenge gene can be predicted by having
seen a biologically near-identical training gene. The report requires the headline target-OOD
split to block **sequence/paralog**, **complex/pathway**, and **close graph-neighborhood**
similarity simultaneously (rule 517, §Phase-1 step 5). All response-derived transforms
(programs, scaling, feature selection) are fit inside training folds only — the split is what
makes that enforceable downstream.

## Scope (report-justified)

**Build now:**
- **Biologically-blocked target-OOD split** — the confirmatory headline. 4 roles.
- **Random diagnostic split** — explicitly required as the sanity baseline (§Training Splits 1;
  "diagnostic only, not a publishable headline").

**Deferred (with reason):**
- Held-out **condition** — "exploratory" (only 3 contexts); trivial leave-one-condition-out, added
  when a consumer needs it.
- Held-out **donor** — "descriptive" only, and DE rows are donor-aggregated; a real donor split
  lives on the donor-pair MuData layer, which is not built.
- **Low-degree / off-target / joint-OOD** — stress *strata* (a tag on the challenge set), not
  separate partitions; derivable from `ppi_degree_*` / q_post flags later.
- **DataSAIL** dependency — connected-components blocking is exact, transparent, and independently
  auditable (which the report permits), so no heavy external splitter is added.

The module is built so a condition/low-degree family is a few lines to add.

## Empirical reality (measured on the real marts — this reshaped the algorithm)

Naive connected-components (single-linkage) over any of the three axes **collapses** because dense
similarity neighborhoods chain transitively into a giant component that cannot be split:

| Axis (over 11,525 target genes) | Largest connected component |
|---|---|
| physical_ppi 1-hop (476k edges) | **10,893 (95%)** — one hairball |
| complex co-membership / co_complex | **2,632 (23%)** — overlapping complexes chain |
| ESM cosine ≥0.95 (raw) | 92% · ≥0.97 → 72% · only ≥0.99 → 14% |
| ESM **centered** cosine ≥0.9 (single-link) | 15% — still chains |
| Louvain community (all axes) | **42%** — physical edges dominate modularity |
| **ESM centered, representative (CD-HIT-style) clustering ≥0.85** | **360 (3.1%)** ✓ splittable |

Two decisions follow, both **report-endorsed** (G1 gate §1470: complex OOD is primary only with
adequate independent clusters, else fall back to family/sequence-blocked; §1136/§762: cluster
sequence/paralog similarity, then *separately* block complexes and *separately stress* PPI
neighborhoods; §6/§9: *publish* the neighborhood similarity distribution):

1. **The hard partition block is the sequence/paralog family axis**, built by **representative
   (non-chaining) clustering** on **centered** ESM embeddings — not connected-components.
2. **Physical-PPI neighborhood is an audit/stress axis, never a partition constraint** (95% of
   targets are in one physical component; blocking it as a partition is impossible). Complex is a
   *capped* merge into the family groups (its 23% supercluster is "inadequate resolution" per G1).

## The two forks the report resolves

- **Family/sequence blocker = external sequence/paralog similarity** (§Phase-1 step 5, §758) via
  the feat-015 ESM-2 embeddings — but **centered + representative-clustered**, since raw cosine + CC
  is a hairball (table above). External, non-response-derived, no new download.
- **Partition = 4 roles**, not 3 (§Phase-1 step 3, §1471): `train` / `val` (model-selection) /
  `calibration` (conformal only) / `challenge` (sequestered "test"). Target fractions
  **~60 / 15 / 10 / 15** by unique eligible group.

## Architecture — `src/tcell_pipeline/splits.py`

Unit of generalization = **target gene** (§1466); all 3 conditions of a gene share one role.

1. **Family groups via size-capped union-find** over must-link edges, applied in priority order:
   - **(a) sequence/paralog** — representative (CD-HIT-style, non-chaining) clustering on
     **centered, L2-normalized** ESM-2 embeddings at cosine ≥ `SEQ_SIM_COSINE_THRESHOLD` (0.85):
     iterate genes in a fixed order, each unassigned gene opens a cluster of all unassigned genes
     within τ of it. Non-chaining → largest family 3.1%.
   - **(b) complex** — direct CORUM co-membership pairs (`complex_membership.parquet`) merged into
     the family groups.
   - A merge is **refused if it would push a component over `GROUP_SIZE_CAP`** (0.05 = 5% of genes),
     preventing any giant group; refused pairs are recorded as residual for the audit. Physical PPI
     is **not** a merge input.
   Genes with no embedding and no complex are singleton groups (low/zero-degree targets, which the
   report wants represented in the challenge set).
2. **Assign whole components to the 4 roles**, seeded, size-proportional greedy (largest first,
   each to the currently most-under-target role). Realized fractions + per-role condition balance
   reported (not hard-constrained; effect-size is q_post — diagnostic only, never a grouping input).
3. **Random diagnostic split** — row-level, seeded, same 4 roles.
4. **Freeze + hash + audit** → `data/splits/` (git-tracked; `.gitignore` already whitelists it):
   - `blocked_target_ood.csv`, `random.csv` — `hgnc_symbol, role` / `row_index, role`.
   - `manifest.json` — seed, thresholds, cap, fractions (target + realized), per-file sha256,
     group/gene counts.
   - `leakage_report.json` (§Phase-1 6/9, §331) — **hard-asserts** no union-find component is split
     across roles, and **publishes** per-axis (sequence, complex, physical-neighbourhood) the max &
     percentile train↔challenge similarity/overlap + count of residual cross-role blocked pairs.
     The neighbourhood distribution is reported, not zeroed (it can't be — the hairball).

## Config additions (`config.py`)

`SPLITS_ROOT = DATA_DIR / "splits"`, `SPLIT_ROLES`, `SPLIT_FRACTIONS` (60/15/10/15),
`SPLIT_SEED = 0`, `SEQ_SIM_COSINE_THRESHOLD = 0.85` (**centered** ESM cosine; measured to give a
3.1% largest family — a tuning knob the leakage report calibrates), `GROUP_SIZE_CAP = 0.05`,
`SPLIT_AUDIT_HOPS = 1` (physical-neighbourhood audit radius), path constants for the artifacts.

## Public interface

- `family_groups(genes, embeddings, complex_df, cap) -> labels` (capped union-find; component id per gene)
- `assign_partitions(labels, genes, fractions, seed) -> dict[gene -> role]`
- `audit_leakage(assignments, edges_df, complex_df, embeddings) -> dict` (fails closed if any
  component is split across roles; publishes per-axis residual distributions)
- `run() -> None` (loads Module 0 marts + embeddings, writes the three artifacts, prints summary)
- `load_split(name) -> pd.DataFrame` (reader for feat-005/006)

## Tests (synthetic, `src/tests/test_splits.py`)

- **Blocking correctness:** genes in the same family group (shared complex / high-cosine paralog)
  never land in different roles.
- **Size cap:** no family group exceeds the cap; a would-be giant merge is refused.
- **Fractions:** realized role fractions ≈ target within tolerance on a synthetic gene set.
- **Determinism:** same seed → identical assignment hash; different seed → different assignment.
- **Leakage audit** passes on the produced split and **fails closed** on a hand-injected
  component-split leak; the physical-neighbourhood residual is *reported*, not asserted zero.
- **Singletons** (no-embedding, no-complex genes) distribute across roles.
- **q_pre only:** no q_post column is consulted when building groups.

## Non-goals / ceiling markers

- Physical-neighbourhood blocking is **audit-only** (`ponytail:` — the 95% hairball makes it an
  unenforceable partition constraint; the report asks only that its distribution be published).
- Centered-cosine threshold + cap are calibration knobs, not learned (`ponytail:` — tune on the
  published paralog-similarity distribution).
- Greedy size-proportional assignment, not optimal balanced partitioning (`ponytail:` — exact
  balance / DataSAIL only if realized fractions drift too far).
- Curated paralog/family download (Ensembl/HGNC) deliberately skipped — centered representative
  ESM clustering gives clean families with no new dependency (`ponytail:` — swap in curated
  families if the embedding proxy proves too coarse).
