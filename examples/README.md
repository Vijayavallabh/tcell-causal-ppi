# Data inspection examples

One self-contained script per artifact in `data/raw/`. Each uses backed/lazy reads
(no full densification), prints the relevant structure + distributions, and asserts
the known-good dims as a self-check. Override the data location with `DATA_ROOT`.

```bash
python examples/dataset_overview.py        # provenance + local-vs-expected inventory
python examples/inspect_de_stats.py        # GWCD4i.DE_stats.h5ad
python examples/inspect_pseudobulk.py      # GWCD4i.pseudobulk_merged.h5ad
python examples/inspect_by_guide.py        # GWCD4i.DE_stats.by_guide.h5mu
python examples/inspect_by_donors.py       # GWCD4i.DE_stats.by_donors.h5mu
python examples/inspect_suppl_tables.py    # 3 core experiment-design tables
python examples/inspect_analysis_tables.py # derived analysis tables (program anchors, etc.)
python examples/inspect_metadata_jsonld.py # metadata/*.jsonld (Croissant)
```

`inspect_suppl_tables.py` covers the 3 tables describing the screen (donors, guides, DE summary);
`inspect_analysis_tables.py` covers the derived biological tables and maps each to its role in the
EG-IPG method (program anchors, regulator→program edges, complex/cluster priors, guide-KD control
weighting, cross-cell reference).

## Dataset provenance

From the [VCP dataset card](https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq):
**MIT license** · v1.0.0, released 2025-12-22 · [preprint](https://www.biorxiv.org/content/10.64898/2025.12.23.696273v1) ·
S3 `s3://genome-scale-tcell-perturb-seq/marson2025_data/` · access via `vcp data search "Primary Human CD4+ T Cell Perturb-seq" --exact`.
`dataset_overview.py` prints the full provenance and the fetch commands for missing artifacts.

## Headline facts (verified against the files, not the README)

| Artifact | Shape | Storage note |
|---|---|---|
| `DE_stats.h5ad` | 33,983 × 10,282 | `.X` empty; 6 **dense float64** layers. zscore in fp32 ≈ 1.4 GB → fits RAM |
| `pseudobulk_merged.h5ad` | 278,684 × 18,129 | `.X` CSR, 2.78B nnz (~55% dense); **only source of NTC controls** (11,018) |
| `by_guide.h5mu` | guide_1 33,488 · guide_2 26,078 (×10,282) | 7,410 single-guide targets absent from guide_2 |
| `by_donors.h5mu` | 6 pairs × ~4,880 × **10,273** | 9 fewer genes than main DE; join on gene_ids |

Gotchas the scripts encode:

- **DE has no control rows** — it is already a per-perturbation contrast. Independent
  control centroids for shared-control-bias-safe metrics must come from pseudobulk.
- **q_post leakage fields** live in DE `.obs` (`ontarget_significant`, `guide/donor_correlation_*`,
  off-target flags, `n_downstream`) — never feed to the H1 predictor.
- **Guide target curation:** 4.9% of guides have designed ≠ validated target; join on `target_gene_id`.
- **Donor key:** use physical CE codes (4 donors). The D#↔CE mapping is consistent
  across runs (R1 = 2-donor batch, R2 = 4-donor batch).
- **The DE CSV is a stale schema** (lacks correlation cols) — prefer the `.h5ad` `.obs`.
- **Supplementary tables:** only 3 are on S3; the rest (guide-KD efficiency, aging /
  Th1-Th2 signatures, regulator coefficients, downstream clustering, K562) come from the
  GitHub analysis repo — see the two-step fetch in the project README. Two tables named in
  the data README (`QC_summaries_per_sample_lane.csv`, `Th1Th2_validation_summary.suppl_table.csv`)
  are not published on S3 or GitHub and cannot be fetched.
- **Cell-level `.h5ad` files (~1.58 TiB) are not downloaded**; only Croissant descriptors
  remain, with S3 URLs and placeholder (all-zero) md5s.

## Analysis tables → EG-IPG method role

Ready-made biological supervision now in `data/raw/suppl_tables/` (see `inspect_analysis_tables.py`):

| Table | Role in the method |
|---|---|
| `CD4T_aging_signature`, `Th2_Th1_polarization_signature`, `IL10IL21bulkRNAseq` | Curated **program anchors** (aging, Th1/Th2, cytokine) for program-level targets |
| `aging/polarization_..._regulator_coefficients`, `clustering_downstream_genes` (1.18M rows) | **Regulator→program** edge supervision (per-context weights; sign-coherent downstream genes) |
| `clustering_results_and_annotations` (112 clusters ↔ CORUM/STRING/KEGG/Reactome), `cluster_autoimmune_enrichment` | **Complex/cluster priors** + biological-alignment / trait-relevance metrics |
| `guide_kd_efficiency` (per-guide KD vs NTC) | **Control weighting** — a `q_post` confidence/QC source, never an H1 input |
| `K562_comparison` | **Cross-cell-type** generalization reference (CD4 vs K562) |
