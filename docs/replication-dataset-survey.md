# Replication dataset survey — can the EG-IPG null be tested outside the Marson CD4 screen?

Written 2026-08-03, before any replication GPU work, as required by the campaign goal spec.

**Method.** Every number below was read from the actual data file, not from the paper describing it.
The scPerturb harmonised `.h5ad` files were downloaded from Zenodo record
[7041849](https://zenodo.org/records/7041849) (CC-BY-4.0, 26.9 GB total) into `data/raw/scperturb/`
and opened with `anndata.read_h5ad(..., backed="r")`; checksums are in
`data/raw/scperturb/SHA256SUMS.txt`. Where a paper claim and the file disagree, the file wins and the
disagreement is noted. This matters: two secondary sources consulted first gave different accounts of
the scPerturb column standard, and a third asserted a column set the files do not have.

---

## 1. What a replication has to match

The reference dataset (Zhu et al. 2025 genome-scale CD4+ T-cell Perturb-seq) supplies, after Module 0:

| Property | Value |
|---|---|
| DE rows (perturbation x condition) | 33,983 |
| Distinct gene targets | **11,526** |
| Experimental conditions | **3** (`Rest`, `Stim8hr`, `Stim48hr`) |
| Measured genes | 10,282 |
| Sequence-similarity family groups (0.75/0.15 split) | 3,282 |

Two of these are load-bearing for the experimental design, and they pull in opposite directions:

- **Target count** drives the *blocked target-OOD split*. The split holds out whole
  sequence-similarity families, so the number of targets sets how many families exist to hold out,
  and therefore the effective sample size of every contrast.
- **Condition count** drives the *condition gate*. `condition_gated` learns
  `nn.Embedding(len(config.CONDITIONS), ...)` and indexes it per row
  (`typed_graph_encoder.py:164,228`). With one condition the embedding is a constant and the arm
  degenerates to `typed_static`, so the h1 contrast stops being a test of anything.

---

## 2. The candidates, as measured

All six were opened and counted directly. "Single targets" excludes controls; Norman's
double perturbations (`GENE1_GENE2`) are counted separately because they are not single-gene targets.
"Targets in ref" is the overlap with the 11,526 reference targets. "Gene overlap" is against the
reference's 10,282 measured genes.

| Dataset | Cells | Measured genes | Single targets | Doubles | Conditions | Condition values | Targets in ref | Gene overlap | Cell type |
|---|---|---|---|---|---|---|---|---|---|
| ShifrutMarson2018 | 52,236 | 33,694 | **21** | 0 | **2** | Stim, NoStim | 18 | 9,777 (95.1%) | **primary human T cells** |
| DatlingerBock2017 | 5,905 | 36,722 | **32** | 0 | **2** | stimulated, unstimulated | 31 | 9,579 (93.2%) | Jurkat (T-cell line) |
| FrangiehIzar2021 | 218,331 | 23,712 | **248** | 0 | **3** | Control, IFNg, Co-culture | 179 | 9,953 (96.8%) | melanocytes (melanoma) |
| PapalexiSatija2021 | 20,729 | 18,649 | 25 | 0 | 1 | — | 24 | 9,338 (90.8%) | THP-1 monocytes |
| NormanWeissman2019 | 111,445 | 33,694 | 105 | 131 | 1 | — | 87 | 9,777 (95.1%) | K562 lymphoblasts |
| ReplogleWeissman2022 (RPE1) | 247,914 | 8,749 | **2,393** | 0 | 1 | — | 1,337 | 6,924 (67.3%) | RPE1 epithelial |

The harmonised obs schema is real and consistent across files. Columns present in every one:
`perturbation`, `perturbation_type`, `celltype`, `tissue_type`, `disease`, `organism`, `nperts`,
`ncounts`, `ngenes`, `percent_mito`, `percent_ribo`. **The second experimental factor is
`perturbation_2`**, with `perturbation_type_2` naming it (`TCR stimulation`, `IFN-g stimulation`).
That column is the condition axis, and it is absent exactly when the dataset is single-condition.
`var.index` is gene symbols throughout, so the join to the reference gene space is a symbol join.

### 2.1 Two findings that change the plan

**(a) The target axis collapses by one to three orders of magnitude.** The reference has 11,526
targets. The largest candidate has 2,393 (a 4.8x reduction), the best multi-condition candidate has
248 (46x), and both T-cell candidates have 21 and 32 (about 400x). A blocked target-OOD split over 21
targets cannot hold out sequence families in any meaningful sense. Whatever else is true, **no
candidate can reproduce the frozen fold's statistical power**, and any replication result is a
different, weaker measurement wearing the same name. This has to be said in the paper rather than
discovered by a reviewer.

**(b) Target count and condition count are anti-correlated across the candidates.** Everything with
enough targets for a real split (Replogle 2,393; Norman 105) is single-condition, and everything with
a condition axis (Frangieh 3; Shifrut/Datlinger 2) has few targets. Frangieh is the only dataset that
has both, and it is the compromise: 248 targets, 3 conditions, 218k cells.

### 2.2 Where the paper claims and the files disagree

- Frangieh is widely cited as "~750 perturbations". The file has **249 unique `perturbation` values
  including `control`**, i.e. 248 gene targets. The larger figure counts guides, not targets. The
  split axis is targets, so 248 is the number that matters.
- Schmidt et al. 2022 (CRISPRa Perturb-seq, primary human T cells, ~56k cells, 70 screen hits,
  **two** conditions: resting and 24 h restimulated; GEO GSE174292/GSE190604/GSE190846) is a good
  design fit and is **not in scPerturb**. It would need ingestion from GEO/Zenodo rather than the
  harmonised path, so it is listed as a stretch candidate, not a first-wave one.

---

## 3. PINNACLE context availability (the "do not silently weaken the graph" check)

The graph's node features include PINNACLE cell-type-contextualised protein embeddings, and the
frozen pipeline pins `config.PINNACLE_CONTEXT = "cd4-positive helper t cell"`. Running another cell
type against the CD4 context would degrade the graph for reasons that have nothing to do with the
hypothesis, and would **manufacture** the null. The installed PINNACLE label dictionary
(`data/raw/pinnacle/pinnacle_embeds/pinnacle_labels_dict.txt`) has 156 plain cell-type contexts. Checked:

| Dataset cell type | Matching PINNACLE context | Verdict |
|---|---|---|
| CD4+ T cells (reference) | `cd4-positive helper t cell` | present |
| Primary human T cells (Shifrut) | `cd4-positive helper t cell` | present (approximate: Shifrut is bulk CD4+/CD8+ T cells) |
| Jurkat (Datlinger) | `cd4-positive helper t cell` | present (approximate: Jurkat is a CD4+ leukemic line) |
| Melanocytes (Frangieh) | `melanocyte` | **present** |
| RPE1 (Replogle) | `retinal pigment epithelial cell` | **present** |
| THP-1 monocytes (Papalexi) | `monocyte`, `classical monocyte` | **present** |
| K562 lymphoblasts (Norman) | none — no `lymphoblast`, `leukemi`, or erythroleukemia context | **ABSENT** |

So the context-matched graph is available for every candidate except Norman/K562. Norman is
therefore the one dataset that must run with ESM-2-only node features and a logged ablation, and its
result must be read as confounded-by-construction with respect to graph quality. Every other dataset
gets a matched context, and the match must be recorded in the run's config, not assumed.

---

## 4. What the pipeline actually consumes (the adapter contract)

The model never sees raw counts. `run_module0` produces, and training reads, a **DE-statistics
matrix**: one row per (perturbation x condition), one column per gene. Concretely the contract is
`config.DE_STATS_PATH`, an `.h5ad` with

- `layers`: `log_fc`, `zscore`, `p_value`, `adj_p_value`, `baseMean`, `lfcSE`, each of shape
  `(DE_N_OBS, DE_N_VARS)` and asserted against `config.DE_N_OBS/DE_N_VARS`
  (`de_extraction.py:73-75`);
- `obs`: the q_pre columns (`culture_condition`, `target_contrast`, `target_contrast_gene_name`,
  `ensembl_id`, `hgnc_symbol`, `uniprot_id`, `entrez_id`, `ppi_degree_*`, `control_baseline_expr`)
  and the q_post columns, which are the leakage-fenced response-derived ones;
- `var`: `gene_name`.

This is good news and bad news. Good: the per-dataset work is a well-defined transformation
(pseudobulk by (target, condition, replicate) -> differential expression against control -> those six
layers), not a rewrite of the model. Bad: it is still a real component, and three constants are
currently module-level rather than dataset-scoped and must be parameterised before a second dataset
can coexist with the first:

1. `config.DE_STATS_PATH` / `PSEUDOBULK_PATH` — hardcoded `GWCD4i.*` filenames (`config.py:24-25`).
2. `config.CONDITIONS` — a closed 3-value vocab baked into two `nn.Embedding` sizes
   (`context_encoder.py:19,42`; `typed_graph_encoder.py:31,164`) and range-checked at
   `typed_graph_encoder.py:228`.
3. `config.PINNACLE_CONTEXT` — pinned to the CD4 context.

`pydeseq2` and `decoupler` are **not installed**; `statsmodels`, `scipy` and `scikit-learn` are. A
pseudobulk + moderated-t path is available without new heavy dependencies, but the DE method chosen
must be fixed in the pre-registration before any of it runs, because "which DE test" is a
researcher-degree-of-freedom that moves every downstream number.

---

## 5. Recommended portfolio

Four datasets, chosen to span a T-cell replication and a different-cell-type transfer as the goal
spec requires, with each dataset's role and its honest limitation stated up front.

| # | Dataset | Role | Primary contrast | Known limitation |
|---|---|---|---|---|
| 1 | **FrangiehIzar2021** | the real replication: 3 conditions and 248 targets | full 4-arm family, h1 headline | melanoma, not T cells; 46x fewer targets than the reference |
| 2 | **ReplogleWeissman2022 (RPE1)** | the only comparably-powered split | **h2a** (`typed_static` vs `expression_only`) | single condition, so h1 is not defined; 8,749 measured genes only (67% gene overlap) |
| 3 | **NormanWeissman2019** | second cell type, moderate target count | **h2a** | single condition; **no PINNACLE context** -> ESM-2-only, logged |
| 4 | **ShifrutMarson2018** | the T-cell replication the goal asks for | h1, reported as preliminary only | 21 targets: cannot support a real family-blocked split; underpowered by construction |

Stretch, in this order if budget allows: `ReplogleWeissman2022_K562_essential` (2,058 perturbations,
a second large single-condition arm), `DatlingerBock2017` (32 targets, 2 conditions, Jurkat),
Schmidt 2022 (needs GEO ingestion).

**The honest headline this portfolio can support** is *not* "the null replicates in four datasets".
It is: h2a (the static typed graph is worse) can be tested at comparable power on Replogle and at
moderate power on Norman; h1 (the evidence-gated graph is at parity) can be tested at moderate power
only on Frangieh, and at token power on Shifrut. A pooled fixed-effect and random-effects estimate
across whatever lands is still the right summary, but the pooling must be over h2a and h1
separately, and must carry the per-dataset target counts so the weights are visible.

---

## 6. Cost, and what is not yet built

Per dataset the work is: adapter (h5ad -> pseudobulk -> DE stats h5ad) → ID mapping to the existing
PPI graph → PINNACLE context swap → program basis refit **inside the train fold** → blocked
target-OOD split → 4 arms x >=4 seeds. The GPU cost is the small part: the reference's arms cost
0.34 (expression_only) to ~11 h (condition_gated) per seed at 20k training rows, and every candidate
except Replogle has far fewer rows. The adapter and the config parameterisation are the schedule
risk, and they are CPU and engineering, exactly as `NEXT_ACTIONS.txt` warned when it scoped a second
dataset at 2-5 days. That estimate still looks right, with the qualification that the harmonised
scPerturb layer removes the per-dataset parsing that dominated it.

Nothing in section 5 has been run. This document is the gate that precedes it, and the
pre-registration in `docs/replication-prereg.md` is the next artifact.

---

## 7bis. Measured outcome (added 2026-08-03, after building the DE matrices)

Section 5 predicted a four-dataset portfolio. Building the actual DE-statistics matrices under the
pre-registered rules cut it to **two**. The prediction and the outcome are both kept here on purpose:
the gap is the finding.

| Dataset | DE rows | Targets kept | Conditions kept | Replicate unit | Verdict |
|---|---|---|---|---|---|
| **FrangiehIzar2021** | 628 | **216** / 248 | **3** | sgRNA | **usable** — the only dataset that can carry an h1 test |
| **NormanWeissman2019** | 102 | **102** / 105 | 1 | gemgroup | **usable** — h2a only; no PINNACLE context, ESM-2-only |
| PapalexiSatija2021 | 24 | 24 / 25 | 1 | hto | preliminary: below the 50-family floor |
| ShifrutMarson2018 | 20 | 20 / 21 | **1** (lost one) | replicate (donor) | preliminary: below the floor |
| ReplogleWeissman2022 (RPE1) | 12 | **12** / 2,393 | 1 | batch | **DROPPED** by prereg section 3 |
| DatlingerBock2017 | 7 | 6 / 32 | 2 | replicate | unusable: 6 targets |

Three results deserve to be stated plainly rather than buried.

**Replogle, the best-powered candidate, is unusable under a defensible replicate-level DE design.**
It has 2,393 targets but only ~103 cells per target, spread across 56 batches, which is ~1.8 cells
per (target, batch) pseudobulk. Only 12 targets clear the 25-cell floor with two or more replicates.
There is no alternative replicate axis: `guide_id` is a dual-guide construct at ~1.1 per target. The
pre-registration says a dataset that cannot yield two replicate pseudobulks is dropped, **not**
switched to a different test, so it is not rescued by splitting cells into random pseudo-replicates.
That rule was written before these numbers existed, and it is the reason this section reads as it
does rather than reporting a rescued 2,393-target replication.

**Shifrut silently lost its second condition.** Its NoStim arm retains only the two pooled control
pseudobulks; every perturbed (target, donor) group there falls below the 25-cell floor. So the one
primary-human-T-cell candidate degenerates to a single condition, and cannot test h1 at all.

**The condition axis and the power axis remain unmet together.** After measurement, exactly one
dataset (Frangieh, 216 targets x 3 conditions) supports the h1 contrast, and exactly two (Frangieh,
Norman) clear the target-axis floor. "Four Perturb-seq replications" is not available at this
protocol strength from the harmonised public corpus, and any paper that reports four should be asked
what its replicate unit and its minimum-cells rule were.

**Adapter validation.** The Frangieh matrix was checked against biology the adapter cannot fake:
knockouts reduce their own transcript (own-gene log2FC mean $-0.632$, 89% negative, versus $-0.014$
and 61% for random genes; paired $p<10^{-300}$), and IFNGR1 knockout blunts interferon-stimulated
genes (STAT1, GBP1, IRF1, CXCL10, HLA-B) by 1.0 to 3.0 log2 units under IFN-gamma and co-culture but
not under Control (all within $\pm0.17$). The second check also confirms the condition axis is
correctly assigned, since a scrambled one would not reproduce it.

**Cell-type-matched PINNACLE contexts are extracted** (2026-08-03) into
`data/intermediate/replication/pinnacle_<context>.parquet` by
`tcell_pipeline.replication.pinnacle_contexts`. That module exists because
`tcell_pipeline.embeddings_pinnacle.run()` takes a `context` argument but always writes to
`config.PINNACLE_EMBEDDINGS_PATH`, so running it for melanocyte would have silently overwritten the
CD4 store that the reference screen's arms read at training time. The CD4 store was sha256-verified
byte-identical before and after. Re-extracting the CD4 context reproduces the frozen store's 1,119
proteins exactly, which is the cross-check that the extraction is faithful.

| Context | Proteins | Mapped to UniProt | Used by |
|---|---|---|---|
| melanocyte | 2,746 | 2,292 | Frangieh |
| monocyte | 2,866 | 2,605 | Papalexi |
| retinal pigment epithelial cell | 2,225 | 1,762 | Replogle (dropped) |
| cd4-positive helper t cell | 1,272 | 1,119 | reference, Shifrut, Datlinger |
| *(none for lymphoblasts)* | | | Norman -> ESM-2-only |

**An asymmetry that has to be carried into the interpretation.** PINNACLE covers **48% to 52%** of
the replication datasets' targets (Frangieh 83/173, Papalexi 12/23, Shifrut 9/18) against **9.2%** of
the reference's DE rows. The reason is selection: these datasets perturb a few dozen to a few hundred
well-studied screen hits, which are the proteins most likely to be in a curated interaction network,
whereas the reference is genome-scale and most of its 11,526 targets are not. So on every replication
dataset the "no-graph" baseline carries substantially **more** graph-derived node information than it
does on the reference. That makes limitation D in the paper stronger there, not weaker, and it means
a null on these datasets is even less informative about whether interaction structure carries signal.
Recorded now, before any replication lane runs, so it cannot be discovered after seeing a result.

**Not yet trained.** These are DE matrices only. Running the four-arm family on them additionally
requires making `config.CONDITIONS`, `config.PINNACLE_CONTEXT` and the DE/pseudobulk paths
dataset-scoped rather than module-level constants (survey section 4), which was deliberately not
attempted while a GPU campaign was in flight on the reference dataset.

## 7. Provenance

- Source: Zenodo record 7041849, "scPerturb Single-Cell Perturbation Data: RNA and protein h5ad
  files", DOI `10.5281/zenodo.7041849`, licence **CC-BY-4.0**. Downloaded 2026-08-03 to
  `data/raw/scperturb/` (6 files, ~3.9 GB); `SHA256SUMS.txt` alongside.
- scPerturb resource paper: Peidli et al., *Nature Methods* (2024), 44 harmonised datasets.
- Frangieh et al., *Nature Genetics* 53:332-341 (2021); count matrices also at Broad Single Cell
  Portal SCP1064.
- Norman et al., *Science* (2019); Replogle et al., *Cell* (2022); Papalexi et al., *Nature
  Genetics* (2021); Shifrut et al., *Cell* (2018); Datlinger et al., *Nature Methods* (2017).
- Schmidt et al., *Science* 375:eabj4008 (2022), GEO GSE174292 / GSE190604 / GSE190846.
- PINNACLE contexts: Li et al. (2024), Figshare article 22708126, local copy under
  `data/raw/pinnacle/pinnacle_embeds/`.

---

# L1 candidate pool, measured 2026-08-10 (supersedes section 5's portfolio)

Prereg Amendment 2 corrected the DE unit, which changed which datasets are admissible. Everything
below is measured from the files by `tcell_pipeline.replication.survey`, not read from papers.

**Coverage numbers are PRE-EXTENSION.** `id_mapping` and the ESM-2 store were built for the reference
screen's 11,526 targets, so replication targets outside that gene space resolve to nothing. The
`L1-PRE` gate (extend both, by copy, never in place) has not run yet. Any "coverage too low" verdict
here is therefore provisional and expected to improve.

| dataset | cell type | targets >=25 cells | ESM-2 usable | 2nd factor | status |
|---|---|---|---|---|---|
| ReplogleWeissman2022_rpe1 | RPE1 epithelial | **2,122** | 1,216 (57%) | none | best-powered; needs L1-PRE |
| FrangiehIzar2021 | melanocytes | 246 | 177 (72%) | **yes** (`perturbation_2`, 3 levels) | **only h1-capable dataset** |
| TianKampmann2021_CRISPRi | iPSC-induced neuron | 184 | 131 (71%) | none | usable now |
| NormanWeissman2019 | K562 lymphoblasts | 105 | 87 (83%) | none | usable; NO PINNACLE context |
| TianKampmann2021_CRISPRa | iPSC-induced neuron | 100 | 66 (66%) | none | borderline; recheck after L1-PRE |
| DatlingerBock2017 | Jurkat T | 31 | 31 (100%) | yes (`perturbation_2`) | below 50-family floor |
| TianKampmann2019_day7neuron | iPSC-induced neuron | 26 | 19 (73%) | none | below floor |
| TianKampmann2019_iPSC | iPSC | 26 | 19 (73%) | none | below floor |
| PapalexiSatija2021 | THP-1 monocytes | 25 | 24 (96%) | replicate-shaped only | below floor |
| ShifrutMarson2018 | primary human T | 21 | 18 (86%) | yes (`perturbation_2`) | below floor |
| Adamson / Dixit / Replogle K562 x2 | K562 | pending | pending | pending | downloading |

## What this means for the design

**Cell-type diversity is achievable.** Clearing the 50-family floor: RPE1 epithelial, melanocyte,
iPSC-induced neuron (x1-2), K562 lymphoblast, plus whatever the pending K562 files add. That is four
to six lineages, which meets the L1 target.

**h1 is the binding constraint, not the dataset count.** Only Frangieh has a confirmed second
experimental factor. Every other admissible dataset is single-condition, where `condition_gated`
degenerates to `typed_static` and **h1 is undefined**. So the multi-dataset result will be a pooled
**h2a** across many cell types plus an h1 that pools over very few. The paper must say this plainly
rather than presenting a pooled "the graph does not help" that quietly mixes the two.
Worth checking the pending downloads specifically for a second factor; it is the cheapest way to widen
the h1 pool, and cheaper than any additional dataset.

**An unplanned contrast worth taking.** TianKampmann2021 supplies CRISPRi and CRISPRa on the same
cell type. That is a within-cell-type comparison of perturbation DIRECTION (knockdown vs activation),
which nothing in the plan anticipated and which the reference screen (CRISPRi only) cannot provide.
If the graph's contribution differs between knockdown and activation, that is a mechanistic result
rather than another null, and it costs nothing extra since both datasets are already downloaded.

## Negative survey results (2026-08-10) — downloaded, measured, NOT usable

Recording these so they are not re-downloaded or re-surveyed. Both were pulled as L1 candidates and
neither helps.

| dataset | cells | targets >=25 | 2nd factor | why it fails |
|---|---|---|---|---|
| AdamsonWeissman2016 (GSM2406681_10X010) | 65,337 | **1** | none | this sub-experiment has essentially one perturbation clearing the floor; the Adamson screen is a small UPR-focused CRISPRi study split across three files, and this one is not a target panel |
| DixitRegev2016 | 51,898 | 33 | none | below the 50-family floor, single condition, K562 (a lineage already covered by Norman) |

Neither adds a cell type the pool lacks and neither widens the h1 pool, which remains the binding
constraint. **Only Frangieh has a confirmed second experimental factor**, so h1 stays poolable over
essentially one dataset while h2a pools over five or more. If widening h1 matters more than adding a
sixth h2a dataset, the remaining options are (a) check the two pending Replogle K562 files for a second
factor, (b) ingest Schmidt 2022 from GEO, which has resting vs 24 h restimulated in primary human T
cells but is not in scPerturb and needs its own adapter, or (c) accept the asymmetry and report it.
Option (c) is honest and free; (b) is the only one that would materially change the h1 claim.

## L1-PRE gate CLOSED (2026-08-10) — extended id_mapping + ESM-2

`id_mapping` and the ESM-2 store were extended BY COPY (frozen files sha256-verified untouched;
12,938 mapping rows, 12,744 accessions). Coverage before -> after:

| dataset | cell type | targets >=25 | coverage before | after | gate |
|---|---|---|---|---|---|
| ReplogleWeissman2022_rpe1 | RPE1 epithelial | 2,122 | 57.3% | **99.5%** | PASS |
| ReplogleWeissman2022_K562_essential | K562 | **2,003** | n/a | **96.0%** | PASS |
| FrangiehIzar2021 | melanocytes | 246 | 72.0% | **94.7%** | PASS |
| TianKampmann2021_CRISPRi | iPSC-induced neuron | 184 | 71.2% | **100%** | PASS |
| NormanWeissman2019 | K562 | 105 | 82.9% | **99.0%** | PASS |
| TianKampmann2021_CRISPRa | iPSC-induced neuron | 100 | 66.0% | **100%** | PASS |
| Datlinger / Papalexi / Shifrut / Tian2019 x2 | various | 21-31 | - | 95-100% | below 50-family floor |

The gate that mattered was Replogle RPE1: 43% of its graph arm would have had no graph node at all,
biasing h1 toward zero. The fix was mostly gene-symbol nomenclature - Perturb-seq screens are annotated
against the gene build current at publication, so an unresolved symbol is usually a rename
(AARS -> AARS1, ATP5A1 -> ATP5F1A, C10orf54 -> VSIR) rather than a gene without a protein. Querying
current symbols only would have shrunk the graph arm preferentially on the OLDER datasets, a bias
correlated with dataset age and invisible in any aggregate.

**Pool status: six datasets clear both gates**, spanning RPE1 epithelial, K562 lymphoblast (x2),
melanocyte and iPSC-induced neuron (x2) - four lineages, meeting the >=5 cell-type target once the
pending genome-wide K562 file is counted separately or a further lineage is added.

**h1 remains constrained to Frangieh alone.** K562_essential has no second experimental factor, like
every other large candidate. The multi-dataset result is therefore a pooled h2a over six datasets and
an h1 over one, and the paper must present them separately rather than merging.

## FINAL L1 POOL (2026-08-10, all downloads complete, both gates applied)

| dataset | cell type | targets >=25 | coverage | 2nd factor | DE rows (approx) |
|---|---|---|---|---|---|
| **ReplogleWeissman2022_K562_gwps** | K562 | **9,730** | 92.9% | none | ~9,730 |
| ReplogleWeissman2022_rpe1 | RPE1 epithelial | 2,122 | 99.5% | none | ~2,122 |
| ReplogleWeissman2022_K562_essential | K562 | 2,003 | 96.0% | none | ~2,003 |
| FrangiehIzar2021 | melanocytes | 246 | 94.7% | **yes (3 levels)** | ~738 |
| TianKampmann2021_CRISPRi | iPSC-induced neuron | 184 | 100% | none | ~184 |
| TianKampmann2021_CRISPRa | iPSC-induced neuron | 100 | 100% | none | ~100 |
| *(reference, for scale)* | *CD4+ T* | *11,526* | - | *3 levels* | *33,983* |

**The genome-wide K562 screen is the headline acquisition.** At 9,730 targets it is 84% of the
reference's target count, so for the first time the replication is not obviously weaker than the thing
being replicated. Section 2.1 of this document claimed "no candidate can reproduce the frozen fold's
statistical power" - that was written under the pre-Amendment-2 rule and before this file was
surveyed, and on the target axis it is now wrong. Superseded.

**Budget is far better than the plan assumed.** Cost scales with DE ROWS, not target count, and every
large candidate is single-condition. gwps has ~9,730 rows against the reference's 33,983, so a seed is
roughly 0.46x the reference cost: condition_gated ~5 h, typed_static ~3 h, expression_only ~0.17 h.
An h2a pair at n=5 is therefore ~16 GPU-h on gwps and less on everything else - the whole six-dataset
h2a sweep is well under 100 GPU-h, not the ~420 the plan budgeted. Re-measure before committing;
this is an extrapolation from row count, and this project has been burned by cost models fitted on
too few points.

**h1 is still Frangieh alone.** Every dataset that clears the target-axis floor at scale is
single-condition. Adding datasets buys h2a power and cell-type breadth; it does not buy h1 power. The
honest framing is a well-powered multi-dataset h2a across four lineages, plus an h1 that remains a
single-dataset result on a 246-target screen.

## On-target QC (2026-08-10): two datasets fail, and the pool has a perturbation-DIRECTION axis

Every DE matrix is checked for whether the perturbations did anything specific to their own targets.
A dataset that fails cannot inform a graph-versus-no-graph contrast, because it would return a null
for reasons having nothing to do with the graph.

| dataset | own-gene log2FC | direction consistency | inferred direction | verdict |
|---|---|---|---|---|
| NormanWeissman2019 | **+1.71** | 97% | **activation** | PASS |
| TianKampmann2021_CRISPRa | +0.66 | 88% | **activation** | PASS |
| TianKampmann2021_CRISPRi | -0.57 | 92% | knockdown | PASS |
| PapalexiSatija2021 | -0.65 | 88% | knockdown | PASS |
| FrangiehIzar2021 | -0.61 | 90% | knockdown | PASS |
| ShifrutMarson2018 | -0.03 | 50% | - | **FAIL** |
| DatlingerBock2017 | -0.02 | 51% | - | **FAIL** |

**The gate had to be rewritten mid-flight.** Its first version asserted the knockdown signature
(own-gene log2FC < 0) and consequently FAILED TianKampmann2021_CRISPRa, which shows $+0.66$ with 88%
of rows positive - a textbook on-target ACTIVATION result. The gate was rejecting a dataset for
working. scPerturb cannot disambiguate this: `perturbation_type` is the string `CRISPR` for CRISPRi,
CRISPRa and knockout alike. The gate now tests magnitude and sign CONSISTENCY and reports the
direction it infers.

**Consequence for the design, unplanned and better than planned.** The pool is not
knockdown-only. Norman (K562) and TianKampmann2021_CRISPRa (neuron) are activation screens; the
reference and the rest are knockdown. TianKampmann2021 supplies both directions in the SAME cell
type. So the replication can ask whether the graph's contribution depends on perturbation DIRECTION,
not only on cell type - a question the CRISPRi-only reference screen cannot pose, and one where a
difference would be mechanistic rather than another null.

**The two failures are the two oldest and smallest T-cell screens** (Datlinger 2017 Jurkat CROP-seq,
Shifrut 2018 primary T), both already below the 50-family floor. They are now excluded for a
principled reason - no detectable perturbation effect - rather than only for being small. Note the
cost of this: the pool loses BOTH of its primary/near-primary T-cell datasets, so the replication
tests transfer to other lineages but contains no T-cell replication of the reference screen. That
should be stated in the paper rather than left for a reader to notice.
