# Code Review — feat-016 (Module 2) + feat-003 (leakage-safe splits)

**Date:** 2026-07-15 · **Effort:** xhigh (workflow-backed) · **Reviewer:** `/code-review` multi-agent
**Scope:** commits `100a505` (Module 2 typed graph encoder) and `35e3999` (leakage-safe splits),
diffed against `b833aa2`.
**Run:** 24 agents · 6 finders + 1 sweep · 15 verifier agents · **23 candidates verified, 0 refuted → 13 reported.**

## Headline

Three of these findings mean the **feat-003 split as committed has weaker leakage-safety than its own
`leakage_report.json` claims** — the audit misses cap-induced family splits, the residual metric is
computed in mismatched coordinate frames (understating leakage), and the pipeline fails open when the
PLM embeddings are absent. Fixing Tier 1 requires **regenerating the frozen `data/splits/` artifacts**
and will change the published effectiveness numbers (the relative "47% reduction" is roughly stable
since the same bias applied to both blocked and random; the absolute 28.1% / 53.5% figures will move).

Severity assessments below are mine (author), layered on top of the verifier verdicts.

## Resolution (2026-07-15, applied)

Tier 1 + Tier 2 + all of Tier 3 (the cheap defenses first, then #10/#11/#12) were addressed and
committed at `7760624`; `data/splits/` was regenerated and **57 pytest** stay green (+3 regression
checks: OOV-condition raises, edge_gates length == E, random split covers all items at small N).

| # | Location | Outcome |
|---|----------|---------|
| 1 | splits.py:186 | **Fixed** — added `_precap_labels`; audit publishes `cap_induced_family_splits` (real data: 1) + `family_challenge_sharing_train_frac` |
| 2 | splits.py:164 | **Fixed** — `_sequence_residual` centers all genes in one global frame; effectiveness now 53.8→26.4 = 51% (was 53.5→28.1 = 47%) |
| 3 | splits.py:241 | **Fixed** — `run()` fails closed when `gene_vec` empty (override `SPLITS_ALLOW_NO_SEQUENCE=1`); manifest gains `sequence_block_active` / `n_genes_with_embedding` |
| 4 | test_graph.py:104 | **Fixed** — seeded + false `<1.0` bound dropped |
| 5 | graph_builder.py:61 | **Fixed** — degrees reordered to `[physical, functional, complex]` |
| 6 | typed_graph_encoder.py:156 | **Fixed** — `encode_one` moves `h_do` to device |
| 7 | graph_builder.py:41 | **Fixed** — `nan_to_num` on PP + membership edge features |
| 8 | graph_builder.py:103 | **Fixed** — `.dropna()` on gene symbols |
| 9 | graph_builder.py:40 | **Fixed** — fail-fast on unknown PPI source |
| 10 | typed_graph_encoder.py:145 | **Fixed** — `_condition_index` raises a legible `ValueError` on OOV (still fail-fast, as intended); over-promising "never crash a batch" docstring narrowed to genes |
| 11 | splits.py:148 | **Fixed** — cumulative-boundary allocation (last boundary == n; no truncated tail). `random.csv` regenerated (blocked split unchanged; effectiveness numbers unchanged — gene-level baseline didn't shift at N=11525) |
| 12 | typed_graph_encoder.py:127 | **Partially fixed** — returned `edge_gates[rel]` is now length E (one per original edge) for *all* relations, aligned to the sub-graph `edge_index` (was 2E-doubled for PP); full gate→(u,v) identity-forwarding API still deferred to Module 4 |
| 13 | config.py:105/108/122 | **Fixed** — `N_RELATION_TYPES` / `RELATION_TYPES` / `SPLIT_AUDIT_HOPS` removed |

Note on #1: the split CSVs are **byte-identical** to the original freeze (both sha256 unchanged) — a
family larger than any single role's budget *must* be split, so the fix discloses the cap-induced
split rather than eliminating it. The residual cosine distribution (26.4%) is the true leakage; the
new `family_challenge_sharing_train_frac` (0.41) is a single-linkage upper bound.

---

## Tier 1 — leakage-safety correctness (the split's core guarantee)

### 1. `splits.py:186` — audit doesn't catch cap-induced family splits `CONFIRMED`
The only hard assertion in `audit_leakage` checks that no **post-cap union-find component** spans >1
role. But when a paralog/complex family exceeds `GROUP_SIZE_CAP` (5%), `_CappedUnionFind.union` *refuses*
the merge, leaving the family in separate components that `assign_partitions` can place in different
roles. Because `grp_roles` is keyed on the post-cap label, `len(rs) > 1` is trivially satisfied and no
`ValueError` is raised.
- **Fires on real data:** the largest family hit exactly the 5.0% cap and **3,986 merges were refused**,
  so some families are genuinely split train↔challenge, undetected by the "fails-closed" gate.
- **Fix:** audit against the **pre-cap** similarity/family structure (representative clusters + complex
  co-membership), not the post-cap components; report cap-induced cross-role family splits explicitly.

### 2. `splits.py:164-165` — sequence-residual centers train/challenge by separate means `CONFIRMED`
`_sequence_residual` calls `_centered(chal, …)` and `_centered(train, …)` independently, so each subset
is centered by its **own** mean and the train↔challenge cosine is computed in two mismatched frames —
systematically **understating** residual similarity (an identical paralog pair reads ~0.94 instead of
1.0, and can drop below the 0.85 threshold entirely).
- **Impact:** distorts the published `leakage_report.json` (`max`/`p99`/`frac_ge_threshold`) and the
  `sequence_effectiveness_vs_random` numbers. Grouping (`_representative_edges`) centers **globally over
  all genes**; the audit must use the same global centering, then index the train/challenge rows.

### 3. `splits.py:241` — fail-open when PLM embeddings are absent `CONFIRMED`
If `PLM_EMBEDDINGS_PATH` doesn't exist, `gene_vec` stays `{}`, `_representative_edges` returns nothing,
and only complex must-links group genes — i.e. the **primary sequence hard-block is silently disabled**.
`run()` still writes `blocked_target_ood.csv` + `manifest.json` + `leakage_report.json` (which merely
omits the sequence axis) and exits 0, publishing a sequence-leaky split as leakage-safe.
- **Fix:** refuse (or emit a prominent warning + a manifest flag) when `gene_vec` is empty.

---

## Tier 2 — active bugs

### 4. `test_graph.py:104` — flaky assertion (author bug) `CONFIRMED`
`assert out.abs().max() < 1.0` is mathematically false: `signed_message = tanh(·) * relu(·)` and `relu`
is unbounded, so `|out|` can exceed 1. The test seeds no RNG → **~12% flake rate** (369/3000 trials,
max observed 2.34). Passed on the original runs by luck.
- **Fix:** seed the RNG and drop the `< 1.0` bound (keep the composition-equality and relu-nonnegativity
  checks, which are the meaningful assertions).

### 5. `graph_builder.py:61` — degree columns misordered vs Module 1 `CONFIRMED`
`_protein_features` fills degrees in relation order `[physical_ppi, co_complex, functional_assoc]` =
`[physical, complex, functional]`, whereas `TargetEncoder.TARGET_SCALAR_KEYS` is
`[physical, functional, complex]`. The graph_builder docstring claims graph and encoder describe a
protein "the same way" — false; functional↔complex are swapped.
- **Latent** now (Module 2 learns its own input projection, so it's internally consistent), but a
  footgun for any future weight-transfer / cross-module comparison relying on the stated invariant.
- **Fix:** reorder the graph degrees to `[physical, functional, complex]` to match Module 1 (cheap), or
  correct the docstring to disclaim the invariant.

### 6. `typed_graph_encoder.py:156` — `encode_one` doesn't move `h_do` to device `CONFIRMED`
`encode_one` reads `device = self.proj.weight.device` and moves `sub`/`h_p`/`node_states` onto it, but
never moves the caller-supplied `h_do`. A direct `encode_one` call (documented public method) with the
encoder on CUDA and `h_do` on CPU raises a device-mismatch `RuntimeError` inside `MultiheadAttention`.
`forward()` is fine (it does `h_do = h_do.to(device)` first).
- **Fix:** add `h_do = h_do.to(device)` in `encode_one` (one line).

---

## Tier 3 — latent robustness (PLAUSIBLE, don't trigger on current data) + cleanup

### 7. `graph_builder.py:41` — no NaN guard on edge features `PLAUSIBLE`
`score`/`is_direct_binary`/`n_supporting_sources` (and membership `confidence`/`is_curated`) flow
straight into `edge_attr`, while `_protein_features` wraps only the baseline in `torch.nan_to_num`. A
NaN in any of those would propagate through `u_r(edge_attr)` + the gate into every message → `h_graph`
NaN, and `run_module2_smoke`'s finiteness check would fail. Real marts are currently clean (smoke passes).
- **Fix:** `torch.nan_to_num` the edge-feature tensors symmetrically (cheap defense).

### 8. `graph_builder.py:103` — NaN gene symbol crashes `sorted(set(...))` `PLAUSIBLE`
`sorted(set(source_gene) | set(target_gene))` raises `TypeError` if any symbol is NaN (str/float
unorderable). `splits.py` guards this with `.dropna()`; `graph_builder` doesn't. Real data has no null
gene symbols today.

### 9. `graph_builder.py:40` — unknown PPI source → `IndexError` `PLAUSIBLE`
A `source` outside the five `PPI_SOURCES` makes `.map(_SOURCE_INDEX)` return NaN, which then fancy-indexes
the one-hot array → `IndexError`. Real edges only use the five known sources.

### 10. `typed_graph_encoder.py:145` — OOV `culture_condition` → `KeyError` `CONFIRMED`
`forward` guards unknown target genes with a zero-`h_graph` fallback but leaves `condition` unguarded;
an out-of-vocab condition raises `KeyError` in `_COND_INDEX[condition]`, aborting the whole batch and
contradicting the docstring's "never crash a batch."
- **Assessment (author):** debatable — only 3 valid conditions exist, so an OOV condition is genuinely
  invalid input and fail-fast is defensible. Worth making consistent with the gene guard if we want the
  stated contract to hold.

### 11. `splits.py:148` — banker's rounding drifts random-split role sizes `CONFIRMED`
`k = round(fractions[r] * len(items))` uses round-half-to-even; on some `N` (e.g. 10) the role sizes
over-allocate and a later role's slice is truncated. Minor — affects only the diagnostic random split's
exact fractions (the test uses N=1000, within tolerance).

### 12. `typed_graph_encoder.py:127` — `edge_gates` length/identity inconsistency for Module 4 `PLAUSIBLE`
Protein-protein `edge_gates` are length `2E` (symmetrized via `cat([ei, ei.flip(0)])`), while
`complex_membership` is length `E`, and no `edge_index`/`orig_idx` is forwarded from `forward()`. Module 4
(the documented consumer, "mechanism attribution") can't map a gate back to a specific interaction.
- **Forward-looking:** Module 4 isn't built yet; fix the `edge_gates` contract (return edge identities,
  consistent lengths) when it is.

### 13. `config.py:105/108/122` — dead constants `CONFIRMED`
`N_RELATION_TYPES`, `RELATION_TYPES`, and `SPLIT_AUDIT_HOPS` are declared but never referenced (the
encoder hardcodes its own `_PP_RELATIONS`/`_MEMBERSHIP`; the audit hardcodes a 1-hop check). Dead knobs
that will silently drift. Remove, or wire them into their consumers.

---

## Triage summary

| # | Location | Sev | Verdict | Recommendation |
|---|----------|-----|---------|----------------|
| 1 | splits.py:186 | Tier 1 | CONFIRMED | Fix — audit pre-cap family integrity |
| 2 | splits.py:164 | Tier 1 | CONFIRMED | Fix — center globally in the residual |
| 3 | splits.py:241 | Tier 1 | CONFIRMED | Fix — fail/warn when embeddings absent |
| 4 | test_graph.py:104 | Tier 2 | CONFIRMED | Fix — seed + drop the `<1.0` bound |
| 5 | graph_builder.py:61 | Tier 2 | CONFIRMED | Fix — reorder degrees to match Module 1 |
| 6 | typed_graph_encoder.py:156 | Tier 2 | CONFIRMED | Fix — move `h_do` to device in `encode_one` |
| 7 | graph_builder.py:41 | Tier 3 | PLAUSIBLE | Cheap defense — `nan_to_num` edge features |
| 8 | graph_builder.py:103 | Tier 3 | PLAUSIBLE | Cheap defense — `dropna` gene symbols |
| 9 | graph_builder.py:40 | Tier 3 | PLAUSIBLE | Cheap defense — guard unknown sources |
| 10 | typed_graph_encoder.py:145 | Tier 3 | CONFIRMED | Optional — make OOV-condition consistent or keep fail-fast |
| 11 | splits.py:148 | Tier 3 | CONFIRMED | Optional — cumulative-boundary allocation |
| 12 | typed_graph_encoder.py:127 | Tier 3 | PLAUSIBLE | Defer — fix `edge_gates` contract with Module 4 |
| 13 | config.py:105/108/122 | Tier 3 | CONFIRMED | Cleanup — remove/wire dead constants |

## Recommended action

1. **Fix Tier 1 (1-3) + Tier 2 (4-6)** and the cheap Tier-3 defenses (7-9).
2. **Regenerate `data/splits/`** and the corrected `leakage_report.json` (Tier 1 changes the artifacts +
   effectiveness numbers); update the feat-003 evidence in `feature_list.json` / handoff accordingly.
3. Drop the dead config constants (13); note the `edge_gates` contract (12) for Module 4.
4. Leave (10) as fail-fast unless we want the "never crash a batch" contract to hold literally; (11) is
   cosmetic for the diagnostic split.
