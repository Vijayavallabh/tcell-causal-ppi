"""feat-003 — Leakage-safe train/val/calibration/challenge splits.

Partitions perturbation targets so no held-out *challenge* gene can be predicted by having seen a
biologically near-identical *training* gene. The hard block is the sequence/paralog family axis:
representative (non-chaining, CD-HIT-style) clustering on centered ESM-2 embeddings, merged with
direct CORUM complex co-membership under a size cap. Physical-PPI neighbourhood is NOT a partition
constraint — 95% of targets sit in one physical connected component, so it is an *audited* axis
whose train-to-challenge similarity distribution is published (report §Phase-1 6/9, §1136, §1470).

See docs/specs/2026-07-15-feat-003-leakage-safe-splits.md for the empirical measurements that
forced this design (naive connected-components collapses to a single giant group on every axis).

    python -m tcell_pipeline.splits
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import pandas as pd

from tcell_pipeline import config


# --- family grouping -------------------------------------------------------

def _centered(genes: list[str], gene_vec: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return (embedded genes, centered+L2-normalized matrix). Centering removes the high mean-pooled
    ESM baseline so cosine is discriminative; empty -> (‑[], (0,0))."""
    embg = [g for g in genes if g in gene_vec]
    if not embg:
        return [], np.zeros((0, 0), dtype=np.float32)
    V = np.stack([np.asarray(gene_vec[g], dtype=np.float32) for g in embg])
    V = V - V.mean(0)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    return embg, V


def _representative_edges(genes: list[str], gene_vec: dict[str, np.ndarray], threshold: float):
    """Greedy representative clustering: each unassigned gene opens a cluster of all still-unassigned
    genes within cosine ``threshold`` of it. Non-chaining (members link to the representative, not to
    each other), so dense neighbourhoods don't transitively merge into a giant component."""
    embg, V = _centered(genes, gene_vec)
    assigned = np.full(len(embg), -1, dtype=int)
    edges: list[tuple[str, str]] = []
    for i in range(len(embg)):
        if assigned[i] >= 0:
            continue
        members = np.where((V @ V[i] >= threshold) & (assigned < 0))[0]
        assigned[members] = i
        edges.extend((embg[i], embg[int(j)]) for j in members if j != i)
    return edges


def _complex_edges(genes: list[str], complex_df: pd.DataFrame):
    """Star must-link edges per CORUM complex (member[0]—member[k]); a star is
    connected-component-equivalent to a clique but O(members) not O(members^2)."""
    gset = set(genes)
    sub = complex_df[complex_df["protein_gene"].isin(gset)]
    edges: list[tuple[str, str]] = []
    for _, grp in sub.groupby("complex_id", sort=True)["protein_gene"]:
        ids = list(dict.fromkeys(grp))  # unique, order-stable
        edges.extend((ids[0], j) for j in ids[1:])
    return edges


class _CappedUnionFind:
    """Union-find that refuses a merge if the resulting component would exceed ``cap`` members —
    keeps every group splittable while blocking as many high-priority pairs as fit."""

    def __init__(self, items: list[str], cap: int) -> None:
        self.parent = {x: x for x in items}
        self.size = {x: 1 for x in items}
        self.cap = cap

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.size[ra] + self.size[rb] > self.cap:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def family_groups(
    genes: list[str],
    gene_vec: dict[str, np.ndarray],
    complex_df: pd.DataFrame,
    cap_frac: float = config.GROUP_SIZE_CAP,
    seq_threshold: float = config.SEQ_SIM_COSINE_THRESHOLD,
) -> tuple[dict[str, int], int]:
    """Assign each gene a family-group id (capped union-find over sequence then complex must-links).
    Returns (gene -> group id, n_refused_merges). Sequence is applied first (tightest signal)."""
    cap = max(2, math.ceil(cap_frac * len(genes)))
    uf = _CappedUnionFind(genes, cap)
    refused = 0
    for edges in (_representative_edges(genes, gene_vec, seq_threshold), _complex_edges(genes, complex_df)):
        for a, b in edges:
            refused += not uf.union(a, b)
    roots = {g: uf.find(g) for g in genes}
    relabel = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {g: relabel[roots[g]] for g in genes}, refused


# --- partition assignment --------------------------------------------------

def assign_partitions(labels: dict[str, int], genes: list[str], seed: int = config.SPLIT_SEED,
                      fractions: dict[str, float] = config.SPLIT_FRACTIONS,
                      roles: tuple[str, ...] = config.SPLIT_ROLES) -> dict[str, str]:
    """Assign whole family groups to roles; each group goes entirely to the currently most-deficit
    role (deficit-greedy keeps balance within the cap). Seed shuffles group order → seed-sensitive."""
    groups: dict[int, list[str]] = {}
    for g in genes:
        groups.setdefault(labels[g], []).append(g)
    n = len(genes)
    target = {r: fractions[r] * n for r in roles}
    current = {r: 0 for r in roles}
    order = sorted(groups.values(), key=lambda m: (-len(m), m[0]))  # deterministic base order
    np.random.default_rng(seed).shuffle(order)
    role_of: dict[str, str] = {}
    for members in order:
        r = max(roles, key=lambda r: target[r] - current[r])
        for g in members:
            role_of[g] = r
        current[r] += len(members)
    return role_of


def assign_random(items: list, seed: int = config.SPLIT_SEED,
                  fractions: dict[str, float] = config.SPLIT_FRACTIONS,
                  roles: tuple[str, ...] = config.SPLIT_ROLES) -> dict:
    """Row-level diagnostic split (sanity baseline; not a headline)."""
    n = len(items)
    perm = np.random.default_rng(seed).permutation(n)
    # cumulative-fraction boundaries: the last boundary is exactly n, so every item is placed once and
    # no role's slice is truncated (independent per-role round() can over-allocate and drop the tail).
    role_of, prev, cum = {}, 0, 0.0
    for r in roles:
        cum += fractions[r]
        j = n if r == roles[-1] else round(cum * n)
        for idx in perm[prev:j]:
            role_of[items[int(idx)]] = r
        prev = j
    return role_of


# --- leakage audit ---------------------------------------------------------

def _precap_labels(genes: list[str], gene_vec: dict[str, np.ndarray], complex_df: pd.DataFrame,
                   seq_threshold: float) -> dict[str, int]:
    """Uncapped connected components over the SAME sequence + complex must-links — the true
    biological families the size cap may have split across roles. Audited (not blocked): a family
    larger than the cap is deliberately splittable, so this reports the residual it leaves."""
    parent = {g: g for g in genes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edges in (_representative_edges(genes, gene_vec, seq_threshold), _complex_edges(genes, complex_df)):
        for a, b in edges:
            parent[find(a)] = find(b)
    roots = {g: find(g) for g in genes}
    relabel = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {g: relabel[roots[g]] for g in genes}



def _sequence_residual(role_of: dict[str, str], genes: list[str], gene_vec: dict[str, np.ndarray],
                       seq_threshold: float) -> dict | None:
    """Distribution of each challenge gene's nearest centered-cosine to any train gene (report §6).

    Centers ALL genes by one shared mean (the same global frame ``_representative_edges`` groups in),
    then indexes the challenge/train rows — centering each subset by its own mean puts the two sides
    in mismatched frames and understates similarity (identical paralogs read ~0.94 instead of 1.0)."""
    embg, V = _centered(genes, gene_vec)
    row = {g: i for i, g in enumerate(embg)}
    ci = [row[g] for g in genes if role_of[g] == "challenge" and g in row]
    ti = [row[g] for g in genes if role_of[g] == "train" and g in row]
    if not (ci and ti):
        return None
    nearest = (V[ci] @ V[ti].T).max(1)
    return {
        "max": float(nearest.max()), "p99": float(np.percentile(nearest, 99)),
        "p90": float(np.percentile(nearest, 90)), "median": float(np.median(nearest)),
        "frac_ge_threshold": float((nearest >= seq_threshold).mean()),
    }


def audit_leakage(role_of: dict[str, str], labels: dict[str, int], genes: list[str],
                  gene_vec: dict[str, np.ndarray], complex_df: pd.DataFrame, edges_df: pd.DataFrame,
                  seq_threshold: float = config.SEQ_SIM_COSINE_THRESHOLD) -> dict:
    """Hard-assert no (post-cap) family group is split across roles — that would be an assignment
    bug, since whole groups are placed atomically. Then publish train↔challenge residual similarity
    for each axis (sequence / complex / physical neighbourhood) AND the cap-induced family splits the
    post-cap check cannot see. Fails closed only on a genuine assignment bug."""
    grp_roles: dict[int, set] = {}
    for g in genes:
        grp_roles.setdefault(labels[g], set()).add(role_of[g])
    split = [l for l, rs in grp_roles.items() if len(rs) > 1]
    if split:
        raise ValueError(f"leakage: {len(split)} family groups span >1 role (assignment bug)")

    train = {g for g in genes if role_of[g] == "train"}
    chal = [g for g in genes if role_of[g] == "challenge"]
    report: dict = {
        "role_counts": {r: sum(role_of[g] == r for g in genes) for r in config.SPLIT_ROLES},
        "n_family_groups": len(set(labels.values())),
    }

    # cap-induced leakage: families the cap refused to keep whole and that landed in >1 role. This is
    # the leakage the post-cap assertion above is blind to (grp_roles keys on the post-cap label).
    precap = _precap_labels(genes, gene_vec, complex_df, seq_threshold)
    fam_members: dict[int, list[str]] = {}
    for g in genes:
        fam_members.setdefault(precap[g], []).append(g)
    split_fams = [m for m in fam_members.values() if len({role_of[g] for g in m}) > 1]
    report["n_precap_families"] = len(fam_members)
    report["cap_induced_family_splits"] = len(split_fams)
    if not chal:
        return report

    leaky_fam_chal = {g for m in split_fams for g in m
                      if role_of[g] == "challenge" and any(role_of[t] == "train" for t in m)}
    report["family_challenge_sharing_train_frac"] = len(leaky_fam_chal) / len(chal)

    seq = _sequence_residual(role_of, genes, gene_vec, seq_threshold)
    if seq:
        report["sequence_train_to_challenge_cosine"] = seq

    # complex: challenge genes sharing a CORUM complex with any train gene
    leaky_c = set()
    for members in complex_df.groupby("complex_id")["protein_gene"].agg(set):
        if members & train:
            leaky_c |= members & set(chal)
    report["complex_challenge_sharing_train_frac"] = len(leaky_c) / len(chal)

    # physical neighbourhood (audit-only): challenge genes with a physical edge to a train gene
    phys = edges_df[edges_df["is_physical"] == 1]
    cset = set(chal)
    a = phys[phys["source_gene"].isin(cset) & phys["target_gene"].isin(train)]["source_gene"]
    b = phys[phys["target_gene"].isin(cset) & phys["source_gene"].isin(train)]["target_gene"]
    report["physical_neighbour_challenge_touching_train_frac"] = len(set(a) | set(b)) / len(chal)
    report["note"] = ("physical neighbourhood is audit-only: 95% of targets share one physical "
                       "component, so it cannot be a hard partition block (see spec).")
    return report


# --- driver ----------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_csv(df: pd.DataFrame, path) -> str:
    text = df.to_csv(index=False)
    config.write_text_atomic(text, path)
    return _sha256(text)


def run() -> bool:
    if not config.PERTURBATION_CONDITION_PATH.exists():
        print("[splits] perturbation_condition mart absent — run run_module0.py first")
        return False
    pc = pd.read_parquet(config.PERTURBATION_CONDITION_PATH,
                         columns=["row_index", "hgnc_symbol", "uniprot_id", "culture_condition"])
    genes = sorted(g for g in pc["hgnc_symbol"].dropna().unique())
    gene_uni = dict(zip(pc["hgnc_symbol"], pc["uniprot_id"].astype("string")))

    gene_vec: dict[str, np.ndarray] = {}
    if config.PLM_EMBEDDINGS_PATH.exists():
        emb = pd.read_parquet(config.PLM_EMBEDDINGS_PATH)
        u2v = {str(u): np.asarray(v, dtype=np.float32) for u, v in zip(emb["uniprot_id"], emb["embedding"])}
        gene_vec = {g: u2v[gene_uni[g]] for g in genes if isinstance(gene_uni.get(g), str) and gene_uni[g] in u2v}
    if not gene_vec:
        # The sequence/paralog family axis is THE hard block. With no embeddings it silently vanishes
        # and only complex must-links group genes — a sequence-leaky split we'd stamp "leakage-safe".
        print("[splits] ABORT: no PLM embeddings (PLM_EMBEDDINGS_PATH missing or no gene mapped). The "
              "sequence hard-block would be disabled. Run the PLM embedding step first, or set "
              "SPLITS_ALLOW_NO_SEQUENCE=1 to override (the resulting split is NOT leakage-safe).")
        if not os.environ.get("SPLITS_ALLOW_NO_SEQUENCE"):
            return False
    complex_df = pd.read_parquet(config.COMPLEX_MEMBERSHIP_PATH, columns=["protein_gene", "complex_id"])
    edges_df = pd.read_parquet(config.PROTEIN_EDGES_PATH, columns=["source_gene", "target_gene", "is_physical"])

    print(f"Target genes: {len(genes)}  ({len(gene_vec)} with ESM embedding)")
    labels, refused = family_groups(genes, gene_vec, complex_df)
    grp_sizes = np.bincount(list(labels.values()))
    print(f"Family groups: {len(set(labels.values()))}  largest {grp_sizes.max()} "
          f"({grp_sizes.max()/len(genes):.1%})  refused merges (over cap): {refused}")

    role_of = assign_partitions(labels, genes)
    blocked = pd.DataFrame({"hgnc_symbol": genes, "role": [role_of[g] for g in genes]})
    row_role = assign_random(pc["row_index"].tolist())
    random_df = pd.DataFrame({"row_index": pc["row_index"], "role": pc["row_index"].map(row_role)})

    config.ensure_dir(config.SPLITS_ROOT)
    h_blocked = _write_csv(blocked, config.BLOCKED_SPLIT_PATH)
    h_random = _write_csv(random_df, config.RANDOM_SPLIT_PATH)

    audit = audit_leakage(role_of, labels, genes, gene_vec, complex_df, edges_df)
    # effectiveness: does blocking actually reduce sequence leakage vs a random gene-level split?
    base = _sequence_residual(assign_random(genes), genes, gene_vec, config.SEQ_SIM_COSINE_THRESHOLD)
    blk = audit.get("sequence_train_to_challenge_cosine")
    if base and blk and base["frac_ge_threshold"]:
        audit["sequence_effectiveness_vs_random"] = {
            "blocked_frac_ge_threshold": blk["frac_ge_threshold"],
            "random_frac_ge_threshold": base["frac_ge_threshold"],
            "reduction": round(1 - blk["frac_ge_threshold"] / base["frac_ge_threshold"], 3),
        }
    realized = {r: round(sum(role_of[g] == r for g in genes) / len(genes), 4) for r in config.SPLIT_ROLES}
    manifest = {
        "seed": config.SPLIT_SEED, "fractions_target": config.SPLIT_FRACTIONS,
        "fractions_realized_blocked": realized, "seq_cosine_threshold": config.SEQ_SIM_COSINE_THRESHOLD,
        "group_size_cap_frac": config.GROUP_SIZE_CAP, "n_genes": len(genes),
        "sequence_block_active": bool(gene_vec), "n_genes_with_embedding": len(gene_vec),
        "n_family_groups": len(set(labels.values())), "refused_merges": refused,
        "sha256": {"blocked_target_ood.csv": h_blocked, "random.csv": h_random},
    }
    config.write_text_atomic(json.dumps(manifest, indent=2), config.SPLIT_MANIFEST_PATH)
    config.write_text_atomic(json.dumps(audit, indent=2), config.SPLIT_LEAKAGE_REPORT_PATH)

    print(f"Blocked split realized fractions: {realized}")
    print(f"Leakage audit: {json.dumps(audit, indent=2)}")
    print(f"Wrote {config.SPLITS_ROOT}/  (blocked_target_ood.csv, random.csv, manifest.json, leakage_report.json)")
    return True


def load_split(path=config.BLOCKED_SPLIT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    import sys

    sys.exit(0 if run() else 1)
