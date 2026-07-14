"""Parse CORUM core complexes into a bipartite (protein, complex) membership table."""
from __future__ import annotations

import pandas as pd

from tcell_pipeline import config

MEMBERSHIP_COLS = ["protein_gene", "complex_id", "source_database", "confidence", "is_curated"]


def parse_corum_complexes(corum: pd.DataFrame) -> pd.DataFrame:
    """Explode CORUM rows into one (protein_gene, complex_id) row per subunit.

    CORUM is a manually curated resource, so every membership is is_curated=1.
    """
    gene_col = next((c for c in corum.columns if "subunits(Gene name)" in c or "Gene name" in c), None)
    id_col = next((c for c in corum.columns if c.lower() in ("complexid", "complex_id", "complexname")), None)
    if gene_col is None or id_col is None:
        return pd.DataFrame(columns=MEMBERSHIP_COLS)

    rows = []
    for cid, genes in zip(corum[id_col], corum[gene_col]):
        if pd.isna(genes):
            continue
        for g in str(genes).split(";"):
            g = g.strip().upper()
            if g:
                rows.append((g, str(cid)))
    out = pd.DataFrame(rows, columns=["protein_gene", "complex_id"]).drop_duplicates()
    out["source_database"] = "CORUM"
    out["confidence"] = 1.0
    out["is_curated"] = 1
    return out[MEMBERSHIP_COLS].reset_index(drop=True)


def run() -> pd.DataFrame:
    from tcell_pipeline.ppi_graph import SOURCE_URLS, _cache

    print("[complex_membership] loading CORUM core complexes ...")
    path = _cache("corum", SOURCE_URLS["corum"])
    if path is None:
        print("[complex_membership] CORUM unreachable; writing empty membership table")
        out = pd.DataFrame(columns=MEMBERSHIP_COLS)
    else:
        out = parse_corum_complexes(pd.read_csv(path, sep="\t", compression="infer"))
    config.write_parquet_atomic(out, config.COMPLEX_MEMBERSHIP_PATH)
    print(f"[complex_membership] wrote {len(out)} memberships "
          f"({out['complex_id'].nunique() if len(out) else 0} complexes) -> {config.COMPLEX_MEMBERSHIP_PATH}")
    return out


if __name__ == "__main__":
    run()
