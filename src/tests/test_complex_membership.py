import pandas as pd

from tcell_pipeline import complex_membership as cm
from tcell_pipeline.ppi_graph import _corum_gene_col

# CORUM 5.x schema (underscored) and the legacy schema must both be understood.
CORUM_5X = pd.DataFrame({
    "complex_id": [1, 2],
    "subunits_gene_name": ["BCL6;HDAC4", "MYC;MAX;MXD1"],
    "subunits_gene_name_synonyms": ["x;y", "a;b;c"],  # must NOT be picked as the gene column
})
CORUM_LEGACY = pd.DataFrame({
    "ComplexID": [7],
    "subunits(Gene name)": ["TP53;MDM2"],
})


def test_gene_col_picks_genes_not_synonyms():
    assert _corum_gene_col(CORUM_5X.columns) == "subunits_gene_name"
    assert _corum_gene_col(CORUM_LEGACY.columns) == "subunits(Gene name)"
    assert _corum_gene_col(["complex_id", "pmid"]) is None


def test_parse_corum_5x_and_legacy():
    out = cm.parse_corum_complexes(CORUM_5X)
    assert list(out.columns) == cm.MEMBERSHIP_COLS
    assert set(out["protein_gene"]) == {"BCL6", "HDAC4", "MYC", "MAX", "MXD1"}
    assert out["complex_id"].nunique() == 2
    assert (out["is_curated"] == 1).all() and (out["source_database"] == "CORUM").all()

    legacy = cm.parse_corum_complexes(CORUM_LEGACY)
    assert set(legacy["protein_gene"]) == {"TP53", "MDM2"}
