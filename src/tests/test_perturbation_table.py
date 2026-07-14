import numpy as np
import pandas as pd

from tcell_pipeline import perturbation_table as pt

DE_OBS = pd.DataFrame({
    "target_contrast_gene_name": ["A1BG", "TP53", "MYC", "A1BG"],
    "culture_condition": ["Rest", "Rest", "Stim48hr", "Stim48hr"],
    "target_contrast": ["ENSG1", "ENSG2", "ENSG3", "ENSG1"],
    "ontarget_significant": [True, False, True, True],
    "n_up_genes": [5, 0, 12, 3],
})

ID_MAP = pd.DataFrame({
    "ensembl_id": ["ENSG1", "ENSG2"],  # ENSG3 intentionally absent -> unmapped, kept
    "hgnc_symbol": ["A1BG", "TP53"],
    "uniprot_id": ["P04217", "P04637"],
    "entrez_id": ["1", "7157"],
    "mapping_status": ["mapped", "mapped"],
})


def test_row_index_unique_contiguous_and_columns():
    t = pt.build_perturbation_table(DE_OBS, ID_MAP)
    assert t["row_index"].tolist() == list(range(len(DE_OBS)))
    assert t["row_index"].is_unique
    for c in ["target_contrast_gene_name", "culture_condition", "ensembl_id",
              "uniprot_id", "control_baseline_expr", *pt.PPI_DEGREE_COLS]:
        assert c in t.columns
    assert "ontarget_significant" in t.columns and "n_up_genes" in t.columns  # q_post carried


def test_unmapped_rows_kept_and_defaults():
    t = pt.build_perturbation_table(DE_OBS, ID_MAP)
    assert len(t) == len(DE_OBS)  # nothing dropped
    unmapped = t[t["ensembl_id"] == "ENSG3"]
    assert len(unmapped) == 1 and unmapped["uniprot_id"].isna().all()
    assert (t[pt.PPI_DEGREE_COLS] == 0).all().all()
    assert t["control_baseline_expr"].isna().all()
