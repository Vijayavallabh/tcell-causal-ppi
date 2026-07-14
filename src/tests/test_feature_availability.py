from tcell_pipeline import config
from tcell_pipeline.feature_availability import classify_columns

COLUMNS = [
    "row_index", "target_contrast_gene_name", "culture_condition", "ensembl_id",
    "uniprot_id", "ppi_degree_physical", "control_baseline_expr",
    "donor_pc_00", "donor_pc_31", "mapping_status",
    *config.Q_POST_COLS,
]


def test_qpre_qpost_disjoint():
    m = classify_columns(COLUMNS)
    assert set(m["q_pre"]).isdisjoint(m["q_post"])


def test_all_qpost_cols_under_qpost():
    m = classify_columns(COLUMNS)
    for c in config.Q_POST_COLS:
        assert c in m["q_post"]
    assert not any(c in m["q_pre"] for c in config.Q_POST_COLS)


def test_donor_pcs_are_qpre_and_partition_is_complete():
    m = classify_columns(COLUMNS)
    assert "donor_pc_00" in m["q_pre"] and "donor_pc_31" in m["q_pre"]
    assert "row_index" in m["metadata"] and "mapping_status" in m["metadata"]
    assert len(m["q_pre"]) + len(m["q_post"]) + len(m["metadata"]) == len(COLUMNS)
