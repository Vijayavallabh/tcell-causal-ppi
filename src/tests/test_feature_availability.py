import pytest

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


def test_config_qpre_qpost_are_disjoint():
    # the real leakage guard: the output-level disjointness (test_qpre_qpost_disjoint) is a
    # structural tautology of classify_columns' if/elif/else. What actually protects the fence
    # is that no name appears in both config lists -- else q_post wins and the feature silently
    # drops out of q_pre while every output-level "disjoint" test stays green.
    assert set(config.Q_PRE_COLS).isdisjoint(config.Q_POST_COLS)


def test_donor_pc_prefix_requires_digits():
    # a response-derived column merely sharing the donor_pc_ prefix must NOT be auto-classified
    # q_pre; it falls through to metadata where the REVIEW tripwire can fire.
    m = classify_columns(["donor_pc_00", "donor_pc_response_score"])
    assert "donor_pc_00" in m["q_pre"]
    assert "donor_pc_response_score" in m["metadata"]


def test_committed_manifest_matches_current_config():
    # drift guard: re-classifying the committed manifest's own columns under the current config
    # must reproduce it exactly. Fails if Q_POST_COLS/Q_PRE_COLS changed without regenerating the
    # tracked YAML (run() is never exercised by tests and its parquet input is untracked).
    p = config.FEATURE_AVAILABILITY_PATH
    if not p.exists():
        pytest.skip("feature_availability.yaml not generated")
    import yaml

    manifest = yaml.safe_load(p.read_text())
    all_cols = manifest["q_pre"] + manifest["q_post"] + manifest["metadata"]
    assert classify_columns(all_cols) == manifest


def test_known_metadata_allowlisted_but_rogue_flagged():
    # the leakage-fence tripwire must stay quiet on known bookkeeping cols and fire on anything else.
    m = classify_columns([*COLUMNS, "mystery_response_col"])
    unexpected = [c for c in m["metadata"] if c not in config.KNOWN_METADATA_COLS]
    assert unexpected == ["mystery_response_col"]
    assert "row_index" not in unexpected and "mapping_status" not in unexpected
