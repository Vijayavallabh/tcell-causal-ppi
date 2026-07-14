import pandas as pd

from tcell_pipeline import id_mapping

TARGETS = pd.DataFrame({"ensembl_id": ["ENSG1", "ENSG2", "ENSG3"],
                        "hgnc_symbol": ["A1BG", "TP53", "MYC"]})
MEASURED = pd.DataFrame({"ensembl_id": ["ENSG3", "ENSG4"],
                         "hgnc_symbol": ["MYC", "GAPDH"]})

REQUIRED = ["ensembl_id", "hgnc_symbol", "uniprot_id", "entrez_id",
            "is_target", "is_measured", "mapping_status"]


def _lookup(ids):
    return {
        "ENSG1": {"symbol": "A1BG", "uniprot": ["P04217"], "entrez": 1},
        "ENSG2": {"symbol": "TP53", "uniprot": ["P04637", "K7PPA8"], "entrez": 7157},
        # ENSG3 intentionally missing -> unmapped; ENSG4 measured-only
        "ENSG4": {"symbol": "GAPDH", "uniprot": ["P04406"], "entrez": 2597},
    }


def test_columns_and_flags():
    mapping, _ = id_mapping.build_id_mapping(TARGETS, MEASURED, _lookup)
    assert list(mapping.columns) == REQUIRED
    row3 = mapping.set_index("ensembl_id").loc["ENSG3"]
    assert row3["is_target"] and row3["is_measured"]  # present in both sets
    assert mapping.set_index("ensembl_id").loc["ENSG4"]["is_target"] == False  # measured only


def test_one_to_many_and_unmapped():
    mapping, report = id_mapping.build_id_mapping(TARGETS, MEASURED, _lookup)
    st = mapping.set_index("ensembl_id")["mapping_status"]
    assert st["ENSG2"] == "one-to-many-uniprot"
    assert st["ENSG3"] == "unmapped"
    assert st["ENSG1"] == "mapped"
    assert "one-to-many uniprot" in report


def test_offline_requires_online_lookup():
    mapping, _ = id_mapping.build_id_mapping(TARGETS, MEASURED, None)
    assert (mapping["mapping_status"] == "requires_online_lookup").all()
    assert mapping["uniprot_id"].isna().all()
    assert mapping["hgnc_symbol"].notna().all()  # symbols still resolved offline
