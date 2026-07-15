import pandas as pd

from tcell_pipeline import id_mapping

TARGETS = pd.DataFrame({"ensembl_id": ["ENSG1", "ENSG2", "ENSG3"],
                        "hgnc_symbol": ["A1BG", "TP53", "MYC"]})
MEASURED = pd.DataFrame({"ensembl_id": ["ENSG3", "ENSG4"],
                         "hgnc_symbol": ["MYC", "GAPDH"]})

REQUIRED = ["ensembl_id", "hgnc_symbol", "uniprot_id", "uniprot_alternatives",
            "uniprot_ambiguous", "entrez_id", "is_target", "is_measured", "mapping_status"]


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
    m = mapping.set_index("ensembl_id")
    st = m["mapping_status"]
    assert st["ENSG2"] == "one-to-many-uniprot"  # 2 candidates, no reviewed lookup -> flagged
    assert m.loc["ENSG2", "uniprot_id"] == "K7PPA8"  # deterministic lexical pick
    assert m.loc["ENSG2", "uniprot_alternatives"] == "P04637"  # the other kept, not lost
    assert st["ENSG3"] == "unmapped"
    assert st["ENSG1"] == "mapped"
    assert "ambiguous uniprot" in report


def test_reviewed_lookup_resolves_multi():
    # TP53's reviewed canonical is P04637 -> collapse to one, drop the ambiguous flag.
    reviewed = {"TP53": {"P04637": 5.0}}
    mapping, _ = id_mapping.build_id_mapping(TARGETS, MEASURED, _lookup, lambda syms: reviewed)
    m = mapping.set_index("ensembl_id")
    assert m.loc["ENSG2", "mapping_status"] == "mapped"
    assert m.loc["ENSG2", "uniprot_id"] == "P04637"
    assert not m.loc["ENSG2", "uniprot_ambiguous"]
    assert m.loc["ENSG2", "uniprot_alternatives"] == "K7PPA8"  # runner-up retained


def test_choose_uniprot_prefers_score_then_lexical():
    assert id_mapping.choose_uniprot([]) == (None, [], False)
    # score-decisive: winner strictly higher -> confidently resolved, NOT ambiguous
    chosen, alts, amb = id_mapping.choose_uniprot(["Q9", "P1"], {"Q9": 5.0, "P1": 2.0})
    assert chosen == "Q9" and alts == ["P1"] and amb is False
    # tied score -> lexical pick; genuinely ambiguous (GNAS/CDKN2A shape)
    chosen, _, amb = id_mapping.choose_uniprot(["P63092", "O95467"], {"P63092": 5.0, "O95467": 5.0})
    assert chosen == "O95467" and amb is True
    # only one accession is the reviewed canonical -> not ambiguous (paralog-family shape)
    chosen, alts, amb = id_mapping.choose_uniprot(["P0DP24", "P0DP23", "P0DP25"], {"P0DP23": 5.0})
    assert chosen == "P0DP23" and amb is False and set(alts) == {"P0DP24", "P0DP25"}


def test_offline_requires_online_lookup():
    mapping, _ = id_mapping.build_id_mapping(TARGETS, MEASURED, None)
    assert (mapping["mapping_status"] == "requires_online_lookup").all()
    assert mapping["uniprot_id"].isna().all()
    assert mapping["hgnc_symbol"].notna().all()  # symbols still resolved offline
