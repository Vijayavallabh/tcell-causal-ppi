import pandas as pd

from tcell_pipeline import ppi_graph


def _frame(rows, source, **flags):
    df = pd.DataFrame(rows, columns=["source_gene", "target_gene"])
    df["source"] = source
    df["evidence_type"] = flags.pop("evidence", source)
    df["score"] = flags.pop("score", 1.0)
    for f in ppi_graph.BIN_FLAGS:
        df[f] = flags.get(f, 0)
    return df


STRING = _frame([("tp53", "MDM2"), ("myc", "max"), ("AAA", "AAA")], "string",
                is_functional=1, score=0.9)
BIOPLEX = _frame([("MDM2", "TP53"), ("egfr", "grb2")], "bioplex", is_physical=1, score=0.7)


def test_score_range_and_binary_flags():
    edges = ppi_graph.harmonize_edges([STRING, BIOPLEX])
    assert edges["score"].between(0.0, 1.0).all()
    for f in ppi_graph.BIN_FLAGS:
        assert edges[f].isin([0, 1]).all()


def test_missing_score_floors_to_zero_not_max():
    scored = _frame([("AAA", "BBB")], "string", is_functional=1, score=float("nan"))
    edges = ppi_graph.harmonize_edges([scored, BIOPLEX])
    pair = edges[(edges["source_gene"] == "AAA") & (edges["source"] == "string")]
    assert (pair["score"] == 0.0).all()  # unknown confidence must not become 1.0


def test_at_least_two_sources_and_dedup():
    edges = ppi_graph.harmonize_edges([STRING, BIOPLEX])
    assert edges["source"].nunique() >= 2
    assert (edges["source_gene"] == edges["target_gene"]).sum() == 0  # self-loop AAA dropped
    assert (edges["source_gene"].str.isupper() & edges["target_gene"].str.isupper()).all()
    # TP53-MDM2 appears in both sources (reversed) -> collapsed, 2 supporting sources
    pair = edges[(edges["source_gene"] == "MDM2") & (edges["target_gene"] == "TP53")]
    assert set(pair["source"]) == {"string", "bioplex"}
    assert (pair["n_supporting_sources"] == 2).all()
