"""Build the Ensembl -> HGNC -> UniProt -> Entrez identifier map with an ambiguity report.

Ensembl IDs and their HGNC symbols come from the DE object's ``.obs``
(perturbed targets) and ``.var`` (measured genes) via BACKED reads — the DE layers
are never loaded. UniProt/Entrez need an online source (mygene.info); when offline
those columns stay null and the row is tagged ``requires_online_lookup``.
"""
from __future__ import annotations

from typing import Callable

import anndata as ad
import pandas as pd

from tcell_pipeline import config

LookupFn = Callable[[list[str]], dict[str, dict[str, object]]]


def _missing(value: object) -> bool:
    """True for None / NaN / empty-ish symbols. astype(str) leaves NaN as float on pandas 3."""
    return pd.isna(value) or str(value) in ("", "nan", "None", "<NA>")


def read_ensembl_symbol_crossref() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (targets, measured) frames of [ensembl_id, hgnc_symbol] from backed DE reads."""
    a = ad.read_h5ad(config.DE_STATS_PATH, backed="r")
    obs = a.obs[["target_contrast", "target_contrast_gene_name"]].astype(str)
    targets = (
        obs.rename(columns={"target_contrast": "ensembl_id", "target_contrast_gene_name": "hgnc_symbol"})
        .drop_duplicates("ensembl_id")
        .reset_index(drop=True)
    )
    var = a.var.reset_index()
    var["ensembl_id"] = var["gene_ids"].astype(str)
    measured = (
        var[["ensembl_id", "gene_name"]]
        .rename(columns={"gene_name": "hgnc_symbol"})
        .astype(str)
        .drop_duplicates("ensembl_id")
        .reset_index(drop=True)
    )
    return targets, measured


def _mygene_lookup(ensembl_ids: list[str]) -> dict[str, dict[str, object]]:
    import mygene

    mg = mygene.MyGeneInfo()
    hits = mg.querymany(
        ensembl_ids, scopes="ensembl.gene",
        fields="symbol,uniprot.Swiss-Prot,entrezgene", species="human", returnall=False,
    )
    out: dict[str, dict[str, object]] = {}
    for h in hits:
        eid = h.get("query")
        if eid is None or h.get("notfound"):
            continue
        up = h.get("uniprot", {}).get("Swiss-Prot") if isinstance(h.get("uniprot"), dict) else None
        rec = out.setdefault(eid, {"symbol": h.get("symbol"), "uniprot": [], "entrez": h.get("entrezgene")})
        for u in ([up] if isinstance(up, str) else (up or [])):
            if u and u not in rec["uniprot"]:
                rec["uniprot"].append(u)
    return out


def build_id_mapping(
    targets: pd.DataFrame, measured: pd.DataFrame, lookup: LookupFn | None = None,
) -> tuple[pd.DataFrame, str]:
    """Merge target/measured Ensembl IDs into one mapping table + an ambiguity report string.

    ``lookup`` maps ensembl_id -> {"symbol", "uniprot": list, "entrez"}. When None the
    mapping is offline: UniProt/Entrez are null and rows are tagged requires_online_lookup.
    """
    target_ids = set(targets["ensembl_id"])
    measured_ids = set(measured["ensembl_id"])
    symbol_map: dict[str, str] = {}
    for df in (targets, measured):
        for eid, sym in zip(df["ensembl_id"], df["hgnc_symbol"]):
            symbol_map.setdefault(eid, sym)

    all_ids = sorted(target_ids | measured_ids)
    online = lookup(all_ids) if lookup is not None else {}

    rows = []
    one_to_many, unmapped_symbol, needs_online = [], [], []
    for eid in all_ids:
        rec = online.get(eid, {})
        hgnc = rec.get("symbol") or symbol_map.get(eid) or None
        uniprots = rec.get("uniprot") or []
        entrez = rec.get("entrez")
        uniprot = uniprots[0] if uniprots else None
        entrez = str(entrez) if entrez not in (None, "") else None

        if lookup is None:
            status = "requires_online_lookup"
            needs_online.append(eid)
        elif not uniprots and entrez is None:
            status = "unmapped"
        elif len(uniprots) > 1:
            status = "one-to-many-uniprot"
            one_to_many.append((eid, uniprots))
        else:
            status = "mapped"
        if _missing(hgnc):
            unmapped_symbol.append(eid)

        rows.append({
            "ensembl_id": eid,
            "hgnc_symbol": None if _missing(hgnc) else hgnc,
            "uniprot_id": uniprot,
            "entrez_id": entrez,
            "is_target": eid in target_ids,
            "is_measured": eid in measured_ids,
            "mapping_status": status,
        })

    mapping = pd.DataFrame(rows, columns=[
        "ensembl_id", "hgnc_symbol", "uniprot_id", "entrez_id",
        "is_target", "is_measured", "mapping_status",
    ])

    report = [
        "# Identifier mapping ambiguity report",
        f"total unique ensembl ids : {len(all_ids)}",
        f"  targets                : {len(target_ids)}",
        f"  measured               : {len(measured_ids)}",
        f"  target & measured      : {len(target_ids & measured_ids)}",
        f"unmapped hgnc symbol     : {len(unmapped_symbol)}",
        f"one-to-many uniprot      : {len(one_to_many)}",
        f"requires_online_lookup   : {len(needs_online)}",
        "",
        "## one-to-many uniprot (ensembl -> uniprots)",
        *[f"  {eid}\t{','.join(ups)}" for eid, ups in one_to_many[:200]],
        "",
        "## unmapped hgnc symbol (first 200)",
        *[f"  {eid}" for eid in unmapped_symbol[:200]],
    ]
    return mapping, "\n".join(report) + "\n"


def run() -> pd.DataFrame:
    print("[id_mapping] reading Ensembl/HGNC crossref from backed DE obs+var ...")
    targets, measured = read_ensembl_symbol_crossref()
    print(f"[id_mapping]   {len(targets)} targets, {len(measured)} measured genes")

    lookup: LookupFn | None = None
    try:
        import mygene  # noqa: F401
        print("[id_mapping] mygene present — attempting online UniProt/Entrez lookup ...")
        lookup = _mygene_lookup
    except Exception as exc:  # offline / not installed
        print(f"[id_mapping] mygene unavailable ({exc}); offline mode, UniProt/Entrez left null")

    try:
        mapping, report = build_id_mapping(targets, measured, lookup)
    except Exception as exc:
        print(f"[id_mapping] online lookup failed ({exc}); falling back to offline crossref")
        mapping, report = build_id_mapping(targets, measured, None)

    config.write_parquet_atomic(mapping, config.ID_MAPPING_PATH)
    config.write_text_atomic(report, config.AMBIGUITY_REPORT_PATH)
    print(f"[id_mapping] wrote {len(mapping)} rows -> {config.ID_MAPPING_PATH}")
    print(f"[id_mapping] wrote ambiguity report -> {config.AMBIGUITY_REPORT_PATH}")
    return mapping


if __name__ == "__main__":
    run()
