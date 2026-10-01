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
ReviewedFn = Callable[[list[str]], dict[str, dict[str, float]]]


def _missing(value: object) -> bool:
    """True for None / NaN / empty-ish symbols. astype(str) leaves NaN as float on pandas 3."""
    return pd.isna(value) or str(value) in ("", "nan", "None", "<NA>")


def choose_uniprot(
    uniprots: list[str], reviewed_scores: dict[str, float] | None = None,
) -> tuple[str | None, list[str], bool]:
    """Pick one Swiss-Prot accession when an Ensembl gene maps to several.

    ``reviewed_scores`` = {accession: annotation_score} for the accessions that are the
    reviewed human canonical of this gene's HGNC symbol (empty when offline/unresolved).
    Preference: reviewed-canonical > higher UniProt annotation score > lexical accession
    (reproducible tie-break). Returns (chosen, alternatives, ambiguous). ``ambiguous`` is
    True ONLY when the top candidate is not strictly better-scored than the runner-up —
    i.e. equal-evidence reviewed entries, the genuine multi-product loci (CDKN2A p16/p14ARF,
    GNAS). Paralog families (one reviewed canonical) and score-decisive picks (a fragment or
    secondary entry loses on annotation score) are confidently resolved and NOT flagged.
    """
    if not uniprots:
        return None, [], False
    scores = reviewed_scores or {}
    pool = [u for u in uniprots if u in scores] or list(uniprots)
    ordered = sorted(pool, key=lambda a: (-scores.get(a, 0.0), a))
    chosen = ordered[0]
    alternatives = [u for u in uniprots if u != chosen]
    ambiguous = len(ordered) > 1 and scores.get(ordered[0], 0.0) == scores.get(ordered[1], 0.0)
    return chosen, alternatives, ambiguous


def _uniprot_reviewed(symbols: list[str]) -> dict[str, dict[str, float]]:
    """symbol -> {accession: annotation_score} for reviewed human entries. Best-effort."""
    import json
    import urllib.parse
    import urllib.request

    out: dict[str, dict[str, float]] = {}
    for sym in symbols:
        query = f"gene_exact:{sym} AND organism_id:9606 AND reviewed:true"
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {"query": query, "fields": "accession,annotation_score", "format": "json", "size": 25})
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.load(resp)
            out[sym] = {e["primaryAccession"]: float(e.get("annotationScore", 0) or 0)
                        for e in data.get("results", [])}
        except Exception:
            out[sym] = {}
    return out


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
    reviewed_lookup: ReviewedFn | None = None,
) -> tuple[pd.DataFrame, str]:
    """Merge target/measured Ensembl IDs into one mapping table + an ambiguity report string.

    ``lookup`` maps ensembl_id -> {"symbol", "uniprot": list, "entrez"}. When None the
    mapping is offline: UniProt/Entrez are null and rows are tagged requires_online_lookup.
    ``reviewed_lookup`` maps hgnc_symbol -> {accession: annotation_score} for reviewed human
    entries; when a gene has >1 candidate accession it is used to pick the canonical one (see
    ``choose_uniprot``). When None, multi-accession genes fall back to a deterministic pick.
    """
    target_ids = set(targets["ensembl_id"])
    measured_ids = set(measured["ensembl_id"])
    symbol_map: dict[str, str] = {}
    for df in (targets, measured):
        for eid, sym in zip(df["ensembl_id"], df["hgnc_symbol"]):
            symbol_map.setdefault(eid, sym)

    all_ids = sorted(target_ids | measured_ids)
    online = lookup(all_ids) if lookup is not None else {}

    candidates: dict[str, tuple[str | None, list[str], str | None]] = {}
    for eid in all_ids:
        rec = online.get(eid, {})
        hgnc = rec.get("symbol") or symbol_map.get(eid) or None
        uniprots = rec.get("uniprot") or []
        entrez = rec.get("entrez")
        entrez = str(entrez) if entrez not in (None, "") else None
        candidates[eid] = (None if _missing(hgnc) else hgnc, uniprots, entrez)

    ambig_syms = sorted({h for h, ups, _ in candidates.values() if h and len(ups) > 1})
    reviewed = reviewed_lookup(ambig_syms) if (reviewed_lookup and ambig_syms) else {}

    rows = []
    one_to_many, resolved_multi, unmapped_symbol, needs_online = [], 0, [], []
    for eid in all_ids:
        hgnc, uniprots, entrez = candidates[eid]
        uniprot, alternatives, ambiguous = choose_uniprot(uniprots, reviewed.get(hgnc))

        if lookup is None:
            status = "requires_online_lookup"
            needs_online.append(eid)
        elif not uniprots and entrez is None:
            status = "unmapped"
        elif ambiguous:
            status = "one-to-many-uniprot"
            one_to_many.append((eid, hgnc, uniprot, alternatives))
        else:
            status = "mapped"
            if len(uniprots) > 1:
                resolved_multi += 1
        if _missing(hgnc):
            unmapped_symbol.append(eid)

        rows.append({
            "ensembl_id": eid,
            "hgnc_symbol": hgnc,
            "uniprot_id": uniprot,
            "uniprot_alternatives": "|".join(alternatives) if alternatives else None,
            "uniprot_ambiguous": bool(ambiguous),
            "entrez_id": entrez,
            "is_target": eid in target_ids,
            "is_measured": eid in measured_ids,
            "mapping_status": status,
        })

    mapping = pd.DataFrame(rows, columns=[
        "ensembl_id", "hgnc_symbol", "uniprot_id", "uniprot_alternatives", "uniprot_ambiguous",
        "entrez_id", "is_target", "is_measured", "mapping_status",
    ])

    report = [
        "# Identifier mapping ambiguity report",
        f"total unique ensembl ids : {len(all_ids)}",
        f"  targets                : {len(target_ids)}",
        f"  measured               : {len(measured_ids)}",
        f"  target & measured      : {len(target_ids & measured_ids)}",
        f"unmapped hgnc symbol     : {len(unmapped_symbol)}",
        f"multi-accession genes    : {len(one_to_many) + resolved_multi}",
        f"  resolved (canonical)   : {resolved_multi}",
        f"  remain ambiguous       : {len(one_to_many)}",
        f"requires_online_lookup   : {len(needs_online)}",
        "",
        "## ambiguous uniprot (ensembl  chosen  <- alternatives)  [reviewed canonical, score/lexical tie-break]",
        *[f"  {eid} {hgnc}\t{chosen}\t<- {','.join(alts)}" for eid, hgnc, chosen, alts in one_to_many[:200]],
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

    reviewed_lookup: ReviewedFn | None = _uniprot_reviewed if lookup is not None else None
    try:
        mapping, report = build_id_mapping(targets, measured, lookup, reviewed_lookup)
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
