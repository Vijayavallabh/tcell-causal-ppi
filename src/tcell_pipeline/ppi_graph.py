"""Download and harmonize typed protein-interaction edges from 5 sources.

Sources carry different edge semantics, preserved as typed flags rather than merged
away: is_physical (direct/co-complex physical), is_functional (STRING association),
is_complex (co-complex from CORUM), is_direct_binary (Y2H / two-hybrid). Any source
that is unreachable is logged and skipped — this step never crashes on a bad download.
Downloads are cached under data/raw/ppi/<source>/.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tcell_pipeline import config

EDGE_IN_COLS = [
    "source_gene", "target_gene", "source", "evidence_type",
    "score", "is_physical", "is_functional", "is_complex", "is_direct_binary",
]
BIN_FLAGS = ["is_physical", "is_functional", "is_complex", "is_direct_binary"]
EDGE_OUT_COLS = EDGE_IN_COLS + ["n_supporting_sources"]

SOURCE_URLS = {
    "bioplex": "https://bioplex.hms.harvard.edu/data/BioPlex_293T_Network_10K_Dec_2019.tsv",
    "huri": "https://interactome-atlas.org/data/HuRI.tsv",
    "biogrid": "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/"
               "BIOGRID-4.4.235/BIOGRID-ORGANISM-4.4.235.tab3.zip",
    "string_links": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
    "string_info": "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
    "corum": "https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/"
             "download_current_file?file_id=human&file_format=txt",
}


def harmonize_edges(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concat typed per-source edges into one undirected, deduplicated edge table.

    Genes are uppercased; each undirected pair is canonicalized (lo, hi) so A-B and B-A
    collapse. One row per (pair, source) keeps max score and OR-ed type flags;
    n_supporting_sources counts distinct sources backing each pair.
    """
    df = pd.concat([f[EDGE_IN_COLS] for f in frames], ignore_index=True)
    for c in ("source_gene", "target_gene"):
        df[c] = df[c].astype(str).str.upper().str.strip()
    bad = {"", "NAN", "NONE"}
    df = df[~df["source_gene"].isin(bad) & ~df["target_gene"].isin(bad)]
    df = df[df["source_gene"] != df["target_gene"]]

    swap = df["source_gene"] > df["target_gene"]
    lo = df["source_gene"].where(~swap, df["target_gene"])
    hi = df["target_gene"].where(~swap, df["source_gene"])
    df["source_gene"], df["target_gene"] = lo, hi

    # Missing/unparseable confidence -> 0.0 (min), never 1.0: binary sources already set 1.0
    # in their parser, so only genuinely-unknown scored values land at the floor.
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    for c in BIN_FLAGS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0, 1)

    agg = df.groupby(["source_gene", "target_gene", "source"], as_index=False).agg(
        evidence_type=("evidence_type", "first"),
        score=("score", "max"),
        is_physical=("is_physical", "max"),
        is_functional=("is_functional", "max"),
        is_complex=("is_complex", "max"),
        is_direct_binary=("is_direct_binary", "max"),
    )
    agg["n_supporting_sources"] = agg.groupby(["source_gene", "target_gene"])["source"].transform("nunique")
    return agg[EDGE_OUT_COLS].reset_index(drop=True)


# --- downloads (best-effort; return None on any failure) ---------------------

# ponytail: the helmholtz CORUM host serves a broken TLS chain (certifi also fails to
# verify it); skip verification for this one source rather than weaken it for all.
_NO_TLS_VERIFY = {"corum"}


def _cache(source: str, url: str) -> Path | None:
    dest = config.PPI_CACHE_ROOT / source / Path(url.split("?")[0]).name
    if dest.exists():
        return dest
    config.ensure_dir(dest.parent)
    try:
        import requests

        verify = source not in _NO_TLS_VERIFY
        if not verify:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        with requests.get(url, stream=True, timeout=60, verify=verify) as r:
            r.raise_for_status()
            tmp = dest.with_name(dest.name + ".tmp")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
        return dest
    except Exception as exc:
        print(f"[ppi_graph]   WARNING {source} unreachable ({exc}); skipping")
        return None


def _edges(pairs: pd.DataFrame, source: str, evidence: str, *, phys=0, func=0, cplx=0, direct=0) -> pd.DataFrame:
    pairs = pairs.rename(columns={pairs.columns[0]: "source_gene", pairs.columns[1]: "target_gene"}).copy()
    pairs["source"] = source
    pairs["evidence_type"] = evidence
    if "score" not in pairs:
        pairs["score"] = 1.0
    pairs["is_physical"], pairs["is_functional"] = phys, func
    pairs["is_complex"], pairs["is_direct_binary"] = cplx, direct
    return pairs[EDGE_IN_COLS]


def _load_bioplex() -> pd.DataFrame | None:
    p = _cache("bioplex", SOURCE_URLS["bioplex"])
    if p is None:
        return None
    df = pd.read_csv(p, sep="\t")
    out = df[["SymbolA", "SymbolB"]].copy()
    out["score"] = pd.to_numeric(df.get("pInt"), errors="coerce").fillna(0.0)
    return _edges(out, "bioplex", "AP-MS", phys=1)


def _extend_ens2sym_online(ensembl_ids: list[str], ens2sym: dict[str, str]) -> dict[str, str]:
    """Best-effort: fill Ensembl IDs missing from the DE-subset map via mygene (no-op offline)."""
    missing = [e for e in ensembl_ids if e not in ens2sym]
    if not missing:
        return ens2sym
    try:
        import mygene

        hits = mygene.MyGeneInfo().querymany(missing, scopes="ensembl.gene", fields="symbol", species="human")
        merged = dict(ens2sym)
        for h in hits:
            if not h.get("notfound") and h.get("symbol"):
                merged[h["query"]] = h["symbol"]
        return merged
    except Exception as exc:
        print(f"[ppi_graph]   huri: mygene extension unavailable ({exc}); mapping DE-subset only")
        return ens2sym


def _load_huri(ens2sym: dict[str, str] | None) -> pd.DataFrame | None:
    p = _cache("huri", SOURCE_URLS["huri"])
    if p is None or not ens2sym:
        if p is not None:
            print("[ppi_graph]   WARNING huri needs an ensembl->symbol map (run id_mapping first); skipping")
        return None
    df = pd.read_csv(p, sep="\t", header=None).iloc[:, :2]
    df.columns = ["a", "b"]
    ensembl_ids = pd.unique(pd.concat([df["a"], df["b"]]).astype(str)).tolist()
    ens2sym = _extend_ens2sym_online(ensembl_ids, ens2sym)
    df["a"] = df["a"].map(ens2sym)
    df["b"] = df["b"].map(ens2sym)
    kept = df.dropna()
    dropped = len(df) - len(kept)
    if dropped:
        print(f"[ppi_graph]   huri: dropped {dropped}/{len(df)} edges with an unmapped Ensembl endpoint")
    return _edges(kept, "huri", "Y2H", phys=1, direct=1)


def _load_biogrid() -> pd.DataFrame | None:
    import zipfile

    p = _cache("biogrid", SOURCE_URLS["biogrid"])
    if p is None:
        return None
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            members = z.namelist()
            member = next((m for m in members if "homo_sapiens" in m.lower()), members[0])
            with z.open(member) as fh:
                df = pd.read_csv(fh, sep="\t", low_memory=False)
    else:
        df = pd.read_csv(p, sep="\t", compression="infer", low_memory=False)
    phys = df[df["Experimental System Type"].str.lower() == "physical"]
    out = phys[["Official Symbol Interactor A", "Official Symbol Interactor B"]].copy()
    is_binary = (phys["Experimental System"].str.contains("hybrid", case=False, na=False)).astype(int)
    edges = _edges(out, "biogrid", "physical", phys=1)
    edges["is_direct_binary"] = is_binary.values
    return edges


def _load_string() -> pd.DataFrame | None:
    links = _cache("string", SOURCE_URLS["string_links"])
    info = _cache("string", SOURCE_URLS["string_info"])
    if links is None or info is None:
        return None
    meta = pd.read_csv(info, sep="\t")
    id2sym = dict(zip(meta["#string_protein_id"], meta["preferred_name"]))
    df = pd.read_csv(links, sep=" ")
    df["a"] = df["protein1"].map(id2sym)
    df["b"] = df["protein2"].map(id2sym)
    df = df.dropna(subset=["a", "b"])
    out = df[["a", "b"]].copy()
    out["score"] = df["combined_score"] / 1000.0
    return _edges(out, "string", "functional-association", func=1)


def _corum_gene_col(columns) -> str | None:
    # matches old "subunits(Gene name)" and CORUM 5.x "subunits_gene_name"; excludes the synonyms column.
    return next((c for c in columns
                 if "gene name" in c.lower().replace("_", " ") and "synonym" not in c.lower()), None)


def _load_corum() -> pd.DataFrame | None:
    p = _cache("corum", SOURCE_URLS["corum"])
    if p is None:
        return None
    df = pd.read_csv(p, sep="\t", compression="infer")
    col = _corum_gene_col(df.columns)
    if col is None:
        return None
    rows = []
    for genes in df[col].dropna():
        members = [g.strip() for g in str(genes).split(";") if g.strip()]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                rows.append((members[i], members[j]))
    pairs = pd.DataFrame(rows, columns=["a", "b"])
    return _edges(pairs, "corum", "co-complex", phys=1, cplx=1)


def run() -> pd.DataFrame:
    ens2sym = None
    if config.ID_MAPPING_PATH.exists():
        m = pd.read_parquet(config.ID_MAPPING_PATH)
        ens2sym = dict(zip(m["ensembl_id"], m["hgnc_symbol"].astype(str)))

    print("[ppi_graph] loading PPI sources (unreachable ones are skipped) ...")
    loaders = [_load_bioplex, lambda: _load_huri(ens2sym), _load_biogrid, _load_string, _load_corum]
    frames: list[pd.DataFrame] = []
    for load in loaders:
        try:
            f = load()
        except Exception as exc:
            print(f"[ppi_graph]   WARNING source failed to parse ({exc}); skipping")
            f = None
        if f is not None and len(f):
            print(f"[ppi_graph]   +{len(f)} edges from {f['source'].iloc[0]}")
            frames.append(f)

    if not frames:
        print("[ppi_graph] no sources available; writing empty edge table")
        edges = pd.DataFrame(columns=EDGE_OUT_COLS)
    else:
        edges = harmonize_edges(frames)
    config.write_parquet_atomic(edges, config.PROTEIN_EDGES_PATH)
    n_src = edges["source"].nunique() if len(edges) else 0
    print(f"[ppi_graph] wrote {len(edges)} edges from {n_src} sources -> {config.PROTEIN_EDGES_PATH}")
    return edges


if __name__ == "__main__":
    run()
