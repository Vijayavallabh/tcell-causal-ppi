"""L1-PRE step 1: extend id_mapping to a replication dataset's targets, BY COPY.

WHY THIS IS A SEPARATE MODULE. `data/intermediate/id_mapping.parquet` is load-bearing for
graph_builder, ppi_graph, perturbation_table, run_screening and reproducibility/manifest. Rewriting it
in place would invalidate every result in the project - the frozen H1, all three folds, all three
re-draws - and the damage would be invisible until numbers stopped reproducing. So this writes a NEW
file that is a strict superset, and sha256-verifies the frozen one before and after.

WHY IT IS NEEDED. id_mapping was built for the reference screen's 11,526 targets. Replication targets
outside that gene space resolve to nothing, and a target with no UniProt has no PPI graph node, so the
graph encoder has nothing to message-pass over and condition_gated silently degenerates to
expression_only for that row while still reporting a number. Measured on Replogle RPE1: 535 of 2,122
targets absent from id_mapping, 371 more without an ESM-2 vector, i.e. 43% of the graph arm would have
had no graph. That biases h1 toward zero - it manufactures the null.

    PYTHONPATH=src python -m tcell_pipeline.replication.extend_id_mapping --symbols-from <survey.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

OUT_DEFAULT = "data/intermediate/replication/id_mapping_extended.parquet"


def _sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _target_symbols(survey_paths: list[str]) -> set[str]:
    """Union of target symbols across surveyed datasets, applying the same guide collapse."""
    import anndata as ad

    syms: set[str] = set()
    for sp in survey_paths:
        for r in json.load(open(sp)):
            f = f"data/raw/scperturb/{r['dataset']}.h5ad"
            if not os.path.exists(f):
                continue
            a = ad.read_h5ad(f, backed="r")
            t = a.obs[r["target_column"]].astype(str)
            t = t[~t.str.lower().isin(["control", "nan", "none", ""])]
            t = t[~t.str.contains("_")]
            if r.get("guide_labels_collapsed_to_genes"):
                t = t.str.replace(r"g\d+$", "", regex=True).str.replace(r"-\d+$", "", regex=True)
            vc = t.value_counts()
            syms |= set(vc[vc >= 25].index)
            a.file.close()
    return {s for s in syms if s and not re.fullmatch(r"(nan|None)", s)}


def main() -> int:
    from tcell_pipeline import config
    from tcell_pipeline.id_mapping import _mygene_lookup, choose_uniprot  # noqa: F401

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-from", nargs="+",
                    default=["data/intermediate/replication/survey_l1.json",
                             "data/intermediate/replication/survey_tian.json"])
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--dry-run", action="store_true", help="report what is missing; resolve nothing")
    a = ap.parse_args()

    frozen_path = str(config.ID_MAPPING_PATH)
    before = _sha256(frozen_path)
    frozen = pd.read_parquet(frozen_path)
    known = set(frozen["hgnc_symbol"].dropna().astype(str))

    wanted = _target_symbols([p for p in a.symbols_from if os.path.exists(p)])
    missing = sorted(wanted - known)
    print(f"[extend] replication targets (>=25 cells): {len(wanted)}")
    print(f"[extend] already in frozen id_mapping    : {len(wanted & known)}")
    print(f"[extend] MISSING, need resolution        : {len(missing)}")
    if missing[:8]:
        print(f"[extend]   e.g. {missing[:8]}")
    if a.dry_run:
        print("[extend] dry run: nothing written")
        return 0

    # Resolve by SYMBOL via mygene. Failures are expected and fine - deprecated symbols and novel
    # transcripts (AC118549.1) have no UniProt, and a target that genuinely has no protein is a real
    # exclusion rather than a bug. They are recorded so the paper can report per-dataset coverage.
    resolved, failed = [], []
    try:
        import mygene

        mg = mygene.MyGeneInfo()

        def _query(syms: list[str], scopes: str, status: str) -> list[str]:
            """Resolve `syms` under `scopes`; append hits to `resolved`, return the misses."""
            if not syms:
                return []
            miss = []
            for h in mg.querymany(syms, scopes=scopes, fields="symbol,uniprot.Swiss-Prot,entrezgene",
                                  species="human", returnall=False):
                sym = h.get("query")
                up = h.get("uniprot", {}).get("Swiss-Prot") if isinstance(h.get("uniprot"), dict) else None
                ups = [up] if isinstance(up, str) else (up or [])
                if not ups:
                    miss.append(sym); continue
                chosen, alts, amb = choose_uniprot(list(ups))
                resolved.append({"ensembl_id": None, "hgnc_symbol": sym, "uniprot_id": chosen,
                                 "uniprot_alternatives": ";".join(alts), "uniprot_ambiguous": amb,
                                 "entrez_id": h.get("entrezgene"), "is_target": True,
                                 "is_measured": False, "mapping_status": status})
            return miss

        # Pass 1: current official symbols.
        miss1 = _query(missing, "symbol", "replication_extension")
        # Pass 2: DEPRECATED SYMBOLS. Perturb-seq screens are often annotated against an older gene
        # build, so a "miss" is frequently a rename rather than a gene without a protein: AARS ->
        # AARS1, ATP5A1 -> ATP5F1A, C10orf54 -> VSIR. Retrying under alias and previous-symbol scopes
        # recovers these. Skipping this pass would silently shrink the graph arm on exactly the older
        # datasets, which is the "manufactured null" hazard arriving through gene nomenclature.
        failed.extend(_query(miss1, "alias,retired,symbol", "replication_extension_via_alias"))
    except Exception as exc:
        print(f"[extend] mygene unavailable ({type(exc).__name__}: {exc}); nothing resolved")
        return 1

    ext = pd.concat([frozen, pd.DataFrame(resolved)], ignore_index=True) if resolved else frozen
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ext.to_parquet(a.out, index=False)

    after = _sha256(frozen_path)
    ok = before == after
    print(f"[extend] resolved {len(resolved)}, unresolvable {len(failed)} "
          f"(deprecated symbols / novel transcripts are expected)")
    print(f"[extend] wrote {a.out}: {len(frozen)} frozen + {len(resolved)} new = {len(ext)} rows")
    print(f"[extend] frozen id_mapping {'VERIFIED byte-identical' if ok else '*** CHANGED - INVESTIGATE ***'}")
    json.dump({"wanted": len(wanted), "already_known": len(wanted & known), "missing": len(missing),
               "resolved": len(resolved), "unresolvable": failed[:200],
               "frozen_sha256_unchanged": ok},
              open(a.out.replace(".parquet", "_report.json"), "w"), indent=2)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
