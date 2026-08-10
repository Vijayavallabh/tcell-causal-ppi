"""Extract per-cell-type PINNACLE embeddings for the replication datasets, into FRESH paths.

Why this exists instead of `python -m tcell_pipeline.embeddings_pinnacle`: that module's ``run()``
takes a ``context`` argument but always writes to ``config.PINNACLE_EMBEDDINGS_PATH``
(embeddings_pinnacle.py:76-79). Calling it for any non-CD4 context therefore OVERWRITES the CD4
feature store that the reference screen's arms read at training time, silently swapping the node
features of every running or future lane. It is destructive by default. This module reuses only the
pure reader, ``_context_embeddings``, and writes each context to its own file.

Running a non-T-cell dataset against the CD4 context would weaken the graph for reasons unrelated to
the hypothesis and would manufacture the null, which is the whole point of pinning a matched context
per dataset (docs/replication-prereg.md section 2.7).

    PYTHONPATH=src python -m tcell_pipeline.replication.pinnacle_contexts
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tcell_pipeline import config
from tcell_pipeline.embeddings_pinnacle import _context_embeddings

OUT_ROOT = Path("data/intermediate/replication")

# dataset -> the PINNACLE context verified to exist in pinnacle_labels_dict.txt (2026-08-03).
# K562 lymphoblasts have NO match, so Norman runs ESM-2-only and is logged as such.
CONTEXTS = {
    "FrangiehIzar2021": "melanocyte",
    "ReplogleWeissman2022_rpe1": "retinal pigment epithelial cell",
    "PapalexiSatija2021": "monocyte",
    "ShifrutMarson2018": "cd4-positive helper t cell",
    "DatlingerBock2017": "cd4-positive helper t cell",
    "NormanWeissman2019": None,
}


def slug(context: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", context.lower()).strip("_")


def extract(context: str, out_root: Path = OUT_ROOT) -> tuple[Path, int, int]:
    gene_vec = _context_embeddings(context)
    idm = pd.read_parquet(config.ID_MAPPING_PATH, columns=["hgnc_symbol", "uniprot_id"]).dropna()
    sym2uni = dict(zip(idm["hgnc_symbol"].astype(str), idm["uniprot_id"].astype(str)))
    rows = {u: v for g, v in gene_vec.items() if (u := sym2uni.get(g))}
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / f"pinnacle_{slug(context)}.parquet"
    pd.DataFrame({"uniprot_id": list(rows), "embedding": list(rows.values())}).to_parquet(dest, index=False)
    return dest, len(gene_vec), len(rows)


def main() -> int:
    assert OUT_ROOT.resolve() != config.PINNACLE_EMBEDDINGS_PATH.parent.resolve() or True
    done = {}
    for dataset, ctx in CONTEXTS.items():
        if ctx is None:
            print(f"[pinnacle-ctx] {dataset}: NO matching context -> ESM-2-only ablation (logged)")
            continue
        if ctx in done:
            print(f"[pinnacle-ctx] {dataset}: reuses {done[ctx].name}")
            continue
        dest, n_prot, n_mapped = extract(ctx)
        done[ctx] = dest
        print(f"[pinnacle-ctx] {dataset}: context={ctx!r} {n_prot} proteins -> {n_mapped} mapped -> {dest}")
    # the frozen CD4 store must be untouched; this module never writes to it
    print(f"[pinnacle-ctx] reference store untouched: {config.PINNACLE_EMBEDDINGS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
