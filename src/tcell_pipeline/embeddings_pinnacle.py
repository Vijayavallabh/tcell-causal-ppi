"""Map real PINNACLE CD4+ T-cell protein embeddings (128-d) onto the mart's UniProt IDs.

PINNACLE (Li et al., Nature Methods 2024) gives context-specific protein representations.
The screen is CD4+ T cells, so we take the ``cd4-positive helper t cell`` context. PINNACLE
keys proteins by gene symbol; we map those to UniProt via id_mapping. Proteins absent from the
context keep the zero fallback (contextual embeddings only cover in-network proteins).

Frozen features, not trainable -> config.PINNACLE_EMBEDDINGS_PATH, where PluggableEmbeddingStore
picks them up. Downloads the 1.2 GB Figshare resource on first run if the raw dir is absent.

    python -m tcell_pipeline.embeddings_pinnacle
"""
from __future__ import annotations

import ast
import zipfile

import numpy as np
import pandas as pd

from tcell_pipeline import config


def download() -> None:
    """Fetch + unzip the PINNACLE embeds resource into DATA_ROOT/pinnacle if missing."""
    if config.PINNACLE_RAW_DIR.exists():
        return
    import urllib.request

    dest = config.DATA_ROOT / "pinnacle"
    config.ensure_dir(dest)
    zip_path = dest / "pinnacle_embeds.zip"
    print(f"[pinnacle] downloading {config.PINNACLE_FIGSHARE_URL} -> {zip_path}", flush=True)
    urllib.request.urlretrieve(config.PINNACLE_FIGSHARE_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    print(f"[pinnacle] unzipped -> {dest}", flush=True)


def _context_embeddings(context: str) -> dict[str, np.ndarray]:
    """{gene_symbol: 128-d vector} for one PINNACLE cell-type context."""
    import torch

    embed = torch.load(config.PINNACLE_RAW_DIR / "pinnacle_protein_embed.pth", map_location="cpu")
    labels = ast.literal_eval((config.PINNACLE_RAW_DIR / "pinnacle_labels_dict.txt").read_text())
    # cell-type contexts are the CCI_-prefixed rows, in order; index i keys the embed dict.
    celltypes = [c for c in labels["Cell Type"] if c.startswith("CCI")]
    names_by_idx = {c.split("CCI_")[1]: i for i, c in enumerate(celltypes)}
    if context not in names_by_idx:
        raise ValueError(f"PINNACLE context {context!r} not found; available e.g. {list(names_by_idx)[:5]}")
    idx = names_by_idx[context]
    # protein rows for this context are row-aligned to embed[idx] in label order.
    genes = [n for n, c in zip(labels["Name"], labels["Cell Type"]) if c == context]
    mat = embed[idx].to(torch.float32).numpy()
    if len(genes) != mat.shape[0]:
        raise ValueError(f"PINNACLE {context}: {len(genes)} gene labels vs {mat.shape[0]} embed rows")
    return dict(zip(genes, mat))


def run(context: str | None = None) -> None:
    context = context or config.PINNACLE_CONTEXT
    download()
    gene_vec = _context_embeddings(context)

    idm = pd.read_parquet(config.ID_MAPPING_PATH, columns=["hgnc_symbol", "uniprot_id"]).dropna()
    # gene symbol -> uniprot; keep the first uniprot per gene (id_mapping is 1 gene : 1 canonical).
    sym2uni = dict(zip(idm["hgnc_symbol"].astype(str), idm["uniprot_id"].astype(str)))

    rows: dict[str, np.ndarray] = {}
    for gene, vec in gene_vec.items():
        uni = sym2uni.get(gene)
        if uni:
            rows[uni] = vec  # last write wins on the rare gene collision; vectors are near-identical
    print(f"[pinnacle] context={context!r}: {len(gene_vec)} proteins -> {len(rows)} mapped to UniProt", flush=True)

    config.write_parquet_atomic(
        pd.DataFrame({"uniprot_id": list(rows), "embedding": list(rows.values())}),
        config.PINNACLE_EMBEDDINGS_PATH,
    )
    print(f"[pinnacle] wrote {len(rows)} embeddings -> {config.PINNACLE_EMBEDDINGS_PATH}", flush=True)


if __name__ == "__main__":
    run()
