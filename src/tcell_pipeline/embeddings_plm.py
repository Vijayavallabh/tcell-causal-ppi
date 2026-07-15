"""Generate real ESM-2 650M per-protein embeddings (1280-d) for the mart's proteins.

Sequences are fetched from the UniProt REST ``accessions`` endpoint; each protein's
embedding is the mean of its final-layer residue representations (BOS/EOS/pad excluded).
Frozen features, not trainable — written to ``config.PLM_EMBEDDINGS_PATH`` where
``PluggableEmbeddingStore`` picks them up (missing proteins keep the zero fallback).

Resumable: proteins already in the output parquet (and sequences already cached) are
skipped, and the parquet is rewritten atomically every ``checkpoint`` proteins. Runs on
GPU when available (falls back to CPU), so ~11k proteins finish quickly on an A100:

    python -m tcell_pipeline.embeddings_plm
"""
from __future__ import annotations

import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from tcell_pipeline import config

SEQUENCE_CACHE_PATH = config.INTERMEDIATE_ROOT / "uniprot_sequences.parquet"
MAX_RESIDUES = 1022          # ESM-2 position budget (1024 incl. BOS/EOS)
_UNIPROT_BATCH = 250         # accessions per REST call (endpoint cap is 500)
ESM_LAYER = 33               # esm2_t33_650M final representation layer


def _unique_uniprot_ids() -> list[str]:
    df = pd.read_parquet(config.PERTURBATION_CONDITION_PATH, columns=["uniprot_id"])
    return sorted({str(u) for u in df["uniprot_id"].dropna().unique()})


def _parse_fasta(text: str) -> dict[str, str]:
    seqs: dict[str, str] = {}
    acc = None
    for line in text.splitlines():
        if line.startswith(">"):
            # header: >db|ACCESSION|NAME ...
            parts = line[1:].split("|")
            acc = parts[1] if len(parts) > 1 else line[1:].split()[0]
            seqs[acc] = ""
        elif acc is not None:
            seqs[acc] += line.strip()
    return seqs


def fetch_sequences(uniprot_ids: list[str]) -> dict[str, str]:
    """Fetch canonical sequences from UniProt, caching to SEQUENCE_CACHE_PATH (resumable)."""
    cache: dict[str, str] = {}
    if SEQUENCE_CACHE_PATH.exists():
        c = pd.read_parquet(SEQUENCE_CACHE_PATH)
        cache = dict(zip(c["uniprot_id"].astype(str), c["sequence"].astype(str)))
    missing = [u for u in uniprot_ids if u not in cache]
    for i in range(0, len(missing), _UNIPROT_BATCH):
        batch = missing[i : i + _UNIPROT_BATCH]
        url = "https://rest.uniprot.org/uniprotkb/accessions?" + urllib.parse.urlencode(
            {"accessions": ",".join(batch), "format": "fasta"}
        )
        req = urllib.request.Request(url, headers={"User-Agent": "tcell-causal-ppi/1.0"})
        text = urllib.request.urlopen(req, timeout=60).read().decode()
        cache.update(_parse_fasta(text))
        print(f"[plm] sequences fetched {min(i + _UNIPROT_BATCH, len(missing))}/{len(missing)}", flush=True)
        # persist after every batch so a kill doesn't refetch
        config.write_parquet_atomic(
            pd.DataFrame({"uniprot_id": list(cache), "sequence": list(cache.values())}),
            SEQUENCE_CACHE_PATH,
        )
    return {u: cache[u] for u in uniprot_ids if u in cache}


def _load_model():
    import esm  # heavy import; keep it lazy so the rest of the pipeline never pays for it
    import torch

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[plm] model on {device}", flush=True)
    return model, alphabet, alphabet.get_batch_converter()


def _embed_batch(model, batch_converter, items: list[tuple[str, str]]) -> dict[str, np.ndarray]:
    import torch

    device = next(model.parameters()).device
    _, _, toks = batch_converter([(u, s[:MAX_RESIDUES]) for u, s in items])
    with torch.no_grad():
        reps = model(toks.to(device), repr_layers=[ESM_LAYER])["representations"][ESM_LAYER]
    out: dict[str, np.ndarray] = {}
    for i, (uid, seq) in enumerate(items):
        n = min(len(seq), MAX_RESIDUES)
        vec = reps[i, 1 : n + 1].mean(0)  # drop BOS(0) and EOS(n+1); pad is beyond n+1
        out[uid] = vec.to("cpu", torch.float32).numpy()
    return out


def _write(embeds: dict[str, np.ndarray]) -> None:
    config.write_parquet_atomic(
        pd.DataFrame({"uniprot_id": list(embeds), "embedding": list(embeds.values())}),
        config.PLM_EMBEDDINGS_PATH,
    )


def run(limit: int | None = None, batch_size: int = 32, checkpoint: int = 256) -> None:
    ids = _unique_uniprot_ids()
    if limit is not None:
        ids = ids[:limit]

    done: dict[str, np.ndarray] = {}
    if config.PLM_EMBEDDINGS_PATH.exists():
        d = pd.read_parquet(config.PLM_EMBEDDINGS_PATH)
        done = {str(u): np.asarray(e, dtype=np.float32) for u, e in zip(d["uniprot_id"], d["embedding"])}
    todo = [u for u in ids if u not in done]
    print(f"[plm] {len(done)} embedded, {len(todo)} to go (of {len(ids)} target proteins)", flush=True)
    if not todo:
        return

    seqs = fetch_sequences(todo)
    items = [(u, seqs[u]) for u in todo if u in seqs and seqs[u]]
    missing_seq = [u for u in todo if u not in seqs or not seqs[u]]
    if missing_seq:
        print(f"[plm] {len(missing_seq)} proteins have no UniProt sequence; left to zero fallback", flush=True)

    model, _, batch_converter = _load_model()
    since_ckpt = 0
    for i in range(0, len(items), batch_size):
        done.update(_embed_batch(model, batch_converter, items[i : i + batch_size]))
        since_ckpt += len(items[i : i + batch_size])
        if since_ckpt >= checkpoint or i + batch_size >= len(items):
            _write(done)
            since_ckpt = 0
            print(f"[plm] embedded {len(done)}/{len(ids)} -> {config.PLM_EMBEDDINGS_PATH}", flush=True)
    print("[plm] done", flush=True)


if __name__ == "__main__":
    run()
