#!/usr/bin/env python
"""Inspect data/raw/metadata/*.jsonld — Croissant metadata describing the 12
cell-level D*_*.assigned_guide.h5ad files.

Those raw single-cell files (~1.58 TiB total) are NOT downloaded (storage-blocked,
as the report scopes). The jsonld descriptors are still here and document each
absent file's declared size, md5, and per-column schema — useful for planning any
future bounded streaming pull without materializing the corpus.

Croissant structure (per descriptor):
  - `distribution[0]`: the cell-level h5ad — S3 contentUrl, encodingFormat, md5.
    NOTE: md5 here is a placeholder (all zeros), so it can't be used for checksum;
    verify size/ETag against the live S3 object at download time instead.
  - `variableMeasured`: 12 dataset-level descriptors (domain, assay, organism,
    tissue, ...), i.e. biosample annotations, NOT the per-cell .obs column schema
    (that lives in data_sharing_readme.md).

Run:  python examples/inspect_metadata_jsonld.py
"""
import json
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "data" / "raw"))
META = DATA_ROOT / "metadata"


def main() -> None:
    files = sorted(META.glob("*.jsonld"))
    print(f"jsonld descriptors: {len(files)}\n")

    urls = []
    for p in files:
        doc = json.loads(p.read_text())
        dist = (doc.get("distribution") or [{}])[0]
        vm = [v.get("name") for v in doc.get("variableMeasured", []) if isinstance(v, dict)]
        url = dist.get("contentUrl")
        urls.append(url)
        print(f"{p.name}")
        print(f"  name={doc.get('name')}  version={doc.get('version')}  published={doc.get('datePublished')}")
        print(f"  h5ad -> {url}  ({dist.get('encodingFormat')}, md5={dist.get('md5')})")
        print(f"  variableMeasured ({len(vm)}): {vm}")

    # --- self-check ---
    stems = {p.stem for p in files}
    donors = {s.split("_")[1] for s in stems}
    conds = {s.split("_")[2].split(".")[0] for s in stems}
    assert len(files) == 12, f"expected 12 descriptors, got {len(files)}"
    assert donors == {"D1", "D2", "D3", "D4"}, donors
    assert conds == {"Rest", "Stim8hr", "Stim48hr"}, conds
    assert all(u and u.startswith("s3://") for u in urls), "missing S3 contentUrl"
    print("\nOK: 12 descriptors, 4 donors x 3 conditions, all with S3 URLs.")


if __name__ == "__main__":
    main()
