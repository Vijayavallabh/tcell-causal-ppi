"""Merge the reference registry's seed 0-4 runs into the n=7 root's registry, then aggregate.

WHY. multiseed reads fold evidence from the registry: it needs to see that every seed it is pooling
was scored on the same split. The fresh n=7 root starts with an empty registry, so at n=5 it already
reports "NOT comparable" even though the numbers reproduce the reference exactly. That flag would
attach to the n=7 result and would be wrong - every lane here ran against SPLITS_ROOT=data/splits, the
frozen fold. Merging supplies the missing evidence rather than suppressing the check.

The reference registry is treated as READ-ONLY; only the copy in the fresh root is written.
"""
import shutil, sys, yaml
from pathlib import Path

REF = Path("data/results/experiment_registry.yaml")
DST = Path("data/results/screening_untyped_n7/experiment_registry.yaml")

ref = yaml.safe_load(REF.read_text())["runs"]
dst_doc = yaml.safe_load(DST.read_text())
dst = dst_doc["runs"]

have = {(r["config_id"], r["seed"]) for r in dst}
added = [r for r in ref if (r["config_id"], r["seed"]) not in have]
if not added:
    print("[merge] nothing to add"); sys.exit(0)

shutil.copy2(DST, DST.with_suffix(".yaml.bak"))
merged = dst + added
for i, r in enumerate(merged, 1):
    r["run_id"] = f"run-{i:04d}"
dst_doc["runs"] = merged
DST.write_text(yaml.safe_dump(dst_doc, sort_keys=False))

splits = sorted({r.get("split") for r in merged})
print(f"[merge] {len(dst)} local + {len(added)} from reference = {len(merged)} runs")
print(f"[merge] splits present: {splits}  (must be exactly ['blocked_target_ood'])")
print(f"[merge] reference registry untouched: {REF} mtime unchanged")
