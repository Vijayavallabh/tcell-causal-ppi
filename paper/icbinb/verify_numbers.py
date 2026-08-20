#!/usr/bin/env python3
"""Guard the paper's numbers across a rewrite.

Two independent checks, because a rewrite can break numbers in two different ways.

1. INVENTORY DRIFT. Every numeric literal in main.tex is extracted with its context. A
   rewrite may re-word freely, but it must not invent a number that was not there before,
   and must not silently drop one. Compare against a snapshot taken from the verified
   state:  --snapshot  writes it,  --check  diffs against it.

2. ARTIFACT AGREEMENT. The tables are re-derived from their JSON artifacts, independent of
   the snapshot, so a number that was wrong BEFORE the rewrite is still caught.

    python verify_numbers.py --snapshot          # from a state you trust
    python verify_numbers.py --check             # after editing
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SNAP = HERE / "numbers.snapshot.json"

# Numeric literals inside math mode: the form every measured quantity is written in.
NUM = re.compile(r"[-+]?\d+(?:\{,\})?\d*(?:\.\d+)?")


def literals(tex: str) -> Counter:
    """Multiset of numeric literals appearing inside $...$ math, comments stripped."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    out = Counter()
    for m in re.finditer(r"\$([^$]*)\$", tex):
        for n in NUM.findall(m.group(1)):
            if n in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                continue          # bare small integers are indices/counts, too noisy
            out[n.replace("{,}", ",")] += 1
    return out


def artifact_checks() -> list[str]:
    """Re-derive the tables from artifacts. Returns a list of failure strings."""
    tex = (HERE / "main.tex").read_text()
    fails: list[str] = []

    def blk(label):
        m = re.search(r"\\label\{" + label + r"\}(.*?)\\end\{tabular\}", tex, re.S)
        return m.group(1) if m else ""

    def rows(label):
        for line in blk(label).split("\\\\"):
            line = line.replace("\\midrule", "").replace("\\toprule", "").strip()
            cells = [c.strip() for c in line.split("&")]
            if len(cells) > 1:
                yield cells

    # --- tab:ksweep against the A3 sweep -------------------------------------------------
    p = ROOT / "data/results/a3_external/k_sweep.json"
    if p.exists():
        ks = json.loads(p.read_text())
        for cells in rows("tab:ksweep"):
            if len(cells) != 5:
                continue
            kr = cells[0].replace("$", "").strip()
            if not (kr == "all" or kr.isdigit()):
                continue
            row = ks.get(str(10282 if kr == "all" else int(kr)))
            if row is None:
                continue
            for cell, key in zip(cells[1:], ["h2a", "h2b", "promotion_margin", "h1_vs_no_graph"]):
                m = re.search(r"([-+]\d\.\d{4})", cell)
                if m and f"{row[key]['mean']:+.4f}" != m.group(1):
                    fails.append(f"tab:ksweep k={kr} {key}: paper {m.group(1)} artifact "
                                 f"{row[key]['mean']:+.4f}")

    # --- tab:bins against the B2 decomposition -------------------------------------------
    p = ROOT / "data/results/b2_deciles/deciles.json"
    if p.exists():
        head = json.loads(p.read_text())["schemes"]["head"]
        for cells in rows("tab:bins"):
            if len(cells) != 4:
                continue
            lab = cells[0].replace("$", "").replace("{,}", "").replace("--", "-")
            for cell, key in ((cells[1], "promotion_margin"), (cells[3], "h2a")):
                c = head["cells"].get(f"{lab}/{key}")
                m = re.search(r"([-+]\d\.\d{4})", cell)
                if c and m and f"{c['mean']:+.4f}" != m.group(1):
                    fails.append(f"tab:bins {lab} {key}: paper {m.group(1)} artifact {c['mean']:+.4f}")

    # --- tab:mechanism against A1 ---------------------------------------------------------
    p = ROOT / "data/results/screening_a1/a1_mechanism.json"
    if p.exists():
        fam = json.loads(p.read_text())["family"]
        for line in blk("tab:mechanism").split("\\\\"):
            if "$D_" not in line:
                continue
            key = "D1" if "$D_1$" in line else "D2"
            c = fam[key]
            got = re.findall(r"[-+]\d\.\d{4}", line)
            want = [f"{c['mean']:+.4f}", f"{c['ci_low']:+.4f}", f"{c['ci_high']:+.4f}"]
            if got != want:
                fails.append(f"tab:mechanism {key}: paper {got} artifact {want}")

    # --- prose-level singletons that the paper leans on -----------------------------------
    p = ROOT / "data/results/screening_b1/b1_message_form.json"
    if p.exists():
        d3 = json.loads(p.read_text())["contrasts"]["D3"]
        for lit, val in ((f"{d3['mean']:+.4f}", "B1a D3 mean"),
                         (f"{d3['ci_low']:+.4f}", "B1a D3 ci_low"),
                         (f"{d3['ci_high']:+.4f}", "B1a D3 ci_high")):
            if f"${lit}$" not in tex and lit not in tex:
                fails.append(f"{val} ({lit}) absent from the paper")
    p = ROOT / "data/results/replication/pooled_with_reference.json"
    if p.exists():
        pm = json.loads(p.read_text())["pooled"]["promotion_margin"]
        i2 = f"{pm['I2']*100:.1f}"
        if f"I^2{{=}}{i2}" not in tex:
            fails.append(f"eight-dataset I^2 should read {i2}% and does not appear")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    tex = (HERE / "main.tex").read_text()
    cur = literals(tex)

    if a.snapshot:
        SNAP.write_text(json.dumps(dict(sorted(cur.items())), indent=1))
        print(f"snapshot: {len(cur)} distinct literals, {sum(cur.values())} occurrences -> {SNAP.name}")
        return 0

    fails = artifact_checks()
    print(f"artifact agreement: {len(fails)} failure(s)")
    for f in fails:
        print("  ", f)

    if a.check and SNAP.exists():
        old = Counter(json.loads(SNAP.read_text()))
        gone = {k: v for k, v in (old - cur).items()}
        new = {k: v for k, v in (cur - old).items()}
        print(f"inventory drift: {len(gone)} dropped, {len(new)} introduced")
        if gone:
            print("   DROPPED (was in the verified paper, now absent):")
            for k, v in sorted(gone.items()):
                print(f"     {k} x{v}")
        if new:
            print("   INTRODUCED (not in the verified paper - justify each against an artifact):")
            for k, v in sorted(new.items()):
                print(f"     {k} x{v}")
        return 1 if (fails or gone or new) else 0
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
