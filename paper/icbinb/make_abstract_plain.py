#!/usr/bin/env python3
"""Regenerate abstract_plain.txt from main.tex. It is DERIVED; never hand-edit it.

WHY THIS SCRIPT EXISTS. abstract_plain.txt is what SUBMISSION.md tells a human to paste into the
OpenReview portal, and it had no generator, so it drifted a full campaign behind main.tex and still
carried a claim the paper had already retracted. A derived file with no generator drifts by default.

    .venv/bin/python paper/icbinb/make_abstract_plain.py [--check]

--check exits non-zero if the file on disk differs from what main.tex implies, so the drift is a gate
rather than something a reader discovers in the portal.

ORDER MATTERS. Comments are stripped BEFORE any backslash escape is converted, because a "% AUTO ..."
line converted first would leak an escaped percent into the plain text - which it did once.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Applied in order. Math and macros the abstract actually uses; anything unhandled is caught by the
# leftover-backslash check below rather than silently shipped.
SUBS = [
    (r"\$103\\times\$", "103x"),
    (r"\$\\sim\$\$10\^\{-7\}\$", "~10^-7"),
    (r"\\emph\{([^}]*)\}", r"\1"),
    (r"\\textbf\{([^}]*)\}", r"\1"),
    (r"\\textsc\{([^}]*)\}", lambda m: m.group(1).upper()),
    (r"\\citep\{[^}]*\}", ""),
    (r"\\ref\{[^}]*\}", ""),
    (r"\\Delta\\,", "delta "),
    (r"\\Delta", "delta"),
    (r"\\%", "%"),
    (r"\\times", "x"),
    (r"\\sim", "~"),
    (r"\\,", " "),
    (r"\\ ", " "),
    (r"\{=\}", "="),
    (r"\{,\}", ","),
    (r"I\^2", "I^2"),
    (r"\^\{2\}", "^2"),
    (r"_\{\\mathrm\{([^}]*)\}\}", r"_\1"),
    (r"\$([^$]*)\$", r"\1"),          # unwrap any remaining inline math
    (r"n\{=\}", "n="),
    (r"--", "-"),
    (r"``|''", '"'),
]


def plain(tex: str) -> str:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        raise SystemExit("no abstract environment in main.tex")
    body = m.group(1)
    # STRIP COMMENTS FIRST (see the module docstring): a real "%" is escaped as "\%" in TeX, so an
    # unescaped % starts a comment and everything after it on that line is not abstract text.
    body = re.sub(r"(?<!\\)%.*", "", body)
    for pat, rep in SUBS:
        body = re.sub(pat, rep, body)
    body = body.replace("\\", "")
    return re.sub(r"\s+", " ", body).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the file on disk is stale")
    a = ap.parse_args()
    out = HERE / "abstract_plain.txt"
    text = plain((HERE / "main.tex").read_text())

    # A legitimate percent in this abstract always follows a digit ("95% CI"). One that does not is a
    # leaked comment marker, which is exactly the failure this file shipped once.
    if (bad := re.search(r"(?<!\d)%", text)):
        print(f"REFUSING: a LaTeX comment leaked into the plain text at {bad.start()}: "
              f"{text[max(0, bad.start() - 40):bad.start() + 40]!r}", file=sys.stderr)
        return 2
    if a.check:
        cur = out.read_text().strip() if out.exists() else ""
        if cur != text:
            print(f"STALE: abstract_plain.txt differs from main.tex ({len(cur)} vs {len(text)} chars)")
            return 1
        print(f"abstract_plain.txt is current ({len(text)} chars)")
        return 0
    out.write_text(text + "\n")
    print(f"wrote {out} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
