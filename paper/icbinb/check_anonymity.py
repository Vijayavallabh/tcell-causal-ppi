#!/usr/bin/env python3
"""GATE 6: refuse to ship a paper that names anybody. Double-blind is not repairable after the deadline.

WHY THIS EXISTS. Nothing in this harness checked anonymity. check_paper.sh gated the build, the page
budget and abstract drift; verify_numbers.py gates numbers. Not one check looked for a NAME. The repo
scrub landed 2026-08-11 (commit d840824, six strings) and the ENTIRE paper was then rewritten section
by section on 2026-08-20 with no anonymity re-check afterwards. Every other gate in this repo guards
something a reviewer would ask us to fix. This one guards a desk reject.

    .venv/bin/python paper/icbinb/check_anonymity.py              scan the paper, exit 1 on any hit
    .venv/bin/python paper/icbinb/check_anonymity.py --repo       also scan every TRACKED file
    .venv/bin/python paper/icbinb/check_anonymity.py --self-test  prove the scan can actually FAIL

--repo EXISTS BECAUSE THE PAPER IS NOT THE ONLY BLINDED ARTIFACT. The venue's requirement is that
linked material preserves anonymity too, and the linked material is an anonymous.4open.science mirror
of this repository, which masks the org name while serving file CONTENT unchanged. A clean PDF beside
a repo that names the author is still a blinding failure. It scans what `git ls-files` reports, so
untracked artifacts are out of scope by construction - which is correct, because the mirror serves
tracked content. (data/results/a3_external/rescored.json does record an absolute home path, and is
untracked, so it never reaches the mirror. Checked 2026-08-21, not assumed.)

THE DENY-LIST IS NOT WRITTEN DOWN, AND THAT IS DELIBERATE. This file is tracked, and
anonymous.4open.science masks the org name while serving file CONTENT unchanged - which is the exact
reasoning that motivated the 2026-08-11 scrub. Pasting the author's name into a tracked build script
to search for it would make this file the seventh deanonymising string. So the identity terms are
DERIVED AT RUNTIME from git config, the checkout path and the environment, and never persisted. What
IS written literally here are STRUCTURAL patterns, which identify nobody: an email address in any
form, a code-host URL, an absolute home path, an ORCID, a non-anonymous \\author block.

github.com IS REJECTED AND anonymous.4open.science IS ACCEPTED. Do not "fix" this the other way
round. The Reproducibility statement links the 4open.science mirror on purpose: that URL is the
anonymised artifact the venue asks for, and it stays a placeholder until a human creates the mirror
(SUBMISSION.md step 1). A github.com URL in its place is the compliance failure.

A MISSING TOOL IS A REFUSAL, NOT A PASS (exit 2). pdftotext and pdfinfo are how the RENDERED artifact
is checked, and the rendered artifact is what gets uploaded. A gate that silently skips its own
strongest check reads as coverage and is worse than no gate.

PDF TIMESTAMPS: examined and deliberately left alone. main.pdf's CreationDate/ModDate carry IST,
which is a weak locality signal. Every LaTeX submission carries the author's local timezone, the
venue asks for anonymity of authorship rather than of geography, and \\pdfinfoomitdate would make this
build differ from the workshop template's for no compliance gain. Recorded here so the next reader
knows it was decided rather than missed.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # the checkout root, for --repo

# Accepted anywhere, each for a stated reason. An allow-list entry without one is how a gate rots.
ALLOW = (
    "anonymous.4open.science",           # the anonymised mirror: the ONE external link the paper carries
    "github.com/ANONYMIZED",             # the placeholder the 2026-08-11 scrub (d840824) left behind
    # A THIRD PARTY's public analysis repository, cited as data provenance for the genome-wide screen.
    # Blinding protects OUR authorship; anonymising someone else's published citation would destroy
    # provenance and is not what the venue asks for.
    "github.com/emdann",
    "api.github.com/repos/emdann",       # the same third party, in REST-endpoint form
)

# STRUCTURAL patterns. Each names a CLASS of leak rather than a person, so writing them here leaks
# nothing. Ordered by how badly each one bites.
STRUCTURAL = [
    (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "an email address"),
    (r"(?:github|gitlab|bitbucket)\.com/[A-Za-z0-9._\-]+", "a code-host URL carrying an account name"),
    (r"orcid\.org/[0-9Xx\-]+", "an ORCID"),
    (r"/(?:home|Users)/[A-Za-z0-9._\-]+", "an absolute home path"),
    (r"/mnt/[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+", "an absolute path on this machine"),
    (r"\\(?:author|affiliation|thanks)\{(?![^}]*[Aa]nonymous)[^}]+\}",
     "an \\author/\\affiliation/\\thanks block that is not Anonymous"),
]

# Path components too generic to search for: hits would be noise, not identity.
_GENERIC = {"home", "users", "user", "root", "data", "work", "repo", "repos", "src", "tmp", "var",
            "opt", "mnt", "media", "projects", "project", "code", "backup", "documents"}


def derived_terms() -> tuple[list[tuple[str, str]], list[str]]:
    """Identity terms read from the environment, never from this file. Returns (terms, unavailable).

    ``unavailable`` is reported rather than dropped: a rule that could not run is a hole in the gate,
    and a hole that prints nothing looks exactly like a pass."""
    terms: list[tuple[str, str]] = []
    missing: list[str] = []

    for key, what in (("user.name", "the git author name"), ("user.email", "the git author email")):
        try:
            v = subprocess.run(["git", "config", "--get", key], cwd=HERE,
                               capture_output=True, text=True).stdout.strip()
        except OSError:
            v = ""
        if len(v) >= 4:
            terms.append((re.escape(v), what))
        else:
            missing.append(f"git config {key}")

    # The checkout's own location. The repo BASENAME is excluded on purpose - it is the project name,
    # it is in the anonymous URL by design, and searching for it would fail the gate on the allow-list.
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        top = ""
    if top:
        for part in Path(top).parent.parts:
            p = part.strip("/")
            if len(p) >= 4 and p.lower() not in _GENERIC:
                terms.append((re.escape(p), f"a component of the checkout path ({len(p)} chars)"))
    else:
        missing.append("git rev-parse --show-toplevel")

    for var in ("USER", "LOGNAME"):
        v = (os.environ.get(var) or "").strip()
        if len(v) >= 4 and v.lower() not in _GENERIC:
            terms.append((re.escape(v), f"the ${var} login name"))
    return terms, missing


def scan(text: str, rules: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Every rule hit, as (what, the offending excerpt).

    A hit is dropped only when it lies INSIDE an allow-listed occurrence. Overlap, rather than "an
    allowed string appears nearby", is what makes the exemption safe: a real leak sitting one line
    below an allowed URL still fails, and it is also what lets one entry cover a URL the pattern
    matches only part of (``api.github.com/repos/<third party>`` matches as ``github.com/repos``)."""
    spans = [m.span() for a in ALLOW for m in re.finditer(re.escape(a), text, re.IGNORECASE)]
    hits = []
    for pattern, what in rules:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if any(lo <= m.start() and m.end() <= hi for lo, hi in spans):
                continue
            frag = text[max(0, m.start() - 30):m.end() + 30].replace("\n", " ")
            hits.append((what, frag.strip()))
    return hits


def pdf_metadata(pdf: Path) -> dict:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    return {k.strip(): v.strip() for k, K, v in
            (line.partition(":") for line in out.splitlines()) if K}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="prove the scan can fail; touches no real file")
    ap.add_argument("--repo", action="store_true",
                    help="also scan every git-tracked file (the mirror serves these unchanged)")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    for tool in ("pdftotext", "pdfinfo"):
        if not shutil.which(tool):
            print(f"REFUSING: {tool} is not on PATH, so the RENDERED pdf cannot be checked. "
                  f"A skipped check is not a pass.", file=sys.stderr)
            return 2

    terms, missing = derived_terms()
    rules = STRUCTURAL + terms
    pdf = HERE / "main.pdf"
    sources = {p.name: p.read_text(errors="ignore") for p in (HERE / "main.tex", HERE / "main.bbl")
               if p.exists()}
    if "main.tex" not in sources:
        print("REFUSING: main.tex is missing", file=sys.stderr)
        return 2
    if not pdf.exists():
        print("REFUSING: main.pdf is missing — build before gating", file=sys.stderr)
        return 2
    sources["main.pdf (rendered text)"] = subprocess.run(
        ["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout

    fails = []
    for name, text in sources.items():
        for what, frag in scan(text, rules):
            fails.append(f"{name}: {what} — ...{frag}...")

    n_tracked = 0
    if a.repo:
        listed = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
        files = [f for f in listed.stdout.split("\n") if f.strip()]
        if not files:
            print("REFUSING: --repo found no tracked files, so it checked nothing", file=sys.stderr)
            return 2
        for rel in files:
            # This file is skipped from its own scan: --self-test's fixtures are DELIBERATELY leaky.
            # They name Ada Lovelace, who is dead, is not an author here, and was chosen precisely so
            # the fixture identifies nobody real.
            if Path(rel).resolve() == Path(__file__).resolve():
                continue
            try:
                text = (ROOT / rel).read_text(errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue                       # binary or unreadable: nothing to read a name out of
            n_tracked += 1
            for what, frag in scan(text, rules):
                fails.append(f"{rel}: {what} — ...{frag}...")

    meta = pdf_metadata(pdf)
    for field in ("Author", "Title", "Subject", "Keywords"):
        if meta.get(field):
            fails.append(f"main.pdf metadata: /{field} is non-empty ({meta[field]!r}); "
                         f"hyperref writes it from \\author/\\title")

    print(f"rules active     : {len(STRUCTURAL)} structural + {len(terms)} derived from this "
          f"environment")
    if missing:
        print(f"rules UNAVAILABLE: {', '.join(missing)} — those identity terms were NOT searched for")
    print(f"scanned          : {', '.join(sources)}"
          + (f", + {n_tracked} git-tracked files" if a.repo else
             "   (paper only; --repo also scans the tracked tree the mirror serves)"))
    print(f"pdf /Author      : {meta.get('Author', '')!r}   /Title: {meta.get('Title', '')!r}   "
          f"(gate: both empty)")
    print(f"allow-listed     : {', '.join(ALLOW)}")
    if fails:
        print("FAIL: the submission is not anonymous:\n  " + "\n  ".join(fails))
        return 1
    print("PASS")
    return 0


def self_test() -> int:
    """The gate is only evidence if it can fail. Same idea as test_inject_signal's leakage guard:
    feed it text that SHOULD trip each rule and assert it does, then feed it the allow-listed URL and
    assert it does not."""
    rules = STRUCTURAL + [(re.escape("Ada Lovelace"), "a derived name")]
    must_fail = [
        "Correspondence to ada.lovelace@example.ac.uk for details",
        "Code at https://github.com/alovelace/tcell-causal-ppi",
        "See https://orcid.org/0000-0002-1825-0097",
        "Run from /home/alovelace/tcell-causal-ppi",
        "Data under /mnt/md0/somebody/tree",
        r"\author{Ada Lovelace}",
        "written by Ada Lovelace",
        # The overlap rule's own trap: an allowed URL must not launder a leak beside it.
        "Mirror at https://anonymous.4open.science/r/x, written by Ada Lovelace",
    ]
    for s in must_fail:
        if not scan(s, rules):
            print(f"SELF-TEST FAILED: nothing tripped on {s!r} — the gate proves nothing")
            return 1
    must_pass = [
        "Code at https://anonymous.4open.science/r/tcell-causal-ppi",
        r"\author{Anonymous Author(s)}",
        "the systema metric is 0.0904 at n=5",
    ]
    for s in must_pass:
        if (h := scan(s, rules)):
            print(f"SELF-TEST FAILED: false positive on {s!r}: {h}")
            return 1
    print(f"self-test PASS: {len(must_fail)} leaks caught, {len(must_pass)} clean strings accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
