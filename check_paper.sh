#!/bin/bash
# Rebuild paper/icbinb and check every gate rail 3 imposes, in one command.
#
# WHY THIS EXISTS. The gates are "0 errors, 0 undefined, 0 overfull >2pt, body EXACTLY 8pp", and each
# one lives in a different place: the log, the log, the log, and the PDF. Checking them by hand takes
# four commands, two of which have to run from the repo root and two from paper/icbinb, which is a
# footgun that has already cost this session two failed invocations.
#
#   ./check_paper.sh            rebuild (2 passes) and check
#   ./check_paper.sh --full     rebuild with bibtex (4 passes) and check — needed after a new \cite
#
# THE BODY PAGE GATE is expressed as "References must start on page 9". That is the operational form of
# "body is exactly 8pp" for this document: the body runs to page 8, and the References heading opens
# page 9. Measure it that way rather than trusting a section label, because \label records where the
# label was typeset and a section pushed to the next page still reports its own number.
set -u
cd "$(dirname "$0")" || exit 1
export PATH=/tmp/tl/current/bin/x86_64-linux:$PATH   # NOT /tmp/tl/bin
command -v pdflatex >/dev/null || {
  echo "REFUSING: no pdflatex on PATH. /tmp is cleaned periodically here, so TeX Live may need"
  echo "reinstalling — check for the BINARY, not for the /tmp/tl directory (it has survived a"
  echo "cleaning that removed the binaries)."; exit 2; }

( cd paper/icbinb || exit 1
  pdflatex -interaction=nonstopmode main >/dev/null 2>&1
  [ "${1:-}" = "--full" ] && bibtex main >/dev/null 2>&1 && pdflatex -interaction=nonstopmode main >/dev/null 2>&1
  pdflatex -interaction=nonstopmode main >/dev/null 2>&1 ) || { echo "BUILD FAILED"; exit 1; }

.venv/bin/python - <<'PY'
import re, subprocess, sys
log = open("paper/icbinb/main.log", errors="ignore").read()
pages = re.search(r"Output written on \S+ \((\d+) pages", log)
overfull = [float(x) for x in re.findall(r"Overfull \\hbox \((\d+\.\d+)pt", log)]
bad = [x for x in overfull if x > 2]
undef_seq = log.count("Undefined control sequence")
undef_ref = "There were undefined references" in log
errors = log.count("\n! ")

txt = subprocess.run(["pdftotext", "paper/icbinb/main.pdf", "-"],
                     capture_output=True, text=True).stdout
refs_page = None
page = 1
# split on "\n" and NOT splitlines(): splitlines() treats the form feed as a line break and eats the
# page separators, so every page count comes out 1 and the gate silently reports the wrong page.
for line in txt.split("\n"):
    page += line.count("\f")
    if line.strip() == "References":
        refs_page = page
        break

fails = []
if bad:        fails.append(f"{len(bad)} overfull hbox >2pt: {bad}")
if undef_seq:  fails.append(f"{undef_seq} undefined control sequences")
if undef_ref:  fails.append("undefined references (run with --full)")
if errors:     fails.append(f"{errors} LaTeX errors")
if refs_page != 9:
    fails.append(f"body page gate: References start on page {refs_page}, must be 9 "
                 f"(fix by MOVING content to an appendix, never by cutting)")

print(f"pages total      : {pages.group(1) if pages else '?'}")
print(f"References start : page {refs_page}  (gate: 9)")
print(f"overfull >2pt    : {len(bad)}   (all overfull: {len(overfull)})")
print(f"undefined        : {undef_seq} control seq, refs={undef_ref}")
print(f"LaTeX errors     : {errors}")
print("PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails))
sys.exit(1 if fails else 0)
PY
