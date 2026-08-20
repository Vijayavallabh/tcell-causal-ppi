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
# THE BODY PAGE GATE is "the body ends by page 8", enforced as: the References heading must be the
# FIRST thing on page 9, with NO body text above it. Measure it from the PDF rather than trusting a
# section label, because \label records where the label was typeset and a section pushed to the next
# page still reports its own number.
#
# THE "ON PAGE 9" FORM OF THIS GATE WAS WRONG AND PASSED FOR WEEKS. It asked only whether the word
# "References" appeared somewhere on page 9, so a body that spilled most of the way down page 9 and
# then began the bibliography still passed. On 2026-08-20 the rendered PDF had checklist items 13-14,
# the whole Limitations section and the whole Conclusion sitting above the References heading: 87 lines
# of body on page 9, a body of roughly 8.9 pages against a venue limit of 8. Counting the lines above
# the heading is the fix.
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
body_lines_above = 0        # non-blank lines on the References page that sit ABOVE the heading
for line in txt.split("\n"):
    if line.count("\f"):
        page += line.count("\f")
        body_lines_above = 0    # new page: restart the count
    if line.strip() == "References":
        refs_page = page
        break
    if line.strip():
        body_lines_above += 1

fails = []
if bad:        fails.append(f"{len(bad)} overfull hbox >2pt: {bad}")
if undef_seq:  fails.append(f"{undef_seq} undefined control sequences")
if undef_ref:  fails.append("undefined references (run with --full)")
if errors:     fails.append(f"{errors} LaTeX errors")
if refs_page != 9:
    fails.append(f"body page gate: References start on page {refs_page}, must be 9 "
                 f"(fix by MOVING content to an appendix, never by cutting)")
elif body_lines_above > 0:
    fails.append(f"body page gate: References open page 9 but {body_lines_above} lines of BODY text "
                 f"sit above them, so the body runs to ~8.{min(9, int(body_lines_above / 10))} pages "
                 f"against a limit of 8 (fix by MOVING content to an appendix, never by cutting)")

print(f"pages total      : {pages.group(1) if pages else '?'}")
print(f"References start : page {refs_page}  (gate: 9, and must be the FIRST thing on it)")
print(f"body above refs  : {body_lines_above} lines   (gate: 0)")
print(f"overfull >2pt    : {len(bad)}   (all overfull: {len(overfull)})")
print(f"undefined        : {undef_seq} control seq, refs={undef_ref}")
print(f"LaTeX errors     : {errors}")
print("PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails))
sys.exit(1 if fails else 0)
PY

# Gate 5: abstract_plain.txt is DERIVED from main.tex and is what SUBMISSION.md tells a human to paste
# into the portal. It had no generator and drifted a full campaign behind the paper, still carrying a
# retracted claim. Making the drift a gate is the only thing that stops it recurring.
GATE5=0
.venv/bin/python paper/icbinb/make_abstract_plain.py --check || GATE5=$?
if [ $GATE5 -ne 0 ]; then
  echo "FAIL: abstract_plain.txt is stale — run .venv/bin/python paper/icbinb/make_abstract_plain.py"
  exit 1
fi
