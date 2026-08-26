# ICBINB-BIO submission — everything needed to file it

Prepared 2026-08-11. Two steps remain and both need your accounts. Everything else is done and
verified; nothing below has to be re-derived.

## Venue facts

**RE-VERIFIED AGAIN 2026-08-26, three days out: deadline UNCHANGED** at 29 Aug 2026, 11:59 p.m. AoE,
with the page limit, double-blind requirement and portal all unchanged. It is still listed under
"Tentative dates" and has not been extended. Checked because the row below says to and because a
deadline move inside the last week is the one venue fact that cannot be recovered from.

**RE-VERIFIED 2026-08-21** against https://icbinb-bio.github.io/submit/ and https://icbinb-bio.github.io/,
eight days out. First verified 2026-08-11. **Every fact below is unchanged.** Re-checked because the
deadline row said to and ten days had passed; an unchecked table that looks checked is the thing to
avoid.

| | |
|---|---|
| deadline | **29 Aug 2026, 11:59 p.m. AoE** — **still tentative on 2026-08-21.** Verbatim: "All deadlines are 11:59 p.m. Anywhere on Earth and remain subject to final confirmation", and every date is additionally marked "Tentative". It has not been confirmed in the ten days since the first check, so do not read the delay as confirmation |
| portal | https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/ICBINB-BIO |
| length | verbatim: "up to **eight pages**, excluding references and appendices" |
| blinding | verbatim: "Submissions are double-blind; linked material must also preserve anonymity" |
| archival | verbatim: "the workshop remains non-archival" (so the AAAI dual submission is fine) |
| LLM | verbatim: "Disclose any use of large language models (LLMs) with a short paragraph describing their role" — REQUIRED |
| tracks | **two**: Full papers (8pp) and Tiny papers (4pp of main text, unlimited appendices that are not required to be reviewed). We are submitting to the **full-paper** track |
| rest of the timeline | review 29 Aug – 21 Sep; acceptance notification 29 Sep; camera-ready and poster 20 Oct; in-person workshop 11 or 12 Dec 2026. All marked tentative |

### The one fact that could NOT be re-verified from this machine

The submit page **no longer names a style file**. It links the template as a Google Drive download,
`Formatting_Instructions_For_ICBINB_Failure_Modes_in_AI_for_Biology___NeurIPS_2026.zip`. The filename
was read from the Drive page; the **zip itself cannot be fetched from here** — Drive serves a sign-in
interstitial rather than the file, and this machine has no credentials for it (same finding as the
OpenReview probe below). So the 2026-08-11 "byte-identical to the workshop's own zip" diff is
**not re-confirmed as of 2026-08-21**.

What IS confirmed, so the gap is bounded rather than open:

- Our `neurips_2026.sty` has not drifted locally. sha256 `cedbda3f16ceae6eeb85b5aacd3e4b4d654de71427dfe82a9b553327f15e9c7c`,
  13,834 bytes, mtime 3 Aug 2026 — untouched since before the 2026-08-11 diff was run against it.
- The preamble still matches the template's required form: `\usepackage[dblblindworkshop]{neurips_2026}`
  plus `\workshoptitle{...}`, both present.
- The workshop's own name on the site, "I Can't Believe It's Not Better: Failure Modes of AI in
  Biology", matches what `\workshoptitle` carries.

**Human step, 2 minutes, worth doing before upload:** download that zip, unzip it, and diff its `.sty`
against ours. If it differs, the paper needs a rebuild against the new one and the page budget
re-checked, because a style change moves the body page count.

## Compliance, checked rather than assumed

- **Body is exactly 8 pages.** References begin on page 9 and are the FIRST thing on it. Total 24
  pages with appendices, which is allowed since references and appendices do not count.
  Verified by `./check_paper.sh`, which counts the body lines sitting above the References heading and
  requires zero. Until 2026-08-20 that gate only asked whether the word "References" appeared somewhere
  on page 9, and the body had in fact been running ~0.9 page past its limit for weeks.
- **Style file is byte-identical** to the workshop's own template. I downloaded their zip and diffed
  it against `neurips_2026.sty` in this directory.
- **Preamble matches their template**: `\usepackage[dblblindworkshop]{neurips_2026}` plus
  `\workshoptitle{...}`. Their template requires BOTH `\title{}` and `\workshoptitle{}`; we have both.
- **Anonymous**: author block is `Anonymous Author(s)`. The repository content has also been scrubbed —
  six identifying strings removed (GitHub username in clone URLs, absolute home paths, and one that
  exposed a third party's directory on the shared machine). A rescan returns zero.
  **This is now a GATE, not a one-off scan** (added 2026-08-21). `./check_paper.sh` gate 6 runs
  `check_anonymity.py` over `main.tex`, `main.bbl` and the RENDERED `pdftotext main.pdf`, and fails the
  build on an email address, a code-host URL, an ORCID, an absolute home path, a non-anonymous
  `\author` block, or a non-empty PDF `/Author` `/Title` `/Subject` `/Keywords`. It ACCEPTS
  `anonymous.4open.science` and REJECTS `github.com`, which is the distinction step 1 below turns on.
  The identity terms are derived at runtime from git config and the checkout path rather than written
  into the tracked script, because a deny-list containing the author's name would itself be the
  seventh deanonymising string. `--self-test` proves the gate can fail before you trust it passing.
  The scrub was dated 2026-08-11 and the paper was rewritten section by section on 2026-08-20 with no
  re-check in between; verified clean on 2026-08-21, so the gate keeps a clean thing clean.
- **PDF timestamps: examined, deliberately left as they are.** `CreationDate`/`ModDate` carry IST, a
  weak locality signal present in essentially every LaTeX submission. The venue asks for anonymity of
  authorship, not of geography, and `\pdfinfoomitdate` would make this build differ from the
  workshop template's for no compliance gain. Recorded so it reads as decided rather than missed.
- **LLM disclosure paragraph is present** and is honest that the code was largely agent-built.
- **Ethics and reproducibility statements** are present and are free at this venue.
- Build is clean: 0 errors, 0 overfull boxes, 0 undefined references, no multiply-defined labels,
  every `\ref` resolves.

## Why an agent cannot do either step (checked 2026-08-11, not assumed)

I verified this on the machine rather than inferring it, so nobody has to repeat the check:

| probe | result |
|---|---|
| `OPENREVIEW_*` env vars | none set |
| `~/.openreview`, `~/.config/openreview*` | absent |
| `openreview-py` in the venv | not installed |
| credential store in the repo (`.env`, `.secrets`, config files) | none |
| `gh` CLI (needed for the 4open.science mirror) | not installed |

So neither step is reachable from here even in principle. Both need your accounts.

## Step 1 — the anonymised repository (do this FIRST)

The paper links `https://anonymous.4open.science/r/tcell-causal-ppi`, which is currently a
**placeholder**. The blinding requirement covers linked material, so submitting with a dead or
non-anonymous link is a compliance failure, not a cosmetic one.

1. Go to https://anonymous.4open.science and create an anonymised mirror of the repo.
2. Paste the URL it gives you over the placeholder in `main.tex` (search for `anonymous.4open.science`;
   it appears in the Reproducibility statement).
3. Rebuild (see below) and confirm the body is still 8 pages.

## Step 2 — submit

Upload `main.pdf` at the portal above. Fields you will be asked for:

- **Title**: The Regularizer That Switched Off the Experiment: Why Our Protein-Interaction Null Was
  About the Encoder, Not the Prior
- **Abstract**: paste from `abstract_plain.txt` in this directory — LaTeX already stripped, percent
  signs and minus signs intact, 439 words (updated 2026-08-26 when C1's floor entered the abstract). That file is DERIVED from `main.tex`; regenerate it after
  any abstract edit rather than editing it by hand. It had drifted a full campaign behind the paper
  once (it still carried the pooled `+0.0031`, `I^2=26%` and the retracted "positive on all five
  datasets"), and since it is the text that gets pasted into the portal, a stale copy submits a claim
  the body of the paper contradicts.
- **Track**: full paper (8 pages), not the 4-page tiny-paper track.

## Rebuilding

```bash
export PATH=/tmp/tl/current/bin/x86_64-linux:$PATH   # NOT /tmp/tl/bin
cd paper/icbinb
pdflatex main; bibtex main; pdflatex main; pdflatex main
```

`/tmp` is cleaned periodically on this machine, so TeX Live may have to be reinstalled. Check for the
binary itself, not the `/tmp/tl` directory — the directory survived a cleaning that removed the
binaries at least once.

## What the paper claims, in one paragraph

Two failures, then a third finding that reframes both. A textbook edge-sparsity regulariser, written as
an unnormalised sum over sampled neighbourhoods, is ~103x the response term and drives every edge gate
to ~1e-7 inside epoch 0, so the graph-vs-no-graph comparison silently became no-graph vs no-graph.
Repaired, the typed gated graph is a bounded null, and that null replicates across seven further
screens including a genome-wide one (pooled +0.0018, random-effects 95% CI [-0.0028, +0.0065],
I^2=39%, Q p=0.13). But an untyped arm — same topology, no typing, no gating — behaves completely
differently and NOT consistently: it beats the baseline on the reference screen at n=7 (+0.0043, 95% CI
[+0.0017, +0.0069]) and on Replogle RPE1 (+0.0675), and LOSES on Norman (-0.0790), all three surviving
Bonferroni and Holm, in a family of eight datasets that do not agree in sign (I^2=88%). The prior is
not what fails; the encoder we built to exploit it is.

Do not restore the earlier version of this paragraph. It claimed the untyped arm was positive on all
five datasets, which the sixth dataset reversed decisively; that retraction is on the record in
`RESULTS_SUMMARY.md` and in `NEXT_ACTIONS.txt` under CORRECTIONS ON RECORD.
