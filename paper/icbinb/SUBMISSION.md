# ICBINB-BIO submission — everything needed to file it

Prepared 2026-08-11. Two steps remain and both need your accounts. Everything else is done and
verified; nothing below has to be re-derived.

## Venue facts (verified 2026-08-11 from https://icbinb-bio.github.io/submit/, not from memory)

| | |
|---|---|
| deadline | **29 Aug 2026, 11:59 p.m. AoE** — the site still says "subject to final confirmation", so re-check |
| portal | https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/ICBINB-BIO |
| length | up to 8 pages, **excluding** references and appendices |
| blinding | double-blind; **linked material must also preserve anonymity** |
| archival | non-archival (so the AAAI dual submission is fine) |
| LLM | short paragraph disclosing LLM role — REQUIRED |

## Compliance, checked rather than assumed

- **Body is exactly 8 pages.** References begin on page 9. Total 17 pages with appendices, which is
  allowed since references and appendices do not count.
- **Style file is byte-identical** to the workshop's own template. I downloaded their zip and diffed
  it against `neurips_2026.sty` in this directory.
- **Preamble matches their template**: `\usepackage[dblblindworkshop]{neurips_2026}` plus
  `\workshoptitle{...}`. Their template requires BOTH `\title{}` and `\workshoptitle{}`; we have both.
- **Anonymous**: author block is `Anonymous Author(s)`. The repository content has also been scrubbed —
  six identifying strings removed (GitHub username in clone URLs, absolute home paths, and one that
  exposed a third party's directory on the shared machine). A rescan returns zero.
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

- **Title**: The Regularizer That Switched Off the Experiment: Diagnosing Why a Protein-Interaction
  Prior Does Not Help T-Cell Perturbation Prediction
- **Abstract**: paste from `abstract_plain.txt` in this directory — LaTeX already stripped, percent
  signs and minus signs intact, 441 words. That file is DERIVED from `main.tex`; regenerate it after
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
