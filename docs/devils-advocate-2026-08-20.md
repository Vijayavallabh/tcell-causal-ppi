# Devil's advocate on paper/icbinb, 2026-08-20

Written adversarially: the goal was to find what a hostile but competent reviewer would
land, and to check each objection against the artifacts and the pre-registration rather
than assert it. Objections that the paper already answers are recorded as answered, so
nobody re-raises them.

## CRITICAL — a pre-registered caveat was promised and never written

`docs/replication-prereg.md` Amendment 6.5 chose `untyped_gnn` over `condition_gated` for
the injected-signal ladder, on cost and on sensitivity, and closed with:

> This is a bound on the pipeline's sensitivity, not on the typed encoder's specifically,
> and **it will be labelled that way**.

The paper never labels it. The abstract said "the null is bounded by a measured
instrument"; Limitations said "the instrument is not the limit". Both are unqualified.

Why it matters, on the paper's own evidence. The null is about the typed, gated arm. The
floor was measured with the untyped arm, which this paper separately shows is the BETTER
detector (h2a $-0.0120$; B1a attributes most of that to the unnormalised sum). So the
measured floor bounds the sensitivity of the arm that works, and the arm the null is
about could need a larger signal. The a-fortiori argument in 6.5 runs the safe direction
(if the best arm needs $\delta$, weaker arms need at least $\delta$), which is exactly
why the claim has to be stated in that direction rather than as "the null is bounded".

FIXED: abstract, Limitations and `app:floor` now attribute the floor to the best graph
arm and say the typed arm's own floor is not measured.

## IMPORTANT — the injected signal has the encoder's own functional form

Each rung adds `delta` times the mean response of a target's PPI neighbours. That is, up
to scaling, the quantity a message-passing layer computes. So the ladder measures
sensitivity to a signal shaped like the model's inductive bias, which is the most
favourable possible case.

The scrambled control does real work here and should not be oversold: it rules out
"the arms respond to injected magnitude regardless of the graph". It does not rule out
"the injected signal was of a form this architecture is built to find". A real biological
graph effect need not be a neighbour mean.

FIXED: stated in `app:floor` as a limit on what the floor generalises to.

## ANSWERED ALREADY — objections that do not land

- *"The typed null cannot generalise when the untyped arm's sign flips by dataset."*
  The two differ in exactly the way that matters: h2a pools at $I^2=39.2\%$, $Q$ $p=0.13$
  (homogeneous), the untyped margin at $I^2=88.7\%$, $Q$ $p<0.001$ (heterogeneous). The
  paper reports both and reads them differently, which is the correct move.
- *"h1 rests on one replication."* True, and stated in Limitations and `app:repl`: the
  condition gate needs two experimental contexts and only Frangieh has them.
- *"B1a's `clears both corrections` is a family of one."* Stated in `app:mechanism`,
  including that it would not clear at $m{=}4$.
- *"The pooled meta-analysis is post-hoc."* Stated, with the per-dataset tests as the
  pre-registered ones.

## NOTED, not fixed

- **Paper-level multiplicity.** Corrections are applied within families (six "family of
  four", one of eight, one of two, a 20-cell and a 36-cell). Each is honest; there is no
  accounting across the paper as a whole. Amendment 5.3 does this within A3 and nowhere
  else. A reviewer may reasonably raise look-elsewhere at the document level.
- **Failure 1 is a bug report.** Its defence is that it is a class of failure with a
  measurable signature, not that unnormalised penalties are novel. Reasonable people can
  disagree on whether that clears the bar; ICBINB is the venue where it plausibly does.
