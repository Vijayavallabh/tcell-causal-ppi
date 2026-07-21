"""Target-grouped train-internal holdout, so hyperparameter selection never touches the val fold.

One target gene spans ~3 rows in this mart (21,262 train rows over 7,208 genes), so a random ROW split
puts nearly every gene on both sides and the "holdout" score is contaminated by memorised targets. That
error produced a false +0.0843 on a tabular bar here; the val-blind number was +0.0783. Whole target
genes therefore move together, and a split that cannot honour the requested fraction distorts the
fraction rather than splitting a gene — LOUDLY, via a warning, because a caller who believes it held out
20% and actually held out 50% will report the wrong thing.

ponytail: grouped by target SYMBOL, not by the sequence/complex family groups `splits.family_groups`
builds for the outer split — two paralogs in the train fold can still land on opposite sides. Upgrade to
family grouping if inner-holdout scores start disagreeing with the outer fold.
"""
from __future__ import annotations

import warnings

import numpy as np

# Beyond this relative gap between requested and achieved holdout fraction, the split is not the one the
# caller asked for and staying silent would misinform whatever gets reported downstream.
_FRACTION_WARN_REL = 0.5


def group_partition(labels, holdout_frac: float = 0.2, seed: int = 0) -> tuple[list[int], list[int]]:
    """Row indices -> (inner_train, inner_holdout), splitting WHOLE label groups.

    Groups are shuffled by ``seed``, then admitted to the holdout only while admitting one moves the
    holdout row count CLOSER to ``holdout_frac`` — so a dominant gene is left whole on the train side
    instead of blowing the holdout up to 96% of the rows. Both sides are guaranteed non-empty.

    ``holdout_frac`` must lie in (0, 0.5]: above 0.5 the "holdout" would be the majority of the data,
    which is not a holdout, and it is also the only regime in which the admission loop could sweep every
    group into the holdout and leave the train side empty. Rejecting it at the boundary is what lets the
    loop below carry no repair branch for a case that can no longer arise.

    A single distinct label (no honest split exists) raises rather than returning an empty side. When
    group sizes make the requested fraction unreachable, the achieved fraction wins and a warning says so.
    """
    labels = list(labels)
    if not labels:
        raise ValueError("no rows to split")
    if not 0.0 < holdout_frac <= 0.5:
        raise ValueError(
            f"holdout_frac must be in (0, 0.5], got {holdout_frac!r} — a holdout larger than half the "
            f"rows is not a holdout, and above 0.5 the split can no longer guarantee a non-empty train side")
    groups: dict[object, list[int]] = {}
    for i, g in enumerate(labels):
        groups.setdefault(g, []).append(i)
    if len(groups) < 2:
        raise ValueError(
            f"only one target gene ({next(iter(groups))!r}) in {len(labels)} rows — no target-grouped "
            f"holdout exists; a split here could only leak that gene across both sides")

    order = sorted(groups)  # deterministic base order, independent of dict insertion
    np.random.default_rng(seed).shuffle(order)
    target = holdout_frac * len(labels)
    held: list[object] = []
    n_held = 0
    for key in order:
        size = len(groups[key])
        if abs(n_held + size - target) < abs(n_held - target):  # admitting it gets us closer
            held.append(key)
            n_held += size
    if not held:  # every group individually overshoots (few, large groups) -> take the smallest
        held = [min(order, key=lambda k: (len(groups[k]), k))]
        n_held = len(groups[held[0]])
    # With holdout_frac <= 0.5 the loop cannot admit every group (admitting the last would require a
    # group larger than 2*n*(1-holdout_frac) >= n rows), so the train side is always non-empty here.

    achieved = n_held / len(labels)
    # Two independent reasons to speak up, because the relative test alone has a blind spot: at
    # holdout_frac=0.5 it tolerates anything up to 75%, so a 74/26 split — the holdout being the clear
    # MAJORITY, the exact thing the argument check above refuses to accept as a request — passed silently.
    # Validate what the caller GETS, not only what they asked for.
    drifted = abs(achieved - holdout_frac) > _FRACTION_WARN_REL * holdout_frac
    inverted = achieved > 0.5
    if drifted or inverted:
        why = ("the holdout is now the MAJORITY of the rows, so the split is inverted: training happens "
               "on the smaller side" if inverted else
               "sizes that cannot be packed closer without splitting a gene")
        warnings.warn(
            f"target-grouped holdout is {achieved:.1%} of rows ({n_held}/{len(labels)}), not the "
            f"requested {holdout_frac:.1%} — {len(groups)} target groups, {why}. The achieved fraction "
            f"is what you have; report that one.",
            stacklevel=2,
        )

    held_set = set(held)
    inner_train = sorted(i for k, rows in groups.items() if k not in held_set for i in rows)
    inner_holdout = sorted(i for k in held_set for i in groups[k])
    return inner_train, inner_holdout


def target_grouped_subsets(dataset, holdout_frac: float = 0.2, seed: int = 0):
    """``PerturbationDataset`` -> (inner_train Subset, inner_holdout Subset), grouped by target symbol.

    Reads the per-row target from ``dataset.pc['hgnc_symbol']`` — the same column the outer split keys
    on — so the inner fence matches the outer one's unit."""
    from torch.utils.data import Subset

    tr, ho = group_partition(dataset.pc["hgnc_symbol"].astype(str).tolist(), holdout_frac, seed)
    return Subset(dataset, tr), Subset(dataset, ho)
