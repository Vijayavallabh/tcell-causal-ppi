"""Statistical-fallacy scan (feat-013): eleven independent detectors run over the study's diagnostic inputs
so a reproduction can assert none of the classic inference traps is silently driving a headline claim
(report §reproducibility 11/11 fallacy scan). Each detector is a small, self-contained numpy check that
returns ``{flagged, evidence}``; ``run_fallacy_scan`` runs whichever inputs are supplied and reports coverage
(the "11/11" is complete coverage, not eleven flags).

The eleven: Simpson, ecological, Berkson, collider, base-rate, regression-to-mean, survivorship,
look-elsewhere, garden-of-forks, correlation≠causation, reverse-causation.

**Two invariants hold across every detector** (both were review findings — the first round of fixes patched
individual detectors and left the class open):

1. *Undefined input raises ``Unevaluable``; it never returns an unflagged "clean" result.* A detector that
   could not evaluate must drop the scan's coverage below 11/11 so the verifier reports PARTIALLY — silently
   certifying a trap that was never examined is the failure this scan exists to prevent.
2. *A degenerate statistic is never encoded as a sentinel value.* ``_corr`` raises rather than returning 0.0
   for "too few pairs" / "a series is constant", because a caller cannot distinguish that sentinel from a
   genuine zero correlation — which is precisely how ``berkson`` came to flag studies with no collider.
"""
from __future__ import annotations

import numpy as np


class Unevaluable(ValueError):
    """A detector's input is degenerate, so the statistic it checks is undefined. Raised (not silently
    returned as an unflagged pass) so ``run_fallacy_scan`` records the detector as errored → coverage drops
    below 11/11 → the verifier reports PARTIALLY rather than certifying a trap it never examined."""


def _finite(name: str, *arrays) -> list[np.ndarray]:
    """Coerce to float arrays and reject empty / all-non-finite input, so a NaN cannot ride through a
    detector and silently produce ``flagged: False`` (or a nonsense statistic) on a clean-looking pass."""
    out = []
    for a in arrays:
        arr = np.asarray(a, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise Unevaluable(f"{name}: empty input")
        if not np.isfinite(arr).any():
            raise Unevaluable(f"{name}: no finite values")
        out.append(arr)
    return out


def _scalar(name: str, v) -> float:
    f = float(v)
    if not np.isfinite(f):
        raise Unevaluable(f"{name}: non-finite value {v!r}")
    return f


def _corr(x, y) -> float:
    """Pearson correlation over the finite pairs.

    RAISES ``Unevaluable`` when the correlation is undefined (fewer than 2 finite pairs, or either series
    constant) instead of returning a 0.0 sentinel: callers cannot tell a sentinel from a real zero, so the
    sentinel read as 'the association vanished' and produced false flags. Non-finite entries are dropped
    pairwise — ``np.ptp`` of an array holding NaN returns NaN and ``nan == 0`` is False, so a NaN would
    otherwise slip the degeneracy guard and reach ``np.corrcoef``."""
    x, y = np.asarray(x, np.float64).reshape(-1), np.asarray(y, np.float64).reshape(-1)
    if x.shape != y.shape:
        raise Unevaluable(f"correlation needs equal-length series (got {x.shape} vs {y.shape})")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        raise Unevaluable(f"correlation needs >=2 finite pairs (got {x.size})")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise Unevaluable("correlation undefined: a series is constant over the finite pairs")
    return float(np.corrcoef(x, y)[0, 1])


def _try_corr(x, y) -> float | None:
    """``_corr`` or None when undefined — for callers where 'no trend' is a legitimate reading."""
    try:
        return _corr(x, y)
    except Unevaluable:
        return None


def _sign(v: float | None, eps: float = 1e-9) -> int:
    if v is None or not np.isfinite(v):  # int(np.sign(nan)) raises; a non-finite statistic has no direction
        return 0
    return 0 if abs(v) < eps else int(np.sign(v))


def simpson(groups, eps: float = 1e-9) -> dict:
    """Sign reversal between the pooled trend and the (consistent) within-group trends. ``groups`` is a list
    of (x, y) pairs, one per subgroup.

    Needs >=2 subgroups: the paradox IS a pooled-vs-within disagreement, which one group cannot exhibit (a
    single group would trivially report 'no reversal' — a clean pass for a check that never ran). A subgroup
    whose own trend is undefined contributes 'no trend' rather than sinking the scan."""
    groups = list(groups)
    if len(groups) < 2:
        raise Unevaluable(f"simpson needs >=2 subgroups to compare pooled vs within (got {len(groups)})")
    within = [_sign(_try_corr(x, y)) for x, y in groups]
    px = np.concatenate([np.asarray(x, np.float64).reshape(-1) for x, _ in groups])
    py = np.concatenate([np.asarray(y, np.float64).reshape(-1) for _, y in groups])
    pooled = _sign(_corr(px, py))  # the pooled trend must be defined, else there is nothing to reverse
    consistent = {s for s in within if s != 0}
    if not consistent:
        raise Unevaluable("simpson: no subgroup has a defined trend to compare against the pooled trend")
    reversed_ = len(consistent) == 1 and pooled != 0 and pooled == -next(iter(consistent))
    return {"flagged": bool(reversed_), "evidence": {"pooled_sign": pooled, "within_signs": within}}


def ecological(x, y, group, inflation: float = 0.3) -> dict:
    """Aggregate (group-mean) correlation inflated or sign-flipped vs the individual-level correlation.

    Needs >=3 groups: the correlation of 2 group means is degenerately +-1 regardless of the individual data,
    so the aggregate carries no information and the comparison is undefined."""
    x, y = _finite("ecological", x, y)
    group = np.asarray(group).reshape(-1)
    if not (x.shape == y.shape == group.shape):
        raise Unevaluable("ecological: x, y and group must be the same length")
    gs = [group == g for g in np.unique(group)]
    if len(gs) < 3:
        raise Unevaluable(f"ecological needs >=3 groups to compare aggregate vs individual (got {len(gs)}); "
                          f"the correlation of <=2 group means is degenerately +-1")
    indiv = _corr(x, y)
    with np.errstate(invalid="ignore"):
        gx = np.array([np.nanmean(x[m]) for m in gs])
        gy = np.array([np.nanmean(y[m]) for m in gs])
    agg = _corr(gx, gy)
    flag = (_sign(agg) != _sign(indiv) and _sign(agg) != 0) or (abs(agg) - abs(indiv) > inflation)
    return {"flagged": bool(flag),
            "evidence": {"individual_corr": indiv, "aggregate_corr": agg, "n_groups": len(gs)}}


def berkson(x, y, selected, margin: float = 0.2, min_selected: int = 3) -> dict:
    """Selection on a collider induces a spurious (more-negative) association within the selected sample.

    Both correlations must be DEFINED: if the within-selection correlation is undefined (too few rows, or x
    or y constant among the selected) this raises rather than reading a 0.0 sentinel as 'selection destroyed
    the association' — the false-flag path a bare row-count guard does not close."""
    x, y = _finite("berkson", x, y)
    selected = np.asarray(selected, bool).reshape(-1)
    if selected.shape != x.shape:
        raise Unevaluable("berkson: selected must be the same length as x/y")
    n_sel = int(selected.sum())
    if n_sel < min_selected:
        raise Unevaluable(f"berkson needs >={min_selected} selected rows to estimate the within-selection "
                          f"correlation (got {n_sel})")
    full = _corr(x, y)
    sel = _corr(x[selected], y[selected])  # raises if undefined — never a 0.0 sentinel
    return {"flagged": bool(sel < full - margin),
            "evidence": {"corr_full": full, "corr_selected": sel, "n_selected": n_sel}}


def _partial_corr(x, y, z) -> float:
    rxy, rxz, ryz = _corr(x, y), _corr(x, z), _corr(y, z)
    den = np.sqrt(max(1 - rxz ** 2, 0.0)) * np.sqrt(max(1 - ryz ** 2, 0.0))
    if den <= 1e-12:  # z is collinear with x or y -> the partial correlation is undefined, not zero
        raise Unevaluable("collider: partial correlation undefined (z is collinear with x or y)")
    return float((rxy - rxz * ryz) / den)


def collider(x, y, z, margin: float = 0.2) -> dict:
    """Conditioning on a collider ``z`` induces association between marginally-independent ``x`` and ``y``.

    Every input correlation must be defined (``_corr`` raises otherwise), so a constant z or a 1-row input
    can no longer return an unflagged clean pass."""
    _finite("collider", x, y, z)
    marginal = _corr(x, y)
    partial = _partial_corr(x, y, z)
    return {"flagged": bool(abs(partial) - abs(marginal) > margin),
            "evidence": {"marginal_corr": marginal, "partial_corr_given_z": partial}}


def base_rate(y_true, y_pred, acc_thresh: float = 0.9, precision_thresh: float = 0.5) -> dict:
    """High accuracy masking poor precision when the positive class is rare (base-rate neglect).

    Requires both classes to be present: with prevalence 0 or 1 there is no base rate to neglect, and the
    precision comparison is meaningless."""
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if yt.size == 0 or yt.shape != yp.shape:
        raise Unevaluable("base_rate: empty or mismatched y_true/y_pred")
    yt, yp = (yt >= 0.5).astype(int), (yp >= 0.5).astype(int)
    prevalence = float(yt.mean())
    if prevalence in (0.0, 1.0):
        raise Unevaluable(f"base_rate: y_true is single-class (prevalence {prevalence}) — no base rate")
    acc = float((yt == yp).mean())
    n_pos_pred = int((yp == 1).sum())
    # precision is 0/0 when nothing is predicted positive; that is the degenerate majority-class predictor,
    # which IS base-rate neglect, so it is scored as precision 0 rather than raising
    precision = float(((yp == 1) & (yt == 1)).sum() / n_pos_pred) if n_pos_pred else 0.0
    flag = acc >= acc_thresh and precision < precision_thresh
    return {"flagged": bool(flag),
            "evidence": {"accuracy": acc, "precision": precision, "prevalence": prevalence,
                         "n_positive_predictions": n_pos_pred}}


def regression_to_mean(baseline, followup, select_frac: float = 0.1, margin: float = 0.1) -> dict:
    """An extreme-selected group reverts toward the mean on retest — apparent change that is only noise.

    Deviations are measured **in each series' own standard-deviation units**. Measuring against a pooled
    grand mean makes any level shift look like regression (``followup = baseline - 10`` correlates 1.0 and
    regresses not at all); measuring in raw units against each series' own mean fixes that but still fires on
    a pure RESCALE (``followup = baseline * 0.5``, also correlation 1.0). Standardising makes the statistic
    invariant to both affine changes, which is what 'regression to the mean' must be blind to."""
    b, f = _finite("regression_to_mean", baseline, followup)
    if b.shape != f.shape:
        raise Unevaluable("regression_to_mean: baseline and followup must be paired (same length)")
    sb, sf = float(np.std(b)), float(np.std(f))
    if sb == 0 or sf == 0:
        raise Unevaluable("regression_to_mean: a series is constant — deviations are undefined")
    k = max(1, int(round(select_frac * len(b))))
    top = np.argsort(b)[-k:]                        # the extreme-high baseline group
    d_base = abs(float(b[top].mean()) - float(b.mean())) / sb      # in baseline SDs...
    d_follow = abs(float(f[top].mean()) - float(f.mean())) / sf    # ...vs in followup SDs
    flag = d_follow < d_base - margin * (abs(d_base) + 1e-9)
    return {"flagged": bool(flag),
            "evidence": {"selected_baseline_dev_sd": d_base, "selected_followup_dev_sd": d_follow,
                         "baseline_sd": sb, "followup_sd": sf}}


def survivorship(values, survived, margin: float = 0.2) -> dict:
    """A metric reported on survivors only differs materially from the full-population value.

    With zero survivors the survivor-only metric is undefined; falling back to the full-population mean would
    report a clean, unflagged pass for a check that never ran."""
    (v,) = _finite("survivorship", values)
    s = np.asarray(survived, bool).reshape(-1)
    if s.shape != v.shape:
        raise Unevaluable("survivorship: survived must be the same length as values")
    if not s.any():
        raise Unevaluable("survivorship has zero survivors — the survivor-only metric is undefined")
    spread = float(np.nanstd(v))
    if spread == 0:
        raise Unevaluable("survivorship: values are constant — no survivor bias is measurable")
    full, surv = float(np.nanmean(v)), float(np.nanmean(v[s]))
    return {"flagged": bool(abs(surv - full) > margin * spread),
            "evidence": {"mean_full": full, "mean_survivors": surv, "n_survivors": int(s.sum())}}


def look_elsewhere(pvalues, alpha: float = 0.05) -> dict:
    """A raw hit that does not survive family-wise (Bonferroni) correction across the tested hypotheses."""
    (p,) = _finite("look_elsewhere", pvalues)
    if not np.isfinite(p).all():
        raise Unevaluable("look_elsewhere: non-finite p-value")
    if ((p < 0) | (p > 1)).any():
        raise Unevaluable("look_elsewhere: p-values outside [0, 1]")
    m = len(p)
    raw_hit = bool((p < alpha).any())
    survives = bool((p < alpha / m).any())
    return {"flagged": bool(raw_hit and not survives),
            "evidence": {"n_tests": m, "min_p": float(p.min()), "bonferroni_alpha": alpha / m}}


def garden_of_forks(estimates, eps: float = 1e-9) -> dict:
    """The effect estimate is fragile across analysis choices — its sign flips, or its spread exceeds its own
    magnitude (researcher degrees of freedom). Needs >=2 forks: a single estimate has no spread to assess."""
    (e,) = _finite("garden_of_forks", estimates)
    if not np.isfinite(e).all():
        raise Unevaluable("garden_of_forks: non-finite estimate")
    if e.size < 2:
        raise Unevaluable(f"garden_of_forks needs >=2 analysis variants to compare (got {e.size})")
    signs = {_sign(v) for v in e if abs(v) > eps}
    sign_flip = len(signs) > 1
    spread = float(e.max() - e.min())
    fragile = sign_flip or spread > abs(float(e.mean())) + eps
    return {"flagged": bool(fragile),
            "evidence": {"sign_flip": sign_flip, "spread": spread, "mean": float(e.mean())}}


def correlation_not_causation(corr, has_interventional_support, threshold: float = 0.3) -> dict:
    """A strong correlation asserted causally without interventional / confounder-adjusted support."""
    c = _scalar("correlation_not_causation", corr)
    return {"flagged": bool(abs(c) >= threshold and not has_interventional_support),
            "evidence": {"corr": c, "interventional_support": bool(has_interventional_support)}}


def reverse_causation(forward_corr, reverse_corr, margin: float = 0.0,
                      min_association: float = 0.1) -> dict:
    """The reverse cross-lagged association (y→x) is as strong as, or stronger than, the claimed forward one
    (x→y) — the causal direction is not identified by the data.

    The floor is on the STRONGER of the two, not on the forward one. Requiring a strong *forward* effect
    would silently unflag the archetypal trap — a weak claimed forward effect dominated by the reverse
    association (f=0.05, r=0.95), which is the case the detector most needs to catch. What must be excluded
    is only the null endpoint, where neither direction shows an association and there is no directional claim
    to invalidate (f=r=0). ``margin`` raises the bar for flagging, matching ``berkson``/``collider``."""
    f = abs(_scalar("reverse_causation.forward_corr", forward_corr))
    r = abs(_scalar("reverse_causation.reverse_corr", reverse_corr))
    flagged = max(f, r) >= min_association and r >= f + margin
    return {"flagged": bool(flagged),
            "evidence": {"forward": f, "reverse": r, "min_association": min_association}}


_DETECTORS = {
    "simpson": simpson,
    "ecological": ecological,
    "berkson": berkson,
    "collider": collider,
    "base_rate": base_rate,
    "regression_to_mean": regression_to_mean,
    "survivorship": survivorship,
    "look_elsewhere": look_elsewhere,
    "garden_of_forks": garden_of_forks,
    "correlation_not_causation": correlation_not_causation,
    "reverse_causation": reverse_causation,
}
FALLACIES: tuple[str, ...] = tuple(_DETECTORS)  # the eleven, in report order


def run_fallacy_scan(inputs: dict) -> dict:
    """Run each detector whose kwargs are supplied in ``inputs`` ({fallacy_name -> kwargs dict}); a missing
    key is 'not evaluated'. ``complete`` is True only when all eleven ran CLEANLY (the 11/11 coverage the
    report requires); ``flagged`` lists the detectors that fired, ``errored`` those that could not run.

    ``unevaluable`` and ``crashed`` are reported separately: an ``Unevaluable`` is an inadequate INPUT (the
    steward's probe cannot answer the question), while any other exception is a BUG in the detector. Both
    drop coverage, but conflating them would hide a real defect behind 'degenerate input'."""
    results: dict = {}
    for name, fn in _DETECTORS.items():
        spec = inputs.get(name)
        if spec is None:
            results[name] = {"evaluated": False}
            continue
        try:
            results[name] = {"evaluated": True, "error": None, **fn(**spec)}
        except Unevaluable as exc:  # inadequate input: NOT a clean pass, and NOT a bug
            results[name] = {"evaluated": False, "unevaluable": True, "error": f"Unevaluable: {exc}"}
        except Exception as exc:  # a real defect in the detector — surfaced, never swallowed as degenerate
            results[name] = {"evaluated": False, "unevaluable": False,
                             "error": f"{type(exc).__name__}: {exc}"}
    succeeded = [n for n, r in results.items() if r.get("evaluated")]  # ran cleanly (errored ones are False)
    errored = [n for n, r in results.items() if r.get("error")]
    crashed = [n for n, r in results.items() if r.get("error") and r.get("unevaluable") is False]
    flagged = [n for n, r in results.items() if r.get("flagged")]
    return {"n_fallacies": len(_DETECTORS), "n_evaluated": len(succeeded),
            "complete": len(succeeded) == len(_DETECTORS), "flagged": flagged, "errored": errored,
            "crashed": crashed, "results": results}
