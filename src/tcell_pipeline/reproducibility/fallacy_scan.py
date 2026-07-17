"""Statistical-fallacy scan (feat-013): eleven independent detectors run over the study's diagnostic inputs
so a reproduction can assert none of the classic inference traps is silently driving a headline claim
(report §reproducibility 11/11 fallacy scan). Each detector is a small, self-contained numpy check that
returns ``{flagged, evidence}``; ``run_fallacy_scan`` runs whichever inputs are supplied and reports coverage
(the "11/11" is complete coverage, not eleven flags).

The eleven: Simpson, ecological, Berkson, collider, base-rate, regression-to-mean, survivorship,
look-elsewhere, garden-of-forks, correlation≠causation, reverse-causation.
"""
from __future__ import annotations

import numpy as np


def _corr(x, y) -> float:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _sign(v: float, eps: float = 1e-9) -> int:
    return 0 if abs(v) < eps else int(np.sign(v))


def simpson(groups, eps: float = 1e-9) -> dict:
    """Sign reversal between the pooled trend and the (consistent) within-group trends. ``groups`` is a list
    of (x, y) pairs, one per subgroup."""
    within = [_sign(_corr(x, y)) for x, y in groups]
    px = np.concatenate([np.asarray(x, np.float64) for x, y in groups])
    py = np.concatenate([np.asarray(y, np.float64) for x, y in groups])
    pooled = _sign(_corr(px, py))
    consistent = {s for s in within if s != 0}
    reversed_ = len(consistent) == 1 and pooled != 0 and pooled == -next(iter(consistent))
    return {"flagged": bool(reversed_), "evidence": {"pooled_sign": pooled, "within_signs": within}}


def ecological(x, y, group, inflation: float = 0.3) -> dict:
    """Aggregate (group-mean) correlation inflated or sign-flipped vs the individual-level correlation.

    Needs >=3 groups: the correlation of 2 group means is degenerately +-1 regardless of the individual data,
    so it would spuriously flag every 2-group study — with <3 groups the aggregate is uninformative and we do
    not flag."""
    x, y, group = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(group)
    indiv = _corr(x, y)
    gs = [group == g for g in np.unique(group)]
    if len(gs) < 3:
        return {"flagged": False, "evidence": {"individual_corr": indiv, "aggregate_corr": None,
                                               "n_groups": len(gs)}}
    gx = np.array([x[m].mean() for m in gs])
    gy = np.array([y[m].mean() for m in gs])
    agg = _corr(gx, gy)
    flag = (_sign(agg) != _sign(indiv) and _sign(agg) != 0) or (abs(agg) - abs(indiv) > inflation)
    return {"flagged": bool(flag),
            "evidence": {"individual_corr": indiv, "aggregate_corr": agg, "n_groups": len(gs)}}


def berkson(x, y, selected, margin: float = 0.2) -> dict:
    """Selection on a collider induces a spurious (more-negative) association within the selected sample."""
    x, y, selected = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(selected, bool)
    full = _corr(x, y)
    sel = _corr(x[selected], y[selected])
    return {"flagged": bool(sel < full - margin), "evidence": {"corr_full": full, "corr_selected": sel}}


def _partial_corr(x, y, z) -> float:
    rxy, rxz, ryz = _corr(x, y), _corr(x, z), _corr(y, z)
    den = np.sqrt(max(1 - rxz ** 2, 0.0)) * np.sqrt(max(1 - ryz ** 2, 0.0))
    return float((rxy - rxz * ryz) / den) if den > 1e-12 else 0.0


def collider(x, y, z, margin: float = 0.2) -> dict:
    """Conditioning on a collider ``z`` induces association between marginally-independent ``x`` and ``y``."""
    marginal = _corr(x, y)
    partial = _partial_corr(x, y, z)
    return {"flagged": bool(abs(partial) - abs(marginal) > margin),
            "evidence": {"marginal_corr": marginal, "partial_corr_given_z": partial}}


def base_rate(y_true, y_pred, acc_thresh: float = 0.9, precision_thresh: float = 0.5) -> dict:
    """High accuracy masking poor precision when the positive class is rare (base-rate neglect)."""
    yt, yp = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    acc = float((yt == yp).mean())
    tp = int(((yp == 1) & (yt == 1)).sum())
    precision = tp / max(int((yp == 1).sum()), 1)
    prevalence = float((yt == 1).mean())
    flag = acc >= acc_thresh and precision < precision_thresh
    return {"flagged": bool(flag),
            "evidence": {"accuracy": acc, "precision": precision, "prevalence": prevalence}}


def regression_to_mean(baseline, followup, select_frac: float = 0.1, margin: float = 0.1) -> dict:
    """An extreme-selected group reverts toward the grand mean on retest — apparent change that is noise."""
    b, f = np.asarray(baseline, np.float64), np.asarray(followup, np.float64)
    grand = float(np.concatenate([b, f]).mean())
    k = max(1, int(round(select_frac * len(b))))
    top = np.argsort(b)[-k:]                       # the extreme-high baseline group
    d_base = abs(b[top].mean() - grand)
    d_follow = abs(f[top].mean() - grand)
    flag = d_follow < d_base - margin * (abs(d_base) + 1e-9)
    return {"flagged": bool(flag),
            "evidence": {"grand_mean": grand, "selected_baseline_dev": d_base, "selected_followup_dev": d_follow}}


def survivorship(values, survived, margin: float = 0.2) -> dict:
    """A metric reported on survivors only differs materially from the full-population value."""
    v, s = np.asarray(values, np.float64), np.asarray(survived, bool)
    full = float(v.mean())
    surv = float(v[s].mean()) if s.any() else full
    spread = np.std(v) + 1e-9
    return {"flagged": bool(abs(surv - full) > margin * spread),
            "evidence": {"mean_full": full, "mean_survivors": surv}}


def look_elsewhere(pvalues, alpha: float = 0.05) -> dict:
    """A raw hit that does not survive family-wise (Bonferroni) correction across the tested hypotheses."""
    p = np.asarray(pvalues, np.float64)
    m = len(p)
    raw_hit = bool((p < alpha).any())
    survives = bool((p < alpha / max(m, 1)).any())
    return {"flagged": bool(raw_hit and not survives),
            "evidence": {"n_tests": m, "min_p": float(p.min()) if m else None,
                         "bonferroni_alpha": alpha / max(m, 1)}}


def garden_of_forks(estimates, eps: float = 1e-9) -> dict:
    """The effect estimate is fragile across analysis choices — its sign flips, or its spread exceeds its
    own magnitude (researcher degrees of freedom)."""
    e = np.asarray(estimates, np.float64)
    signs = {_sign(v) for v in e if abs(v) > eps}
    sign_flip = len(signs) > 1
    spread = float(e.max() - e.min())
    fragile = sign_flip or spread > abs(float(e.mean())) + eps
    return {"flagged": bool(fragile),
            "evidence": {"sign_flip": sign_flip, "spread": spread, "mean": float(e.mean())}}


def correlation_not_causation(corr, has_interventional_support, threshold: float = 0.3) -> dict:
    """A strong correlation asserted causally without interventional / confounder-adjusted support."""
    return {"flagged": bool(abs(float(corr)) >= threshold and not has_interventional_support),
            "evidence": {"corr": float(corr), "interventional_support": bool(has_interventional_support)}}


def reverse_causation(forward_corr, reverse_corr, margin: float = 0.0) -> dict:
    """The reverse cross-lagged association (y→x) is as strong as, or stronger than, the claimed forward one
    (x→y) — the causal direction is not identified by the data."""
    f, r = abs(float(forward_corr)), abs(float(reverse_corr))
    return {"flagged": bool(r >= f - margin), "evidence": {"forward": f, "reverse": r}}


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
    key is 'not evaluated'. ``complete`` is True only when all eleven ran (the 11/11 coverage the report
    requires); ``flagged`` lists the detectors that fired."""
    results: dict = {}
    for name, fn in _DETECTORS.items():
        spec = inputs.get(name)
        if spec is None:
            results[name] = {"evaluated": False}
            continue
        try:
            results[name] = {"evaluated": True, "error": None, **fn(**spec)}
        except Exception as exc:  # a malformed input for one detector must not sink the whole scan — but it is
            # NOT a clean pass: a detector that raised did NOT run, so it must not count toward 11/11 coverage
            # (else a crashed check would silently certify a study as REPRODUCIBLE with a fallacy unexamined).
            results[name] = {"evaluated": False, "error": f"{type(exc).__name__}: {exc}"}
    succeeded = [n for n, r in results.items() if r.get("evaluated")]  # ran cleanly (errored ones are False)
    errored = [n for n, r in results.items() if r.get("error")]
    flagged = [n for n, r in results.items() if r.get("flagged")]
    return {"n_fallacies": len(_DETECTORS), "n_evaluated": len(succeeded),
            "complete": len(succeeded) == len(_DETECTORS), "flagged": flagged, "errored": errored,
            "results": results}
