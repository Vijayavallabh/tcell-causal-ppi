"""Near-null-signal freeze gate for the Stage-B fits (feat-008 §c).

A Stage-B calibration head or a rationale head is fitted in a regime where the graph signal is at
statistical parity with no-graph, so a head can look good purely by fitting noise. This gate decides
whether such a fit may be FROZEN / promoted, and it decides it from a PAIRED comparison against the
fit's own control — never from a headline number read on its own.

    evaluate_gate({"name": {"fit": {unit: value}, "control": {unit: value},
                            "higher_is_better": bool, "units": [...]}}) -> report

Per contrast the per-unit advantage (oriented so positive == the fit is better) goes through the SAME
paired-t core the multi-seed campaign uses (``screening.multiseed.paired_delta_summary``), then the
family of simultaneous contrasts gets BOTH Bonferroni and Holm — a freeze requires both, so the
correction cannot be shopped after seeing which one rescues the claim.

Three outcomes, deliberately distinct (``None`` is not a negative):

    freeze       every required contrast has mean > 0, a CI excluding zero, and survives family-wise
    refuse       some required contrast was COMPUTED and did not clear (crosses zero, dies under
                 correction, or is significant on the WRONG side — a fit that reliably loses)
    undecidable  some required contrast could not be decided at all (absent, n<2, zero variance,
                 every unit non-finite). Absence of evidence is never a pass.

``exit_code`` maps those to 0 / 1 / 2, so an unattended run that cannot freeze never exits 0.
"""
from __future__ import annotations

# the paired-t core and the family-wise correction are the hardened ones from the multi-seed campaign
# (zero variance -> undecidable, missing unit -> dropped loudly, BOTH corrections recorded). Reusing
# them keeps one implementation of those guards; `_apply_family_wise` is private only because nothing
# outside that module needed it before. Its unit axis is named "seed" there and stays so in the report.
from tcell_pipeline.screening.multiseed import _apply_family_wise, paired_delta_summary

FREEZE = "freeze"
REFUSE = "refuse"
UNDECIDABLE = "undecidable"

_EXIT = {FREEZE: 0, REFUSE: 1, UNDECIDABLE: 2}


def require_unique_units(units, what: str = "units") -> list:
    """Unit ids key the paired comparison; duplicates collapse into one dict entry and silently shrink n,
    leaving an underpowered contrast that still reports a clean verdict. Raise instead."""
    units = list(units)
    if len(set(units)) != len(units):
        raise ValueError(f"{what} are not uniquely identified ({len(units)} values, "
                         f"{len(set(units))} distinct) — the paired contrast would drop rows")
    return units


def _stub(name: str) -> dict:
    """A required contrast that was never computed: undecidable, and it says so."""
    return {"n": 0, "seeds_used": [], "dropped": [], "deltas": [], "mean": None, "sd": None,
            "se": None, "t": None, "p_value": None, "ci_low": None, "ci_high": None,
            "ci_excludes_zero": None, "fit_mean": None, "control_mean": None,
            "verdict": f"not computed: contrast {name!r} is required but absent from the gate inputs"}


def _summarise(spec: dict, alpha: float) -> dict:
    """Paired advantage of ``fit`` over ``control``, oriented so positive always means "the fit won"."""
    fit, control = spec["fit"], spec["control"]
    higher_is_better = bool(spec.get("higher_is_better", True))
    better, worse = (fit, control) if higher_is_better else (control, fit)
    # pass the REQUESTED unit universe: without it a unit missing from BOTH arms vanishes silently and
    # n shrinks with an empty `dropped` list
    units = spec.get("units")
    out = paired_delta_summary(better, worse, alpha=alpha,
                               seeds=require_unique_units(units) if units is not None
                               else sorted(set(fit) | set(control)))
    used = out["seeds_used"]
    # Arm means over the USED units only, so they reconcile exactly with the reported advantage — but
    # the advantage is ORIENTED (positive always means the fit won), so it equals fit_mean - control_mean
    # only when higher is better, and control_mean - fit_mean when lower is better. These stay the RAW
    # arm values (a reader wants the actual NLL / distance), so ``higher_is_better`` below is what makes
    # the subtraction unambiguous; ``render`` prints it beside every row for the same reason.
    out["fit_mean"] = sum(float(fit[u]) for u in used) / len(used) if used else None
    out["control_mean"] = sum(float(control[u]) for u in used) / len(used) if used else None
    out["higher_is_better"] = higher_is_better
    return out


def _decide(c: dict) -> str:
    """One contrast's call. Significance alone is NOT a pass: a fit that is reliably WORSE than its
    control also excludes zero and also survives correction — direction is what separates them."""
    # absent / no finite unit -> mean is None; n<2 / zero variance -> the paired core leaves
    # ci_excludes_zero None rather than inventing a CI. An explicit `n < 2` clause here would be
    # decoration: no input can reach it that these two do not already catch (pinned by
    # test_single_unit_is_undecidable, which holds the behaviour if that core ever changes).
    if c.get("mean") is None or c.get("ci_excludes_zero") is None:
        return UNDECIDABLE
    if c["mean"] > 0 and c["ci_excludes_zero"] and c.get("survives_family_wise"):
        return FREEZE
    return REFUSE


def evaluate_gate(contrasts: dict, *, alpha: float = 0.05, required=None) -> dict:
    """Decide whether these fits may be frozen. ``required`` defaults to every supplied contrast; a
    required name with no inputs is UNDECIDABLE, and an EMPTY contrast set is undecidable too (``all([])``
    is vacuously true — a gate with nothing to check must never read as a pass)."""
    required = tuple(contrasts) if required is None else tuple(required)
    out = {name: _summarise(spec, alpha) for name, spec in contrasts.items()}
    for name in required:
        out.setdefault(name, _stub(name))
    _apply_family_wise(out, alpha)
    for c in out.values():
        c["decision"] = _decide(c)

    calls = [out[name]["decision"] for name in required]
    if not calls:
        decision = UNDECIDABLE                   # nothing was required -> nothing was shown
    elif REFUSE in calls:
        decision = REFUSE                        # a computed negative is a decision, not a gap
    elif UNDECIDABLE in calls:
        decision = UNDECIDABLE
    else:
        decision = FREEZE
    return {"decision": decision, "alpha": alpha, "required": list(required), "contrasts": out}


def exit_code(report: dict) -> int:
    """0 only for a freeze. A missing/unknown decision is not a pass (see the campaign that printed
    'NOT comparable' and then returned 0)."""
    return _EXIT.get(report.get("decision"), _EXIT[UNDECIDABLE])


def fmt(v, spec: str = "+.4f") -> str:
    """None-safe: the degenerate case is exactly when these are None, and a `None:+.4f` in the verdict
    print crashes on the one input the guard exists for."""
    return "n/a" if v is None else format(v, spec)


def render(report: dict) -> str:
    """Markdown summary. Every headline is printed WITH its control on the same row — a fit number
    without its control is not a result."""
    lines = [f"decision: **{report.get('decision')}** (alpha={report.get('alpha')}, "
             f"required={', '.join(report.get('required', [])) or 'none'})", "",
             "| contrast | direction | n | fit | control | advantage | 95% CI | raw p | Bonferroni | Holm | call |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in report.get("contrasts", {}).items():
        ci = (f"[{fmt(c.get('ci_low'))}, {fmt(c.get('ci_high'))}]"
              if c.get("ci_low") is not None else "n/a")
        # the advantage is oriented (positive == the fit won), so without the direction a reader cannot
        # tell whether it is fit-minus-control or control-minus-fit
        direction = "n/a" if c.get("higher_is_better") is None else \
            ("higher is better" if c["higher_is_better"] else "lower is better")
        lines.append(
            f"| {name} | {direction} | {c.get('n')} | {fmt(c.get('fit_mean'))} | {fmt(c.get('control_mean'))} | "
            f"{fmt(c.get('mean'))} | {ci} | {fmt(c.get('p_value'), '.4f')} | "
            f"{fmt(c.get('p_bonferroni'), '.4f')} | {fmt(c.get('p_holm'), '.4f')} | {c.get('decision')} |")
    lines += ["", "verdicts:"] + [f"- {n}: {c.get('verdict')}" for n, c in report.get("contrasts", {}).items()]
    return "\n".join(lines)
