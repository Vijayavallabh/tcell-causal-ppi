"""Figure 1 (paper/figures/lambda_sweep.pdf) from its artifact, so it can be regenerated and checked.

One panel, three lines: the mean edge gate per epoch with no penalty, with the weakest penalty tried and
with the default one, against the dead-gate threshold. The caption's statements about the other five
weights (when each first falls below the threshold, and that none recovers) are recomputed from the
same file by verify_numbers.py. An earlier two-panel version also plotted the final gate and the
fraction of dead gates against every weight; the plain-language revision of 2026-10-28 dropped it for
readability.

    .venv/bin/python paper/icbinb/make_lambda_sweep.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/lambda_sweep.pdf"
DEAD = 1e-3   # config.GATE_DEAD: a mean gate below this is switched off
SHOWN = [(1e-2, "default penalty ($10^{-2}$)", "#b4451f"),
         (1e-5, "weakest penalty tried ($10^{-5}$)", "#e8a15a"),
         (0.0, "no penalty (our fix)", "tab:green")]


def main() -> None:
    runs = {r["lambda_graph"]: r["trajectory"]
            for r in json.loads((ROOT / "data/results/q4_lambda_sweep_22ep.json").read_text())["runs"]}

    # pdf.fonttype 42 embeds TrueType; the default (3) leaves Type 3 fonts, which some PDF checkers and
    # print shops reject.
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 9,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    for lam, label, color in SHOWN:
        t = runs[lam]
        ax.plot([e["epoch"] + 1 for e in t], [e["val_gate"] for e in t], "-", lw=1.8, color=color,
                label=label)
    ax.axhline(DEAD, color="0.4", ls="--", lw=1.0)
    ax.text(22.4, DEAD * 1.6, "off below this line", fontsize=7.5, color="0.3", ha="right")
    ax.set(yscale="log", ylim=(1e-10, 3), xlim=(0.5, 22.5), xlabel="epoch (pass over the data)",
           ylabel="mean edge gate", xticks=[1, 5, 10, 15, 20])
    # The y-axis runs down to 1e-10 to leave an empty band below every line (the lowest point is ~9e-7)
    # for the legend, and the threshold label sits at the right, so neither hides a line.
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)   # paper/figures/ is not in a fresh clone
    fig.savefig(OUT)
    end = {lam: runs[lam][-1]["val_gate"] for lam, _, _ in SHOWN}
    print(f"wrote {OUT} (final gates: " + ", ".join(f"{k:g}: {v:.3g}" for k, v in end.items()) + ")")


if __name__ == "__main__":
    main()
