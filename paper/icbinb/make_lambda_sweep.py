"""Figure 1 (paper/figures/lambda_sweep.pdf) from its artifacts, so it can be regenerated and checked.

The submitted figure was drawn ad hoc with no generator, and its "campaign band 0.57-0.77" matched no
definition of the live gates: the five scored lambda_graph=0 checkpoints span 0.27 to 0.78. The band
is now read from camera_ready/facts.json and the sweep from q4_lambda_sweep_22ep.json, and the y-axis
no longer clips the points at and beyond the default weight that the caption describes.

    .venv/bin/python paper/icbinb/make_lambda_sweep.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/lambda_sweep.pdf"


def main() -> None:
    runs = json.loads((ROOT / "data/results/q4_lambda_sweep_22ep.json").read_text())["runs"]
    facts = json.loads((ROOT / "data/results/camera_ready/facts.json").read_text())
    live = [lane["gate_at_best"] for lane in facts["training"]["condition_gated"].values()]
    fix = next(r for r in runs if r["lambda_graph"] == 0)["trajectory"][-1]["val_gate"]
    pts = sorted((r["lambda_graph"], r["trajectory"][-1]) for r in runs if r["lambda_graph"] > 0)
    lam = [p[0] for p in pts]

    # pdf.fonttype 42 embeds TrueType; the default (3) leaves Type 3 fonts, which some PDF checkers and
    # print shops reject.
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    # Labelled through the legend, not as text on the plot, where a label hides the markers under it.
    ax.axhspan(min(live), max(live), color="tab:green", alpha=0.15, lw=0,
               label=f"live campaign: {min(live):.2f} to {max(live):.2f}")
    ax.axhline(fix, color="tab:green", lw=1.2, label=rf"$\lambda{{=}}0$ (fix): gate $={fix:.2f}$")
    ax.plot(lam, [p[1]["val_gate"] for p in pts], "o-", color="#1f4e79", ms=4, label="mean edge gate")
    ax.axvline(1e-2, color="darkred", ls=":", lw=1.2)
    ax.set(xscale="log", yscale="log", ylim=(4e-8, 3), xlabel=r"edge-sparsity weight $\lambda_{\rm graph}$",
           ylabel="mean edge gate")
    ax2 = ax.twinx()
    ax2.plot(lam, [p[1]["train_frac_dead"] for p in pts], "s--", color="#d9822b", ms=4,
             label="fraction dead")
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_ylabel("fraction of gates dead", color="#b8650f")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", bbox_to_anchor=(1.0, 0.5), fontsize=6.5, frameon=True,
              facecolor="white", edgecolor="none", framealpha=1.0,
              handlelength=1.6)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)   # paper/figures/ is not in a fresh clone
    fig.savefig(OUT)
    print(f"wrote {OUT} (live band {min(live):.3f} to {max(live):.3f}, fix gate {fix:.3f})")


if __name__ == "__main__":
    main()
