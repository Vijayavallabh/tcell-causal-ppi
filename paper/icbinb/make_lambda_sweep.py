"""Figure 2 (paper/figures/lambda_sweep.pdf) from its artifacts, so it can be regenerated and checked.

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
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8,
                         "pdf.fonttype": 42})
    # Orange is the gated arm in every figure of the paper; these are its gates.
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.0), gridspec_kw={"width_ratios": [1.15, 1]})
    # Labelled through the legend, not as text on the plot, where a label hides the markers under it.
    ax.axhspan(min(live), max(live), color="tab:green", alpha=0.15, lw=0,
               label=f"live campaign: {min(live):.2f} to {max(live):.2f}")
    ax.axhline(fix, color="tab:green", lw=1.2, label=rf"$\lambda{{=}}0$ (fix): gate $={fix:.2f}$")
    ax.plot(lam, [p[1]["val_gate"] for p in pts], "o-", color="#d9822b", ms=4, label="mean edge gate")
    ax.axvline(1e-2, color="darkred", ls=":", lw=1.2)
    ax.set(xscale="log", yscale="log", ylim=(4e-8, 3), xlabel=r"edge-sparsity weight $\lambda_{\rm graph}$",
           ylabel="mean edge gate at end")
    ax2 = ax.twinx()
    ax2.plot(lam, [p[1]["train_frac_dead"] for p in pts], "s--", color="0.3", ms=3.5,
             label="fraction dead")
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_ylabel("fraction of gates dead", color="0.3")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # One row under both panels: inside panel (a) no free region is large enough, and a legend there
    # hides the very curve the figure exists to show.
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, fontsize=6.5, frameon=False,
               handlelength=1.6, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    ax.set_title(r"(a) final gate against $\lambda_{\rm graph}$", fontsize=8, loc="left")

    # (b) the same runs epoch by epoch: how fast each weight kills the gates, and that none recovers.
    cmap = plt.get_cmap("Oranges")
    for i, r in enumerate(sorted((r for r in runs if r["lambda_graph"] > 0), key=lambda r: r["lambda_graph"])):
        t = r["trajectory"]
        bx.plot([e["epoch"] + 1 for e in t], [e["val_gate"] for e in t], "-", lw=1.1,
                color=cmap(0.35 + 0.65 * i / max(1, len(pts) - 1)))
        if r["lambda_graph"] in (1e-5, 1e-2, 1e-1):
            bx.text(t[-1]["epoch"] + 1.4, t[-1]["val_gate"],
                    rf"$10^{{{round(__import__('math').log10(r['lambda_graph']))}}}$", fontsize=6.5, va="center")
    zero = next(r for r in runs if r["lambda_graph"] == 0)["trajectory"]
    bx.plot([e["epoch"] + 1 for e in zero], [e["val_gate"] for e in zero], "-", color="tab:green", lw=1.2)
    bx.text(zero[-1]["epoch"] + 1.4, zero[-1]["val_gate"], "0", fontsize=6.5, va="center")
    bx.axhline(1e-3, color="0.45", ls="--", lw=0.8)
    bx.set(yscale="log", ylim=(4e-8, 3), xlim=(0.5, len(zero) + 4.6), xlabel="epoch",
           ylabel="mean edge gate")
    bx.set_title(r"(b) gate per epoch, by $\lambda_{\rm graph}$", fontsize=8, loc="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)   # paper/figures/ is not in a fresh clone
    fig.savefig(OUT)
    print(f"wrote {OUT} (live band {min(live):.3f} to {max(live):.3f}, fix gate {fix:.3f})")


if __name__ == "__main__":
    main()
