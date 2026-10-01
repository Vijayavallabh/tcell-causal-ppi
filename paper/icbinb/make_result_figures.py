"""Figures 3 to 5 (paper/figures/folds_floor.pdf, replication_forest.pdf, causes.pdf) from their artifacts.

Figure 3 puts h1 and h2a on the three folds beside both graph arms' injected-signal ladder; Figure 4
is the per-dataset replication of the typed and untyped contrasts with their pooled estimates; Figure 5
is the evidence for causes C and D.
Every plotted value is read from the artifact the paper's tables and prose quote, and the dataset
order and the K labels come from the same DE_stats_v2 provenance and built bases that Table 5's check
reads, so the figures cannot drift from the text.

    .venv/bin/python paper/icbinb/make_result_figures.py
"""
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data/results"
REPL = ROOT / "data/intermediate/replication"
FIG = ROOT / "paper/figures"
# One colour per arm across every figure: purple typed static, orange gated, blue untyped.
COL = {"typed_static": "#7b3294", "condition_gated": "#d9822b", "untyped_gnn": "#1f78b4"}
NAMES = {"reference_screen_n7": r"CD4$^{+}$ T, reference ($n{=}7$)",
         "ReplogleWeissman2022_K562_gwps": "Replogle K562 genome-wide",
         "ReplogleWeissman2022_rpe1": "Replogle RPE1",
         "ReplogleWeissman2022_K562_essential": "Replogle K562 essential",
         "FrangiehIzar2021_RNA": "Frangieh melanoma",
         "TianKampmann2021_CRISPRi": "Tian CRISPRi",
         "NormanWeissman2019_filtered": "Norman K562",
         "TianKampmann2021_CRISPRa": "Tian CRISPRa"}


def load(rel: str):
    return json.loads((RES / rel).read_text())


def point(ax, x, y, lo, hi, color, filled, marker="o"):
    ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt="none", ecolor=color, elinewidth=1.1)
    ax.plot(x, y, marker, ms=4.2, mec=color, mfc=color if filled else "white", mew=1.1)


def vpoint(ax, x, y, lo, hi, color, filled, marker="o"):
    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=color, elinewidth=1.1)
    ax.plot(x, y, marker, ms=4.2, mec=color, mfc=color if filled else "white", mew=1.1)


def diamond(ax, y, est, lo, hi, color):
    ax.fill([lo, est, hi, est], [y, y - 0.3, y, y + 0.3], color=color, alpha=0.85, lw=0)


def dataset_order(per: dict) -> list:
    """Reference first, then the replication screens by built target count, largest first."""
    n = {}
    for ds in per:
        prov = REPL / f"{ds}.DE_stats_v2.provenance.json"
        if prov.exists():
            n[ds] = json.loads(prov.read_text())["n_targets"]
    return ["reference_screen_n7"] + sorted(n, key=n.get, reverse=True)


def program_k(ds: str):
    p = REPL / ds / "program_response.parquet"
    return pd.read_parquet(p).shape[1] - 1 if p.exists() else None


def forest() -> None:
    """Figure 4. Pooled diamonds: random effects over the seven replication screens, and over all
    eight datasets including the reference screen, for both arms, so neither pool is chosen per arm."""
    with_ref = load("replication/pooled_with_reference.json")
    seven = load("replication/pooled.json")["pooled"]
    order = dataset_order(with_ref["per_dataset"]["h2a"])
    assert set(order) == set(NAMES), f"dataset set changed: {sorted(set(order) ^ set(NAMES))}"
    labels = []
    for ds in order:
        k = program_k(ds) if ds != "reference_screen_n7" else 128
        labels.append(NAMES[ds] + ("" if k == 128 else rf" ($K{{=}}{k}$)"))
    labels += ["pooled, 7 replication screens", "pooled, all 8 datasets"]
    ys = list(range(len(order))) + [len(order) + 0.6, len(order) + 1.6]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), sharey=True)
    panels = (("h2a", "typed_static", r"(a) typed static $-$ no graph"),
              ("promotion_margin", "untyped_gnn", r"(b) untyped $-$ no graph"))
    for ax, (key, arm, title) in zip(axes, panels):
        per = with_ref["per_dataset"][key]
        for y, ds in zip(ys, order):
            c = per[ds]
            point(ax, c["mean"], y, c["ci"][0], c["ci"][1], COL[arm], c["survives_family_wise"])
        for y, pool in zip(ys[-2:], (seven[key], with_ref["pooled"][key])):
            lo, hi = pool["random_ci"]
            diamond(ax, y, pool["random_effect"], lo, hi, COL[arm])
            ax.text(hi + 0.006, y, rf"$I^2{{=}}{100 * pool['I2']:.0f}\%$", va="center", fontsize=6.5)
        ax.axvline(0, color="0.45", lw=0.8, ls="--", zorder=0)
        ax.axhline(len(order) - 0.2, color="0.8", lw=0.6)
        ax.set_xlim(-0.175, 0.175)
        ax.set_xticks([-0.1, 0, 0.1])
        ax.set_xticks([-0.15, -0.05, 0.05, 0.15], minor=True)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel(r"$\Delta$ SYSTEMA vs. no graph")
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=0.6)
    out = FIG / "replication_forest.pdf"
    fig.savefig(out)
    print(f"wrote {out}: {len(order)} datasets, pooled RE h2a 7={seven['h2a']['random_effect']:+.4f} "
          f"8={with_ref['pooled']['h2a']['random_effect']:+.4f}, untyped "
          f"7={seven['promotion_margin']['random_effect']:+.4f} "
          f"8={with_ref['pooled']['promotion_margin']['random_effect']:+.4f}")


def oriented(c: dict, arm: str):
    """robustness_5seed.json stores better-minus-worse; return arm-minus-baseline and its CI."""
    s = 1.0 if c["better"] == arm else -1.0
    lo, hi = sorted((s * c["ci_low"], s * c["ci_high"]))
    return s * c["mean"], lo, hi


def folds_floor() -> None:
    """Figure 3. (a) h1 and h2a on the three folds, n=5 each; (b) the injected-signal ladder."""
    folds = (("frozen\n(0.85/0.05)", "screening_lambda0"),
             ("intermediate\n(0.80/0.10)", "screening_c080c10_h1"),
             ("harder\n(0.75/0.15)", "screening_c075c15_n5"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 1.95), gridspec_kw={"width_ratios": [1, 1.35]})
    for y, (_, root) in enumerate(folds):
        cs = load(f"{root}/robustness_5seed.json")["contrasts"]
        for key, arm, dy in (("h1_vs_no_graph", "condition_gated", -0.14), ("h2a", "typed_static", 0.14)):
            m, lo, hi = oriented(cs[key], arm)
            point(a, m, y + dy, lo, hi, COL[arm], cs[key]["survives_family_wise"])
    a.axvline(0, color="0.45", lw=0.8, ls="--", zorder=0)
    a.set_yticks(range(len(folds)))
    a.set_yticklabels([f[0] for f in folds], fontsize=7)
    a.set_ylim(2.55, -0.95)   # inverted, with headroom for the legend
    a.set_xlabel(r"$\Delta$ SYSTEMA vs. no graph")
    a.set_xlim(-0.042, 0.022)
    a.set_xticks([-0.04, -0.02, 0, 0.02])
    a.tick_params(axis="x", labelsize=6.5)
    a.set_title("(a) h1 and h2a on three folds", fontsize=8, loc="left")
    a.plot([], [], "o", mec=COL["condition_gated"], mfc=COL["condition_gated"], ms=4, label="h1 (gated)")
    a.plot([], [], "o", mec=COL["typed_static"], mfc=COL["typed_static"], ms=4, label="h2a (typed)")
    a.legend(fontsize=6.5, loc="upper left", ncol=2, frameon=False, handletextpad=0.2, columnspacing=0.8,
             borderaxespad=0.2)

    lad = {"untyped_gnn": load("a2_ladder/floor.json"),
           "condition_gated": load("c1_ladder/floor_condition_gated.json")}
    rungs = ["zero_point", "d020", "d050", "d100", "d200", "d400", "permuted_d400"]
    ref = lad["untyped_gnn"]["rungs"]
    ticks = ["0"] + [f"{ref[r]['delta']:.2f}" for r in rungs[1:-1]] + [f"{ref['permuted_d400']['delta']:.2f}\nscr."]
    for arm, dx in (("untyped_gnn", -0.13), ("condition_gated", 0.13)):
        d = lad[arm]
        for x, r in enumerate(rungs):
            c = d["zero_point"] if r == "zero_point" else d["contrasts"][r]
            filled = r != "zero_point" and c["survives_family_wise"]
            vpoint(b, x + dx, c["mean"], c["ci_low"], c["ci_high"], COL[arm], filled,
                   marker="s" if r == "zero_point" else "o")
    b.axhline(0, color="0.45", lw=0.8, ls="--", zorder=0)
    b.axvline(len(rungs) - 1.5, color="0.8", lw=0.6)
    b.set_xticks(range(len(rungs)))
    b.set_xticklabels(ticks, fontsize=7)
    b.tick_params(axis="y", labelsize=7)
    b.set_xlabel(r"injected signal $\delta$ (response SDs)")
    b.set_ylabel(r"$\Delta$ SYSTEMA vs. no graph", fontsize=7.5)
    b.set_title("(b) injected-signal ladder, four seeds", fontsize=8, loc="left")
    b.plot([], [], "o", mec=COL["untyped_gnn"], mfc=COL["untyped_gnn"], ms=4, label="untyped")
    b.plot([], [], "o", mec=COL["condition_gated"], mfc=COL["condition_gated"], ms=4, label="gated")
    b.legend(fontsize=6.5, loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.0)
    out = FIG / "folds_floor.pdf"
    fig.savefig(out)
    print(f"wrote {out}: untyped floor {lad['untyped_gnn']['floor']}, gated floor status "
          f"{lad['condition_gated']['floor_status']}")


def tci(values: list) -> tuple:
    """Mean and 95% t interval of paired per-seed differences, as the paper's tables compute them."""
    m, sd, n = st.mean(values), st.stdev(values), len(values)
    h = stats.t.ppf(0.975, n - 1) * sd / n ** 0.5
    return m, m - h, m + h


def causes() -> None:
    """Figure 5. (a) the n=7 family on the frozen fold, cause C; (b) cause D: what the graph-derived
    features inside the "no graph" arm are worth (intact minus zeroed), beside message passing (h1 at
    n=5, the same five seeds and baseline lanes) and the paired test of the two, which is not a
    pre-registered contrast and is drawn hollow and grey."""
    n7 = load("screening_n7_live/robustness_5seed.json")["contrasts"]
    rows_a = (("typed static $-$ no graph", "h2a", "typed_static"),
              ("gated $-$ typed static", "h2b", "condition_gated"),
              ("gated $-$ no graph (h1)", "h1_vs_no_graph", "condition_gated"),
              ("untyped $-$ no graph", "promotion_margin", "untyped_gnn"))
    fab = {v["variant"]: v for v in load("feature_ablation_report.json")}
    h1 = load("screening_lambda0/robustness_5seed.json")["contrasts"]["h1_vs_no_graph"]
    h1s = dict(zip(h1["seeds_used"], (d if h1["better"] == "condition_gated" else -d for d in h1["deltas"])))
    # value of a channel = intact - zeroed; the artifact stores zeroed - intact
    val = {v: {int(s): -d for s, d in fab[v]["per_seed"].items()} for v in fab}
    seeds = sorted(set(val["nograph"]) & set(h1s))
    assert len(seeds) == 5 and seeds == sorted(h1s), f"ablation and h1 seeds differ: {seeds}, {sorted(h1s)}"
    paired = tci([val["nograph"][s] - h1s[s] for s in seeds])
    grey = "0.35"
    rows_b = [("graph features, all", tci(list(val["nograph"].values())), fab["nograph"]["survives"] == "True", grey),
              ("degree scalars only", tci(list(val["nodegree"].values())), fab["nodegree"]["survives"] == "True", grey),
              ("PINNACLE only", tci(list(val["nopinnacle"].values())), fab["nopinnacle"]["survives"] == "True", grey),
              ("gated $-$ no graph (h1)", oriented(h1, "condition_gated"), h1["survives_family_wise"],
               COL["condition_gated"])]

    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 1.5), gridspec_kw={"width_ratios": [1, 1]})
    for y, (label, key, arm) in enumerate(rows_a):
        m, lo, hi = oriented(n7[key], arm)
        point(a, m, y, lo, hi, COL[arm], n7[key]["survives_family_wise"])
    a.set_yticks(range(len(rows_a)))
    a.set_yticklabels([r[0] for r in rows_a], fontsize=7)
    a.set_ylim(len(rows_a) - 0.5, -0.6)
    a.set_title(r"(a) the $n{=}7$ family, frozen fold", fontsize=8, loc="left")
    for y, (label, (m, lo, hi), filled, col) in enumerate(rows_b):
        point(b, m, y, lo, hi, col, filled)
    yd = len(rows_b) + 0.35
    point(b, paired[0], yd, paired[1], paired[2], "0.6", False, marker="D")
    b.axhline(len(rows_b) - 0.33, color="0.8", lw=0.6)
    b.set_yticks(list(range(len(rows_b))) + [yd])
    b.set_yticklabels([r[0] for r in rows_b] + ["features $-$ h1, paired"], fontsize=7)
    b.set_ylim(yd + 0.6, -0.6)
    b.set_title(r"(b) graph features and h1, $n{=}5$", fontsize=8, loc="left")
    for ax in (a, b):
        ax.axvline(0, color="0.45", lw=0.8, ls="--", zorder=0)
        ax.set_xlabel(r"$\Delta$ SYSTEMA", fontsize=7.5)
        ax.tick_params(axis="x", labelsize=6.5)
    fig.tight_layout(w_pad=1.2)
    # Start each title over its row labels, not over the plot, which the long labels push right.
    r = fig.canvas.get_renderer()
    for ax in (a, b):
        x = ax.transAxes.inverted().transform((ax.yaxis.get_tightbbox(r).x0, 0))[0]
        ax.set_title(ax.get_title(loc="left"), loc="left", x=x, fontsize=8)
    out = FIG / "causes.pdf"
    fig.savefig(out)
    print(f"wrote {out}: n=7 h2a {oriented(n7['h2a'], 'typed_static')[0]:+.4f}; features worth "
          f"{rows_b[0][1][0]:+.4f}, h1 {rows_b[3][1][0]:+.4f}, paired {paired[0]:+.4f} "
          f"[{paired[1]:+.4f},{paired[2]:+.4f}]")


def main() -> None:
    # Same settings as Figure 2; pdf.fonttype 42 embeds TrueType rather than Type 3 fonts.
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8,
                         "pdf.fonttype": 42})
    FIG.mkdir(parents=True, exist_ok=True)   # paper/figures/ is not in a fresh clone
    forest()
    folds_floor()
    causes()


if __name__ == "__main__":
    main()
