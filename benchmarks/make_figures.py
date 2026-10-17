#!/usr/bin/env python3
"""
Figures for benchmarks/training_methods.md: relative L2 error against time,
one line per method, for each example of the study.

The numbers are the Error Analysis results of the GPU runs (benchmarks/
figures/runs.json, one record per run: example, method label, [t, L2] pairs).
Rebuild the figures with
    python3 benchmarks/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

# categorical slots, fixed order (validated palette); the plain baseline is a
# neutral gray, dashed -- it is the reference line, not a method
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASE = "#8a8a86"
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"


def chart(fig_name, title, subtitle, runs, xlabel="$t$", note=None):
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=130)
    fig.patch.set_facecolor("white")
    k = 0
    for r in runs:
        ts = [p[0] for p in r["l2"] if p[0] > 0]
        ls = [p[1] * 100 for p in r["l2"] if p[0] > 0]
        if r.get("baseline"):
            ax.plot(ts, ls, color=BASE, lw=1.6, ls=(0, (4, 3)), marker="o", ms=4.5, label=r["label"], zorder=2)
            continue
        c = SERIES[k % len(SERIES)]
        k += 1
        best = r.get("best")
        ax.plot(ts, ls, color=c, lw=3.0 if best else 2.0, marker="o", ms=6.5 if best else 5,
                markeredgecolor="white", markeredgewidth=1.0, label=r["label"] + ("  (template)" if r.get("template") else ""),
                zorder=4 if best else 3)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    subs = (1.0,) if hi / lo > 300 else (1.0, 2.0, 5.0)       # label 2x and 5x steps on short ranges
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel("relative $L_2$ error (log scale)", color=INK2, fontsize=10)
    fig.text(0.012, 0.965, title, fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=9, color=INK2, ha="left", va="top")
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8.6, labelcolor=INK)
    for h in leg.legend_handles:
        h.set_alpha(1)
    if note:
        fig.text(0.012, 0.015, note, fontsize=8, color=INK2, ha="left", va="bottom")
    fig.subplots_adjust(left=0.09, right=0.6, top=0.82, bottom=0.17 if note else 0.13)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, fig_name)
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def main():
    data = json.load(open(os.path.join(OUT, "runs.json")))
    for fig in data["figures"]:
        print(chart(fig["file"], fig["title"], fig["subtitle"], fig["runs"], fig.get("xlabel", "$t$"), fig.get("note")))


if __name__ == "__main__":
    main()
