#!/usr/bin/env python3
"""
Figures for benchmarks/training_methods.md: relative L2 error against time,
one line per method, for each example of the study.

The numbers are the Error Analysis results of the GPU runs (benchmarks/
figures/runs.json, one record per run: example, method label, [t, L2] pairs).
Each figure is written twice: a PNG for the documentation and a vector PDF
(figures/pdf/) for papers and slides. Captions live in the markdown, not in
the figure. Rebuild with
    python3 benchmarks/make_figures.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

# categorical colours in a fixed order (validated palette); the reference
# line (plain PINN, or whatever the figure compares against) is a dashed gray
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<"]
BASE = "#7d7d78"
INK, INK2, GRID, FRAME = "#1f1f1e", "#4a4a46", "#e8e7e2", "#8c8b86"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "ytick.minor.size": 2,
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 8,
    "savefig.dpi": 220,
    "pdf.fonttype": 42,          # editable text in the PDF
})


def _pct(v, _):
    return f"{v:g}%"


def _legend(ax, loc, ncol):
    leg = ax.legend(loc=loc, ncol=ncol, frameon=True, fancybox=False, framealpha=0.95,
                    edgecolor="#cfcec8", borderpad=0.6, handlelength=2.6, labelspacing=0.45,
                    columnspacing=1.4, borderaxespad=0.7)
    leg.get_frame().set_linewidth(0.6)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _overlaps(fig, ax, leg):
    """True if any plotted line (sampled along its segments, in screen space)
    passes under the legend box."""
    fig.canvas.draw()
    box = leg.get_window_extent().expanded(1.02, 1.06)
    for line in ax.get_lines():
        xy = ax.transData.transform(list(zip(*line.get_data())))
        for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
            for f in np.linspace(0.0, 1.0, 25):
                if box.contains(x0 + f * (x1 - x0), y0 + f * (y1 - y0)):
                    return True
        if len(xy) == 1 and box.contains(*xy[0]):
            return True
    return False


def _place_legend(fig, ax, spec):
    """Legend inside the axes, never on top of the data: the first free
    corner, else two columns at the top with the y range raised to make room."""
    if "legend" in spec:
        _legend(ax, spec["legend"], spec.get("legend_cols", 1))
        return
    for loc in ("upper right", "lower right", "upper left", "lower left", "center right"):
        leg = _legend(ax, loc, 1)
        if not _overlaps(fig, ax, leg):
            return
        leg.remove()
    lo, hi = ax.get_ylim()
    for _ in range(12):
        leg = _legend(ax, "upper center", 2)
        if not _overlaps(fig, ax, leg):
            return
        leg.remove()
        hi *= 1.6
        ax.set_ylim(lo, hi)
    _legend(ax, "upper center", 2)


def chart(spec):
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    k = 0
    for r in spec["runs"]:
        pts = [p for p in r["l2"] if p[0] > 0]          # t = 0 is not drawn (log scale, exact IC)
        ts = [p[0] for p in pts]
        ls = [p[1] * 100 for p in pts]
        label = r["label"] + (" (template)" if r.get("template") else "")
        if r.get("baseline"):
            ax.plot(ts, ls, color=BASE, lw=1.4, ls=(0, (5, 3)), marker="o", ms=4, mfc="white",
                    mew=1.1, label=label, zorder=2)
            continue
        c, m = SERIES[k % len(SERIES)], MARKERS[k % len(MARKERS)]
        k += 1
        best = r.get("best")
        ax.plot(ts, ls, color=c, lw=2.4 if best else 1.4, marker=m, ms=6 if best else 4.5,
                mec="white" if best else c, mew=0.8, label=label, zorder=4 if best else 3)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    decades = hi / lo > 300
    if decades:                                   # every point between two labelled decades
        bottom = min(p[1] * 100 for r in spec["runs"] for p in r["l2"] if p[0] > 0)
        ax.set_ylim(min(lo, 10 ** np.floor(np.log10(bottom)) * 0.9), hi)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,) if decades else (1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    top = max(p[1] * 100 for r in spec["runs"] for p in r["l2"] if p[0] > 0)
    # no labels far above the data (the y range may be raised for the legend)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _pct(v, _) if v <= top * 2.5 else ""))
    ax.grid(True, which="major", axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(FRAME)
    ax.tick_params(colors=INK2, which="both")
    ax.set_xlabel(spec.get("xlabel", "$t$"), color=INK)
    ax.set_ylabel("Relative $L_2$ error", color=INK)
    if "ylim" in spec:
        ax.set_ylim(*spec["ylim"])
    fig.tight_layout(pad=0.4)
    _place_legend(fig, ax, spec)
    os.makedirs(os.path.join(OUT, "pdf"), exist_ok=True)
    png = os.path.join(OUT, spec["file"])
    fig.savefig(png, facecolor="white")
    fig.savefig(os.path.join(OUT, "pdf", spec["file"].replace(".png", ".pdf")), facecolor="white")
    plt.close(fig)
    return png


def main():
    data = json.load(open(os.path.join(OUT, "runs.json")))
    for spec in data["figures"]:
        print(chart(spec))


if __name__ == "__main__":
    main()
