"""Figure 7 candidates for the paper: the training-methods study (benchmarks/figures/study.json).

(a) one row per phase-field problem: the relative L2 error at the last reference time of every training setup
    the study ran, on a log axis. Colour = the method added (none, RBA, SA-PINN, causal, causal + MsFFN),
    shape = whole time interval (circle) or time windows (square); the plain PINN (loss BC + loss IC) is the
    open black circle and the best setup a star, with the factor between them.
(b) the relative L2 error against time for three problems: the plain PINN, the best setup on the whole
    interval and the best setup with time windows, which shows where the error builds up.

    python tools/make_methods_figure.py          # benchmarks/figures/paper_methods_{a,b}.{png,pdf}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "benchmarks", "figures", "study.json")
OUT = os.path.join(ROOT, "benchmarks", "figures")
WIDTH = 6.5

ROWS = [("ac1d", "Allen-Cahn 1D (W&Z)"), ("ac2d_wz", "Allen-Cahn 2D (W&Z), $t=10$"),
        ("ac2d_mg", "Allen-Cahn 2D (M&G)"), ("ac3d", "Allen-Cahn 3D (W&Z)"),
        ("ch1d", "Cahn-Hilliard 1D (W&Z)"), ("ch1d_mg", "Cahn-Hilliard 1D (M&G)"),
        ("ch2d_wz", "Cahn-Hilliard 2D (W&Z)"), ("ch2d_mg", "Cahn-Hilliard 2D (M&G)")]
# colour-blind-safe (Okabe-Ito); grey for "no method added"
C = {"none": "#6b6b6b", "RBA": "#0072B2", "SA-PINN": "#D55E00", "causal": "#009E73", "causal + MsFFN": "#CC79A7"}
INK, MUTED, GRID = "#1f2933", "#5f6b7a", "#e4e7eb"


def method_of(setup):
    s = setup.replace("Time windows: ", "")
    for m in ("causal + MsFFN", "SA-PINN", "RBA", "causal"):
        if "+ " + m in s:
            return m
    return "none"


def is_plain(setup):
    return setup.startswith("Loss BC + loss IC")


def last(c):
    return c["l2"][-1][1]


def pct(x):
    v = 100 * x
    return f"{v:.2g}%" if v < 10 else f"{v:.0f}%"


def style():
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.7, "pdf.fonttype": 42,
                         "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK})


def fig_a(ex):
    style()
    fig, ax = plt.subplots(figsize=(WIDTH, 4.6))
    for i, (eid, label) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        e = ex[eid]
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f4f5f7", zorder=0, lw=0)
        best = min(e["core"], key=last)
        plain = next((c for c in e["core"] if is_plain(c["setup"])), None)
        others = sorted((c for c in e["core"] if c is not best and c is not plain), key=last)
        dy, prev, k = {}, None, 0
        for c in others:                      # points that would sit on top of each other: spread up and down
            k = k + 1 if (prev is not None and last(c) / last(prev) < 1.12) else 0
            dy[id(c)] = (0, 0.17, -0.17, 0.34, -0.34)[k % 5]
            prev = c
        for c in others:
            m, win = method_of(c["setup"]), c["setup"].startswith("Time windows")
            ax.scatter(last(c), y + dy[id(c)], s=26, marker="s" if win else "o", color=C[m], edgecolor="white", lw=0.5, zorder=3,
                       alpha=0.95)
        if plain is not None:
            ax.scatter(last(plain), y, s=34, marker="o", facecolor="white", edgecolor=INK, lw=1.1, zorder=4)
        bm = method_of(best["setup"])
        ax.scatter(last(best), y, s=120, marker="*", color=C[bm], edgecolor=INK, lw=0.5, zorder=5)
        if plain is not None and last(plain) > 1.5 * last(best):
            ax.annotate("", xy=(last(best) * 1.25, y + 0.28), xytext=(last(plain), y + 0.28),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7), zorder=2)
            ax.text((last(plain) * last(best)) ** 0.5, y + 0.33, f"×{last(plain) / last(best):.0f}", ha="center",
                    va="bottom", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(ROWS)), [lab for _, lab in ROWS][::-1])
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 4)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_xlabel("Relative $L_2$ error at the last reference time")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    mk = lambda **k: Line2D([], [], ls="", **k)
    h = ([mk(marker="o", markersize=6, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.1),
          mk(marker="*", markersize=10, color=MUTED, markeredgecolor=INK, markeredgewidth=0.5)]
         + [mk(marker="o", markersize=6, color=C[m]) for m in C]
         + [mk(marker="o", markersize=6, markerfacecolor="none", markeredgecolor=MUTED),
            mk(marker="s", markersize=6, markerfacecolor="none", markeredgecolor=MUTED)])
    lab = (["Plain PINN (loss BC + loss IC)", "Best setup"]
           + ["Exact BC/IC only" if m == "none" else "+ " + m for m in C]
           + ["Whole time interval", "Time windows"])
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=7.3, handletextpad=0.3,
               columnspacing=1.4, bbox_to_anchor=(0.6, 1.0))
    fig.subplots_adjust(left=0.27, right=0.98, top=0.80, bottom=0.10)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"paper_methods_a.{ext}"), dpi=300)
    plt.close(fig)


def fig_b(ex):
    style()
    picks = [("ac1d", "(a) Allen-Cahn 1D (W&Z)"), ("ac2d_wz", "(b) Allen-Cahn 2D (W&Z)"),
             ("ch1d_mg", "(c) Cahn-Hilliard 1D (M&G)")]
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.75), sharey=True)
    short = lambda c: c["setup"].replace("Time windows: ", "").replace("Exact BC + exact IC", "exact BC/IC") \
        .replace("Exact BC + loss IC", "exact BC").replace("exact BC + loss IC", "exact BC")
    for ax, (eid, title) in zip(axs, picks):
        e = ex[eid]
        plain = next((c for c in e["core"] if is_plain(c["setup"])), None)
        whole = min((c for c in e["core"] if not c["setup"].startswith("Time windows")), key=last)
        wins = [c for c in e["core"] if c["setup"].startswith("Time windows")]
        win = min(wins, key=last) if wins else None
        for c, ls, lab_on in ((plain, (0, (3, 2)), False), (whole, "-", True), (win, (0, (1, 1.2)), True)):
            if c is None:
                continue
            col = INK if c is plain else C[method_of(c["setup"])]
            t = [p[0] for p in c["l2"] if p[0] > 0]
            v = [max(p[1], 1e-5) for p in c["l2"] if p[0] > 0]
            ax.plot(t, v, ls=ls, color=col, lw=1.6, marker="o", ms=2.8)
            if lab_on:
                ax.annotate(short(c), xy=(t[-1], v[-1]), xytext=(-2, 8 if c is whole else -13), textcoords="offset points",
                            ha="right", fontsize=6.3, color=col)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 5)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("$t$")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("Relative $L_2$ error")
    h = [Line2D([], [], color=INK, ls=(0, (3, 2)), lw=1.6), Line2D([], [], color=MUTED, ls="-", lw=1.6),
         Line2D([], [], color=MUTED, ls=(0, (1, 1.2)), lw=1.6)]
    fig.legend(h, ["Plain PINN (loss BC + loss IC)", "Best setup, whole time interval", "Best setup, time windows"],
               loc="upper center", ncol=3, frameon=False, fontsize=7.3, bbox_to_anchor=(0.53, 1.0))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.78, bottom=0.17, wspace=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"paper_methods_b.{ext}"), dpi=300)
    plt.close(fig)


def main():
    d = json.load(open(DATA, encoding="utf-8"))
    ex = {e["id"]: e for e in d["examples"]}
    fig_a(ex)
    fig_b(ex)
    print("wrote", os.path.join(OUT, "paper_methods_a.png"), "and _b")


if __name__ == "__main__":
    main()
