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
    axs[0].set_ylabel("$L_2$ error")
    h = [Line2D([], [], color=INK, ls=(0, (3, 2)), lw=1.6), Line2D([], [], color=MUTED, ls="-", lw=1.6),
         Line2D([], [], color=MUTED, ls=(0, (1, 1.2)), lw=1.6)]
    fig.legend(h, ["Plain PINN (loss BC + loss IC)", "Best setup, whole time interval", "Best setup, time windows"],
               loc="upper center", ncol=3, frameon=False, fontsize=7.3, bbox_to_anchor=(0.53, 1.0))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.78, bottom=0.17, wspace=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"paper_methods_b.{ext}"), dpi=300)
    plt.close(fig)


# Figure 7 (chosen design): the same five setups in every panel
SETUPS = [("Loss BC + loss IC", "Plain PINN", INK, (0, (4, 2)), "o"),
          ("Exact BC + exact IC", "Exact BC + IC", "#8a8f98", "-", "s"),
          ("Exact BC + exact IC + RBA", "Exact BC + IC + RBA", C["RBA"], "-", "^"),
          ("Exact BC + exact IC + SA-PINN", "Exact BC + IC + SA-PINN", C["SA-PINN"], "-", "D"),
          ("Time windows: exact BC + loss IC", "Time windows", C["causal"], (0, (6, 1.5, 1.5, 1.5)), "v")]
PANELS = [("ac1d", "(a)"), ("ch1d", "(b)"), ("ac2d_mg", "(c)"), ("ch2d_mg", "(d)")]

# the confirmation reruns (benchmarks/plans/fig7_confirm.json): within each problem every setup has the plain
# PINN's network, points, learning rate, seed and steps; time windows share those steps out over the windows
RERUN = os.path.join(ROOT, "benchmarks", "figures", "fig7_v2.json")          # the common protocol (plans/fig7_v2.json)
RERUN_OLD = os.path.join(ROOT, "benchmarks", "figures", "fig7_rerun.json")   # matched per problem (plans/fig7_confirm.json)
PREFIX = {RERUN: "f7v2", RERUN_OLD: "f7"}
RERUN_ID = {"ac1d": "ac1d", "ch1d": "ch1d", "ac2d_mg": "ac2dmg", "ch2d_mg": "ch2dmg"}
SETUPS6 = SETUPS + [("bcpinn", "Time windows + backward (bc-PINN)", C["causal + MsFFN"], (0, (1.2, 1.2)), "P")]
RERUN_KEY = dict(zip([s[0] for s in SETUPS6], ("plain", "exact", "rba", "sa", "windows", "bcpinn")))
# Supplementary: the time windows at full and at equal compute, against the best whole-interval setup
SETUPS_EQ = [("Exact BC + exact IC + SA-PINN", "Exact BC + IC + SA-PINN (whole interval)", C["SA-PINN"], "-", "D"),
             ("Time windows: exact BC + loss IC", "Time windows, 4 × the steps", C["causal"], (0, (6, 1.5, 1.5, 1.5)), "v"),
             ("windowsEq", "Time windows, equal steps", C["causal"], (0, (1.5, 1.5)), "o"),
             ("bcpinn", "bc-PINN windows, 4 × the steps", C["causal + MsFFN"], (0, (6, 1.5, 1.5, 1.5)), "P"),
             ("bcpinnEq", "bc-PINN windows, equal steps", C["causal + MsFFN"], (0, (1.5, 1.5)), "s")]
RERUN_KEY.update(windowsEq="windowsEq", bcpinnEq="bcpinnEq")


def rerun_examples(path=None, setups=None):
    """{eid: {"core": [{"setup", "l2"}]}} from fig7_rerun.json (the main output, not mu); seed median when there are
    several seeds (runs named <name>_s<seed>)."""
    import statistics
    path = path or RERUN
    setups = setups or SETUPS6
    if not os.path.exists(path):
        return None
    r = json.load(open(path, encoding="utf-8"))
    pre = PREFIX[path]
    out = {}
    for eid, pid in RERUN_ID.items():
        core = []
        for st, *_ in setups:
            runs = [v for k, v in r.items() if k == f"{pre}_{pid}_{RERUN_KEY[st]}" or k.startswith(f"{pre}_{pid}_{RERUN_KEY[st]}_s")]
            runs = [v for v in runs if v.get("rc") == 0]
            if not runs:
                continue
            series = [next(c for g, c in v["l2"].items() if g != "mu") for v in runs]
            ts = [t for t, _ in series[0]]
            core.append(dict(setup=st, n=len(series),
                             l2=[[t, statistics.median(c[i][1] for c in series)] for i, t in enumerate(ts)]))
        out[eid] = dict(core=core)
    return out


def fig_c(ex, setups=SETUPS, name="paper_methods_c"):
    """Figure 7: 2 x 2, Allen-Cahn 1D | Cahn-Hilliard 1D (top), Allen-Cahn 2D | Cahn-Hilliard 2D (bottom); the
    same five setups in every panel (time windows were not run for 2D Cahn-Hilliard); each panel its own
    y range; t from 0 (the exact initial condition has zero error at t = 0: drawn on the bottom edge)."""
    import math
    style()
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 4.9))
    for ax, (eid, letter) in zip(axs.ravel(), PANELS):
        e = ex[eid]
        curves = [(next((c for c in e["core"] if c["setup"] == st), None), col, ls, mk) for st, _, col, ls, mk in setups]
        vals = [p[1] for c, *_ in curves if c for p in c["l2"] if p[0] > 0]
        lo = 10 ** math.floor(math.log10(min(vals)) - 0.6)
        hi = max(vals) * 2.2
        for c, col, ls, mk in curves:
            if c is None:
                continue
            pts = [(p[0], max(p[1], lo)) for p in c["l2"]]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, color=col, lw=1.5, marker=mk, ms=3.6,
                    mec="white", mew=0.5, zorder=3, clip_on=False)
            if c["l2"] and c["l2"][0][0] == 0 and c["l2"][0][1] < lo:   # exact IC: zero error at t = 0, on the lower edge
                ax.plot([0], [lo], ls="", marker=mk, ms=4.4, mfc="white", mec=col, mew=1.0, zorder=4, clip_on=False)
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
        ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom")
        ax.set_xlabel("$t$", labelpad=1)
        ax.set_ylabel("$L_2$ error", labelpad=2)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=7.8)
    h = [Line2D([], [], color=col, ls=ls, lw=1.5, marker=mk, ms=3.6, mec="white", mew=0.5) for _, _, col, ls, mk in setups]
    nc = 5 if len(setups) <= 5 and max(len(s[1]) for s in setups) < 26 else 3
    fig.legend(h, [lab for _, lab, _, _, _ in setups], loc="upper center", ncol=nc, frameon=False, fontsize=7.4,
               handlelength=2.4, handletextpad=0.35, columnspacing=0.9, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.885 if nc == 5 else 0.86, bottom=0.08, wspace=0.3, hspace=0.42)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def main():
    d = json.load(open(DATA, encoding="utf-8"))
    ex = {e["id"]: e for e in d["examples"]}
    fig_a(ex)
    fig_b(ex)
    rr = rerun_examples()
    if rr:
        fig_c(rr, SETUPS6)                                 # Figure 7: the common protocol
        fig_c(rerun_examples(path=RERUN, setups=SETUPS_EQ), SETUPS_EQ, name="paper_methods_c_windows")   # Suppl.
        old = rerun_examples(path=RERUN_OLD)
        if old:
            fig_c(old, SETUPS6, name="paper_methods_c_matched")   # earlier: settings matched per problem
        fig_c(ex, name="paper_methods_c_study")           # earlier: the original study
    else:
        fig_c(ex)
    print("wrote", os.path.join(OUT, "paper_methods_a.png"), "and _b")


if __name__ == "__main__":
    main()
