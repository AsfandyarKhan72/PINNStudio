#!/usr/bin/env python3
"""
Figures and tables for benchmarks/training_methods.md, drawn from
benchmarks/figures/study.json (one record per run: example, setup, and the
relative L2 error of the main output at each reference time).

Every setup has one colour, marker and line style in every figure: the
colour is the weighting method (none, RBA, SA-PINN, causal, causal +
MsFFN), a solid line trains on the whole time interval at once and a dashed
line in time windows, and the two references without an exact initial
condition (loss BC + loss IC, exact BC + loss IC) are gray.

Each figure is written as a PNG for the documentation and a vector PDF
(figures/pdf/) for papers and slides; captions live in the markdown. The
core tables in the markdown sit between <!-- core:ID --> markers and are
rewritten from the same data. Rebuild with
    python3 benchmarks/make_figures.py
"""
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
MD = os.path.join(HERE, "training_methods.md")

INK, INK2, GRID, FRAME = "#1f1f1e", "#4a4a46", "#e8e7e2", "#8c8b86"
GRAY, GRAY2 = "#7d7d78", "#a9a8a2"

# The study's setups, in table order: (name, panel, colour, marker, line style)
SETUPS = [
    ("Loss BC + loss IC", "whole", GRAY, "o", "-"),
    ("Exact BC + loss IC", "whole", GRAY2, "s", "-"),
    ("Exact BC + exact IC", "whole", "#6250d6", "D", "-"),
    ("Exact BC + exact IC + RBA", "whole", "#2a78d6", "o", "-"),
    ("Exact BC + exact IC + SA-PINN", "whole", "#eb6834", "s", "-"),
    ("Exact BC + exact IC + causal", "whole", "#1baf7a", "^", "-"),
    ("Exact BC + exact IC + causal + MsFFN", "whole", "#008300", "v", "-"),
    ("Time windows: exact BC + loss IC", "windows", "#6250d6", "D", (0, (5, 2.5))),
    ("Time windows: exact BC + loss IC + RBA", "windows", "#2a78d6", "o", (0, (5, 2.5))),
    ("Time windows: exact BC + loss IC + SA-PINN", "windows", "#eb6834", "s", (0, (5, 2.5))),
]
# setups of the problems without periodicity or an initial-condition formula
PLAIN_SETUPS = [
    ("Plain", "whole", GRAY, "o", "-"),
    ("Plain + RBA", "whole", "#2a78d6", "o", "-"),
    ("Plain + SA-PINN", "whole", "#eb6834", "s", "-"),
    ("Plain + causal", "whole", "#1baf7a", "^", "-"),
    ("Plain + MsFFN", "whole", "#e87ba4", "P", "-"),
    ("Plain + causal + MsFFN", "whole", "#008300", "v", "-"),
]
STYLE = {s[0]: s[1:] for s in SETUPS + PLAIN_SETUPS}
ORDER = [s[0] for s in SETUPS] + [s[0] for s in PLAIN_SETUPS]

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
    "legend.fontsize": 7.5,
    "savefig.dpi": 220,
    "pdf.fonttype": 42,          # editable text in the PDF
})


def _pct(v):
    return f"{v:g}%"


def _legend(ax, loc, ncol):
    leg = ax.legend(loc=loc, ncol=ncol, frameon=True, fancybox=False, framealpha=0.95,
                    edgecolor="#cfcec8", borderpad=0.5, handlelength=2.4, labelspacing=0.35,
                    columnspacing=1.2, borderaxespad=0.6)
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


def _place_legend(fig, ax):
    """Legend inside the axes, never on top of the data: the first free
    corner, else two columns at the top with the y range raised to make room."""
    for loc in ("upper right", "lower right", "upper left", "lower left", "center right"):
        leg = _legend(ax, loc, 1)
        if not _overlaps(fig, ax, leg):
            return
        leg.remove()
    lo, hi = ax.get_ylim()
    for _ in range(14):
        leg = _legend(ax, "upper center", 2)
        if not _overlaps(fig, ax, leg):
            return
        leg.remove()
        hi *= 1.6
        ax.set_ylim(lo, hi)
    _legend(ax, "upper center", 2)


def _series(r):
    pts = [p for p in r["l2"] if p[0] > 0]       # t = 0 is not drawn (log scale, exact IC)
    return [p[0] for p in pts], [p[1] * 100 for p in pts]


def _draw(ax, recs):
    for r in recs:
        color, marker, ls = STYLE[r["setup"]][1:]
        ts, ls_ = _series(r)
        if not ts:
            continue
        tpl = r.get("template")
        ref = color in (GRAY, GRAY2)          # the two references: open markers
        ax.plot(ts, ls_, color=color, ls=ls, lw=2.4 if tpl else 1.4, marker=marker, ms=6 if tpl else 4.5,
                mfc="white" if ref else color, mec="white" if (tpl and not ref) else color, mew=0.9 if ref else 0.8,
                label=r.get("label", r["setup"]) + (" (template)" if tpl else ""), zorder=4 if tpl else (2 if ref else 3))


def _style_axes(ax, xlabel, ylabel=True):
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(FRAME)
    ax.tick_params(colors=INK2, which="both")
    ax.set_xlabel(xlabel, color=INK)
    if ylabel:
        ax.set_ylabel("Relative $L_2$ error", color=INK)


def _log_ticks(axes, recs):
    vals = [p[1] * 100 for r in recs for p in r["l2"] if p[0] > 0]
    lo, hi = min(vals), max(vals)
    lo_d, hi_d = 10 ** np.floor(np.log10(lo)), 10 ** np.ceil(np.log10(hi))
    decades = hi_d / lo_d > 300
    for ax in axes:
        ax.set_ylim(lo_d * 0.85 if decades else lo * 0.7, hi * 1.5)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,) if decades else (1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        # no labels far above the data (the y range may be raised for the legend)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _pct(v) if v <= hi * 2.5 else ""))


def example_figure(e):
    """All core setups in one plot: colour = weighting method, solid = the
    whole time interval at once, dashed = time windows. The legend sits
    under the plot, in two columns, in the order of the table."""
    core = sorted(e["core"], key=lambda r: ORDER.index(r["setup"]))
    plain = e.get("plain_setups")
    fig, ax = plt.subplots(figsize=(6.6, 4.0) if plain else (7.6, 5.6))
    _draw(ax, core)
    _style_axes(ax, e["xlabel"])
    _log_ticks([ax], core)
    if plain:
        fig.tight_layout(pad=0.4)
        _place_legend(fig, ax)
    else:
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False,
                        handlelength=3.0, columnspacing=1.6, labelspacing=0.5, fontsize=7.5)
        for t in leg.get_texts():
            t.set_color(INK)
        fig.tight_layout(pad=0.4)
    return _save(fig, "study_" + e["id"] + ".png")


def _save(fig, name):
    os.makedirs(os.path.join(OUT, "pdf"), exist_ok=True)
    png = os.path.join(OUT, name)
    fig.savefig(png, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(OUT, "pdf", name.replace(".png", ".pdf")), facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return png


def _at(r, t):
    for tt, v in r["l2"]:
        if abs(tt - t) < 1e-9:
            return v
    return None


def fmt(v):
    """A relative error as a percentage, two significant digits."""
    if v is None:
        return "–"
    p = v * 100
    if p >= 10:
        return f"{p:.0f}%"
    digits = max(1 - int(np.floor(np.log10(p))), 0)
    return f"{p:.{digits}f}%"


def core_table(e):
    times = e["table_times"]
    tl = "t'" if "t'" in e["xlabel"] else "t"
    head = "| Setup | " + " | ".join(f"${tl} = {t:g}$" for t in times) + " |"
    lines = [head, "|---|" + "---|" * len(times)]
    recs = {r["setup"]: r for r in e["core"]}
    setups = [s for s in ORDER if s in recs] if e.get("plain_setups") else [s[0] for s in SETUPS]
    best = min((r for r in e["core"]), key=lambda r: _at(r, times[-1]) or 9e9)
    for s in setups:
        r = recs.get(s)
        name = (r.get("label", s) if r else s) + (" (template)" if r and r.get("template") else "")
        if r is None:
            lines.append(f"| {name} | " + " | ".join("–" for _ in times) + " |")
            continue
        cells = [fmt(_at(r, t)) for t in times]
        if r is best:
            name, cells = f"**{name}**", [f"**{c}**" for c in cells]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


OVERVIEW_COLS = [s[0] for s in SETUPS]
OVERVIEW_SHORT = ["Loss BC\nloss IC", "Exact BC\nloss IC", "Exact BC\nexact IC", "Exact BC\nexact IC\n+ RBA",
                  "Exact BC\nexact IC\n+ SA-PINN", "Exact BC\nexact IC\n+ causal", "Exact BC\nexact IC\n+ causal\n+ MsFFN",
                  "Windows:\nexact BC\nloss IC", "Windows:\nexact BC\nloss IC\n+ RBA", "Windows:\nexact BC\nloss IC\n+ SA-PINN"]


def overview(examples):
    """Error at the last time, every phase-field example x every core setup."""
    rows = [e for e in examples if not e.get("plain_setups")]
    M = np.full((len(rows), len(OVERVIEW_COLS)), np.nan)
    for i, e in enumerate(rows):
        t = e["table_times"][-1]
        for r in e["core"]:
            v = _at(r, t)
            if v is not None:
                M[i, OVERVIEW_COLS.index(r["setup"])] = v * 100
    fig, ax = plt.subplots(figsize=(10.4, 0.48 * len(rows) + 1.5))
    L = np.log10(np.clip(M, 1e-3, 300))
    cmap = matplotlib.colormaps["Blues_r"].copy()
    cmap.set_bad("#f3f2ee")
    im = ax.imshow(np.ma.masked_invalid(L), cmap=cmap, vmin=-2.2, vmax=2.4, aspect="auto")
    for i in range(M.shape[0]):
        best = np.nanmin(M[i])
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", color=INK2, fontsize=7.5)
                continue
            txt = fmt(v / 100)
            dark = L[i, j] < 0.2
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if dark else INK, fontweight="bold" if v == best else "normal")
    ax.set_xticks(range(len(OVERVIEW_COLS)), OVERVIEW_SHORT, fontsize=7.5, color=INK)
    ax.set_yticks(range(len(rows)), [f"{e['title']}, ${'t' + chr(39) if chr(39) in e['xlabel'] else 't'} = "
                                     f"{e['table_times'][-1]:g}$" for e in rows], fontsize=7.5, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, M.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.axvline(6.5, color=INK2, lw=0.8)
    fig.tight_layout(pad=0.4)
    return _save(fig, "study_overview.png")


def update_markdown(examples):
    """Rewrite the generated tables between <!-- core:ID --> and <!-- /core --> markers."""
    if not os.path.exists(MD):
        return
    s = open(MD).read()
    for e in examples:
        s = re.sub(rf"(<!-- core:{e['id']} -->\n)(?:(?!<!-- core:).)*?(<!-- /core -->)",
                   lambda m: m.group(1) + core_table(e) + "\n" + m.group(2), s, flags=re.S)
    open(MD, "w").write(s)


def main():
    data = json.load(open(os.path.join(OUT, "study.json")))
    for e in data["examples"]:
        print(example_figure(e))
    print(overview(data["examples"]))
    update_markdown(data["examples"])


if __name__ == "__main__":
    main()
