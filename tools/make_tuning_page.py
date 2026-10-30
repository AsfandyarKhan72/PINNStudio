#!/usr/bin/env python3
"""Write benchmarks/tuning_study.md (and its figure and tuned problem files)
from the results.jsonl files tools/tuning_study.py makes.

    python tools/make_tuning_page.py runs/pilot/results.jsonl runs/batch2/results.jsonl ...

The same run (example, mode, variant, seed) found in several files counts
once. Write the notes about single examples in NOTES below; everything
else on the page comes from the results.
"""
import argparse
import copy
import dataclasses
import json
import os
import re
import statistics
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))

import tuning_study as ts  # noqa: E402
from pinnstudio.core import deepxde_examples as dx  # noqa: E402
from pinnstudio.core.example_catalog import display_name  # noqa: E402

PAGE = os.path.join(_REPO, "benchmarks", "tuning_study.md")
FIG_DIR = os.path.join(_REPO, "benchmarks", "figures")
PROBLEM_DIR = os.path.join(_REPO, "benchmarks", "tuning")

VARIANT_TEXT = {
    "base": "the example's own settings",
    "iters": "3× the iterations of every training phase",
    "net": "hidden layers twice as wide (at most 512) and one layer more",
    "points": "3× the domain, boundary and initial points",
    "lbfgs": "an L-BFGS phase (15,000 iterations) after Adam",
    "all": "all of the changes above together",
    "dataw": "10× the weight of the measured data (inverse problems)",
    "lbfgs3": "3× the L-BFGS iterations, or an L-BFGS phase of 45,000 added (inverse problems)",
    "times": "measured data at every reference time after t = 0, not only one (inverse problems)",
    "setup": "the best whole-interval setup of the forward problem from the training methods study (inverse problems)",
    "invall": "10× data weight, 3× L-BFGS and the wider, deeper network together (inverse problems)",
    "windows2": "twice as many time windows (time-adaptive training)",
    "rar": "residual-based adaptive refinement: 5 rounds of 500 points with the largest residual",
    "msffn": "a multi-scale Fourier-feature network (scales 1 and 10)",
    "fracres": "3× the resolution of the fractional operator (fractional problems)",
}

# One or two sentences per example, keyed by (example, inverse). Only what the
# results show; leave an example out when there is nothing to add.
NOTES = {
    ("Helmholtz sound-hard scattering (absorbing BC)", False):
        "No variant gets below about 7.7%, although the training loss falls about 800 times. The limit is the "
        "problem itself: the absorbing boundary condition only approximates the infinite domain the exact "
        "solution is written for, so a better-trained network converges to a slightly different answer.",
    ("Inverse: Brinkman-Forchheimer model", False):
        "The field improves, but the effective viscosity stays about 18% too high in every seed, while K comes "
        "out within 0.5%. With the measured velocity at only a few points, ν_e is weakly determined by the data.",
    ("Klein-Gordon equation", False):
        "The wider network alone is the best and steadiest change; with all the changes together the error is "
        "about three times higher.",
    ("Heat equation", False):
        "With one seed the wider network looked best (5.8e-3), but over three seeds it varies from 5.8e-3 to "
        "3.3e-2, while all the changes together stay between 1.0e-2 and 1.6e-2.",
    ("Wave propagation – space-time multi-scale Fourier features", False):
        "Adding L-BFGS reaches almost the lowest training loss but a worse error (0.16), one of the cases where "
        "the training loss and the error do not agree. With every change together the error is about 150 times "
        "lower, and it barely changes between seeds while the example's own settings give 0.14 to 0.76.",
    ("Helmholtz 2D with a hole", False):
        "An L-BFGS phase after Adam is all it needs: about 15 times lower error for 7 seconds more training.",
    ("Diffusion equation with resampling", False):
        "Again L-BFGS alone gives most of the gain, at almost no extra time.",
    ("Second-order ODE", True):
        "PINNStudio's own inverse mode: a and b of u'' - a u' + b u = 5t from u at 30 times. With the example's "
        "settings b comes out 7% off (9.65 against 9); the wider network finds a = 10.0003 and b = 8.993.",
    ("ODE system", True):
        "PINNStudio's own inverse mode: w in y2' = -w y1 from y1 and y2 at 40 times. Already within 0.03% with "
        "the example's settings.",
    ("Lotka-Volterra equation", True):
        "PINNStudio's own inverse mode: the prey growth rate a and the predator death rate d from both "
        "populations. Both are found to 0.001% with the example's settings; no change does better.",
    ("Euler beam", True):
        "PINNStudio's own inverse mode: the bending stiffness EI from the deflection at 20 points. The example's "
        "settings leave EI 2.3% off; every change together brings it to 0.08%.",
    ("Diffusion-reaction equation", True):
        "PINNStudio's own inverse mode: the diffusivity D from u at four times.",
    ("Allen-Cahn equation", True):
        "PINNStudio's own inverse mode: the diffusion coefficient d and the reaction coefficient c from u at "
        "four times, both found to within 0.06% after tuning.",
    ("Kovasznay flow", True):
        "PINNStudio's own inverse mode: the Reynolds number from u and v at 500 points, within 0.01% with "
        "every setting tried.",
    ("Beltrami flow", True):
        "PINNStudio's own inverse mode: the viscosity from u, v and w at two times. The error is that of u; "
        "the pressure is only known up to a constant here, so its own error is large and not a fault. Every "
        "change together ran out of memory on the 24 GB GPU.",
    ("Helmholtz 2D (Dirichlet)", True):
        "PINNStudio's own inverse mode: the wavenumber k0 = 4π from u on a 31 × 31 grid. The source term is the "
        "known load and keeps the true wavenumber, so k0 is unknown only in the k0² u term (an earlier version "
        "also had k0 inside the source and failed). k0 is found to 0.01% with the example's settings and to "
        "0.004% after tuning.",
    ("Linear elastostatic plate", True):
        "PINNStudio's own inverse mode: the Lamé parameters λ and μ from the displacements on a 19 × 19 grid. "
        "The body force is the known load and keeps the true values, so λ and μ are unknown only in the "
        "stress-strain law (with them in the body force too, an earlier version ended 24% off). Both are found "
        "to within 0.03%.",
    ("Schrödinger equation", True):
        "Only |h| is measured, at one time, and with the example's own network c depends on the seed (0.4964 to "
        "0.515 over twelve seeds). The wider network recovers c to within 0.01% and the field to 0.1%.",
    ("Template: 2D Burgers (Mathias)", False):
        "None of the changes helps: every variant ends at about 1.8% (worst output and time). With every change "
        "together the run did not finish within the 4-hour limit.",
    ("Template: 2D Allen-Cahn (Wight & Zhao)", False):
        "The template's own settings are the best here; more iterations or a bigger network make the error 4 to 7 "
        "times larger. Its settings come from the training methods study, where time windows with exact "
        "periodicity and RBA weights were found to be needed. With every change together the run did not finish "
        "within the 4-hour limit.",
    ("Template: 2D Cahn-Hilliard (Wight & Zhao)", False):
        "The worst error is at t = 0, where the initial field is close to zero and the relative error is not a "
        "good measure; none of the changes improves it.",
    ("Template: 1D Cahn-Hilliard (Mattey & Ghosh) | best setup", False):
        "The study's best setup (time windows, exact periodicity, SA-PINN weights) gives 0.39% at t = 1, as in "
        "the training methods study. Every change makes it worse, and the wider network is a warning: its "
        "training loss is the lowest of all (1.5e-12) while the solution has collapsed to a nearly flat field "
        "(99% error). A tiny loss does not mean a right answer for this problem.",
    ("Template: 2D Allen-Cahn (Mattey & Ghosh) | best setup", False):
        "Switching from the template's time windows to the study's best setup (exact periodicity and IC, SA-PINN "
        "weights, whole interval) lowers the error from 6.7e-3 to 3.6e-4 and trains in 4.5 minutes. More "
        "iterations give about the same; the variant with the lowest training loss (the wider network) is a "
        "little worse.",
    ("Template: 2D Allen-Cahn (Wight & Zhao) | best setup", False):
        "The study's best setup (time windows with RBA weights, soft IC, exact periodicity) gives 1.6e-3 at "
        "t = 10, against 7.9e-3 for the template as it loads. The gain is only at the last time: at t = 2.5 and "
        "t = 5 both runs are the same (9.3e-3 and 2.1e-2), and at t = 10 neither run shows the small drop left "
        "in the reference, so the lower number mostly reflects a flatter field. More iterations and two time "
        "windows give the same errors at every time; the wider network fails (120% at t = 10). The run with "
        "every change together did not finish within the 4-hour limit.",
    ("Template: 1D Burgers | best setup", False):
        "Causal training on the whole interval, then every change together, gives 5.2e-4 at t = 1, against 1.3e-3 "
        "for the template as it loads.",
    ("Poisson over an L-shaped domain", False):
        "Training loss picks the wider network, but all the changes together give the lower error; the "
        "re-entrant corner keeps the error near 1–2% either way.",
}


def _load(paths):
    rows, seen = [], set()
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            key = (r["example"], r["inverse"], r["variant"], r["seed"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


PRIMARY = {}   # (run folder, variant, seed) -> L2 of the main output at the last reference time


def _maxl2(r):
    """The error a run is judged by: the main output (u, c, phi, ...) at the
    last reference time, read from the run's log (see --primary), else the
    worst error the run printed."""
    key = (ts._slug(r["example"]) + ("_inv" if r["inverse"] else ""), r["variant"], str(r["seed"]))
    if key in PRIMARY:
        return PRIMARY[key]
    return max(r["l2"]) if r.get("l2") else None


def _perr(r):
    errs = [abs(v - r["true"][k]) / abs(r["true"][k]) for k, v in r.get("params", {}).items()
            if r.get("true", {}).get(k) not in (None, 0)]
    return max(errs) if errs else None


def _med(rs, f):
    v = [f(r) for r in rs if f(r) is not None]
    return statistics.median(v) if v else None


def _rng(rs, f):
    v = [f(r) for r in rs if f(r) is not None]
    return (min(v), max(v)) if len(v) > 1 else None


def e(x):
    if x is None:
        return "–"
    m, p = f"{x:.1e}".split("e")
    return f"{m} × 10<sup>{int(p)}</sup>"


def pct(x):
    return "–" if x is None else (f"{100 * x:.2g}%" if x < 0.1 else f"{100 * x:.0f}%")


def _title(name, inverse):
    if " | " in name:
        base, extra = name.split(" | ", 1)
        return _title(base, inverse) + f" ({extra})"
    if name.startswith(ts.TEMPLATE_PREFIX):
        t = display_name(name[len(ts.TEMPLATE_PREFIX):]) + " (PINNStudio template)"
    else:
        t = display_name(name)
    return t + (", inverse mode" if inverse else "")


def _slug(s):
    return "tune-" + re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _readme_anchor(title_name):
    """The README's <details id=...> for this example, when there is one."""
    try:
        text = open(os.path.join(_REPO, "README.md"), encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(r'<details id="([^"]+)">\s*<summary><strong>' + re.escape(title_name) + r"</strong>", text)
    return m.group(1) if m else None


def _phases_text(r):
    return " → ".join(f"{p['optimizer'].replace('lbfgs', 'L-BFGS').replace('adam', 'Adam')} {int(p['iterations']):,}"
                      for p in r["phases"])


def summarize(rows):
    groups = {}
    for r in rows:
        if r["rc"] == 0 and r["final_loss"] is not None:
            groups.setdefault((r["example"], r["inverse"]), {}).setdefault(r["variant"], []).append(r)
    out = []
    for (name, inv), vs in groups.items():
        if "base" not in vs:
            continue
        # choose by the training loss of the first seed every variant has, so
        # a variant run with more seeds is not favoured
        first = {v: min(rs, key=lambda r: r["seed"] != 2026) for v, rs in vs.items()}
        chosen = min(first, key=lambda v: first[v]["final_loss"])
        out.append(dict(name=name, inverse=inv, variants=vs, chosen=chosen,
                        base=_med(vs["base"], _maxl2), base_rng=_rng(vs["base"], _maxl2),
                        tuned=_med(vs[chosen], _maxl2), tuned_rng=_rng(vs[chosen], _maxl2),
                        base_p=_med(vs["base"], _perr), tuned_p=_med(vs[chosen], _perr),
                        t_base=_med(vs["base"], lambda r: r["seconds"]),
                        t_tuned=_med(vs[chosen], lambda r: r["seconds"]),
                        n_base=len(vs["base"]), n_tuned=len(vs[chosen])))
    out.sort(key=lambda s: -(s["base"] / s["tuned"]) if s["base"] and s["tuned"] else 0)
    return out


# Summary figure, in the layout of the paper's Fig. 4: one row per problem,
# grouped by kind; forward (circles) and inverse (diamonds) on one error axis,
# the worst parameter error on the right. Open grey markers: the example's own
# settings; filled: tuned. Formatting options:
FIG_WIDTH = 7.3
ROW_HEIGHT = 0.19
C_FORWARD, C_INVERSE, C_OWN = "#1f5fa8", "#d1701c", "#aab2bd"
C_TEXT, C_GRID, C_BAND = "#1f2933", "#e4e7eb", "#f3f5f7"
XLIM_ERROR = (1e-6, 3)
XLIM_PARAM = (1e-4, 300)
MISLED_MARK = " ‡"
TUNED_MARK = " †"


# How the study's parameter names are written in the figure (mathtext).
PARAM_SYMBOLS = {"lambda_": r"$\lambda$", "diff_coeff": r"$\lambda\varepsilon^2$", "eps_sq": r"$\varepsilon^2$",
                 "c1_sq": r"$c_1^2$", "c2": r"$c_2$", "source_term": r"$f$", "alpha": r"$\alpha$",
                 "nu": r"$\nu$", "v_e": r"$\nu_e$", "K": r"$K$", "C": r"$C$", "a": r"$a$", "b": r"$b$",
                 "c": r"$c$", "d": r"$d$", "D": r"$D$", "w": r"$\omega$", "EI": r"$EI$", "Re": r"$Re$",
                 "gamma": r"$\gamma$", "k0": r"$k_0$", "lmbd": r"$\lambda$", "mu": r"$\mu$"}
README_RESULTS = os.path.join(_REPO, "benchmarks", "tuning", "readme_results.json")


def _group_of(entry):
    if entry.dim == "ODE":
        return "ODEs"
    if entry.dim == "3D":
        return "3D"
    return f"{entry.dim}, {'steady' if entry.regime == 'Steady' else 'time-dependent'}"


class Found(list):
    """[(symbol, median found value, true value)] of a variant's runs, with the errors taken seed by seed:
    .err = median over the seeds of each seed's worst relative parameter error (the inverse benchmark metric);
    .per_param = {symbol: median over the seeds of that parameter's relative error};
    .per_seed = [(seed, {symbol: (found, relative error)}, worst relative error)]."""
    err = None
    per_param = None
    per_seed = None


def _found(x):
    """Median of each learned parameter over the chosen variant's runs: [(symbol, found, true)] (a Found, which
    also carries the seed-by-seed errors)."""
    rs = x["variants"][x["chosen"]]
    out = Found()
    keys = [(k, tv) for k, tv in (rs[0].get("true") or {}).items() if tv not in (None, 0)]
    for k, tv in keys:
        vals = [r["params"][k] for r in rs if k in r.get("params", {})]
        if vals:
            out.append((PARAM_SYMBOLS.get(k, k), statistics.median(vals), tv))
    seeds = []
    for r in rs:
        e = {PARAM_SYMBOLS.get(k, k): (r["params"][k], abs(r["params"][k] - tv) / abs(tv))
             for k, tv in keys if k in r.get("params", {})}
        if e:
            seeds.append((r.get("seed"), e, max(v[1] for v in e.values())))
    if seeds:
        out.per_seed = seeds
        out.err = statistics.median(w for _, _, w in seeds)
        out.per_param = {sym: statistics.median(e[sym][1] for _, e, _ in seeds if sym in e) for sym, _, _ in out}
    return out


def _fig_rows(summ):
    """Every example of the catalogue, in its order: the tuning study's result
    where the problem is in it (forward: the template's best setup when run,
    against the template as it loads), else the README's (readme_results.json).
    A DeepXDE "Inverse: ..." example is an inverse row though it trains in
    Forward mode."""
    from pinnstudio.core import example_catalog as cat
    by = {(x["name"], x["inverse"]): x for x in summ}
    try:
        readme = json.load(open(README_RESULTS, encoding="utf-8"))
    except OSError:
        readme = {}

    def entry(x):
        if not x:
            return None
        errs = [_med(rs, _maxl2) for rs in x["variants"].values()]
        best = min([v for v in errs if v], default=None)
        return dict(own=x["base"], tuned=x["tuned"], tuned_run=x["chosen"] != "base", study=True,
                    misled=bool(best and x["tuned"] and x["tuned"] > 2 * best),
                    own_p=x["base_p"], params=_found(x))

    rows = []
    for e in cat.ENTRIES:
        if e.hidden:
            continue
        key = e.key if e.source == cat.DEEPXDE else ts.TEMPLATE_PREFIX + e.key
        rd = readme.get(e.key, {})
        is_inv_example = e.key.startswith("Inverse:")
        plain, best = entry(by.get((key, False))), entry(by.get((key + " | best setup", False)))
        inv = entry(by.get((key, True))) or (plain if is_inv_example else None)
        if is_inv_example:
            plain = None
        fwd, only_best = plain, False
        if best:
            only_best = plain is None
            fwd = dict(best, own=plain["own"] if plain else (rd.get("f") or best["own"]), tuned_run=True)
            if plain is None and rd.get("f"):
                only_best = False          # the README's template result is the own-settings marker
        if fwd is None and rd.get("f") is not None and not is_inv_example:
            fwd = dict(own=None, tuned=rd["f"], tuned_run=False, misled=False)
        if inv is None and (rd.get("i") is not None or rd.get("params")):
            inv = dict(own=None, tuned=rd.get("i"), tuned_run=False, misled=False, own_p=None,
                       params=[tuple(p) for p in rd.get("params", [])])
        if is_inv_example and inv is None and rd.get("f") is not None:
            inv = dict(own=None, tuned=rd["f"], tuned_run=False, misled=False, own_p=None,
                       params=[tuple(p) for p in rd.get("params", [])])
        rows.append(dict(name=e.key, label=e.name + (", best setup" if only_best else ""),
                         group=_group_of(e), f=fwd, i=inv))
    return rows


def figure(summ, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    R = _fig_rows(summ)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "pdf.fonttype": 42,
                         "xtick.major.size": 3, "xtick.minor.size": 1.6})
    ypos, heads, groups, yy = {}, [], [], 0.0
    for i, r in enumerate(R):
        if not groups or groups[-1][0] != r["group"]:
            if groups:
                yy += 0.35
            heads.append((r["group"], yy)); yy += 0.9
            groups.append([r["group"], i, i])
        groups[-1][2] = i
        ypos[i] = yy; yy += 1.0
    H = yy
    Y = {i: H - v for i, v in ypos.items()}
    h_in = ROW_HEIGHT * H + 1.0
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(FIG_WIDTH, h_in), sharey=True,
                                 gridspec_kw={"width_ratios": [3.0, 1.45], "wspace": 0.08})
    for gi, (g, a, b) in enumerate(groups):
        for axx in (ax, bx):
            axx.axhspan(Y[b] - 0.5, Y[a] + 1.4, color=C_BAND if gi % 2 == 0 else "white", zorder=0, lw=0)
    for g, v in heads:
        ax.text(-0.012, H - v - 0.05, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=7.6, fontweight="bold", color=C_TEXT)

    def pair(axx, own, tuned, y, marker, color, ms):
        if own and tuned:
            axx.plot([min(own, tuned), max(own, tuned)], [y, y], color=color, lw=0.9, alpha=0.35, zorder=1)
        if own:
            axx.scatter(own, y, s=ms * 0.75, marker=marker, facecolor="white", edgecolor=C_OWN, lw=0.9, zorder=2)
        if tuned:
            axx.scatter(tuned, y, s=ms, marker=marker, color=color, edgecolor="white", lw=0.5, zorder=3)

    for i, r in enumerate(R):
        y = Y[i]
        f, v = r["f"], r["i"]
        pts = [p for p in ((f or {}).get("tuned"), (v or {}).get("tuned")) if p]
        if len(pts) == 2:
            ax.plot(pts, [y, y], color=C_GRID, lw=1.0, zorder=1)
        if f:
            pair(ax, f["own"] if f.get("tuned_run") else None, f["tuned"], y, "o", C_FORWARD, 27)
        if v:
            pair(ax, v["own"] if v.get("tuned_run") else None, v["tuned"], y, "D", C_INVERSE, 22)
            es = [(sym, max(abs(fo - tr) / abs(tr) * 100, 1.2e-4)) for sym, fo, tr in v.get("params") or []]
            if es:
                right = max(e_ for _, e_ in es)
                if v.get("tuned_run") and v.get("own_p") is not None:
                    o = max(100 * v["own_p"], 1.2e-4)
                    bx.plot([min(o, max(e for _, e in es)), max(o, max(e for _, e in es))], [y, y],
                            color=C_INVERSE, lw=0.9, alpha=0.35, zorder=1)
                    bx.scatter(o, y, s=15, marker="D", facecolor="white", edgecolor=C_OWN, lw=0.9, zorder=2)
                    right = max(right, o)
                for _, e_ in es:
                    bx.scatter(e_, y, s=20, marker="D", color=C_INVERSE, edgecolor="white", lw=0.5, zorder=3)
                txt = ", ".join(sym for sym, _ in sorted(es, key=lambda q: q[1]))
                bx.text(right * 1.9, y, txt, va="center", ha="left", fontsize=6.6,
                        color="#52606d")
    labels = [r["label"] + (TUNED_MARK if any(r[k] and r[k].get("study") for k in ("f", "i")) else "")
              + (MISLED_MARK if any(r[k] and r[k].get("misled") for k in ("f", "i")) else "") for r in R]
    ax.set_yticks([Y[i] for i in range(len(R))], labels, fontsize=7.2, color=C_TEXT)
    ax.set_ylim(Y[len(R) - 1] - 0.7, H + 0.15)
    ax.set_xscale("log"); ax.set_xlim(*XLIM_ERROR)
    ax.set_xticks([10.0 ** k for k in range(-6, 1)])
    bx.set_xscale("log"); bx.set_xlim(*XLIM_PARAM)
    bx.set_xticks([1e-3, 1e-1, 10], ["0.001", "0.1", "10"])
    ax.set_xlabel("Relative $L_2$ error of the main output at the last reference time", fontsize=8)
    bx.set_xlabel("Worst parameter error (%)", fontsize=8)
    for axx in (ax, bx):
        axx.grid(axis="x", color=C_GRID, lw=0.6, zorder=0)
        axx.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            axx.spines[sp].set_visible(False)
        axx.tick_params(axis="y", length=0)
        axx.tick_params(axis="x", labelsize=7.2)
    h = [Line2D([], [], ls="", marker="o", color=C_FORWARD, markersize=5, label="Forward"),
         Line2D([], [], ls="", marker="D", color=C_INVERSE, markersize=4.5, label="Inverse"),
         Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=C_OWN, markersize=4.5,
                label="Example's own settings, where a change helped")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=7.2, bbox_to_anchor=(0.62, 1.0))
    fig.text(0.30, 0.0, "† in this study (filled: the chosen run); rows without † show the README's result.  "
             "‡ the run with the lowest training loss (shown) is more than 2× less accurate than another run.",
             fontsize=6.4, color="#52606d", ha="left", va="bottom")
    fig.subplots_adjust(left=0.33, right=0.985, top=1 - 0.32 / h_in, bottom=0.55 / h_in)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def problem_file(s):
    """The chosen variant as a PINNStudio problem file, when it can be made
    portable (reference paths written as @REF/...). Returns its file name."""
    try:
        from pinnstudio.core.portable_paths import to_portable
    except ImportError:
        return None
    name, inv = s["name"], s["inverse"]
    plain, _, extra = name.partition(" | ")
    tkey = plain[len(ts.TEMPLATE_PREFIX):] if plain.startswith(ts.TEMPLATE_PREFIX) else None
    if tkey:
        base = ts.template_config(tkey, inv)
        if extra == "best setup":
            base = ts.apply_switches(base, ts.BEST_FORWARD[tkey])
    else:
        ex = {e.name: e for e in dx.EXAMPLES}[plain]
        base = dx.build_config(ex, ts.REF_ROOT, inverse=inv)
    cfg = ts.apply_variant(base, s["chosen"], tkey)
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)
    cfg.save_dir = ""
    cfg.sweep_save_dir = ""
    title = _title(name, inv)
    stem = re.sub(r'\s*[\\/:*?"<>|]+\s*', " - ", title) + " (tuned)"
    payload = {"pinnstudio_problem_format": 1, "problem_name": stem,
               "config": to_portable(dataclasses.asdict(cfg), ts.REF_ROOT)}
    os.makedirs(PROBLEM_DIR, exist_ok=True)
    fn = stem + ".pinn.json"
    with open(os.path.join(PROBLEM_DIR, fn), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return fn


FIG_ROOT = os.path.join(FIG_DIR, "tuning")


def _figures(s):
    """Figures of the example's own-settings run and the chosen run (seed 2026),
    from benchmarks/figures/tuning/<run folder>/own_*.png and tuned_*.png: the
    surface comparisons when there are any (else the line comparisons), and the
    parameter convergence of an inverse run."""
    slug = ts._slug(s["name"]) + ("_inv" if s["inverse"] else "")
    folder = os.path.join(FIG_ROOT, slug)
    if not os.path.isdir(folder):
        return []
    names = sorted({f[4:] if f.startswith("own_") else f[6:] for f in os.listdir(folder) if f.endswith(".png")})
    surf = [n for n in names if n.startswith("surface_comparison")]
    line = [n for n in names if n.startswith("line_comparison")]
    kinds = (surf or line) + [n for n in names if n == "param_plot.png"]
    out = []
    for n in kinds:
        what = ("Parameter convergence" if n == "param_plot.png" else
                ("Surface" if n.startswith("surface") else "Line") + " comparison")
        sub = re.sub(r"^(surface|line)_comparison_?|\.png$|^param_plot", "", n.replace(".png", ""))
        if sub:
            what += f", {sub}"
        pairs = [("own", "the example's own settings"), ("tuned", f"tuned ({s['chosen']})")]
        if s["chosen"] == "base":  # nothing beat the own settings, so show them once
            pairs = pairs[:1]
        for tag, label in pairs:
            f = os.path.join(folder, f"{tag}_{n}")
            if os.path.isfile(f):
                rel = os.path.relpath(f, os.path.dirname(PAGE)).replace(os.sep, "/")
                out.append(f"*{what}, {label}:*\n\n![{what}, {label}]({rel.replace(' ', '%20')})\n")
    return out


def page(summ, rows, with_files):
    n_seeds = max((s["n_tuned"] for s in summ), default=1)
    better = sum(1 for s in summ if s["base"] and s["tuned"] and s["base"] / s["tuned"] >= 1.5)
    ratios = [s["base"] / s["tuned"] for s in summ if s["base"] and s["tuned"]]
    L = []
    a = L.append
    a("# Tuning study\n")
    a("*PINNStudio, October 2026*\n")
    a("DeepXDE's examples are written to show how the library is used, so many train briefly, on a small network "
      "or with Adam only. This study asks how far ordinary tuning in PINNStudio takes them: each example is run "
      "with its own settings and with the same five changes, and the change that trains best is reported next to "
      "the example's own result. It complements the [training methods study](training_methods.md), which tests "
      "specialised methods (causal training, per-point weights, time windows) on the hardest problems.\n")
    a(f"{len(summ)} problems are included, chosen as the ones with the largest errors in the "
      f"[README results](../README.md). {better} of them improve by 1.5× or more; the median improvement is "
      f"{statistics.median(ratios):.0f}×." if ratios else "")
    a("")
    a("## How it is done\n")
    a("| Variant | What changes |\n|---|---|")
    for v, t in VARIANT_TEXT.items():
        a(f"| **{v}** | {t} |")
    a("")
    a("- **Choosing a variant.** For each problem the variant with the lowest final training loss is chosen. "
      "The reference solution is not looked at, so the choice is one a user could make without knowing the "
      "answer. The best error any variant reached is listed too, so it is visible when the training loss "
      "picks a less accurate variant. (The *Tuned* settings shipped with the examples and shown on the "
      "[benchmarks page](README.md) use a second rule: the variant with the lowest error against the reference, "
      "the fastest one where several are within 25% of it. The two rules agree for most problems.)")
    a("- **The error** is the relative $L_2$ error from PINNStudio's Error Analysis of the main output (u, c, "
      "phi, ...) at the last reference time, against the reference data the example ships with, the same "
      "measure as the paper's Fig. 4. (The worst over all outputs and times would often be an output such as a "
      "chemical potential at t = 0, where the reference is close to zero and a relative error says little.) "
      "For inverse problems the worst relative error of the learned parameters is given as well.")
    a(f"- **Seeds.** The example's own settings and the chosen variant are run with up to {n_seeds} random seeds "
      "(1, 2 and 2026); tables give the median and, in brackets, the range. The screening of all variants uses "
      "seed 2026.")
    a("- **Hardware.** One NVIDIA RTX 4090, PyTorch backend, PINNStudio 1.5. Times are wall-clock training time "
      "per run.")
    a("- **Reproducing a result.** Each problem's section links the tuned settings as a problem file in "
      "[`tuning/`](tuning/): open it in PINNStudio (*Open a saved problem…*) and press Solve. The runs were made "
      "with [`tools/tuning_study.py`](../tools/tuning_study.py) and this page is written from its results by "
      "[`tools/make_tuning_page.py`](../tools/make_tuning_page.py).\n" if with_files else
      "- **Reproducing a result.** The runs were made with [`tools/tuning_study.py`](../tools/tuning_study.py), "
      "and this page is written from its results by [`tools/make_tuning_page.py`](../tools/make_tuning_page.py).\n")
    a("## Summary\n")
    a("The figure has the layout of the paper's Fig. 4 and lists every example of PINNStudio, in the order of "
      "the README: one row per problem, the forward (circles) and inverse (diamonds) results on one axis, and "
      "each learned parameter's error on the right, labelled with its symbol. Rows marked † are in this study: "
      "the filled marker is the chosen run, and an open grey marker the example's own settings where a change "
      "helped. Rows without † show the README's result (kept in "
      "[`tuning/readme_results.json`](tuning/readme_results.json)) until they are run here. For a template "
      "run from the training methods study's best setup, the grey marker is the template as it loads.\n")
    a("![Error with the example's own settings and tuned in PINNStudio](figures/tuning_summary.png)\n")
    a("| Problem | Own settings: $L_2$ | Tuned: $L_2$ | Improvement | Chosen variant | Parameter error: own → tuned | "
      "Time (s): own → tuned |")
    a("|---|---|---|---|---|---|---|")
    for s in summ:
        title = _title(s["name"], s["inverse"])
        r = s["base"] / s["tuned"]
        imp = f"{r:,.0f}×" if r >= 1.5 else ("none" if r > 1 / 1.5 else f"worse ({1 / r:.0f}× higher)")
        par = f"{pct(s['base_p'])} → {pct(s['tuned_p'])}" if s["base_p"] is not None else "–"
        a(f"| [{title}](#{_slug(title)}) | {e(s['base'])} | {e(s['tuned'])} | {imp} | {s['chosen']} | {par} | "
          f"{s['t_base']:.0f} → {s['t_tuned']:.0f} |")
    a("")
    a("## Problems\n")
    a("Click a problem to open its results.\n")
    for s in summ:
        title = _title(s["name"], s["inverse"])
        a(f'<details id="{_slug(title)}">')
        a(f"<summary><strong>{title}</strong></summary>\n")
        anchor = _readme_anchor(display_name(s["name"]) if not s["name"].startswith(ts.TEMPLATE_PREFIX) else title)
        if anchor:
            a(f"Its own settings and results are in the README: [{display_name(s['name'])}](../README.md#{anchor}).\n")
        b = s["variants"]["base"][0]
        a(f"**Own settings:** network {b['layers']}, {_phases_text(b)}, "
          f"{b['points'][0]:,} domain / {b['points'][1]:,} boundary / {b['points'][2]:,} initial points.\n")
        a("| Variant | Final training loss | $L_2$ | Parameter error | Time (s) |")
        a("|---|---|---|---|---|")
        for v in ts.VARIANTS + ts.INVERSE_VARIANTS + ts.TA_VARIANTS:
            rs = [r for r in s["variants"].get(v, []) if r["seed"] == 2026] or s["variants"].get(v, [])[:1]
            if not rs:
                continue
            r = rs[0]
            mark = "**" if v == s["chosen"] else ""
            a(f"| {mark}{v}{mark} | {r['final_loss']:.2e} | {e(_maxl2(r))} | {pct(_perr(r))} | {r['seconds']:.0f} |")
        a("")
        c = s["variants"][s["chosen"]][0]
        a(f"**Chosen: {s['chosen']}** ({VARIANT_TEXT[s['chosen']]}): network {c['layers']}, {_phases_text(c)}, "
          f"{c['points'][0]:,} / {c['points'][1]:,} / {c['points'][2]:,} points.\n")

        def rg(r):
            return f" [{e(r[0])} – {e(r[1])}]" if r else ""
        a(f"Over seeds: own settings {e(s['base'])}{rg(s['base_rng'])} ({s['n_base']} seeds), "
          f"tuned {e(s['tuned'])}{rg(s['tuned_rng'])} ({s['n_tuned']} seeds).")
        if s["base_p"] is not None:
            vals = {}
            for r in s["variants"][s["chosen"]]:
                for k, v in r["params"].items():
                    vals.setdefault(k, []).append(v)
            txt = ", ".join(f"{k} = {statistics.median(v):.6g} (true {s['variants']['base'][0]['true'].get(k)})"
                            for k, v in vals.items())
            a(f" Learned parameters (tuned, median): {txt}.")
        a("")
        note = NOTES.get((s["name"], s["inverse"]))
        if note:
            a(note + "\n")
        figs = _figures(s)
        if figs:
            a("\n".join(figs) + "\n")
        if s.get("file"):
            a(f"Tuned settings: [`{s['file']}`](tuning/{s['file'].replace(' ', '%20')})\n")
        a("</details>\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--no-files", action="store_true", help="do not write the tuned problem files")
    ap.add_argument("--primary", default="", help="primary.json: each run's main-output error at the last time, "
                                                  "read from the run logs")
    args = ap.parse_args()
    if args.primary:
        for slug, vs in json.load(open(args.primary, encoding="utf-8")).items():
            for variant, runs in vs.items():
                for r in runs:
                    PRIMARY[(slug, variant, str(r["seed"]))] = r["l2"]
    rows = _load(args.results)
    summ = summarize(rows)
    # examples left out of PINNStudio's lists (e.g. DeepXDE's Allen-Cahn 1D) are left out here too
    from pinnstudio.core import example_catalog as _cat
    summ = [x for x in summ if not getattr(_cat.BY_KEY.get(x["name"].split(" | ")[0]), "hidden", False)]
    with_files = False
    if not args.no_files:
        for s in summ:
            s["file"] = problem_file(s)
        with_files = any(s.get("file") for s in summ)
    figure(summ, os.path.join(FIG_DIR, "tuning_summary.png"))
    with open(PAGE, "w", encoding="utf-8") as f:
        f.write(page(summ, rows, with_files))
    print(f"wrote {PAGE} ({len(summ)} problems)")


if __name__ == "__main__":
    main()
