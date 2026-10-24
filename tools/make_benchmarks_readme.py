#!/usr/bin/env python3
"""Write benchmarks/README.md: every example of PINNStudio, forward and inverse
together, with its reference result and PINNStudio's best.

    python tools/make_benchmarks_readme.py RESULTS.jsonl ... --primary primary.json \
        [--papers papers.jsonl] [--spacetime spacetime.json]

RESULTS are the tuning study's results.jsonl files (tools/tuning_study.py);
--papers its runs of the papers' methods (variants plain / paper_*,
tools/paper_methods.py); --spacetime each run's error over all reference
times. A problem not yet run shows the README's result
(benchmarks/tuning/readme_results.json) and its README figure.

For a DeepXDE example the page compares DeepXDE's own settings with
PINNStudio's best; for a template from a paper, a base PINN, the paper's
reported error, the paper's method run in PINNStudio and PINNStudio's best.
"""
import argparse
import json
import os
import re
import statistics
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))

import make_tuning_page as mtp  # noqa: E402
import tuning_study as ts  # noqa: E402
from pinnstudio.core import example_catalog as cat  # noqa: E402

PAGE = os.path.join(_REPO, "benchmarks", "README.md")
FIG = os.path.join(_REPO, "benchmarks", "figures", "benchmarks_summary.png")
README = os.path.join(_REPO, "README.md")

GROUPS = ["ODEs", "1D, steady", "1D, time-dependent", "2D, steady", "2D, time-dependent", "3D"]

# What the papers report (relative L2 over space and time), with where.
PAPER_REPORTED = {
    "1D Allen-Cahn (whole interval)": [("Wight & Zhao (2021), Table 3.1", 2.33e-2, "paper_wz"),
                                       ("Mattey & Ghosh (2022), Table 2", 1.68e-2, "paper_mg")],
    "1D Allen-Cahn": [("Wight & Zhao (2021), Table 3.1", 2.33e-2, "paper_wz"),
                      ("Mattey & Ghosh (2022), Table 2", 1.68e-2, "paper_mg")],
    "1D Cahn-Hilliard": [("Wight & Zhao (2021), Sec. 3.2", 9.51e-3, "paper_wz")],
    "1D Cahn-Hilliard (Mattey & Ghosh)": [("Mattey & Ghosh (2022), Sec. 5.2", 1.86e-2, "paper_mg")],
    "2D Allen-Cahn (Mattey & Ghosh)": [("Mattey & Ghosh (2022), Sec. 6.1.1", 2.5e-2, "paper_mg")],
    "2D Allen-Cahn (Wight & Zhao)": [("Wight & Zhao (2021), Fig. 3.8", None, "paper_wz")],
    "2D Cahn-Hilliard (Wight & Zhao)": [("Wight & Zhao (2021), Fig. 3.13", None, "paper_wz")],
    "2D Cahn-Hilliard (Mattey & Ghosh)": [("Mattey & Ghosh (2022), Sec. 6.2.1", None, "paper_mg")],
    "3D Allen-Cahn (Wight & Zhao)": [("Wight & Zhao (2021), Fig. 3.9", None, "paper_wz")],
    "1D Burgers": [("Raissi et al. (2019), Sec. 2.1", 6.7e-4, "paper_raissi")],
    "1D Schrödinger": [("Raissi et al. (2019), Sec. 2.2", 1.97e-3, "paper_raissi")],
    "2D Burgers (Mathias)": [("Mathias et al. (2022)", None, "paper_mathias")],
}
PAPER_NAME = {"paper_wz": "Wight & Zhao's method", "paper_mg": "Mattey & Ghosh's bc-PINN",
              "paper_raissi": "Raissi et al.'s settings", "paper_mathias": "Mathias et al.'s settings"}


def sci(x):
    if x is None:
        return "–"
    m, p = f"{x:.1e}".split("e")
    return f"{m} × 10<sup>{int(p)}</sup>"


def _par_txt(params):
    """[(symbol, found, true)] -> "λ = 9.983 (10)"; mathtext symbols to plain text."""
    out = []
    for sym, found, true in params or []:
        s = re.sub(r"\$|\\", "", str(sym)).replace("varepsilon", "ε").replace("lambda", "λ").replace("alpha", "α") \
            .replace("gamma", "γ").replace("omega", "ω").replace("kappa", "κ").replace("nu", "ν").replace("mu", "μ") \
            .replace("^2", "²").replace("_1", "₁").replace("_2", "₂").replace("_3", "₃").replace("_0", "₀") \
            .replace("_e", "ₑ").replace("_f", "_f")
        out.append(f"{s} = {found:.6g} ({true:.6g})")
    return "; ".join(out)


def _perr(params):
    e = [abs(f - t) / abs(t) for _, f, t in params or [] if t]
    return max(e) if e else None


def _readme_images(name):
    """(forward images, inverse images) of the README section titled `name`: [(alt, path)]."""
    try:
        text = open(README, encoding="utf-8").read()
    except OSError:
        return [], []
    m = re.search(r"<summary><strong>" + re.escape(name) + r"</strong></summary>(.*?)</details>", text, re.S)
    if not m:
        return [], []
    imgs = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", m.group(1))
    inv = [i for i in imgs if re.search(r"nverse|param", i[0] + i[1])]
    fwd = [i for i in imgs if i not in inv]
    return fwd, inv


def _anchor(name):
    try:
        text = open(README, encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(r'<details id="([^"]+)">\s*<summary><strong>' + re.escape(name) + r"</strong>", text)
    return m.group(1) if m else None


def _figs_md(s, which):
    """The tuning study's figures of entry s: which = "own" or "tuned"."""
    if s is None:
        return []
    slug = ts._slug(s["name"]) + ("_inv" if s["inverse"] else "")
    folder = os.path.join(mtp.FIG_ROOT, slug)
    if not os.path.isdir(folder):
        return []
    names = sorted(f for f in os.listdir(folder) if f.startswith(which + "_") and f.endswith(".png"))
    surf = [n for n in names if "surface_comparison" in n]
    line = [n for n in names if "line_comparison" in n]
    keep = (surf or line) + [n for n in names if n.endswith("param_plot.png")]
    return [os.path.relpath(os.path.join(folder, n), os.path.dirname(PAGE)).replace(os.sep, "/") for n in keep]


def _img(path, alt):
    return f"![{alt}]({path.replace(' ', '%20')})"


SHORT = {"base": None, "iters": "3× iterations", "net": "Larger network (2× width, one more layer)",
         "points": "3× training points", "lbfgs": "L-BFGS phase added", "all": "All of the above together",
         "dataw": "10× weight on the measured data", "lbfgs3": "3× L-BFGS iterations",
         "times": "Measured data at every reference time", "setup": "Best whole-interval setup",
         "invall": "10× data weight, 3× L-BFGS and the larger network", "windows2": "2× time windows"}
ORDER = ["base", "iters", "net", "points", "lbfgs", "windows2", "dataw", "lbfgs3", "times", "setup", "all", "invall"]


def _t(q):
    return f"{q['seconds']:.0f}" if q and q.get("seconds") is not None else "–"


def _pct(x):
    return "–" if x is None else (f"{100 * x:.2g}%" if x < 0.1 else f"{100 * x:.0f}%")


def _sweep_rows(s, ref_name, inverse, prefix=""):
    """[(label, L2, params, seconds)] for every variant the study ran of entry s."""
    if not s:
        return []
    out = []
    for v in sorted(s["variants"], key=lambda v: ORDER.index(v) if v in ORDER else 99):
        rs = s["variants"][v]
        l2 = mtp._med(rs, mtp._maxl2)
        t = mtp._med(rs, lambda r: r["seconds"])
        if v == "base":
            lbl = ref_name if not prefix else "Best setup of the training methods study"
        else:
            lbl = prefix + ((SHORT.get(v) or v)[0].lower() + (SHORT.get(v) or v)[1:] if prefix else (SHORT.get(v) or v))
        if len(rs) > 1:
            lbl += f" ({len(rs)} seeds, median)"
        out.append((lbl, l2, mtp._found(dict(s, chosen=v)) if inverse else None,
                    f"{t:.0f}" if t is not None else None))
    return out


def _best_label(r, inverse):
    s = r.get("i_s") if inverse else r.get("f_best_s")
    if not s:
        return "no tuning needed"
    kept = r.get("f_keep") if not inverse else s.get("chosen") == "base" and s.get("tuned") != s.get("base")
    v = s["chosen"]
    if not inverse and r.get("f_keep"):
        return "the starting settings (the run with the lowest training loss was less accurate)"
    if v == "base":
        if len(s["variants"]) == 1:
            return "the example's own settings (tuning runs queued)"
        return ("the best setup as it is" if (not inverse and r.get("f_best_is_setup"))
                else "the example's own settings (no change did better)")
    del kept
    return (SHORT.get(v) or v) if v != "all" else "all of the changes together"



SAME = 1.25    # runs within 25% of the lowest error count as equally accurate; the fastest of them is the best


def _pick(s, inverse):
    """PINNStudio's best of entry s: the lowest error (for an inverse problem the lowest parameter error,
    else L2), and among the runs within SAME of it the fastest. A copy of s with chosen/tuned set."""
    if not s:
        return s
    stats = {}
    for v, rs in s["variants"].items():
        l2 = mtp._med(rs, mtp._maxl2)
        pe = _perr(mtp._found(dict(s, chosen=v))) if inverse else None
        err = pe if pe is not None else l2
        if err is not None:
            stats[v] = (err, mtp._med(rs, lambda r: r["seconds"]) or 0.0, l2)
    if not stats:
        return s
    low = min(e for e, _, _ in stats.values())
    ok = [v for v, (e, _, _) in stats.items() if e <= SAME * low]
    v = min(ok, key=lambda v: stats[v][1])
    return dict(s, chosen=v, tuned=stats[v][2])



def build(summ, papers, spacetime, readme):
    """One record per catalogue example (catalogue order)."""
    by = {(x["name"], x["inverse"]): x for x in summ}
    prow = {}
    for r in papers:
        if r["rc"] == 0 and r.get("final_loss") is not None:
            prow.setdefault((r["example"], r["variant"]), []).append(r)
    out = []
    for e in cat.ENTRIES:
        if e.hidden:
            continue
        deepxde = e.source == cat.DEEPXDE
        key = e.key if deepxde else ts.TEMPLATE_PREFIX + e.key
        inv_example = e.key.startswith("Inverse:")
        rd = readme.get(e.key, {})
        plain = by.get((key, False))
        best = by.get((key + " | best setup", False))
        inv = by.get((key, True)) or (plain if inv_example else None)
        if inv_example:
            plain = None
        rec = dict(entry=e, deepxde=deepxde, key=key, group=mtp._group_of(e), rd=rd,
                   paper=PAPER_REPORTED.get(e.key) if not deepxde else None)
        # forward: reference (DeepXDE's / the template's own settings) and best
        f_ref = plain["base"] if plain else (rd.get("f") if not inv_example else None)
        f_ref_src = "study" if plain else ("README" if f_ref is not None else None)
        f_best_s = _pick(best or plain, False)    # a template's best setup, when it was run
        f_best = f_best_s["tuned"] if f_best_s else f_ref   # not tuned: the reference result
        f_keep = False
        rec.update(f_ref=f_ref, f_ref_src=f_ref_src, f_best=f_best, f_best_s=f_best_s, f_plain_s=plain,
                   f_best_variant=(f_best_s["chosen"] if f_best_s else None),
                   f_best_is_setup=bool(best), f_keep=f_keep)
        # inverse
        if inv:
            base_par = mtp._found(dict(inv, chosen="base"))
            ip = _pick(inv, True)
            rec.update(i_ref=inv["base"], i_ref_par=base_par, i_best=ip["tuned"], i_best_par=mtp._found(ip),
                       i_s=ip, i_src="study", i_variant=ip["chosen"])
        elif rd.get("i") is not None or rd.get("params") or (inv_example and rd.get("f") is not None):
            par = [tuple(p) for p in rd.get("params", [])]
            iv = rd.get("i") if rd.get("i") is not None else (rd.get("f") if inv_example else None)
            rec.update(i_ref=iv, i_ref_par=par, i_best=iv, i_best_par=par, i_s=None, i_src="README",
                       i_variant=None)
        else:
            rec.update(i_ref=None, i_best=None, i_s=None, i_src=None)
        # the papers' methods run in PINNStudio
        pr = {}
        for v in ("plain", "paper_wz", "paper_mg", "paper_raissi", "paper_mathias", "base"):
            rs = prow.get((key, v))
            if rs:
                l2 = statistics.median(mtp._maxl2(r) for r in rs if mtp._maxl2(r) is not None)
                st = [spacetime.get(f"{ts._slug(key)}/{v}_s{r['seed']}") for r in rs]
                st = [x for x in st if x is not None]
                pr[v] = dict(l2=l2, st=statistics.median(st) if st else None, seconds=statistics.median(
                    r["seconds"] for r in rs))
        rec["paper_runs"] = pr
        out.append(rec)
    return out


def figure(recs, path):
    """The paper's Fig. 4 layout: grey open = DeepXDE's settings / the paper's method in PINNStudio
    (the template's own settings where that has not run yet); filled = PINNStudio's best."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = []
    for r in recs:
        e = r["entry"]
        ref_f = r["f_ref"]
        if r["paper"] and r["paper_runs"]:
            pm = [r["paper_runs"][v]["l2"] for _, _, v in r["paper"] if v in r["paper_runs"]]
            ref_f = min(pm) if pm else ref_f
        par_best = r.get("i_best_par") or []
        rows.append(dict(label=e.name, group=r["group"], f_ref=ref_f, f_best=r["f_best"],
                         i_ref=r.get("i_ref"), i_best=r.get("i_best"),
                         p_ref=_perr(r.get("i_ref_par")), params=par_best))
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "pdf.fonttype": 42})
    ypos, heads, groups, yy = {}, [], [], 0.0
    for i, r in enumerate(rows):
        if not groups or groups[-1][0] != r["group"]:
            if groups:
                yy += 0.35
            heads.append((r["group"], yy)); yy += 0.9
            groups.append([r["group"], i, i])
        groups[-1][2] = i
        ypos[i] = yy; yy += 1.0
    H = yy
    Y = {i: H - v for i, v in ypos.items()}
    h_in = mtp.ROW_HEIGHT * H + 1.0
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(mtp.FIG_WIDTH, h_in), sharey=True,
                                 gridspec_kw={"width_ratios": [3.0, 1.45], "wspace": 0.08})
    for gi, (g, a, b) in enumerate(groups):
        for axx in (ax, bx):
            axx.axhspan(Y[b] - 0.5, Y[a] + 1.4, color=mtp.C_BAND if gi % 2 == 0 else "white", zorder=0, lw=0)
    for g, v in heads:
        ax.text(-0.012, H - v - 0.05, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=7.6, fontweight="bold", color=mtp.C_TEXT)

    def pair(axx, ref, best, y, marker, color, ms):
        if ref and best and abs(ref - best) / max(ref, best) > 0.05:
            axx.plot([min(ref, best), max(ref, best)], [y, y], color=color, lw=0.9, alpha=0.35, zorder=1)
            axx.scatter(ref, y, s=ms * 0.75, marker=marker, facecolor="white", edgecolor=mtp.C_OWN, lw=0.9, zorder=2)
        if best:
            axx.scatter(best, y, s=ms, marker=marker, color=color, edgecolor="white", lw=0.5, zorder=3)

    for i, r in enumerate(rows):
        y = Y[i]
        pair(ax, r["f_ref"], r["f_best"], y, "o", mtp.C_FORWARD, 27)
        pair(ax, r["i_ref"], r["i_best"], y, "D", mtp.C_INVERSE, 22)
        es = [(s, max(abs(f - t) / abs(t) * 100, 1.2e-4)) for s, f, t in r["params"] if t]
        if es:
            right = max(e for _, e in es)
            if r["p_ref"] is not None:
                o = max(100 * r["p_ref"], 1.2e-4)
                if abs(o - right) / max(o, right) > 0.05:
                    bx.plot([min(o, right), max(o, right)], [y, y], color=mtp.C_INVERSE, lw=0.9, alpha=0.35, zorder=1)
                    bx.scatter(o, y, s=15, marker="D", facecolor="white", edgecolor=mtp.C_OWN, lw=0.9, zorder=2)
                    right = max(right, o)
            for _, e_ in es:
                bx.scatter(e_, y, s=20, marker="D", color=mtp.C_INVERSE, edgecolor="white", lw=0.5, zorder=3)
            bx.text(right * 1.9, y, ", ".join(s for s, _ in sorted(es, key=lambda q: q[1])), va="center",
                    ha="left", fontsize=6.6, color="#52606d")
    ax.set_yticks([Y[i] for i in range(len(rows))], [r["label"] for r in rows], fontsize=7.2, color=mtp.C_TEXT)
    ax.set_ylim(Y[len(rows) - 1] - 0.7, H + 0.15)
    ax.set_xscale("log"); ax.set_xlim(*mtp.XLIM_ERROR)
    ax.set_xticks([10.0 ** k for k in range(-6, 1)])
    bx.set_xscale("log"); bx.set_xlim(*mtp.XLIM_PARAM)
    bx.set_xticks([1e-3, 1e-1, 10], ["0.001", "0.1", "10"])
    ax.set_xlabel("Relative $L_2$ error of the main output at the last reference time", fontsize=8)
    bx.set_xlabel("Parameter error (%)", fontsize=8)
    for axx in (ax, bx):
        axx.grid(axis="x", color=mtp.C_GRID, lw=0.6, zorder=0)
        axx.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            axx.spines[sp].set_visible(False)
        axx.tick_params(axis="y", length=0)
        axx.tick_params(axis="x", labelsize=7.2)
    h = [Line2D([], [], ls="", marker="o", color=mtp.C_FORWARD, markersize=5, label="Forward, PINNStudio best"),
         Line2D([], [], ls="", marker="D", color=mtp.C_INVERSE, markersize=4.5, label="Inverse, PINNStudio best"),
         Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=mtp.C_OWN, markersize=4.5,
                label="DeepXDE's settings / the paper's method")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=7.2, bbox_to_anchor=(0.62, 1.0))
    fig.subplots_adjust(left=0.33, right=0.985, top=1 - 0.32 / h_in, bottom=0.45 / h_in)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _section(r):
    e = r["entry"]
    L = []
    a = L.append
    paper = next((t[-1] for t in cat._TEMPLATES if t[0] == e.key), "")
    src = "DeepXDE example" if r["deepxde"] else ("PINNStudio template, after " + paper if paper
                                                   else "PINNStudio template")
    a(f'<details id="bench-{re.sub(r"[^a-z0-9]+", "-", e.name.lower()).strip("-")}">')
    a(f"<summary><strong>{e.name}</strong> · {src}</summary>\n")
    links = []
    anc = _anchor(e.name)
    if anc:
        links.append(f"[problem and settings in the README](../README.md#{anc})")
    for s in (r["f_plain_s"], r["f_best_s"], r["i_s"]):
        if s:
            t = mtp._title(s["name"], s["inverse"])
            links.append(f"[{'inverse' if s['inverse'] else 'forward'} runs in the tuning study]"
                         f"(tuning_study.md#{mtp._slug(t)})")
    if links:
        a(" · ".join(dict.fromkeys(links)) + "\n")

    has_f = r["f_ref"] is not None or r["f_best"] is not None
    has_i = r.get("i_best") is not None or r.get("i_best_par")
    ref_name = "DeepXDE's settings" if r["deepxde"] else "The template's own settings"
    paper_rows = bool(r["paper"]) and not r["deepxde"]
    pr = r["paper_runs"]

    if has_f:
        a("**Forward problem**\n")
        if paper_rows:
            a("| Training | L₂, last time | L₂, all times | Time (s) |\n|---|---|---|---|")
            p = pr.get("plain")
            a(f"| Base PINN | {sci(p and p['l2'])} | {sci(p and p['st'])} | {_t(p)} |")
            for ref, val, v in r["paper"]:
                a(f"| {PAPER_NAME[v]}, as reported ({ref}) | – | {sci(val) if val else 'figures only'} | – |")
                q = pr.get(v)
                a(f"| {PAPER_NAME[v]}, run in PINNStudio | {sci(q and q['l2'])} | {sci(q and q['st'])} | {_t(q)} |")
            pre = "| "
            cols = " | – "
        else:
            a("| Training | L₂ | Time (s) |\n|---|---|---|")
            cols = ""
        rows = _sweep_rows(r["f_plain_s"], ref_name, False)
        if not r["f_plain_s"] and not paper_rows:
            rows = [(ref_name + (" (README)" if r["f_ref_src"] == "README" else ""), r["f_ref"], None, None)]
        if r["f_best_s"] and r["f_best_is_setup"]:
            rows += _sweep_rows(r["f_best_s"], "Best setup of the training methods study", False, prefix="Best setup + ")
        for lbl, l2, _, t in rows:
            a(f"| {lbl} | {sci(l2)}{cols} | {t if t is not None else '–'} |")
        lab = _best_label(r, False)
        if paper_rows and lab == "no tuning needed":
            lab = "the template's own settings"
        a(f"| **PINNStudio best** (Settings: Tuned): {lab} | **{sci(r['f_best'])}**{cols} | |")
        a("")
        if paper_rows and not pr:
            a("*The base PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*\n")
    if has_i:
        a("**Inverse problem**\n")
        a("| Training | L₂ | Parameters found (true) | Worst parameter error | Time (s) |\n|---|---|---|---|---|")
        if r.get("i_s"):
            for lbl, l2, par, t in _sweep_rows(r["i_s"], ref_name, True):
                a(f"| {lbl} | {sci(l2)} | {_par_txt(par) or '–'} | {_pct(_perr(par))} | {t if t is not None else '–'} |")
        else:
            a(f"| {ref_name} (README) | {sci(r.get('i_ref'))} | {_par_txt(r.get('i_ref_par')) or '–'} | "
              f"{_pct(_perr(r.get('i_ref_par')))} | – |")
        a(f"| **PINNStudio best** (Settings: Tuned): {_best_label(r, True)} | **{sci(r.get('i_best'))}** | "
          f"{_par_txt(r.get('i_best_par')) or '–'} | **{_pct(_perr(r.get('i_best_par')))}** | |")
        a("")
        if r["deepxde"] and r.get("i_s") and not e.key.startswith("Inverse:") and e.problem != cat.INVERSE:
            a("*DeepXDE's example is forward only; this inverse problem is PINNStudio's inverse mode of it.*\n")

    # figures: reference first, then PINNStudio's best; forward, then inverse
    ref_lbl = "DeepXDE's settings" if r["deepxde"] else "the template's own settings"
    fwd_r, inv_r = _readme_images(e.name)
    blocks = []
    if has_f:
        own = _figs_md(r["f_plain_s"], "own") if r["f_plain_s"] else []
        best_s = r["f_best_s"]
        tuned = ((_figs_md(best_s, "own") if r["f_keep"] else _figs_md(best_s, "tuned"))
                 or (_figs_md(best_s, "own") if r["f_best_is_setup"] else [])) if best_s else []
        if best_s and (best_s["chosen"] == "base" or r["f_keep"]) and not r["f_best_is_setup"]:
            tuned = []
        if own:
            blocks.append((f"Forward, {ref_lbl}", own))
        elif fwd_r:
            blocks.append(("Forward, the README's result", [p for _, p in fwd_r[:1]]))
        if tuned:
            blocks.append(("Forward, PINNStudio best", tuned))
    if has_i:
        own = _figs_md(r.get("i_s"), "own") if r.get("i_s") else []
        tuned = _figs_md(r.get("i_s"), "tuned") if r.get("i_s") and r["i_s"]["chosen"] != "base" else []
        if own:
            blocks.append((f"Inverse, {ref_lbl}", own))
        elif inv_r:
            blocks.append(("Inverse, the README's result", [p for _, p in inv_r[:1]]))
        if tuned:
            blocks.append(("Inverse, PINNStudio best", tuned))
    for title, paths in blocks:
        a(f"*{title}:*\n")
        for pth in paths:
            pth = pth if pth.startswith(("figures/", "../", "http")) else "../" + pth
            a(_img(pth, f"{e.name}: {title}") + "\n")
    a("</details>\n")
    return "\n".join(L)


def page(recs):
    L = []
    a = L.append
    n_t = sum(1 for r in recs if r["f_best_s"] or r.get("i_s"))
    a("# PINNStudio benchmarks\n")
    a("*Every example of PINNStudio, forward and inverse, with the result of its reference settings and "
      "PINNStudio's best. Updated as the runs finish.*\n")
    a("**How to read an example.** Each example opens to one table for the forward problem and one for the "
      "inverse problem. The first row is the reference: **DeepXDE's own settings** for a DeepXDE example, the "
      "**template's own settings** for a PINNStudio template. The next rows are the tuning study's changes, each "
      "tried on its own and then all together: 3× the iterations, a larger network (twice as wide, one more "
      "layer), 3× the training points, an L-BFGS phase after Adam (for inverse problems also 10× the weight of "
      "the measured data and 3× the L-BFGS iterations). The last row, **PINNStudio best**, is the most accurate "
      "run, and when several runs are about as accurate (within 25% of the lowest error) the fastest of them, so "
      "a change that costs time without helping is not chosen. For an inverse problem, accuracy is the error of "
      "the learned parameters. These are the **Tuned** settings. Then come the figures: the "
      "reference first, then PINNStudio's best, forward and then inverse.\n")
    a("For a **template taken from a paper**, the forward table also lists a **base PINN** — the baseline "
      "each paper compares against, with the paper's network and points for it, soft initial and boundary "
      "conditions, the whole time interval, equal loss weights and no adaptive technique, the error the paper reports and the "
      "paper's own method run in PINNStudio ([how they are set up](paper_comparison.md)); for a template run "
      "from the [training methods study](training_methods.md)'s best setup, the changes are tried on that "
      "setup. Where a DeepXDE example has no inverse problem of its own, the inverse table is PINNStudio's "
      "inverse mode of it.\n")
    a("**The error** is the relative L₂ error of the main output (u, φ, h, …) at the last reference time, against "
      "the example's reference data; papers report one error over all times, so their rows give that too. Where "
      "a row ran with 3 seeds (1, 2, 2026) it gives the median, otherwise seed 2026. Times are training seconds "
      "on one RTX 4090.\n")
    a(f"The sweep was run on the {n_t} examples with the largest errors. The others were already accurate with "
      "their own settings: their result is the README's, marked **no tuning needed**.\n")
    a("![Every example: the reference settings and PINNStudio's best, forward and inverse, and the error of each "
      "learned parameter](figures/benchmarks_summary.png)\n")
    a("*Open grey markers: DeepXDE's settings, or the paper's method run in PINNStudio; filled: PINNStudio's best "
      "(one filled marker where the two are the same, or where no tuning was needed). Right: the error of each "
      "learned parameter, labelled with its symbol.*\n")
    for g in GROUPS:
        rs = [r for r in recs if r["group"] == g]
        if not rs:
            continue
        a(f"## {g[0].upper() + g[1:]}\n")
        for r in rs:
            a(_section(r))
    a("---\n")
    a("Written by [`tools/make_benchmarks_readme.py`](../tools/make_benchmarks_readme.py) from the runs of "
      "[`tools/tuning_study.py`](../tools/tuning_study.py).\n")
    return "\n".join(L)


TUNED_JSON = os.path.join(_REPO, "pinnstudio", "resources", "tuned_settings.json")


def write_tuned(recs):
    """Examples -> Settings: Tuned: each example's PINNStudio best as the changes to its own settings
    (pinnstudio/core/example_settings.py)."""
    out = {}
    for r in recs:
        e, ent = r["entry"], {}
        fs = r.get("f_best_s")
        if fs and (fs["chosen"] != "base" or r.get("f_best_is_setup")):
            ent["forward"] = dict(switches=ts.BEST_FORWARD.get(e.key) if r.get("f_best_is_setup") else None,
                                  variant=fs["chosen"],
                                  error=f"L2 {r['f_best']:.1e}" + (f" vs {r['f_ref']:.1e} with its own settings"
                                                                  if r.get("f_ref") else ""))
        iv = r.get("i_s")
        if iv and iv["chosen"] != "base":
            v = iv["chosen"]
            pe = _perr(r.get("i_best_par"))
            ent["inverse"] = dict(switches=ts.BEST_WHOLE.get(e.key) if v == "setup" else None,
                                  variant="base" if v == "setup" else v,
                                  error=(f"parameter error {100 * pe:.2g}%" if pe is not None
                                         else f"L2 {r['i_best']:.1e}"))
        if ent:
            out[e.key] = ent
    with open(TUNED_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--primary", default="")
    ap.add_argument("--papers", default="", help="results.jsonl of the papers' methods (variants plain/paper_*)")
    ap.add_argument("--spacetime", default="", help="json: run folder (slug/variant_sSEED) -> error over all times")
    args = ap.parse_args()
    if args.primary:
        for slug, vs in json.load(open(args.primary, encoding="utf-8")).items():
            for variant, runs in vs.items():
                for r in runs:
                    mtp.PRIMARY[(slug, variant, str(r["seed"]))] = r["l2"]
    rows = mtp._load(args.results)
    summ = mtp.summarize(rows)
    papers = mtp._load([args.papers]) if args.papers else []
    spacetime = json.load(open(args.spacetime, encoding="utf-8")) if args.spacetime else {}
    try:
        readme = json.load(open(mtp.README_RESULTS, encoding="utf-8"))
    except OSError:
        readme = {}
    recs = build(summ, papers, spacetime, readme)
    write_tuned(recs)
    figure(recs, FIG)
    with open(PAGE, "w", encoding="utf-8") as f:
        f.write(page(recs))
    print(f"wrote {PAGE} ({len(recs)} examples)")


if __name__ == "__main__":
    main()
