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


def _math_sym(s):
    """A raw parameter name (lambda_1, gamma_2) as mathtext; symbols already in $...$ stay."""
    s = str(s)
    if "$" in s:
        return s
    m = re.fullmatch(r"(lambda|gamma|alpha|mu|nu|omega|kappa|C|c)_?(\d)", s)
    if m:
        b = m.group(1)
        return f"${chr(92) + b if len(b) > 1 else b}_{m.group(2)}$"
    return s


def _perr(params):
    e = [abs(f - t) / abs(t) for _, f, t in params or [] if t]
    return max(e) if e else None


README_ALT = {}   # README figure file -> its caption


def _readme_images(name, inverse_only=False):
    """(forward images, inverse images) of the README section titled `name`, as paths: every output's
    comparison figure (surface or slice, else line) and the learned parameters' plot; solution animations
    only when there is nothing else. An image is the inverse problem's when its caption or file name says
    so (or the example is inverse only)."""
    try:
        text = open(README, encoding="utf-8").read()
    except OSError:
        return [], []
    m = re.search(r"<summary><strong>" + re.escape(name) + r"</strong></summary>(.*?)</details>", text, re.S)
    if not m:
        return [], []
    imgs = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", m.group(1))
    for a_, p_ in imgs:              # the README's own caption, for the figure's description
        README_ALT[os.path.basename(p_)] = a_
    if any("whole_domain" in os.path.basename(p_) for _, p_ in imgs):
        # the template's own settings are its whole-interval setup (the time-adaptive one was merged into it)
        imgs = [(a_, p_) for a_, p_ in imgs if "whole_domain" in os.path.basename(p_)
                or re.search(r"[Ii]nverse|param", a_ + os.path.basename(p_))]
    is_inv = lambda alt, path: inverse_only or bool(re.search(r"[Ii]nverse|param_|parameter convergence|inferred", alt + " " + os.path.basename(path)))

    def pick(sel):
        comp = [p for a, p in sel if re.search(r"(surface|slice|line)_comparison|view3d|3d_view|isosurface|radius", os.path.basename(p))]
        by_out = {}
        for p in comp:          # one per output: surface/slice/3D before the line plot
            k = re.sub(r"(surface|slice|line)_comparison|view3d|3d_view", "", os.path.basename(p))   # isosurface, radius: own
            rank = 0 if re.search(r"surface|slice|view3d|3d_view|isosurface|radius", os.path.basename(p)) else 1
            if k not in by_out or rank < by_out[k][0]:
                by_out[k] = (rank, p)
        keep = [p for _, p in by_out.values()]
        if WITH_LINES[0]:       # 1D: the line plots too, after the surfaces
            keep += [p for p in comp if "line_comparison" in os.path.basename(p) and p not in keep]
        par = [p for a, p in sel if re.search(r"param_|parameter convergence|inferred", a + " " + os.path.basename(p))]
        par = [p for p in par if p.endswith(".png")] or par[:1]
        out = keep + [p for p in par if p not in keep]
        return out or [p for _, p in sel[:1]]
    inv = [(a, p) for a, p in imgs if is_inv(a, p)]
    fwd = [(a, p) for a, p in imgs if not is_inv(a, p)]
    return pick(fwd) if fwd else [], pick(inv) if inv else []


def _problem_md(name):
    """The problem as the main README states it (source, equations, domain, initial and boundary conditions,
    and the inverse mode's unknowns), without its settings, results and figures, with links made relative
    to this page."""
    try:
        text = open(README, encoding="utf-8").read()
    except OSError:
        return ""
    m = re.search(r"<summary><strong>" + re.escape(name) + r"</strong></summary>(.*?)</details>", text, re.S)
    if not m:
        return ""
    out, skip_para = [], False
    for ln in m.group(1).strip().split("\n"):
        st = ln.strip()
        if st.startswith(("**Result", "**Inverse result", "**Results", "Result:", "**Note", "*Without",
                          "The figure", "The same run", "The plot", "The animation", "The figures", "The drop's radius", "The inverse run's",
                          "**Why", "DeepXDE's own script")):
            skip_para = True
            continue
        if not st:
            skip_para = False
            out.append("")
            continue
        if skip_para or st.startswith("![") or re.fullmatch(r"\*[^*].*[^*]\*", st) and not st.startswith("*Inverse mode*"):
            continue
        mb = re.match(r"- \*\*([^*:]+)", st)
        if mb and ("loss term" in mb.group(1) or not re.match(r"(Initial|Boundary|Geometry|Domain|Exact solution|Constants|Problem dimension|"
                               r"Unknown|Data|Reference|Source|Load|Material|Coefficient|Parameter|Forcing|Equation)",
                               mb.group(1).strip())):
            continue                     # the settings bullets: the tables give the settings each run used
        ln = re.sub(r"[;,]? the live [^.:]*below[^.:]*", "", ln)
        ln = re.sub(r"\s*The (live )?(convergence )?animation below[^:]*(:|$)", "", ln)
        ln = re.sub(r"\s*(The live |the )?(live )?convergence animation[^.]*\.", "", ln)
        ln = re.sub(r";\s*the convergence animation[^.]*", "", ln)
        out.append(re.sub(r"\s*\*\*(Inverse )?[Rr]esults?\b.*$", "", ln))     # a result in the same paragraph
    md = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()

    def fix(mm):
        url = mm.group(2)
        if url.startswith(("http", "mailto:", "../")):
            return mm.group(0)
        url = "../README.md" + url if url.startswith("#") else "../" + url
        return f"{mm.group(1)}({url})"
    return re.sub(r"(\]|`\])\(([^)\s]+)\)", fix, md)


def _anchor(name):
    try:
        text = open(README, encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(r'<details id="([^"]+)">\s*<summary><strong>' + re.escape(name) + r"</strong>", text)
    return m.group(1) if m else None


WITH_LINES = [False]   # set per example: 1D and ODE problems also show the line plots
NEEDED = []   # (run folder, variant) whose figures the page shows: benchmarks/tuning/figures_needed.txt


def _figs_md(s, variant, raw_chosen=None, slug=None):
    """Figures of one run: <variant>_*.png in the run folder's figure folder (older collections: own_* for the
    reference settings, tuned_* for the study's own choice raw_chosen). Notes the run in NEEDED."""
    if s is None and slug is None:
        return []
    slug = slug or (ts._slug(s["name"]) + ("_inv" if s["inverse"] else ""))
    NEEDED.append((slug, variant))
    folder = os.path.join(mtp.FIG_ROOT, slug)
    if not os.path.isdir(folder):
        return []
    files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    names = [f for f in files if f.startswith(variant + "_")]
    if not names and variant == "base":
        names = [f for f in files if f.startswith("own_")]
    if not names and raw_chosen == variant and variant != "base":
        names = [f for f in files if f.startswith("tuned_")]
    names.sort(key=lambda n: (bool(re.search(r"_(mu|v|w|p|q)\.png$", n)), n))   # the main output first
    surf = [n for n in names if "surface_comparison" in n or "view3d" in n]
    line = [n for n in names if "line_comparison" in n]
    slices = [n for n in names if "slice_comparison" in n]
    iso = [n for n in names if "isosurface" in n or "radius" in n]     # 3D drop: interface, cut-away error, radius
    keep = (slices + iso) if iso else (surf or slices or line) + (line if WITH_LINES[0] and (surf or slices) else [])
    keep += [n for n in names if n.endswith("param_plot.png")]
    return [os.path.relpath(os.path.join(folder, n), os.path.dirname(PAGE)).replace(os.sep, "/") for n in keep]


def _caption(path):
    """A few words on what a figure shows, from its file name."""
    f = os.path.basename(path).lower()
    out = re.search(r"_(?!comparison)([a-z0-9]+)\.(png|gif)$", f.split("comparison")[-1] if "comparison" in f else "")
    o = out.group(1) if out else ""
    o = SYM.get(o, o)
    pre = f"{o} — " if o else ""
    if "surface_comparison" in f:
        return pre + "PINN, reference and error over space and time"
    if "slice_comparison" in f:
        return pre + "PINN, reference and error on slices through the domain"
    if "line_comparison" in f:
        return pre + "PINN against the reference along a line, at the reference times"
    if "view3d" in f:
        return "3D view: the solution and its error in the domain"
    if "isosurface" in f:
        return "the drop (φ = 0) in the cube, and the error on the cube with one corner cut away"
    if "radius" in f:
        return "the drop's radius over time: PINN, reference and the sharp-interface limit"
    if "param" in f:
        return "the learned parameters during training"
    if "animation" in f:
        return pre + "the solution over time"
    alt = README_ALT.get(os.path.basename(path), "")      # otherwise the README's own caption
    return alt.split(" — ", 1)[-1] if " — " in alt else alt
    return ""


def _img(path, alt):
    return f"![{alt}]({path.replace(' ', '%20')})"


SHORT = {"base": None, "iters": "3× iterations", "net": "Larger network (2× width, one more layer)",
         "points": "3× training points", "lbfgs": "L-BFGS phase added", "all": "All of the above together",
         "dataw": "10× weight on the measured data", "lbfgs3": "3× L-BFGS iterations",
         "times": "Measured data at every reference time", "setup": "Best whole-interval setup",
         "invall": "10× data weight, 3× L-BFGS and the larger network", "windows2": "2× time windows",
         "rar": "Residual-based adaptive refinement", "msffn": "Multi-scale Fourier features",
         "fracres": "3× fractional resolution"}
ORDER = ["base", "iters", "net", "points", "lbfgs", "windows2", "rar", "msffn", "fracres", "dataw", "lbfgs3", "times",
         "setup", "all", "invall"]


def _t(q):
    return f"{q['seconds']:.0f}" if q and q.get("seconds") is not None else "–"


def _pct(x):
    return "–" if x is None else (f"{100 * x:.2g}%" if x < 0.1 else f"{100 * x:.0f}%")


def _sweep_rows(s, ref_name, inverse, prefix="", mark=None):
    """[(label, L2, params, seconds, picked)] for every variant the study ran of entry s; picked marks
    the run chosen as PINNStudio best when s is `mark` (the entry the best was picked from)."""
    if not s:
        return []
    out = []
    for v in sorted(s["variants"], key=lambda v: ORDER.index(v) if v in ORDER else 99):
        rs = s["variants"][v]
        l2 = mtp._med(rs, mtp._maxl2)
        t = mtp._med(rs, lambda r: r["seconds"])
        st = [SPACETIME.get(f"{ts._slug(r['example'])}{'_inv' if r['inverse'] else ''}/{v}_s{r['seed']}") for r in rs]
        st = [x for x in st if x is not None]
        if v == "base":
            lbl = ref_name if not prefix else "Best setup of the training methods study"
        else:
            lbl = prefix + ((SHORT.get(v) or v)[0].lower() + (SHORT.get(v) or v)[1:] if prefix else (SHORT.get(v) or v))
        if len(rs) > 1:
            lbl += f" ({len(rs)} seeds, median)"
        out.append((lbl, l2, mtp._found(dict(s, chosen=v)) if inverse else None,
                    f"{t:.0f}" if t is not None else None, bool(_same(s, mark)) and v == mark.get("chosen"),
                    statistics.median(st) if st else None, _outs(rs, v)))
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



OUTS = {}   # (run folder, variant, seed) -> {output: L2 at its last reference time} (--primary)
SYM = {"mu": "μ", "phi": "φ", "sxx": "σₓₓ", "syy": "σᵧᵧ", "sxy": "σₓᵧ", "ux": "uₓ", "uy": "uᵧ", "psi": "ψ",
       "theta": "θ", "rho": "ρ", "T": "T"}
MAIN = ("u", "c", "phi", "h", "U", "solution")


def _outs(rs, v):
    """Median over runs rs (of variant v) of each output's L2 at its last reference time."""
    acc = {}
    for r in rs:
        d = OUTS.get((ts._slug(r["example"]) + ("_inv" if r["inverse"] else ""), v, str(r["seed"]))) or {}
        for k, x in d.items():
            acc.setdefault(k, []).append(x)
    return {k: statistics.median(xs) for k, xs in acc.items()}


SPACETIME = {}  # run folder/variant_sSEED -> error over all reference times (--spacetime)
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
        if err is not None and err == err:          # a run that diverged (NaN) is never the best
            stats[v] = (err, mtp._med(rs, lambda r: r["seconds"]) or 0.0, l2)
    if not stats:
        return s
    low = min(e for e, _, _ in stats.values())
    ok = [v for v, (e, _, _) in stats.items() if e <= SAME * low]
    v = min(ok, key=lambda v: stats[v][1])
    return dict(s, chosen=v, tuned=stats[v][2], _raw_chosen=s.get("chosen"))



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
                    r["seconds"] for r in rs), outs=_outs(rs, v))
        rec["paper_runs"] = pr
        # the base PINN and the paper's method are PINNStudio runs too: one of them can be PINNStudio best
        rec["f_best_paper"] = None
        if pr and rec["f_best"] is not None:
            fs = rec["f_best_s"]
            tsec = mtp._med(fs["variants"][fs["chosen"]], lambda q: q["seconds"]) if fs else None
            cand = [(rec["f_best"], tsec or 0.0, None)] + [(q["l2"], q["seconds"], v) for v, q in pr.items()
                                                          if q.get("l2") is not None and v != "base"]
            low = min(c[0] for c in cand)
            win = min((c for c in cand if c[0] <= SAME * low), key=lambda c: c[1])
            if win[2]:
                rec["f_best_paper"] = win[2]
                rec["f_best"] = win[0]
        out.append(rec)
    return out


PAPER_SIZE = (6.5, 7.5)   # inches: the paper figure's printed size
# shorter row names in the paper figure
PAPER_SHORT = {"Wave 1D – space-time multi-scale Fourier features": "Wave 1D – multi-scale Fourier features",
               "Lorenz ODE – unknown C₁, C₂, C₃": "Lorenz ODE",
               "Fractional Poisson 1D – unknown order α": "Fractional Poisson 1D",
               "Poisson 1D – unknown forcing field": "Poisson 1D – unknown forcing",
               "Black-Scholes 1D – European call": "Black-Scholes 1D",
               "Navier-Stokes 2D – Kovasznay flow": "Navier-Stokes 2D – Kovasznay",
               "Navier-Stokes 3D – Beltrami flow": "Navier-Stokes 3D – Beltrami",
               "Linear elasticity 2D – plate": "Linear elasticity 2D"}
P_FLOOR = 1e-4            # parameter errors below it are drawn at it
P_XLIM = (3e-5, 1500)     # room for the markers at the floor and the symbols on the right


def figure(recs, path, stars=False, paper=False):
    """The paper's Fig. 4 layout (stars: also the error each paper reports, for its templates): open = DeepXDE's settings / the paper's method in PINNStudio
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
        # the open marker is DeepXDE's settings or a paper's method run in PINNStudio only; a PINNStudio
        # template's own starting settings are not an outside reference (and a paper's method not run yet)
        if not r["deepxde"] and not (r["paper"] and r["paper_runs"] and pm):
            ref_f = None
        ext_inv = r["deepxde"]      # the papers have no inverse problems: a template's inverse has no reference
        par_best = r.get("i_best_par") or []
        rep = [val for _, val, _ in (r["paper"] or []) if val] if not r["deepxde"] else []
        rows.append(dict(label=PAPER_SHORT.get(e.name, e.name) if paper else e.name, group=r["group"], f_ref=ref_f, f_best=r["f_best"],
                         i_ref=r.get("i_ref") if ext_inv else None, i_best=r.get("i_best"),
                         p_ref=_perr(r.get("i_ref_par")) if ext_inv else None, params=par_best, reported=min(rep) if rep else None))
    # paper: drawn at its printed size (6.5 x 7.5 in, a Word page with 1 in margins and room for a caption)
    F = dict(name=8.5, head=8.8, tick=8.0, axis=9.0, legend=9.0, sym=8.0, ms=1.35) if paper else \
        dict(name=7.2, head=7.6, tick=7.2, axis=8.0, legend=7.2, sym=6.6, ms=1.0)
    plt.rcParams.update({"font.size": F["tick"], "axes.linewidth": 0.7, "pdf.fonttype": 42})
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
    W_in, h_in = (PAPER_SIZE if paper else (mtp.FIG_WIDTH, mtp.ROW_HEIGHT * H + 1.0))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W_in, h_in), sharey=True,
                                 gridspec_kw={"width_ratios": [3.0, 1.6], "wspace": 0.12 if paper else 0.08})
    for gi, (g, a, b) in enumerate(groups):
        for axx in (ax, bx):
            axx.axhspan(Y[b] - 0.5, Y[a] + 1.4, color=mtp.C_BAND if gi % 2 == 0 else "white", zorder=0, lw=0)
    for g, v in heads:
        ax.text(-0.012, H - v - 0.05, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=F["head"], fontweight="bold", color=mtp.C_TEXT)

    def pair(axx, ref, best, y, marker, color, ms):
        if paper and marker == "D":     # the paper figure: inverse = PINNStudio's best parameter error only
            if best:
                axx.scatter(best, y, s=ms, marker=marker, color=color, edgecolor="white", lw=0.5, zorder=3)
            return
        """Open marker (the mode's colour) = DeepXDE's settings / the paper's method; filled = PINNStudio's
        best. Where the two are the same (or nothing needed tuning) the open one is a ring around the filled."""
        if best and ref and abs(ref - best) / max(ref, best) <= 0.05:
            axx.scatter(best, y, s=ms * 2.2, marker=marker, facecolor="none", edgecolor=color, lw=0.9, zorder=2)
        elif ref:
            axx.scatter(ref, y, s=ms * 0.95, marker=marker, facecolor="white", edgecolor=color, lw=1.2, zorder=2)
        if best:
            axx.scatter(best, y, s=ms, marker=marker, color=color, edgecolor="white", lw=0.5, zorder=3)

    for i, r in enumerate(rows):
        y = Y[i]
        pair(ax, r["f_ref"], r["f_best"], y, "o", mtp.C_FORWARD, 27 * F["ms"])
        if not paper:   # the inverse L2 is a fit to the data it trains on: not in the paper figure
            pair(ax, r["i_ref"], r["i_best"], y, "D", mtp.C_INVERSE, 22 * F["ms"])
        if stars and r["reported"]:
            ax.scatter(r["reported"], y, s=46 * F["ms"], marker="*", color="#f59f00", edgecolor="white", lw=0.4, zorder=4)
        es = [(s, max(abs(f - t) / abs(t) * 100, P_FLOOR)) for s, f, t in r["params"] if t]
        if es:
            right = max(e for _, e in es)
            o = max(100 * r["p_ref"], P_FLOOR) if r["p_ref"] is not None else None
            # the worst parameter carries the comparison with the reference (open diamond or ring)
            pair(bx, o, right, y, "D", mtp.C_INVERSE, 20 * F["ms"])
            right = max(right, 0 if paper else (o or 0))
            for _, e_ in es:
                bx.scatter(e_, y, s=20 * F["ms"], marker="D", color=mtp.C_INVERSE, edgecolor="white", lw=0.5, zorder=3)
            bx.text(right * (3.0 if paper else 2.2), y, ", ".join(_math_sym(s) for s, _ in sorted(es, key=lambda q: q[1])), va="center",
                    ha="left", fontsize=F["sym"], color=mtp.C_TEXT)
    ax.set_yticks([Y[i] for i in range(len(rows))], [r["label"] for r in rows], fontsize=F["name"], color=mtp.C_TEXT)
    ax.set_ylim(Y[len(rows) - 1] - 0.7, H + 0.15)
    ax.set_xscale("log"); ax.set_xlim(*mtp.XLIM_ERROR)
    ax.set_xticks([10.0 ** k for k in range(-6, 1)])
    bx.set_xscale("log"); bx.set_xlim(*P_XLIM)
    bx.set_xticks([1e-3, 1e-1, 10], ["0.001", "0.1", "10"])
    ax.set_xlabel("Forward problems\nrelative $L_2$ error, last reference time" if paper else
                  "Relative $L_2$ error of the main output at the last reference time", fontsize=F["axis"])
    bx.set_xlabel("Inverse problems\nparameter error (%)" if paper else "Parameter error (%)", fontsize=F["axis"])
    for axx in (ax, bx):
        axx.grid(axis="x", color=mtp.C_GRID, lw=0.6, zorder=0)
        axx.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            axx.spines[sp].set_visible(False)
        axx.tick_params(axis="y", length=0)
        axx.tick_params(axis="x", labelsize=F["tick"])
    from matplotlib.legend_handler import HandlerTuple
    lab = (["PINNStudio best", "DeepXDE / paper's method", "Both the same", "Paper's reported error"] if paper else
           ["PINNStudio best (forward, inverse)", "DeepXDE's settings / the paper's method",
            "Both the same (no tuning needed)", "Reported in the paper"])
    mk = lambda m, c, ms, **k: Line2D([], [], ls="", marker=m, markersize=ms, **k)
    h = [(mk("o", 0, 5, color=mtp.C_FORWARD), mk("D", 0, 4.5, color=mtp.C_INVERSE)),
         (mk("o", 0, 4.8, markerfacecolor="white", markeredgecolor=mtp.C_FORWARD, markeredgewidth=1.2),
          mk("D", 0, 4.4, markerfacecolor="white", markeredgecolor=mtp.C_INVERSE, markeredgewidth=1.2)),
         (mk("o", 0, 8, markerfacecolor="none", markeredgecolor=mtp.C_FORWARD, markeredgewidth=0.9),
          mk("D", 0, 7, markerfacecolor="none", markeredgecolor=mtp.C_INVERSE, markeredgewidth=0.9))]
    # the ring: a small filled marker inside it
    class _Over(tuple):     # drawn on top of each other in the legend
        pass
    h[2] = (_Over((h[2][0], mk("o", 0, 4, color=mtp.C_FORWARD))), _Over((h[2][1], mk("D", 0, 3.5, color=mtp.C_INVERSE))))
    if stars:
        h.append(mk("*", 0, 8, color="#f59f00"))
    if paper:
        h = [mk("o", 0, 6, color=mtp.C_FORWARD),
             mk("o", 0, 5.6, markerfacecolor="white", markeredgecolor=mtp.C_FORWARD, markeredgewidth=1.2),
             _Over((mk("o", 0, 9, markerfacecolor="none", markeredgecolor=mtp.C_FORWARD, markeredgewidth=0.9),
                    mk("o", 0, 5, color=mtp.C_FORWARD))),
             mk("D", 0, 5.4, color=mtp.C_INVERSE)] + h[3:]
        lab = ["PINNStudio best", "DeepXDE / paper's method", "Both the same", "Inverse: parameter error", lab[3]]
    lab = lab[:len(h)]
    hm = {tuple: HandlerTuple(ndivide=None, pad=0.6), _Over: HandlerTuple(ndivide=1)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not paper:
        fig.legend(h, lab, handler_map=hm, loc="upper center", ncol=len(h), frameon=False,
                   fontsize=F["legend"], handlelength=2.6, bbox_to_anchor=(0.6, 1.0))
        fig.subplots_adjust(left=0.33, right=0.985, top=1 - 0.32 / h_in, bottom=0.45 / h_in)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        return
    # exact printed size: margins from the measured labels, no bbox cropping
    leg = fig.legend(h, lab, handler_map=hm, loc="upper center", ncol=3, frameon=False, fontsize=F["legend"],
                     handlelength=1.2, handletextpad=0.4, columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.3, right=0.98, top=0.95, bottom=0.07)
    rd = fig.canvas.get_renderer()
    pad = 0.04 * fig.dpi
    for _ in range(3):
        fig.canvas.draw()
        w_lab = max(t.get_window_extent(rd).width for t in ax.get_yticklabels() + ax.texts)
        over = max([t.get_window_extent(rd).x1 for t in bx.texts + [bx.xaxis.label]] + [bx.get_window_extent(rd).x1])
        W, Hh = fig.bbox.width, fig.bbox.height
        p = fig.subplotpars
        left = (w_lab + 2.5 * pad) / W
        right = p.right - max(0.0, over + pad - W) / W
        top = (leg.get_window_extent(rd).y0 - pad) / Hh
        xl = min(a.xaxis.label.get_window_extent(rd).y0 for a in (ax, bx))
        bottom = p.bottom + max(0.0, pad - xl) / Hh
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def _same(s, mark):
    # the same study entry (_pick returns a copy with its choice)
    return s if (s and mark and s.get("name") == mark.get("name") and s.get("inverse") == mark.get("inverse")) else None


def _row(cells, picked):
    """A table row; the run picked as PINNStudio best is marked 🔴 and in bold."""
    if picked:
        cells = ["🔴 **" + str(cells[0]) + "**"] + [f"**{c}**" if c not in ("–", "") else c for c in cells[1:]]
    return "| " + " | ".join(str(c) for c in cells) + " |"


# ---- each example's settings, Default and Tuned, as PINNStudio loads them --------------------------------
def _n(x):
    return f"{int(x):,}"


def _phases_txt(c):
    if c.optimizer_scheduler and c.scheduler_phases:
        ph = json.loads(c.scheduler_phases)
    else:
        ph = [{"optimizer": c.optimizer, "iterations": c.iterations, "lr": c.learning_rate, "loss": "MSE"}]
        if c.optimizer2 not in ("none", "", None):
            ph.append({"optimizer": c.optimizer2, "iterations": c.iterations2, "loss": "MSE"})
    name = {"adam": "Adam", "lbfgs": "L-BFGS", "adamw": "AdamW", "sgd": "SGD", "nncg": "NNCG", "rmsprop": "RMSprop"}
    parts = []
    for q in ph:
        o = name.get(q.get("optimizer"), q.get("optimizer"))
        t = f"{o} {_n(q.get('iterations', 0))}"
        if o != "L-BFGS" and q.get("lr") not in (None, 0.001, 1e-3):
            t += f" (lr {q['lr']:g})"
        if q.get("loss", "MSE") != "MSE":
            t += f", {q['loss']} loss"
        parts.append(t)
    w = {q.get("weights") for q in ph if q.get("weights")}
    return " → ".join(parts), (w.pop() if len(w) == 1 else None)


def _settings_md(c, inverse):
    """A short list of what a configuration trains with (network, points, training, methods)."""
    L = []
    lay = list(c.layers)
    net = {"FNN": "FNN", "MsFFN": "multi-scale Fourier features (MsFFN)", "ResNet": "ResNet"}.get(c.network_type, c.network_type)
    hid = lay[1:-1]
    shape = f"{len(hid)} × {hid[0]}" if hid and len(set(hid)) == 1 else str(lay)
    extra = f", scales {c.msffn_sigmas}" if c.network_type == "MsFFN" else ""
    L.append(f"**Network:** {net} {shape} ({lay[0]} input{'s' if lay[0] != 1 else ''}, {lay[-1]} output{'s' if lay[-1] != 1 else ''}), {c.activation}, {c.kernel_initializer}{extra}")
    pts = [f"{_n(c.num_domain)} domain"]
    if not c.steady_state and c.problem_dim != "ODE" or c.num_initial:
        if c.num_initial:
            pts.append(f"{_n(c.num_initial)} initial")
    if c.num_boundary:
        pts.append(f"{_n(c.num_boundary)} boundary")
    L.append("**Points:** " + ", ".join(pts))
    phases, w = _phases_txt(c)
    L.append(f"**Training:** {phases}")
    if w and set(w.split(",")) != {"1.0"}:
        from pinnstudio.core import paper_settings as _ps
        vals = [float(x) for x in w.split(",")]
        n_eq, n_bc, n_ic = _ps._n_terms(c)
        groups = [("equations", vals[:n_eq]), ("boundary", vals[n_eq:n_eq + n_bc]),
                  ("initial", vals[n_eq + n_bc:n_eq + n_bc + n_ic]), ("measured data", vals[n_eq + n_bc + n_ic:])]
        txt = [f"{g} {', '.join(f'{x:g}' for x in v) if len(set(v)) > 1 else f'{v[0]:g}'}" for g, v in groups if v]
        L.append("**Loss weights:** " + "; ".join(txt))
    m = []
    act = (c.ic_active or "").split(",")
    hard = (c.ic_hard or "").split(",")
    names = c.output_names.split(",")
    if "True" in hard:
        m.append("exact initial condition (built into the network) for " + ", ".join(
            n for n, h in zip(names, hard) if h == "True"))
    elif c.output_transform_enabled and c.output_transform_mode == "formula" and (c.output_transform_formula or "").strip("|"):
        m.append("initial and boundary conditions built into the output (output transform formula)")
    if c.periodic_inputs:
        m.append("exact periodicity (periodic input features)")
    if c.causal_training:
        m.append("causal training")
    if c.point_weights not in ("None", "", None):
        m.append(f"{c.point_weights} per-point weights")
    if c.adapt_method == "RAR" and not c.time_adaptive:
        m.append(f"residual-based adaptive refinement ({c.rar_cycles} rounds of {_n(c.rar_add_points)} points)")
    if c.time_adaptive:
        try:
            g = json.loads(c.ta_step_groups) if c.ta_step_groups else []
        except ValueError:
            g = []
        n = sum(int(x["steps"]) for x in g) if g else c.ta_num_steps
        how = ("one network on a growing time range" if c.ta_growing else
               "a network per window, each starting from the previous one" if c.ta_transfer_learning else
               "a separate network per window")
        t = f"time-adaptive training: {n} time windows, {how}"
        if c.ta_backward:
            t += f", backward-compatibility term (weight {c.ta_backward_weight:g})"
        if getattr(c, "ta_resample_rounds", 0):
            t += (f", resampling in each window ({c.ta_resample_rounds} rounds of {_n(c.ta_resample_points)} "
                  f"points, {_n(c.ta_resample_iters)} Adam iterations each)")
        if c.ta_icgl_fraction:
            t += f", initial-condition guided learning ({c.ta_icgl_fraction:g} of Adam)"
        m.append(t)
    if c.ic_pretrain:
        m.append("initial condition fitted first (IC pre-training)")
    if m:
        L.append("**Methods:** " + "; ".join(m))
    if inverse:
        try:
            v = json.loads(c.inverse_variables_json or "[]")
            f = json.loads(c.inverse_obs_files_json or "[]")
        except ValueError:
            v, f = [], []
        if v:
            L.append("**Unknowns:** " + ", ".join(f"{x['name']} (initial guess {x.get('init')}, true {x.get('true')})"
                                                   for x in v))
        if f:
            L.append("**Measured data:** " + "; ".join(
                f"{os.path.basename(x.get('path') or '') or '(file to pick)'} for {names[int(x.get('output_idx', 0))] if int(x.get('output_idx', 0)) < len(names) else 'output'}"
                f" (weight {x.get('weight', 100):g})" for x in f))
    return "\n".join("- " + x for x in L)


def _base_config(e, inverse):
    from pinnstudio.core import deepxde_examples as dx
    if e.source == cat.DEEPXDE:
        ex = dx.BY_NAME[e.key]
        return dx.build_config(ex, ts.REF_ROOT, inverse=inverse and ex.inverse is not None)
    return ts.template_config(e.key, inverse)


def _paper_note(key, variant):
    """What the paper states for its method and what PINNStudio chooses where it does not: the comments of
    the method's function in pinnstudio/core/paper_settings.py."""
    import inspect
    from pinnstudio.core import paper_settings as ps
    f = ps.PAPER.get((key, variant))
    if f is None:
        return ""
    src = inspect.getsource(f)
    lines = [ln.strip()[1:].strip() for ln in src.splitlines() if ln.strip().startswith("#")]
    if "_wz_approach2(" in src:
        lines = [ln.strip()[1:].strip() for ln in inspect.getsource(ps._wz_approach2).splitlines()
                 if ln.strip().startswith("#")] + lines
    txt = " ".join(lines).replace("Delta t", "Δt").replace("->", "→").replace("x128", " × 128")
    return f"*From the paper, and PINNStudio's choices where it is silent:* {txt}" if txt else ""


def _tm_anchor(name):
    """The training methods study's section for an example (benchmarks/training_methods.md), or None:
    the section whose title words (dimension, equation, authors) all appear in the example's name."""
    try:
        text = open(os.path.join(os.path.dirname(PAGE), "training_methods.md"), encoding="utf-8").read()
    except OSError:
        return None
    words = lambda x: {w for w in re.findall(r"[a-z0-9-]+", x.lower()) if w not in ("et", "al", "and")}
    have = words(name)
    if "unknown" in have:            # an inverse example of its own, not the study's forward problem
        return None
    best = None
    for h in re.findall(r"^### (.+)$", text, re.M):
        hw = words(h)
        if hw and hw <= have and (best is None or len(hw) > len(words(best))):
            best = h
    if best is None:
        return None
    return best, re.sub(r"[^a-z0-9 -]", "", best.lower()).replace(" ", "-")


def _settings_toggles(r):
    """Two toggles. Default: the source's own settings (DeepXDE's, or the paper's method) -- only the problems
    the source itself solves (a paper or a DeepXDE forward example has no inverse problem; PINNStudio's own
    templates have both). PINNStudio best (Settings: Tuned): forward and inverse. Each problem says how to
    load it in PINNStudio."""
    from pinnstudio.core import example_settings as es, paper_settings as ps
    e = r["entry"]
    inv_only = e.key.startswith("Inverse:") or e.problem == cat.INVERSE
    paper = ps.TEMPLATE_PAPER.get(e.key)
    has_inv = inv_only or r.get("i_best") is not None or bool(r.get("i_best_par"))
    probs = ([] if inv_only else [False]) + ([True] if has_inv else [])
    # does the source itself have the inverse problem?
    default_inv = inv_only or (not r["deepxde"] and not paper)
    source = ("DeepXDE's settings" if r["deepxde"] else f"{PAPER_NAME[paper]}, as in the paper" if paper
              else "the template's own settings")

    def load(inverse, mode):
        cat_name = "Inverse" if (inverse or inv_only) else "Forward"
        return f"*Load it:* toolbar → Examples: **{cat_name}** → **{e.name}** → **{mode}**, then Solve."

    def cfg_for(inverse, mode):
        base = _base_config(e, inverse and not inv_only)
        return base, es.settings_for(base, e.key, inverse and not inv_only, mode)[0]

    def study_link(inverse):
        """Where PINNStudio best was found: the tuning study's runs (and the training methods study for a
        template's best setup)."""
        st = r.get("i_s") if (inverse or inv_only) else (r.get("f_best_s") or r.get("f_plain_s"))
        links = []
        if not (inverse or inv_only) and r.get("f_best_paper"):
            return ""
        if st:
            t = mtp._title(st["name"], st["inverse"])
            links.append(f"[tuning study, {'inverse' if st['inverse'] else 'forward'} runs](tuning_study.md#{mtp._slug(t)})")
        tm = _tm_anchor(e.name)
        if not (inverse or inv_only) and tm:
            links.append(f"[training methods study: {tm[0]}](training_methods.md#{tm[1]})")
        elif not (inverse or inv_only) and r.get("f_best_is_setup"):
            links.append("[training methods study](training_methods.md)")
        return ("*Found in:* " + " · ".join(links)) if links else ""

    # which problems have a Tuned choice (otherwise Default is PINNStudio best: no Tuned toggle for it)
    tuned_probs = [p for p in probs if es.has_tuned(e.key, p and not inv_only) or (p and not default_inv)]
    same_probs = [p for p in probs if p not in tuned_probs]

    out = []
    # Default
    body = []
    for inverse in [p for p in probs if (not p) or default_inv]:
        head = "Inverse problem" if inverse else "Forward problem"
        try:
            _, cfg = cfg_for(inverse, es.DEFAULT)
        except Exception as ex:
            body.append(f"*{head}:* not available here ({str(ex)[:80]}).\n")
            continue
        also = " — also PINNStudio best" if inverse in same_probs else ""
        why = ""
        if also:
            ran = r.get("i_s") if inverse or inv_only else (r.get("f_plain_s") or r.get("f_best_s"))
            why = (" (no change in the tuning study did better)" if ran else
                   " (already accurate, so it was not tuned)")
            lk = study_link(inverse) if ran else ""
            why += ("\n\n" + lk) if lk else ""
        body.append(f"*{head}{also}*{why}\n\n" + _settings_md(cfg, inverse or inv_only) + "\n\n"
                    + load(inverse, "Default") + "\n")
    if paper:
        body.append(_paper_note(e.key, paper) + "\n")
    title = source + (" (also PINNStudio best)" if not tuned_probs else "")
    out.append(f"<details>\n<summary>⚙️ <em>Default settings — {title}</em></summary>\n\n" + "\n".join(body)
               + "\n</details>\n")
    if not tuned_probs:
        return "\n".join(out)
    # PINNStudio best
    body = []
    for inverse in tuned_probs:
        head = "Inverse problem" if inverse else "Forward problem"
        try:
            base, cfg = cfg_for(inverse, es.TUNED)
        except Exception as ex:
            body.append(f"*{head}:* not available here ({str(ex)[:80]}).\n")
            continue
        note = ""
        if inverse and not default_inv:     # an inverse problem PINNStudio added: not in Default
            note = " — PINNStudio's own inverse setup" + ("" if cfg is base else ", tuned")
        elif cfg is base:
            note = " — the template's own settings"
        lk = study_link(inverse)
        body.append(f"*{head}{note}*\n\n" + _settings_md(cfg, inverse or inv_only) + "\n\n" + load(inverse, "Tuned")
                    + ("\n\n" + lk if lk else "") + "\n")
    for inverse in same_probs:
        if inverse or True:
            head = "Inverse problem" if inverse else "Forward problem"
            body.append(f"*{head}:* PINNStudio best is the Default settings above (no separate Tuned settings).\n")
    out.append("<details>\n<summary>⚙️ <em>PINNStudio best — Tuned settings</em></summary>\n\n" + "\n".join(body)
               + "\n</details>\n")
    return "\n".join(out)


def _section(r):
    e = r["entry"]
    L = []
    a = L.append
    paper = next((t[-1] for t in cat._TEMPLATES if t[0] == e.key), "")
    src = "DeepXDE example" if r["deepxde"] else ("PINNStudio template, after " + paper if paper
                                                   else "PINNStudio template")
    a(f'<details id="bench-{re.sub(r"[^a-z0-9]+", "-", e.name.lower()).strip("-")}">')
    a(f"<summary><strong>{e.name}</strong> · {src}</summary>\n")
    WITH_LINES[0] = e.dim in ("1D", "ODE")
    prob = _problem_md(e.name)
    if prob:
        a("**The problem**\n")
        a(prob + "\n")
    try:
        a(_settings_toggles(r))
    except Exception as ex:
        a(f"*Settings: not available ({ex}).*\n")

    has_f = r["f_ref"] is not None or r["f_best"] is not None
    has_i = r.get("i_best") is not None or r.get("i_best_par")
    ref_name = "DeepXDE's settings" if r["deepxde"] else "The template's starting settings"
    paper_rows = bool(r["paper"]) and not r["deepxde"]
    pr = r["paper_runs"]

    if has_f:
        a("**Forward problem**\n")
        fmark = r["f_best_s"] if not r.get("f_best_paper") else None   # _pick's copy: its "chosen" is PINNStudio best
        rows = _sweep_rows(r["f_plain_s"], ref_name, False, mark=fmark)
        if not r["f_plain_s"] and not paper_rows:
            rows = [(ref_name + (" (README)" if r["f_ref_src"] == "README" else ""), r["f_ref"], None, None, False,
                     None, {})]
        if r["f_best_s"] and r["f_best_is_setup"]:
            rows += _sweep_rows(r["f_best_s"], "Best setup of the training methods study", False,
                                prefix="Best setup + ", mark=fmark)
        # every output's error when the problem has more than one (the main output first; it decides the best)
        names = []
        for d in [x[6] for x in rows] + [q.get("outs") or {} for q in pr.values()]:
            names += [k for k in d if k not in names]
        main = next((k for k in MAIN if k in names), names[0] if names else None)
        names = ([main] + [k for k in names if k != main]) if len(names) > 1 else []
        last = ", last time" if paper_rows else ""
        heads = ([f"L₂ {SYM.get(k, k)}{last}" for k in names] if names else [f"L₂{last}"]) + \
            ([f"L₂ {SYM.get(main, main)}, all times" if names else "L₂, all times"] if paper_rows else []) + ["Time (s)"]
        a("| Training | " + " | ".join(heads) + " |\n|" + "---|" * (len(heads) + 1))

        def cells(l2, outs, st, t):
            c = [sci((outs or {}).get(k, l2 if k == main else None)) for k in names] if names else [sci(l2)]
            return c + ([sci(st)] if paper_rows else []) + [t if t is not None else "–"]
        if paper_rows:
            p = pr.get("plain")
            a(_row(["Base PINN"] + cells(p and p["l2"], p and p.get("outs"), p and p["st"], _t(p) if p else None),
                   r.get("f_best_paper") == "plain"))
            for ref, val, v in r["paper"]:
                a(_row([f"{PAPER_NAME[v]}, as reported ({ref})"] + ["–"] * (max(len(names), 1)) +
                       [sci(val) if val else "figures only", "–"], False))
                q = pr.get(v)
                a(_row([f"{PAPER_NAME[v]}, run in PINNStudio"] +
                       cells(q and q["l2"], q and q.get("outs"), q and q["st"], _t(q) if q else None),
                       r.get("f_best_paper") == v))
        best = None
        for row in rows:
            lbl, l2, _, t, picked, st, outs = row
            if picked:
                best = row
            a(_row([lbl] + cells(l2, outs, st, t), picked))
        lab = _best_label(r, False)
        if paper_rows and lab == "no tuning needed":
            lab = "the template's starting settings"
        if r.get("f_best_paper"):
            q = pr[r["f_best_paper"]]
            lab = "the base PINN" if r["f_best_paper"] == "plain" else f"{PAPER_NAME[r['f_best_paper']]}, run in PINNStudio"
            bc = cells(q["l2"], q.get("outs"), q["st"], _t(q))
        else:
            bc = cells(r["f_best"], best[6] if best else {}, best[5] if best else None, best[3] if best else None)
        a(_row([f"**PINNStudio best** (Settings: Tuned): {lab}"] + [f"**{c}**" if c != "–" else c for c in bc], False))
        a("")
        if names:
            a(f"*Each output's relative L₂ at its last reference time; the best run is chosen by "
              f"{SYM.get(main, main)}.*\n")
        if paper_rows and not pr:
            a("*The base PINN and the paper's method are queued on the GPU; their rows fill in as they finish.*\n")
    if has_i:
        a("**Inverse problem**\n")
        a("| Training | L₂ fit to measured data | Parameters found (true) | Worst parameter error | Time (s) |\n"
          "|---|---|---|---|---|")
        best_t = None
        if r.get("i_s"):
            for lbl, l2, par, t, picked, _st, _o in _sweep_rows(r["i_s"], ref_name, True, mark=r["i_s"]):
                if picked:
                    best_t = t
                a(_row([lbl, sci(l2), _par_txt(par) or '–', _pct(_perr(par)), t if t is not None else '–'], picked))
        else:
            a(f"| {ref_name} (README) | {sci(r.get('i_ref'))} | {_par_txt(r.get('i_ref_par')) or '–'} | "
              f"{_pct(_perr(r.get('i_ref_par')))} | – |")
        a(f"| **PINNStudio best** (Settings: Tuned): {_best_label(r, True)} | **{sci(r.get('i_best'))}** | "
          f"{_par_txt(r.get('i_best_par')) or '–'} | **{_pct(_perr(r.get('i_best_par')))}** | "
          f"{('**' + best_t + '**') if best_t else '–'} |")
        a("")
        a("*L₂ fit to measured data: the inverse run trains on measured values of the solution (in most examples at "
          "every reference time), so this L₂ is mostly a check that the fit worked, not a prediction error. The "
          "parameter error measures the accuracy and decides the best run.*\n")
        if r["deepxde"] and r.get("i_s") and not e.key.startswith("Inverse:") and e.problem != cat.INVERSE:
            a("*DeepXDE's example is forward only; this inverse problem is PINNStudio's inverse mode of it.*\n")

    # figures, in two toggles: forward, then inverse. Forward: the reference (DeepXDE's settings; for a
    # template from a paper the paper's method run in PINNStudio; else the template's starting settings) and
    # PINNStudio best. Inverse: DeepXDE's settings and PINNStudio best for DeepXDE's own inverse examples,
    # else PINNStudio best (the papers have no inverse problem; DeepXDE's forward examples have no inverse
    # setup of their own).
    fwd_r, inv_r = _readme_images(e.name, e.key.startswith("Inverse:") or e.problem == cat.INVERSE)

    def toggle(title, blocks):
        blocks = [(t, p) for t, p in blocks if p or t.startswith("*")]
        if not blocks:
            return
        a(f"<details>\n<summary>{title}</summary>\n")
        for t, paths in blocks:
            if t.startswith("*"):          # a note in place of figures
                a(t + "\n")
                continue
            a(f"*{t}:*\n")
            for pth in paths:
                pth = pth if pth.startswith(("figures/", "../", "http")) else "../" + pth
                cap = _caption(pth)
                if cap and len(paths) > 1:
                    a(f"<sub>{cap}</sub>\n")
                a(_img(pth, f"{e.name}: {t}" + (f" — {cap}" if cap else "")) + "\n")
        a("</details>\n")

    if has_f:
        blocks = []
        best_s, raw = r["f_best_s"], (r["f_plain_s"] or {})
        best_v = best_s["chosen"] if best_s else None
        best_raw = (raw.get("chosen") if not r["f_best_is_setup"] else (best_s or {}).get("_raw_chosen"))
        best_figs = _figs_md(best_s, best_v, best_raw) if best_s else []
        if r.get("f_best_paper"):
            best_figs = []                  # the paper's run is PINNStudio best: shown once, below
        if paper_rows:
            for _, _, v in r["paper"]:
                if v in pr:
                    both = " (also PINNStudio best)" if r.get("f_best_paper") == v else ""
                    blocks.append((f"{PAPER_NAME[v]}, run in PINNStudio{both}",
                                   _figs_md(None, v, slug=ts._slug(r["key"]))))
                else:
                    blocks.append((f"*{PAPER_NAME[v]}, run in PINNStudio: the figures come when its run finishes.*", []))
        else:
            ref = "DeepXDE's settings" if r["deepxde"] else "The template's starting settings"
            same = best_s and best_v == "base" and not r["f_best_is_setup"]
            if raw:
                blocks.append((ref + (" (also PINNStudio best)" if same else ""), _figs_md(raw, "base")))
            elif fwd_r:
                blocks.append((ref + " (README)", fwd_r))
            if same:
                best_figs = []
        if best_s and best_figs:
            blocks.append((f"PINNStudio best — {_best_label(r, False)}", best_figs))
        elif r.get("f_best_paper") == "plain":
            blocks.append(("PINNStudio best — the base PINN", _figs_md(None, "plain", slug=ts._slug(r["key"]))))
        elif not best_s and paper_rows and fwd_r:
            blocks.append(("PINNStudio best (the template's own settings, README)", fwd_r))
        toggle("📊 <em>Forward results — figures</em>", blocks)
    if has_i:
        blocks = []
        i_s = r.get("i_s")
        own_inverse = r["deepxde"] and (e.key.startswith("Inverse:") or e.problem == cat.INVERSE)
        if i_s:
            v = i_s["chosen"]
            raw_v = i_s.get("_raw_chosen")
            if own_inverse and v != "base":
                blocks.append(("DeepXDE's settings", _figs_md(i_s, "base")))
            lab = _best_label(r, True)
            blocks.append((f"PINNStudio best — {lab}", _figs_md(i_s, v, raw_v)))
        elif inv_r:
            blocks.append(("PINNStudio best (README)", inv_r))
        toggle("📊 <em>Inverse results — figures</em>", blocks)
    a("</details>\n")
    return "\n".join(L)


PAPER_LIST = os.path.join(_REPO, "benchmarks", "paper_figure_examples.txt")
PAPER_FIG = os.path.join(_REPO, "benchmarks", "figures", "paper_figure.png")


PAPER_FWD = os.path.join(_REPO, "benchmarks", "figures", "paper_forward.png")
PAPER_INV = os.path.join(_REPO, "benchmarks", "figures", "paper_inverse.png")


def _paper_examples_md():
    """The two toggles with the paper's 3 x 3 example figures (tools/make_paper_examples.py) and the problems."""
    if not (os.path.exists(PAPER_FWD) and os.path.exists(PAPER_INV)):
        return ""
    fwd = f"""<details>
<summary>🧪 <em>Forward examples for the paper — 1D, 2D, 3D</em></summary>

Three forward problems, one per dimension and each from a different area: a multi-scale wave, a phase-field equation and 3D fluid flow. Each row is PINNStudio's best run of that problem (the same run as in its table below), drawn from its saved model.

**(a) Wave 1D — multi-scale Fourier features** ([DeepXDE example](#bench-wave-1d-space-time-multi-scale-fourier-features))

$$\\frac{{\\partial^2 u}}{{\\partial t^2}} = C^2\\frac{{\\partial^2 u}}{{\\partial x^2}}, \\qquad x \\in [0, 1],\\ t \\in [0, 1],\\ C = 10$$

with $u(0, t) = u(1, t) = 0$, $u(x, 0) = \\sin(\\pi x) + \\sin(2\\pi x)$ and $u_t(x, 0) = 0$. The exact solution $u = \\sin(\\pi x)\\cos(10\\pi t) + \\sin(2\\pi x)\\cos(20\\pi t)$ oscillates ten and twenty times in $t$, which a plain network struggles to represent. PINNStudio: space-time multi-scale Fourier feature network 4 × 200, Adam 30,000 → L-BFGS 15,000. $L_2$ = 1.5 × 10⁻³ at $t = 1$ (median of 3 seeds) against 3.0 × 10⁻¹ with DeepXDE's settings; over the whole space-time domain (this seed) 3.3 × 10⁻³.

**(b) Allen-Cahn 2D — Mattey & Ghosh** ([PINNStudio template](#bench-allen-cahn-2d-mattey-ghosh))

$$\\frac{{\\partial h}}{{\\partial t}} = c_1^2\\nabla^2 h - c_2\\left(h^3 - h\\right), \\qquad (x, y) \\in [0, 1]^2,\\ t \\in [0, 1],\\ c_1^2 = 10^{{-4}},\\ c_2 = 1$$

with $h(x, y, 0) = \\sin(4\\pi x)\\cos(4\\pi y)$ and periodic boundaries. PINNStudio: 4 × 128 network on periodic input features, the initial condition built into the output, self-adaptive (SA-PINN) point weights, Adam 20,000 → L-BFGS 10,000 over the whole time interval. $L_2$ = 3.4 × 10⁻⁴ over all reference times; Mattey & Ghosh report 2.5 × 10⁻² for their bc-PINN.

**(c) Navier-Stokes 3D — Beltrami flow** ([DeepXDE example](#bench-navier-stokes-3d-beltrami-flow))

$$\\frac{{\\partial \\mathbf u}}{{\\partial t}} + (\\mathbf u\\cdot\\nabla)\\mathbf u + \\nabla p = \\nabla^2\\mathbf u, \\qquad \\nabla\\cdot\\mathbf u = 0, \\qquad (x, y, z) \\in [-1, 1]^3,\\ t \\in [0, 1]$$

with the exact Beltrami solution, e.g. $u = -\\left(e^{{x}}\\sin(y + z) + e^{{z}}\\cos(x + y)\\right)e^{{-t}}$, on the boundary and at $t = 0$. Four outputs $(u, v, w, p)$ from one network (4 × 50, DeepXDE's settings, already accurate). Shown: $u$ on the three visible faces of the cube at $t = 1$. Over the whole cube at $t = 1$: $L_2$ = 1.4 × 10⁻³ ($u$), 7.3 × 10⁻⁴ ($v$), 1.2 × 10⁻³ ($w$) and 2.4 × 10⁻² ($p$, which is fixed only up to a constant, so compared after removing the mean).

![Forward examples: Wave 1D, Allen-Cahn 2D, Navier-Stokes 3D — PINNStudio, reference and error](figures/paper_forward.png{_ver(PAPER_FWD)})

**Caption for the paper:**

> **Figure X.** Forward problems solved with PINNStudio. (a) Wave equation in 1D, solved with a space-time multi-scale Fourier feature network: $u(x, t)$ over the whole space-time domain (relative $L_2$ error 3.3 × 10⁻³). (b) Allen-Cahn equation in 2D (Mattey & Ghosh, 2022): $h(x, y)$ at $t = 1$ (relative $L_2$ error over all reference times 3.4 × 10⁻⁴). (c) Navier-Stokes equations in 3D (Beltrami flow): the velocity component $u$ on the visible faces of the cube $[-1, 1]^3$ at $t = 1$ (relative $L_2$ error of $u$ over the cube 1.4 × 10⁻³). Left to right: PINNStudio's prediction, the reference solution (exact for (a) and (c), numerical for (b)) and the absolute error. The prediction and the reference share a colour scale.

*PDF: [`figures/paper_forward.pdf`](figures/paper_forward.pdf). Made with [`tools/make_paper_examples.py`](../tools/make_paper_examples.py) from the saved models (the arrays are in [`paper_examples/`](paper_examples)).*

</details>
"""
    inv = f"""<details>
<summary>🔍 <em>Inverse examples for the paper — 1D, 2D, 3D</em></summary>

Three inverse problems, one per dimension: the classic PINN benchmark, material constants of an elastic plate, and the coefficients of a 3D phase-field equation. The unknown coefficients are trained together with the network from measured values of the solution. The third column shows how close each one gets to its true value during training.

**(a) Burgers 1D — Raissi et al.** ([PINNStudio template](#bench-burgers-1d-raissi-et-al))

$$\\frac{{\\partial u}}{{\\partial t}} + \\lambda_1 u\\frac{{\\partial u}}{{\\partial x}} - \\frac{{\\lambda_2}}{{\\pi}}\\frac{{\\partial^2 u}}{{\\partial x^2}} = 0, \\qquad x \\in [-1, 1],\\ t \\in [0, 1]$$

with $u(x, 0) = -\\sin(\\pi x)$ and $u(\\pm 1, t) = 0$; true $\\lambda_1 = 1$, $\\lambda_2 = 0.01$ (initial guesses 0.1 and 1). Measured: $u$ at $t = 1$. PINNStudio: 4 × 40 network, Adam 15,000 → L-BFGS 30,000. Found $\\lambda_1$ = 0.99994, $\\lambda_2$ = 0.010010 (errors 0.006% and 0.099%). The solution is drawn over the whole $(x, t)$ domain, so everything before $t = 1$ is a prediction ($L_2$ = 1.1 × 10⁻³ against the exact Cole-Hopf solution).

**(b) Linear elasticity 2D — plate** ([DeepXDE example](#bench-linear-elasticity-2d-plate))

$$\\frac{{\\partial \\sigma_{{xx}}}}{{\\partial x}} + \\frac{{\\partial \\sigma_{{xy}}}}{{\\partial y}} = f_x, \\quad \\frac{{\\partial \\sigma_{{xy}}}}{{\\partial x}} + \\frac{{\\partial \\sigma_{{yy}}}}{{\\partial y}} = f_y, \\quad \\sigma_{{xx}} = (\\lambda + 2\\mu)\\varepsilon_{{xx}} + \\lambda\\varepsilon_{{yy}},\\ \\sigma_{{yy}} = (\\lambda + 2\\mu)\\varepsilon_{{yy}} + \\lambda\\varepsilon_{{xx}},\\ \\sigma_{{xy}} = 2\\mu\\varepsilon_{{xy}}$$

on $[0, 1]^2$, with the boundary conditions built into the output and the known body force. The Lamé constants $\\lambda$ (true 1, initial guess 2) and $\\mu$ (true 0.5, initial guess 1) are unknown in the stress-strain law. Measured: $u_x$ and $u_y$ on a 19 × 19 grid. PINNStudio: parallel network (PFNN) 4 × 40 with five outputs, Adam 20,000 → L-BFGS 20,000. Found $\\lambda$ = 0.99974, $\\mu$ = 0.50009 (errors 0.026% and 0.018%). Shown: $u_x$.

**(c) Allen-Cahn 3D — Wight & Zhao** ([PINNStudio template](#bench-allen-cahn-3d-wight-zhao))

$$\\frac{{\\partial \\phi}}{{\\partial t}} = \\lambda\\varepsilon^2\\nabla^2\\phi - \\lambda\\left(\\phi^3 - \\phi\\right), \\qquad (x, y, z) \\in [0, 1]^3,\\ t \\in [0, 1]$$

a spherical drop (radius 0.35 at $t = 0$) shrinking in a periodic cube. Unknown: $\\lambda$ (true 10) and $\\lambda\\varepsilon^2$ (true 0.025), both starting from 1. Measured: $\\phi$ at $t = 1$ only, on every second point of the 51³ grid. PINNStudio: 4 × 128 network on periodic input features, Adam 20,000 → L-BFGS 20,000. Found $\\lambda$ = 9.969, $\\lambda\\varepsilon^2$ = 0.02498 (errors 0.31% and 0.07%). Shown: $\\phi$ at $t = 0.5$, a time with no measured data, in the cube with one corner cut away so the drop inside is visible ($L_2$ over the cube 3.7 × 10⁻⁴); the drop's radius over time follows the reference within 0.02% ([radius plot](../assets/Images/3D_AllenCahn-Wight/radius.png)).

![Inverse examples: Burgers 1D, linear elasticity 2D, Allen-Cahn 3D — PINNStudio, reference and the unknown parameters](figures/paper_inverse.png{_ver(PAPER_INV)})

**Caption for the paper:**

> **Figure Y.** Inverse problems solved with PINNStudio. (a) Burgers equation in 1D: $u(x, t)$, with $\\lambda_1$ and $\\lambda_2$ learned from $u$ measured at $t = 1$ only. (b) Linear elasticity in 2D: the displacement $u_x(x, y)$, with the Lamé constants $\\lambda$ and $\\mu$ learned from the measured displacements. (c) Allen-Cahn equation in 3D: $\\phi$ at $t = 0.5$ in the cube $[0, 1]^3$ with one corner cut away, with $\\lambda$ and $\\lambda\\varepsilon^2$ learned from $\\phi$ measured at $t = 1$ only. Left to right: PINNStudio's prediction, the reference solution and the relative error $|p - p_{{\\mathrm{{true}}}}|/p_{{\\mathrm{{true}}}}$ of each unknown during training. Final values: $\\lambda_1$ = 0.99994 and $\\lambda_2$ = 0.010010 (true 1 and 0.01); $\\lambda$ = 0.99974 and $\\mu$ = 0.50009 (true 1 and 0.5); $\\lambda$ = 9.969 and $\\lambda\\varepsilon^2$ = 0.02498 (true 10 and 0.025).

*PDF: [`figures/paper_inverse.pdf`](figures/paper_inverse.pdf).*

</details>
"""
    return fwd + "\n" + inv


def _ver(path):
    """?v=<content hash>: a new figure gets a new link, so browsers do not show a cached old one."""
    import hashlib
    try:
        return "?v=" + hashlib.md5(open(path, "rb").read()).hexdigest()[:8]
    except OSError:
        return ""


def _paper_names():
    """The examples of the paper's figure (benchmarks/paper_figure_examples.txt: one name per line)."""
    try:
        return [ln.strip() for ln in open(PAPER_LIST, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    except OSError:
        return []


def _results_table(recs):
    """Every example in one table: the reference and PINNStudio's best, forward and inverse."""
    L = ["| Example | Source | Forward: reference L₂ | Forward: PINNStudio best L₂ | L₂, all times | Time (s) | "
         "Inverse: parameters found (true) | Worst parameter error | Time (s) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        for r in [x for x in recs if x["group"] == g]:
            e = r["entry"]
            anc = "bench-" + re.sub(r"[^a-z0-9]+", "-", e.name.lower()).strip("-")
            src = "DeepXDE" if r["deepxde"] else ("Paper" if r["paper"] else "PINNStudio")
            fref = r["f_ref"]
            if r["paper"] and r["paper_runs"]:
                pm = [r["paper_runs"][v]["l2"] for _, _, v in r["paper"] if v in r["paper_runs"]]
                fref = min(pm) if pm else fref
            fs = r.get("f_best_s")
            ft = None
            if r.get("f_best_paper"):
                ft = r["paper_runs"][r["f_best_paper"]]["seconds"]
            elif fs:
                ft = mtp._med(fs["variants"][fs["chosen"]], lambda q: q["seconds"])
            st = None
            if fs:
                vals = [SPACETIME.get(f"{ts._slug(q['example'])}/{fs['chosen']}_s{q['seed']}") for q in fs["variants"][fs["chosen"]]]
                vals = [v for v in vals if v is not None]
                st = statistics.median(vals) if vals else None
            ist = r.get("i_s")
            it = mtp._med(ist["variants"][ist["chosen"]], lambda q: q["seconds"]) if ist else None
            has_f = r["f_best"] is not None
            has_i = r.get("i_best") is not None or r.get("i_best_par")
            L.append(f"| [{e.name}](#{anc}) | {src} | {sci(fref) if has_f else '–'} | "
                     f"{('**' + sci(r['f_best']) + '**') if has_f else '–'} | {sci(st) if st else '–'} | "
                     f"{f'{ft:.0f}' if ft else '–'} | {_par_txt(r.get('i_best_par')) or ('–' if not has_i else 'field')} | "
                     f"{_pct(_perr(r.get('i_best_par'))) if has_i else '–'} | {f'{it:.0f}' if it else '–'} |")
    return "\n".join(L) + "\n"


def page(recs):
    L = []
    a = L.append
    n_t = sum(1 for r in recs if r["f_best_s"] or r.get("i_s"))
    a("# PINNStudio benchmarks\n")
    a("*Every example of PINNStudio, forward and inverse, with the result of its reference settings and "
      "PINNStudio's best. Updated as the runs finish.*\n")
    a("**How to read an example.** Each example opens to one table for the forward problem and one for the "
      "inverse problem. The first row is the reference: **DeepXDE's own settings** for a DeepXDE example, the "
      "**template's starting settings** for a PINNStudio template (the settings PINNStudio's template was "
      "written with, before the tuning study; for a template from a paper this is not the paper's method, "
      "which has its own row). The next rows are the tuning study's changes, each "
      "tried on its own and then all together: 3× the iterations, a larger network (twice as wide, one more "
      "layer), 3× the training points, an L-BFGS phase after Adam (for inverse problems also 10× the weight of "
      "the measured data and 3× the L-BFGS iterations). The last row, **PINNStudio best**, is the most accurate "
      "run, and when several runs are about as accurate (within 25% of the lowest error) the fastest of them, so "
      "a change that costs time without helping is not chosen; that run is marked 🔴 in the table and its time "
      "is repeated in the last row. For an inverse problem, accuracy is the error of "
      "the learned parameters. These are the **Tuned** settings. Above the tables, two ⚙️ toggles list the settings themselves, each with how to load "
      "it: **Default** — the source's own settings, DeepXDE's or the paper's method, for the problems the source "
      "solves (an inverse problem PINNStudio added appears only under PINNStudio best) — and **PINNStudio best** "
      "(Settings: Tuned). Below the tables, two toggles hold the figures: **Forward results** "
      "(DeepXDE's settings, or for a template from a paper the paper's method run in PINNStudio, then PINNStudio's best) "
      "and **Inverse results** (PINNStudio's best; for DeepXDE's own inverse examples DeepXDE's settings first).\n")
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
      f"learned parameter](figures/benchmarks_summary.png{_ver(FIG)})\n")
    a("*Circles: forward problems; diamonds: inverse problems. Filled: PINNStudio's best; open: DeepXDE's settings, "
      "or the paper's method run in PINNStudio; a ring around a filled marker: the two are the same (no tuning "
      "was needed). PINNStudio's own templates, and paper templates whose method has not run yet, have no open "
      "marker: there is no outside reference to compare with. Right: the error of each "
      "learned parameter, labelled with its symbol. ⭐: the error the paper reports (over all times) for the "
      "templates taken from a paper.*\n")
    a("<details>\n<summary>📋 <em>All results — one table</em></summary>\n\n"
      "*Reference: DeepXDE's settings, the paper's method run in PINNStudio (paper templates) or the template's own "
      "settings. L₂ at the last reference time; all times where computed. Each name links to its section below.*\n\n"
      + _results_table(recs) + "\n</details>\n")
    names = _paper_names()
    if names:
        a("<details>\n<summary>📄 <em>Figure for the paper — " + f"{len(names)} examples" + "</em></summary>\n\n"
          "**Why these examples.** The full figure has 63 examples, which is too many for one page in a paper, "
          "so this one keeps one example of each kind of problem. They go from ODEs to 1D, 2D and 3D, and in each "
          "dimension from steady to time-dependent problems. Together they cover the main equation types: "
          "elliptic (Poisson, Helmholtz, linear elasticity), parabolic (heat, diffusion-reaction, Black-Scholes), "
          "hyperbolic and dispersive (wave, Klein-Gordon, Schrödinger), nonlinear flow (Burgers, Navier-Stokes "
          "in 2D and 3D), stiff phase-field equations (Allen-Cahn, and the fourth-order Cahn-Hilliard), and "
          "the less common fractional and integro-differential operators. Most of them (18 of the 25) have both "
          "a forward and an inverse problem, so one row shows both modes. A few are forward only (the "
          "integro-differential ODE, Poisson 1D, Cahn-Hilliard 1D, Black-Scholes), and Lorenz, the unknown "
          "forcing field and the fractional order α are inverse problems only. The set mixes DeepXDE's examples "
          "with templates taken from papers (Raissi et al.; Wight & Zhao; Mattey & Ghosh; Mathias et al.). The "
          "left panel shows the forward problems only: PINNStudio's best and DeepXDE's settings or the paper's "
          "method. Inverse problems are judged by their parameter error (right panel), not by their L₂, because an "
          "inverse run trains on measured values of the solution (in most examples at every reference time), so "
          "its L₂ is largely a fit to that data rather than a prediction (it is still in the tables, as \"L₂ fit to measured data\"). For "
          "the paper templates, the ⭐ is the error the paper itself reports, where it gives one (Burgers 1D, Allen-Cahn 1D, Cahn-Hilliard 1D, Allen-Cahn 2D of "
          "Mattey & Ghosh). Note that the papers report one error over all times, while the x-axis is the error "
          "at the last reference time, so the ⭐ is a guide rather than an exact like-for-like point; the "
          "all-times errors are in the tables below.\n\n"
          "**For Word.** The figure is drawn at its printed size, 6.5 × 7.5 in (the text width of a Word page "
          "with 1 in margins), with 8.5–9 pt text, so insert it at 100% and it leaves about 1.5 in for the caption "
          "on the same page. The PDF keeps it sharp at any zoom. A caption to start from:\n\n"
          "> *PINNStudio on 25 representative problems, from ODEs to 3D PDEs. Left: relative L₂ error of the "
          "forward problems with PINNStudio's best settings (filled) and with DeepXDE's settings or, for templates "
          "taken from a paper, that paper's method run in PINNStudio (open; a ring where the two are the same). "
          "Stars mark the error reported in the original paper. Right: the relative error of each parameter "
          "identified in the inverse problems, labelled with its symbol.*\n\n"
          "*The list: [`paper_figure_examples.txt`](paper_figure_examples.txt) (one name per line; change a line "
          "to swap an example). PDF: [`figures/paper_figure.pdf`](figures/paper_figure.pdf).*\n\n"
          f"![The paper's figure: 25 examples, forward and inverse](figures/paper_figure.png{_ver(PAPER_FIG)})\n\n"
          "</details>\n")
        a(_paper_examples_md())
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
        if r.get("f_best_paper"):
            ent["forward"] = dict(switches=None, variant=r["f_best_paper"], error=f"L2 {r['f_best']:.1e}")
        elif fs and (fs["chosen"] != "base" or r.get("f_best_is_setup")):
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
    ap.add_argument("--spacetime", default="", help="json file(s), comma-separated: run folder (slug/variant_sSEED) -> error over all times")
    args = ap.parse_args()
    if args.primary:
        for slug, vs in json.load(open(args.primary, encoding="utf-8")).items():
            for variant, runs in vs.items():
                for r in runs:
                    mtp.PRIMARY[(slug, variant, str(r["seed"]))] = r["l2"]
                    if r.get("outs"):
                        OUTS[(slug, variant, str(r["seed"]))] = r["outs"]
    rows = mtp._load(args.results)
    summ = mtp.summarize(rows)
    papers = mtp._load([x for x in args.papers.split(",") if x]) if args.papers else []   # later files first
    spacetime = {}
    for f in [x for x in args.spacetime.split(",") if x]:
        spacetime.update(json.load(open(f, encoding="utf-8")))
    SPACETIME.update(spacetime)
    try:
        readme = json.load(open(mtp.README_RESULTS, encoding="utf-8"))
    except OSError:
        readme = {}
    recs = build(summ, papers, spacetime, readme)
    write_tuned(recs)
    figure(recs, FIG, stars=True)
    names = _paper_names()
    if names:
        by = {r["entry"].name: r for r in recs}
        missing = [n for n in names if n not in by]
        if missing:
            print("paper figure: not in the catalogue:", missing)
        figure([by[n] for n in names if n in by], PAPER_FIG, stars=True, paper=True)
    with open(PAGE, "w", encoding="utf-8") as f:
        f.write(page(recs))
    # the runs whose figures the page shows, for the figure collection on the GPU machine
    need = os.path.join(os.path.dirname(PAGE), "tuning", "figures_needed.txt")
    with open(need, "w", encoding="utf-8") as f:
        f.write("".join(f"{slug} {v}\n" for slug, v in dict.fromkeys(NEEDED)))
    print(f"wrote {PAGE} ({len(recs)} examples)")


if __name__ == "__main__":
    main()
