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


def _maxl2(r):
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


def figure(summ, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = [x for x in summ if x["base"] and x["tuned"]][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(s) + 1.2), dpi=200)
    y = range(len(s))
    for i, x in enumerate(s):
        ax.plot([x["tuned"], x["base"]], [i, i], color="#c9ccd1", lw=2, zorder=1, solid_capstyle="round")
        for key, rk, col in (("base", "base_rng", "#8a9099"), ("tuned", "tuned_rng", "#2266cc")):
            if x[rk]:
                ax.plot(x[rk], [i, i], color=col, lw=1.2, alpha=0.6, zorder=2)
    ax.scatter([x["base"] for x in s], list(y), s=34, color="#8a9099", zorder=3, label="Example's own settings",
               edgecolors="white", linewidths=1)
    ax.scatter([x["tuned"] for x in s], list(y), s=34, color="#2266cc", zorder=4, label="Tuned in PINNStudio",
               edgecolors="white", linewidths=1)
    ax.set_xscale("log")
    ax.set_yticks(list(y))
    ax.set_yticklabels([_title(x["name"], x["inverse"]).replace(" (PINNStudio template)", " (template)")
                        for x in s], fontsize=7.5)
    ax.set_xlabel("Relative $L_2$ error (worst output and time; median over seeds, line = range)", fontsize=8)
    ax.grid(axis="x", which="major", color="#e6e8eb", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(loc="lower left", fontsize=7.5, frameon=False, bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
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
    a(f"So far {len(summ)} problems are included, chosen as the ones with the largest errors in the "
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
      "picks a less accurate variant.")
    a("- **The error** is the relative $L_2$ error from PINNStudio's Error Analysis, the worst over the outputs "
      "and the reference times, against the reference data the example ships with. For inverse problems the "
      "worst relative error of the learned parameters is given as well.")
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
        a("| Variant | Final training loss | $L_2$ (worst) | Parameter error | Time (s) |")
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
        if s.get("file"):
            a(f"Tuned settings: [`{s['file']}`](tuning/{s['file'].replace(' ', '%20')})\n")
        a("</details>\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--no-files", action="store_true", help="do not write the tuned problem files")
    args = ap.parse_args()
    rows = _load(args.results)
    summ = summarize(rows)
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
