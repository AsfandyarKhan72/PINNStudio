#!/usr/bin/env python3
"""Tuning study: does ordinary tuning in PINNStudio improve DeepXDE's examples?

Each example runs with DeepXDE's own settings ("base") and with the same
fixed set of changes for every example:

    iters   3x the iterations of every training phase
    net     hidden layers twice as wide (at most 512) and one layer more
    points  3x the domain, boundary and initial points
    lbfgs   an L-BFGS phase after Adam (only when the example has none)
    all     every change above together

The best variant is chosen by its final training loss, not by the error
against the reference solution, so the choice does not look at the answer.
Its L2 errors (and, for inverse problems, the inferred parameters) are
then reported next to the base run. --seeds runs each variant with several
random seeds.

    python tools/tuning_study.py --out runs/tuning --names "Klein-Gordon equation|Heat equation"
    python tools/tuning_study.py --out runs/tuning --names "..." --variants base,all --seeds 1,2,3
    python tools/tuning_study.py --summary runs/tuning

Results are appended to <out>/results.jsonl, one line per run, so an
interrupted study continues where it stopped.
"""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from pinnstudio.core import deepxde_examples as dx  # noqa: E402
from pinnstudio.core.codegen import generate_script  # noqa: E402

REF_ROOT = os.path.join(_REPO, "reference_data")
VARIANTS = ["base", "iters", "net", "points", "lbfgs", "all"]
# inverse problems: changes aimed at the unknown parameters
INVERSE_VARIANTS = ["dataw", "lbfgs3", "times", "setup", "invall"]
TA_VARIANTS = ["windows2"]

# The best setup of each PINNStudio template in the training methods study
# (benchmarks/training_methods.md), as switches of benchmarks/study_runner.py.
# --best starts the forward runs from it. BEST_WHOLE is the best setup that
# trains on the whole interval, which is what Inverse mode can use; the
# inverse "setup" variant switches to it.
BEST_FORWARD = {
    "2D Allen-Cahn (Wight & Zhao)": ["soft_ic", "exact_bc", "windows", "rba"],
    "2D Allen-Cahn (Mattey & Ghosh)": ["exact_ic", "exact_bc", "whole", "sa"],
    "1D Cahn-Hilliard (Mattey & Ghosh)": ["soft_ic", "exact_bc", "windows", "sa"],
    "1D Burgers": ["whole", "causal"],
}
BEST_WHOLE = {
    "2D Allen-Cahn (Wight & Zhao)": ["exact_ic", "exact_bc", "whole", "rba"],
    "2D Allen-Cahn (Mattey & Ghosh)": ["exact_ic", "exact_bc", "whole", "sa"],
    "1D Burgers": ["whole", "causal"],
}


def apply_switches(cfg, switches):
    """study_runner's setup switches on a copy of cfg."""
    sys.path.insert(0, os.path.join(_REPO, "benchmarks"))
    import study_runner as sr
    cfg = copy.deepcopy(cfg)
    key = "_tuning_" + "_".join(switches)
    sr.SETUPS[key] = list(switches)
    sr.apply_setup(cfg, key, {})
    return cfg
LBFGS_PHASE = {"optimizer": "lbfgs", "iterations": 15000, "lr": 0.001, "decay_type": "none"}


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _phases(cfg):
    if cfg.optimizer_scheduler and cfg.scheduler_phases:
        return json.loads(cfg.scheduler_phases)
    ph = [{"optimizer": cfg.optimizer, "iterations": cfg.iterations, "lr": cfg.learning_rate, "decay_type": "none"}]
    if cfg.optimizer2 and cfg.optimizer2 != "none":
        ph.append({"optimizer": cfg.optimizer2, "iterations": cfg.iterations2, "lr": cfg.learning_rate,
                   "decay_type": "none"})
    return ph


def _set_phases(cfg, ph):
    cfg.optimizer_scheduler = True
    cfg.scheduler_phases = json.dumps(ph)


def _has_lbfgs(cfg):
    return any(p.get("optimizer", "").lower().startswith("l") for p in _phases(cfg))


def apply_variant(cfg, variant, tkey=None):
    """Return a changed copy of cfg, or None when the variant does not apply."""
    if variant == "plain" or variant.startswith("paper_"):     # the papers comparison (paper_methods.py)
        import paper_methods
        return paper_methods.apply(cfg, variant, tkey)
    cfg = copy.deepcopy(cfg)
    if variant == "windows2":                        # twice the time windows
        if not cfg.time_adaptive or not (cfg.ta_step_groups or "").strip():
            return None
        groups = json.loads(cfg.ta_step_groups)
        for g in groups:
            g["steps"] = int(g["steps"]) * 2
        cfg.ta_step_groups = json.dumps(groups)
        cfg.ta_num_steps = sum(g["steps"] for g in groups)
        return cfg
    if variant == "setup":                           # the best whole-interval setup (inverse)
        if not tkey or tkey not in BEST_WHOLE or not (cfg.inverse_obs_files_json or "").strip():
            return None
        return apply_switches(cfg, BEST_WHOLE[tkey])
    if variant == "times":                           # measured data at every reference time after t = 0
        obs = json.loads(cfg.inverse_obs_files_json or "[]")
        if not obs:
            return None
        import glob as _glob
        new = []
        for o in obs:
            folder, fname = os.path.split(o["path"])
            m = re.match(r"t_([0-9.]+)(.*)\.txt$", fname)
            if not m:
                new.append(o)
                continue
            for f in sorted(_glob.glob(os.path.join(folder, "t_*" + m.group(2) + ".txt"))):
                m2 = re.match(r"t_([0-9.]+)" + re.escape(m.group(2)) + r"\.txt$", os.path.basename(f))
                if m2 and float(m2.group(1)) > 0 and not any(x["path"] == f for x in new):
                    new.append(dict(o, path=f))
        if len(new) <= len(obs):
            return None
        cfg.inverse_obs_files_json = json.dumps(new)
        return cfg
    if variant in ("dataw", "lbfgs3", "invall"):
        if not (cfg.inverse_obs_files_json or "").strip():
            return None
        if variant in ("dataw", "invall"):           # 10x the weight of every measured-data term
            obs = json.loads(cfg.inverse_obs_files_json)
            for o in obs:
                o["weight"] = float(o.get("weight", 1.0)) * 10
            cfg.inverse_obs_files_json = json.dumps(obs)
        if variant in ("lbfgs3", "invall"):          # 3x the L-BFGS phase, or add one
            ph = _phases(cfg)
            lb = [p for p in ph if p.get("optimizer", "").lower().startswith("l")]
            if lb:
                for p in lb:
                    p["iterations"] = int(p["iterations"]) * 3
            else:
                ph.append(dict(LBFGS_PHASE, iterations=45000))
            _set_phases(cfg, ph)
        if variant == "invall":                      # and the wider, deeper network
            cfg = apply_variant(cfg, "net")
        return cfg
    parts = ["iters", "net", "points", "lbfgs"] if variant == "all" else [variant]
    if variant == "lbfgs" and _has_lbfgs(cfg):
        return None
    for part in parts:
        if part == "base":
            pass
        elif part == "iters":
            ph = _phases(cfg)
            for p in ph:
                p["iterations"] = int(p["iterations"]) * 3
            _set_phases(cfg, ph)
            cfg.rar_adam_iters *= 3
            cfg.rar_lbfgs_iters *= 3
        elif part == "net":
            lay = list(cfg.layers)
            hidden = [min(512, int(h) * 2) for h in lay[1:-1]]
            hidden.append(hidden[-1])
            cfg.layers = [lay[0]] + hidden + [lay[-1]]
        elif part == "points":
            cfg.num_domain *= 3
            cfg.num_boundary *= 3
            cfg.num_initial *= 3
        elif part == "lbfgs":
            if not _has_lbfgs(cfg):
                ph = _phases(cfg)
                ph.append(dict(LBFGS_PHASE))
                _set_phases(cfg, ph)
    return cfg


def _parse(out):
    l2 = [float(x) for x in re.findall(r"L(?:2|₂)=([0-9.eE+-]+)", out)]
    loss = re.findall(r"Final loss: ([0-9.eE+-]+)", out)
    params = {n: float(v) for n, v in re.findall(r"=== (\w+) final value: (\S+) ===", out)}
    return l2, (float(loss[-1]) if loss else None), params


def _done(path):
    seen = set()
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
                seen.add((r["example"], r["inverse"], r["variant"], r["seed"]))
            except (ValueError, KeyError):
                pass
    return seen


TEMPLATE_PREFIX = "Template: "
_app = None


def template_config(key, inverse):
    """The settings PINNStudio's own template `key` loads with, read from the
    GUI (offscreen) the way the Examples list loads it."""
    global _app
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(["pinnstudio"])
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    _app.processEvents()
    if inverse:
        w.quick_examples_category_combo.setCurrentText("Inverse")
        _app.processEvents()
    else:
        w.quick_examples_category_combo.setCurrentText("Forward")
        _app.processEvents()
    combo = w.quick_examples_combo
    i = combo.findData(key)
    if i < 0:
        raise SystemExit(f"unknown template {key!r} ({'Inverse' if inverse else 'Forward'} list)")
    combo.setCurrentIndex(i)
    combo.activated.emit(i)
    t0 = time.time()
    while time.time() - t0 < 1.0:
        _app.processEvents()
    cfg = w._build_config()
    w.close()
    return cfg


# ---------------------------------------------------------------- GUI mode
# --gui runs every case through the PINNStudio window itself (offscreen): the
# example is picked from the Examples list, the variant is put into the window
# the way an opened problem file is, Solve is pressed, and the results are read
# from the Training Log. Differences between the window's settings and the
# expected ones are recorded as "gui_diffs" -- each one is a possible bug.
_IGNORE = {"save_dir", "sweep_save_dir", "sweep_label", "random_seed", "gpu_device_index", "gpu_memory_fraction",
           "train_device", "cpu_threads", "plot_colormap", "template_type"}
_WIN = None


def _window():
    global _WIN, _app
    if _WIN is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        _app = QApplication.instance() or QApplication(["pinnstudio"])
        from pinnstudio.ui.main_window import MainWindow
        _WIN = MainWindow()
        _WIN._maybe_save_changes = lambda: True
        # the study starts from each example's own settings: Examples -> Settings (Default would load
        # a paper's method for its templates, Tuned the study's own result) stays out of it
        _WIN._apply_example_settings_mode = lambda key: None
        _pump(1.0)
    return _WIN


def _pump(seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        _app.processEvents()
        time.sleep(0.02)


def _pick(w, key, inverse):
    w.quick_examples_category_combo.setCurrentText("Inverse" if inverse else "Forward")
    _pump(0.2)
    combo = w.quick_examples_combo
    i = combo.findData(key)
    if i < 0 and not inverse:          # an inverse-only DeepXDE example is in the Inverse list
        w.quick_examples_category_combo.setCurrentText("Inverse")
        _pump(0.2)
        i = combo.findData(key)
    if i < 0:
        raise SystemExit(f"{key!r} is not in the {'Inverse' if inverse else 'Forward'} list")
    combo.setCurrentIndex(i)
    combo.activated.emit(i)
    _pump(1.0)


def _norm(v):
    """A setting in a comparable form: JSON and Python-literal strings parsed,
    numbers as floats, so "[1, 2]" and "[1,2]" or 2 and 2.0 compare equal."""
    import ast
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{(":
            for parse in (json.loads, ast.literal_eval):
                try:
                    return _norm(parse(s))
                except (ValueError, SyntaxError, TypeError):
                    pass
        return s
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 12)
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _same(a, b):
    """a and b mean the same: a dict in b may carry extra keys that a leaves
    to their defaults (the window writes every key out)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return all(k in b and _same(v, b[k]) for k, v in a.items())
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


# settings the window fills with its defaults when an example leaves them unset;
# they only take effect when the feature they belong to is switched on
_FILLED = {"bc_boundary_json", "bc_edge_json", "sweep_parameters", "ta_step_groups", "inverse_variables_json",
           "inverse_obs_files_json", "inverse_data_file", "inverse_param_init", "inverse_param_name", "ic_hard",
           "loss_weights_multi"}


def _diffs(expected, got):
    import dataclasses
    a, b = dataclasses.asdict(expected), dataclasses.asdict(got)
    out = {}
    for k in a:
        if k in _IGNORE or k.startswith(("plot_", "fig_", "line_", "surface_", "anim_")):
            continue
        va, vb = _norm(a[k]), _norm(b.get(k))
        if k in _FILLED and va in ("", [], None, {}):
            continue
        if k == "loss_weights_multi" and isinstance(va, str) and isinstance(vb, str):
            try:
                if [float(x) for x in va.split(",")] == [float(x) for x in vb.split(",")]:
                    continue
            except ValueError:
                pass
        if _same(va, vb):
            continue
        if isinstance(va, float) and isinstance(vb, (int, float)) and abs(va - vb) <= 1e-12 * max(1, abs(va)):
            continue
        if va != vb:
            out[k] = [str(va)[:200], str(vb)[:200]]
    return out


def _script_diff(expected, got):
    """The lines that differ between the scripts the two configurations
    generate (save folder, seed and device settings made equal): settings
    that differ only in form generate the same script and are not listed."""
    import difflib
    a, b = copy.deepcopy(expected), copy.deepcopy(got)
    for k in _IGNORE:
        if hasattr(b, k):
            setattr(b, k, getattr(a, k))
    sa, sb = generate_script(a).splitlines(), generate_script(b).splitlines()
    return [l for l in difflib.unified_diff(sa, sb, lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))][:40]


def gui_base_config(name, tkey, inverse):
    """The example as the window loads it, and how that differs from the
    example's own definition (DeepXDE examples only)."""
    w = _window()
    _pick(w, tkey or name, inverse)
    cfg = w._build_config()
    loaded_diffs = {}
    if tkey is None:
        ex = {e.name: e for e in dx.EXAMPLES}[name]
        loaded_diffs = _diffs(dx.build_config(ex, REF_ROOT, inverse=inverse), cfg)
    return cfg, loaded_diffs


def gui_solve(cfg, run_dir, timeout):
    """Put cfg into the window, press Solve, wait, return (rc, log text, round-trip diffs)."""
    w = _window()
    w._apply_config(cfg)
    _pump(0.3)
    w.save_dir_input.setText(run_dir)
    got = w._build_config()
    rt = _diffs(cfg, got)
    w._on_solve()
    t0 = time.time()
    while getattr(w, "thread", None) is None or not hasattr(w.thread, "isRunning"):
        _pump(0.2)
        if time.time() - t0 > 30:
            return "not started", w.log_box.toPlainText(), rt
    while w.thread.isRunning():
        _pump(0.5)
        if time.time() - t0 > timeout:
            w.thread.stop()
            _pump(2)
            return "timeout", w.log_box.toPlainText(), rt
    _pump(1.5)                      # let the window finish reading results
    log = w.log_box.toPlainText()
    rc = 0 if "Training complete" in log else 1
    return rc, log, rt


def run(args):
    names = [s for s in args.names.split("|") if s]
    by_name = {e.name: e for e in dx.EXAMPLES}
    templates = [s for s in args.templates.split("|") if s]
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"unknown example(s): {missing}")
    variants = [v for v in args.variants.split(",") if v]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, "results.jsonl")
    seen = _done(res_path)
    jobs = [(n, None) for n in names] + [(TEMPLATE_PREFIX + k, k) for k in templates]
    for name, tkey in jobs:
        if tkey is None:
            ex = by_name[name]
            inverse = args.inverse and ex.inverse is not None
            base = dx.build_config(ex, REF_ROOT, inverse=inverse)
        else:
            inverse = args.inverse
            base = template_config(tkey, inverse)
        loaded_diffs = {}
        if args.gui:
            base, loaded_diffs = gui_base_config(name, tkey, inverse)
            if loaded_diffs:
                print(f"!!  {name[:50]:50s} the window loads it differently from its definition: "
                      f"{json.dumps(loaded_diffs, ensure_ascii=False)[:600]}", flush=True)
        if args.best and tkey in BEST_FORWARD and not inverse:   # start from the study's best setup
            base = apply_switches(base, BEST_FORWARD[tkey])
            name = name + " | best setup"
        true = {v["name"]: v.get("true") for v in json.loads(base.inverse_variables_json or "[]")}
        for variant in variants:
            cfg0 = apply_variant(base, variant, tkey)
            if cfg0 is None:
                print(f"--  {name[:50]:50s} {variant:7s} does not apply", flush=True)
                continue
            for seed in seeds:
                if (name, inverse, variant, seed) in seen:
                    continue
                d = os.path.abspath(os.path.join(args.out, _slug(name) + ("_inv" if inverse else ""),
                                                 f"{variant}_s{seed}"))
                os.makedirs(d, exist_ok=True)
                cfg = copy.deepcopy(cfg0)
                cfg.save_dir = d
                cfg.random_seed = seed
                if args.smoke:
                    ph = _phases(cfg)
                    for p in ph:
                        p["iterations"] = min(int(p["iterations"]), 30)
                    _set_phases(cfg, ph)
                    cfg.rar_cycles = min(cfg.rar_cycles, 1)
                t0 = time.time()
                rt_diffs = {}
                if args.gui:
                    rc, out, rt_diffs = gui_solve(cfg, d, args.timeout)
                    err = ""
                    if rt_diffs:
                        print(f"!!  {name[:50]:50s} {variant}: the window changed settings: "
                              f"{json.dumps(rt_diffs, ensure_ascii=False)[:600]}", flush=True)
                else:
                  with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as fh:
                    fh.write(generate_script(cfg))
                  try:
                    p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True,
                                       timeout=args.timeout)
                    rc, out, err = p.returncode, p.stdout, p.stderr
                  except subprocess.TimeoutExpired as e:
                    rc = "timeout"
                    out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
                    err = ""
                dt = time.time() - t0
                with open(os.path.join(d, "log.txt"), "w", encoding="utf-8") as fh:
                    fh.write(out + "\n--- stderr ---\n" + err)
                l2, loss, params = _parse(out)
                rec = dict(example=name, inverse=inverse, variant=variant, seed=seed, rc=rc, seconds=round(dt, 1),
                           l2=l2, final_loss=loss, params=params, true=true,
                           layers=list(cfg.layers), phases=_phases(cfg),
                           points=[cfg.num_domain, cfg.num_boundary, cfg.num_initial],
                           mode="gui" if args.gui else "script", gui_diffs=rt_diffs, loaded_diffs=loaded_diffs)
                with open(res_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                worst = max(l2) if l2 else float("nan")
                par = " ".join(f"{k}={v:.6g}(true {true.get(k)})" for k, v in params.items())
                print(f"{'OK ' if rc == 0 else 'ERR'} {name[:50]:50s} {variant:7s} s{seed} {dt:7.0f}s "
                      f"loss={loss} maxL2={worst:.3e} {par}", flush=True)


def _param_err(r):
    errs = [abs(v - r["true"][k]) / abs(r["true"][k]) for k, v in r["params"].items()
            if r["true"].get(k) not in (None, 0)]
    return max(errs) if errs else None


def summary(out):
    import statistics
    rows = [json.loads(line) for line in open(os.path.join(out, "results.jsonl"), encoding="utf-8")]
    groups = {}
    for r in rows:
        if r["rc"] == 0 and r["final_loss"] is not None:
            groups.setdefault((r["example"], r["inverse"]), {}).setdefault(r["variant"], []).append(r)

    def med(rs, f):
        vals = [f(r) for r in rs if f(r) is not None]
        return statistics.median(vals) if vals else None

    def fmt(x, p=".2e"):
        return "—" if x is None else format(x, p)
    print("| Example | Base max L2 | Base param err | Chosen (by loss) | Chosen max L2 | Chosen param err | "
          "Best max L2 (any variant) | Time base → chosen (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for (name, inv), vs in groups.items():
        maxl2 = lambda r: max(r["l2"]) if r["l2"] else None  # noqa: E731
        b = vs.get("base", [])
        chosen = min(vs, key=lambda v: med(vs[v], lambda r: r["final_loss"]))
        best = min((med(rs, maxl2) for rs in vs.values() if med(rs, maxl2) is not None), default=None)
        print(f"| {name}{' (inverse)' if inv else ''} | {fmt(med(b, maxl2))} | {fmt(med(b, _param_err))} | "
              f"{chosen} | {fmt(med(vs[chosen], maxl2))} | {fmt(med(vs[chosen], _param_err))} | {fmt(best)} | "
              f"{fmt(med(b, lambda r: r['seconds']), '.0f')} → {fmt(med(vs[chosen], lambda r: r['seconds']), '.0f')} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--names", default="", help="internal example names, '|' separates several")
    ap.add_argument("--templates", default="", help="PINNStudio template names as the Examples list shows them, "
                                                    "'|' separates several")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default="2026")
    ap.add_argument("--inverse", action="store_true", help="run the Inverse-mode version where there is one")
    ap.add_argument("--best", action="store_true",
                    help="forward templates: start from the training methods study's best setup (BEST_FORWARD)")
    ap.add_argument("--gui", action="store_true",
                    help="run every case through the PINNStudio window (offscreen) instead of its script")
    ap.add_argument("--smoke", action="store_true", help="30 iterations per phase, to check that it runs")
    ap.add_argument("--timeout", type=float, default=10800)
    ap.add_argument("--summary", default="", help="print the table for a finished study folder")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary)
    else:
        if not args.out or not (args.names or args.templates):
            raise SystemExit("--out and --names or --templates are needed")
        run(args)


if __name__ == "__main__":
    main()
