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


def apply_variant(cfg, variant):
    """Return a changed copy of cfg, or None when the variant does not apply."""
    cfg = copy.deepcopy(cfg)
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
    l2 = [float(x) for x in re.findall(r"L2=([0-9.eE+-]+)", out)]
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
        true = {v["name"]: v.get("true") for v in json.loads(base.inverse_variables_json or "[]")}
        for variant in variants:
            cfg0 = apply_variant(base, variant)
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
                with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as fh:
                    fh.write(generate_script(cfg))
                t0 = time.time()
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
                           points=[cfg.num_domain, cfg.num_boundary, cfg.num_initial])
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
