#!/usr/bin/env python3
"""
Runs the training-methods study (benchmarks/training_methods.md): each run is
a PINNStudio example loaded as the GUI loads it, then switched to one of the
study's setups (see SETUPS), written out as the script PINNStudio would run,
and trained. Several runs share the GPU at once (--jobs).

    python benchmarks/study_runner.py PLAN.json OUT_DIR [--jobs 3] [--smoke]

PLAN.json is a list of {"example": <example key>, "setup": <setup name>,
"name": <folder name>, optional "seed", "windows", "phases", "extra"}. Each
run's folder gets run.py, log.txt and result.json (the relative L2 error of
every output group at every reference time). --smoke cuts iterations and
points so that a whole plan can be checked in minutes.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

# The study's setups. Every example is run with the core ones; a setup is a
# list of switches applied on top of the example's own settings.
SETUPS = {
    "Plain": ["soft_ic", "soft_bc", "whole"],
    "Exact periodicity": ["soft_ic", "exact_bc", "whole"],
    "Exact": ["exact_ic", "exact_bc", "whole"],
    "Exact + RBA": ["exact_ic", "exact_bc", "whole", "rba"],
    "Exact + SA-PINN": ["exact_ic", "exact_bc", "whole", "sa"],
    "Exact + causal": ["exact_ic", "exact_bc", "whole", "causal"],
    "Exact + causal + MsFFN": ["exact_ic", "exact_bc", "whole", "causal", "msffn"],
    "Time-adaptive": ["soft_ic", "exact_bc", "windows"],
    "Time-adaptive + RBA": ["soft_ic", "exact_bc", "windows", "rba"],
    "Time-adaptive + SA-PINN": ["soft_ic", "exact_bc", "windows", "sa"],
    "Template": [],
}
IC_WEIGHT = 100.0


def _phases(cfg):
    return json.loads(cfg.scheduler_phases or "[]")


def _set_ic_weights(cfg, n_out_ic):
    """PDE and BC weights as they are, then IC_WEIGHT for each IC loss term."""
    n_eq = len([e for e in (cfg.pde_expressions or "").split("|")])
    n_bc = len(json.loads(cfg.custom_bc_json or "[]"))
    ph = _phases(cfg)
    for p in ph:
        w = [v for v in p.get("weights", "").split(",") if v.strip()]
        w = (w + ["1.0"] * (n_eq + n_bc))[:n_eq + n_bc] + [str(IC_WEIGHT)] * n_out_ic
        p["weights"] = ",".join(w)
    cfg.scheduler_phases = json.dumps(ph)


def apply_setup(cfg, setup, run):
    setup = setup or "Template"
    n_out = len(cfg.output_names.split(","))
    for sw in SETUPS[setup]:
        if sw == "soft_ic":          # the main output's IC as a loss term
            cfg.ic_hard = ",".join(["False"] * n_out)
            cfg.ic_active = ",".join(["True"] + ["False"] * (n_out - 1))
            _set_ic_weights(cfg, 1)
        elif sw == "exact_ic":
            cfg.ic_hard = ",".join(["True"] + ["False"] * (n_out - 1))
            cfg.ic_active = ",".join(["False"] * n_out)
            _set_ic_weights(cfg, 0)
        elif sw == "soft_bc":
            cfg.periodic_inputs = False
        elif sw == "exact_bc":
            cfg.periodic_inputs = True
        elif sw == "whole":
            cfg.time_adaptive = False
            cfg.adapt_method = "None"
        elif sw == "windows":
            cfg.time_adaptive = True
            cfg.adapt_method = "Time Adaptive"
            cfg.ta_transfer_learning = True
            if run.get("windows"):
                cfg.ta_step_groups = json.dumps(run["windows"])
                cfg.ta_num_steps = sum(g["steps"] for g in run["windows"])
        elif sw == "rba":
            cfg.point_weights, cfg.point_weights_lbfgs = "RBA", "keep"
        elif sw == "sa":
            cfg.point_weights, cfg.point_weights_lbfgs, cfg.sa_normalize = "SA-PINN", "keep", False
        elif sw == "causal":
            cfg.causal_training, cfg.causal_anneal = True, True
        elif sw == "msffn":
            cfg.network_type, cfg.msffn_sigmas = "MsFFN", "1, 5"
    if "weighting" not in run.get("extra", {}) and not any(s in SETUPS[setup] for s in ("rba", "sa")) and setup != "Template":
        cfg.point_weights = "None"
    if "causal" not in SETUPS[setup] and setup != "Template":
        cfg.causal_training = False
    if "msffn" not in SETUPS[setup] and setup != "Template":
        cfg.network_type = "FNN"
    if run.get("phases"):            # [[adam_iters, lbfgs_iters]] per window / whole run
        a, l = run["phases"]
        ph = _phases(cfg)
        for p in ph:
            p["iterations"] = a if p["optimizer"] == "adam" else l
        cfg.scheduler_phases = json.dumps(ph)
    for k, v in run.get("extra", {}).items():
        if k == "weighting":
            continue
        assert hasattr(cfg, k), k
        setattr(cfg, k, v)
    if "seed" in run:
        cfg.random_seed = run["seed"]


def _smoke(cfg):
    ph = _phases(cfg)
    for p in ph:
        p["iterations"] = 30 if p["optimizer"] == "adam" else 5
    cfg.scheduler_phases = json.dumps(ph)
    cfg.num_domain = min(cfg.num_domain, 500)
    cfg.num_boundary = min(getattr(cfg, "num_boundary", 100), 100)
    if hasattr(cfg, "num_initial"):
        cfg.num_initial = min(cfg.num_initial, 100)


def build(plan, out, smoke):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from pinnstudio.ui.main_window import MainWindow
    from pinnstudio.core.codegen import generate_script
    w = MainWindow()
    app.processEvents()
    dirs = []
    for run in plan:
        if "settings" in run:          # an example given by its settings (deepxde_examples style)
            from pinnstudio.core import deepxde_examples as dx
            ex = dx.Example(run["name"], dx.FORWARD, "", run["settings"], "", None, "")
            cfg = dx.build_config(ex, os.path.join(os.path.dirname(HERE), "reference_data"))
        else:
            w.quick_examples_combo.setCurrentText(run["example"])
            app.processEvents()
            assert w.quick_examples_combo.currentText() == run["example"], run["example"]
            cfg = w._build_config()
        d = os.path.join(out, run["name"])
        os.makedirs(d, exist_ok=True)
        cfg.save_dir = d
        apply_setup(cfg, run["setup"], run)
        if smoke:
            _smoke(cfg)
        errs = cfg.validate()
        meta = dict(run, ic_hard=cfg.ic_hard, ic_active=cfg.ic_active, periodic=cfg.periodic_inputs,
                    ta=cfg.time_adaptive, windows=cfg.ta_step_groups if cfg.time_adaptive else None,
                    point_weights=cfg.point_weights, causal=cfg.causal_training, net=cfg.network_type,
                    phases=[(p["optimizer"], p["iterations"], p.get("weights", "")) for p in _phases(cfg)],
                    validate=errs)
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print(f"== {run['name']}: {run.get('setup')} | {meta['phases']} ta={cfg.time_adaptive} "
              f"ic_hard={cfg.ic_hard} per={cfg.periodic_inputs} validate={errs}", flush=True)
        if errs:
            continue
        open(os.path.join(d, "run.py"), "w").write(generate_script(cfg))
        dirs.append(d)
    w.close()
    return dirs


def parse(log):
    """{group: [[t, L2], ...]} from the Error Analysis lines of a log."""
    res, group = {}, "u"
    for line in log.splitlines():
        m = re.search(r"Output group[^:]*:\s*(\S+)", line) or re.search(r"── (\S+) ──", line)
        if m:
            group = m.group(1).strip("'\":,")
        m = re.search(r"t=([-\d.eE+]+)\s+—\s+L2=([-\d.eE+]+)", line)
        if m:
            res.setdefault(group, []).append([float(m.group(1)), float(m.group(2))])
    return res


def run_one(d):
    t0 = time.time()
    p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True)
    log = p.stdout + "\n--- stderr ---\n" + p.stderr
    open(os.path.join(d, "log.txt"), "w").write(log)
    res = dict(rc=p.returncode, seconds=round(time.time() - t0), l2=parse(p.stdout))
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
    first = res["l2"].get("u") or next(iter(res["l2"].values()), [])
    print(f"-- {os.path.basename(d)} rc={p.returncode} {res['seconds']} s  "
          + "  ".join(f"t={t:g}:{v:.2e}" for t, v in first), flush=True)
    if p.returncode:
        print(p.stderr[-1200:], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    plan = json.load(open(a.plan))
    dirs = build(plan, out, a.smoke)
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(run_one, dirs))


if __name__ == "__main__":
    main()
