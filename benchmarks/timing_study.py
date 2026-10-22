#!/usr/bin/env python3
"""
Timing study: the same problem solved in PINNStudio and as a plain script.

Each problem is a PINNStudio example, loaded as the window loads it, with
its training phases cut to a fixed number of iterations (the same in every
mode). It is then solved

  script   the plain DeepXDE script (File -> Export as DeepXDE Script),
           run with python
  solve    the script that Solve runs, run with python (no window)
  window   Solve in the PINNStudio window: the same script, plus the
           window's live loss plot, log and checks while it trains
  window-quiet  as window, with the live loss plot and log updates off
                (Settings: "Live loss plot while training" off)

on the CPU and on the GPU. Two times are taken: training alone (the sum of
DeepXDE's "'train' took ... s" lines) and the whole run (start to finish,
with Python start-up, plots and saving).

    python benchmarks/timing_study.py OUT.json [--device cpu|gpu|both]
           [--modes script,solve,window] [--repeat 2] [--only NAME]
"""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# name, (category, example), iterations per phase {optimizer: n}
PROBLEMS = [
    ("Heat 1D", ("Forward", "Heat 1D – Dirichlet"), {"adam": 5000, "lbfgs": 1000}),
    ("Burgers 1D", ("Forward", "Burgers 1D – Raissi et al."), {"adam": 5000, "lbfgs": 1000}),
    ("Navier-Stokes 2D (Kovasznay)", ("Forward", "Navier-Stokes 2D – Kovasznay flow"),
     {"adam": 5000, "lbfgs": 1000}),
    ("Allen-Cahn 1D (causal, MsFFN)", ("Forward", "Allen-Cahn 1D – Wight & Zhao (whole interval)"),
     {"adam": 5000, "lbfgs": 1000}),
    ("Diffusion 1D, inverse", ("Inverse", "Diffusion 1D – unknown coefficient"),
     {"adam": 5000, "lbfgs": 1000}),
]

TRAIN_RE = re.compile(r"'train' took ([\d.]+) s")


def train_seconds(log):
    return round(sum(float(x) for x in TRAIN_RE.findall(log)), 2)


STEP_RE = re.compile(r"^\s*(\d+)\s+\[", re.M)


def steps_taken(log):
    """The last step DeepXDE reports (its step count runs on across the
    phases): L-BFGS can stop before its limit, so runs are compared per step."""
    found = [int(x) for x in STEP_RE.findall(log)]
    return max(found) if found else None


def _env(device):
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", DDE_BACKEND="pytorch")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def _window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox
    app = QApplication.instance() or QApplication(["pinnstudio"])
    QMessageBox.exec = lambda self: 0
    QDialog.exec = lambda self: QDialog.DialogCode.Accepted.value
    for n in ("question", "warning", "information"):
        setattr(QMessageBox, n, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1600, 950)
    w.show()
    app.processEvents()
    return app, w


def load(app, w, example):
    cat, name = example
    w.quick_examples_category_combo.setCurrentText(cat)
    app.processEvents()
    i = w.quick_examples_combo.findText(name)
    assert i >= 0, f"no example {name!r}"
    w.quick_examples_combo.setCurrentIndex(i)
    for _ in range(20):
        app.processEvents()


def config_for(w, iters, out):
    cfg = w._build_config()
    phases = json.loads(cfg.scheduler_phases or "[]")
    for p in phases:
        p["iterations"] = iters.get(p.get("optimizer"), iters["adam"])
    cfg.scheduler_phases = json.dumps(phases)
    cfg.iterations, cfg.iterations2 = iters["adam"], iters["lbfgs"]
    cfg.save_dir = out
    return cfg


def run_script(text, device, workdir, profile=False):
    path = os.path.join(workdir, "run.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    prof = os.path.join(workdir, "run.prof")
    cmd = [sys.executable] + (["-m", "cProfile", "-o", prof] if profile else []) + [path]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=workdir, env=_env(device),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    total = time.perf_counter() - t0
    log = p.stdout + "\n" + p.stderr
    with open(os.path.join(workdir, "log.txt"), "w", encoding="utf-8") as f:
        f.write(log)
    r = dict(rc=p.returncode, total=round(total, 2), train=train_seconds(log), steps=steps_taken(log))
    if profile and os.path.exists(prof):
        import io
        import pstats
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(25)
        r["profile"] = buf.getvalue()[-6000:]
    return r


def run_window(app, w, cfg, device, quiet=False):
    """Solve in the window, timed from the click until the window shows the
    results (Solve enabled again)."""
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["PYTHONUNBUFFERED"] = "1"
    if hasattr(w, "live_loss_cb"):
        w.live_loss_cb.setChecked(not quiet)
    real_build = w._build_config
    w._build_config = lambda: copy.deepcopy(cfg)     # Solve uses this run's settings
    t0 = time.perf_counter()
    try:
        w._on_solve()
    finally:
        w._build_config = real_build
    while not w.solve_btn.isEnabled():
        app.processEvents()
        time.sleep(0.01)
    total = time.perf_counter() - t0
    log = w.log_box.toPlainText()
    ok = "Training complete" in log
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    return dict(rc=0 if ok else 1, total=round(total, 2), train=train_seconds(log), steps=steps_taken(log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--device", default="both", choices=["cpu", "gpu", "both"])
    ap.add_argument("--modes", default="script,solve,window")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--profile", action="store_true", help="run the scripts under cProfile (slower)")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads for PyTorch (OMP_NUM_THREADS); 0 = PyTorch's default (every core)")
    ap.add_argument("--tag", default="", help="added to the mode name in the results")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the iterations (a quick check: 0.05)")
    ap.add_argument("--cpu-scale", type=float, default=0.2,
                    help="multiply the iterations again on the CPU, which is many times slower")
    a = ap.parse_args()
    devices = ["cpu", "gpu"] if a.device == "both" else [a.device]
    if a.threads:
        os.environ["OMP_NUM_THREADS"] = str(a.threads)      # read by PyTorch in the training process
    modes = a.modes.split(",")
    from pinnstudio.core.codegen import generate_clean_script, generate_script
    app, w = _window()
    results = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["problem"], r["device"], r["mode"], r["rep"]) for r in results}
    base = tempfile.mkdtemp(prefix="pinnstudio_timing_")
    for name, example, iters in PROBLEMS:
        if a.only and a.only.lower() not in name.lower():
            continue
        load(app, w, example)
        base_iters = iters
        for device in devices:
            f = a.scale * (a.cpu_scale if device == "cpu" else 1.0)
            iters = {k: max(1, int(v * f)) for k, v in base_iters.items()}
            for rep in range(a.repeat):
                for mode in modes:
                    if (name, device, mode + a.tag, rep) in done:
                        continue
                    d = os.path.join(base, f"{len(results):03d}")
                    os.makedirs(d)
                    cfg = config_for(w, iters, d)
                    if mode == "script":
                        r = run_script(generate_clean_script(cfg), device, d, a.profile)
                    elif mode == "solve":
                        r = run_script(generate_script(cfg), device, d, a.profile)
                    elif mode in ("window", "window-quiet"):
                        r = run_window(app, w, cfg, device, quiet=(mode == "window-quiet"))
                    else:
                        raise SystemExit(f"unknown mode {mode}")
                    r.update(problem=name, device=device, mode=mode + a.tag, rep=rep, iters=iters, threads=a.threads)
                    results.append(r)
                    print(f"{name:32s} {device:3s} {mode:12s} rep {rep}: train {r['train']:8.2f} s  "
                          f"total {r['total']:8.2f} s  rc={r['rc']}", flush=True)
                    with open(a.out, "w") as f:
                        json.dump(results, f, indent=1)
    w.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)     # skip Qt's teardown of the window at interpreter exit (it can crash there)
