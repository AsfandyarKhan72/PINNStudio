#!/usr/bin/env python3
"""Every example, Forward and Inverse, loaded as the window loads it and run
briefly, as Solve runs it and as the exported plain script.

For each example: the settings must pass validation, both scripts must be
written, and both must train to the end (a few iterations, fewer points)
and exit without an error. A summary table is printed and saved as JSON.

    python tools/all_examples_check.py OUT.json [--jobs 4] [--only TEXT] [--modes solve,script]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def _short(cfg):
    phases = json.loads(cfg.scheduler_phases or "[]")
    for p in phases:
        p["iterations"] = 40 if p.get("optimizer") != "lbfgs" else 5
    cfg.scheduler_phases = json.dumps(phases)
    cfg.iterations, cfg.iterations2 = 40, 5
    cfg.num_domain = min(cfg.num_domain, 400)
    cfg.num_boundary = min(getattr(cfg, "num_boundary", 100), 100)
    if hasattr(cfg, "num_initial"):
        cfg.num_initial = min(cfg.num_initial, 100)
    cfg.num_test = min(getattr(cfg, "num_test", 1000) or 1000, 500)
    if getattr(cfg, "time_adaptive", False) and not (getattr(cfg, "ta_step_groups", "") or "").strip():
        cfg.ta_num_steps = min(cfg.ta_num_steps, 2)
    cfg.rar_cycles = min(cfg.rar_cycles, 1)
    cfg.rar_candidates = min(cfg.rar_candidates, 2000)
    cfg.rar_add_points = min(cfg.rar_add_points, 50)
    cfg.rar_adam_iters, cfg.rar_lbfgs_iters = min(cfg.rar_adam_iters, 20), min(cfg.rar_lbfgs_iters, 5)
    cfg.ic_pretrain_iterations = min(cfg.ic_pretrain_iterations, 20)
    cfg.ta_grid_size = min(cfg.ta_grid_size, 41)
    return cfg


def build(out, only):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox
    app = QApplication.instance() or QApplication(["pinnstudio"])
    QMessageBox.exec = lambda self: 0
    QDialog.exec = lambda self: QDialog.DialogCode.Accepted.value
    for n in ("question", "warning", "information", "critical"):
        setattr(QMessageBox, n, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    from pinnstudio.core.codegen import generate_clean_script, generate_script
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    app.processEvents()
    runs = []
    cats = w.quick_examples_category_combo
    for ci in range(cats.count()):
        cat = cats.itemText(ci)
        cats.setCurrentIndex(ci)
        app.processEvents()
        names = [w.quick_examples_combo.itemText(i) for i in range(w.quick_examples_combo.count())]
        for name in names:
            if name in ("None", "") or name.startswith("──"):
                continue
            if only and only.lower() not in f"{cat} {name}".lower():
                continue
            rec = dict(category=cat, example=name)
            try:
                w.quick_examples_combo.setCurrentIndex(w.quick_examples_combo.findText(name))
                for _ in range(10):
                    app.processEvents()
                cfg = w._build_config()
                rec["validate"] = cfg.validate()
                d = os.path.join(out, f"{len(runs):03d}")
                os.makedirs(d, exist_ok=True)
                cfg.save_dir = d
                _short(cfg)
                for mode, gen in (("solve", generate_script), ("script", generate_clean_script)):
                    try:
                        with open(os.path.join(d, f"{mode}.py"), "w", encoding="utf-8") as f:
                            f.write(gen(cfg))
                    except Exception as e:      # the generator itself failed
                        rec[f"{mode}_gen_error"] = f"{type(e).__name__}: {e}"
                rec["dir"] = d
            except Exception as e:
                rec["load_error"] = f"{type(e).__name__}: {e}"
            runs.append(rec)
            print(f"built {cat:8s} {name}", flush=True)
    w.close()
    return runs


def run_one(rec, mode):
    path = os.path.join(rec["dir"], f"{mode}.py")
    if not os.path.isfile(path):
        return
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, path], cwd=rec["dir"], env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=900)
        rc, log = p.returncode, p.stdout + "\n--- stderr ---\n" + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, log = "timeout", str(e.stdout or "")[-3000:]
    with open(os.path.join(rec["dir"], f"{mode}_log.txt"), "w", encoding="utf-8") as f:
        f.write(log)
    rec[f"{mode}_rc"] = rc
    rec[f"{mode}_s"] = round(time.time() - t0, 1)
    if rc != 0:
        tb = [l for l in log.splitlines() if l.strip()]
        rec[f"{mode}_error"] = "\n".join(tb[-6:])
    print(f"{'ok ' if rc == 0 else 'ERR'} {mode:6s} {rec['category']:8s} {rec['example']}  ({rec[f'{mode}_s']} s)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--modes", default="solve,script")
    a = ap.parse_args()
    base = tempfile.mkdtemp(prefix="pinnstudio_examples_")
    runs = build(base, a.only)
    tasks = [(r, m) for r in runs if "dir" in r for m in a.modes.split(",")]
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda t: run_one(*t), tasks))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=1)
    bad = [r for r in runs if r.get("validate") or r.get("load_error") or any(
        k.endswith("_error") for k in r)]
    print(f"\n{len(runs)} examples, {len(bad)} with a problem")
    for r in bad:
        print(f"\n## {r['category']} / {r['example']}")
        for k, v in r.items():
            if k.endswith("error") or (k == "validate" and v):
                print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)     # skip Qt's teardown of the window at interpreter exit (it can crash there)
