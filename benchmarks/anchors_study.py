#!/usr/bin/env python3
"""With and without anchors: is checking the equation at the data points
worth it? Each run is a PINNStudio example loaded as the window loads it,
written out as the script Solve runs, and trained as it is, once with the
data points as anchors and once without.

  inverse  every Inverse example with measured data; inverse_obs_anchors
           True (the default) and False
  ic       Forward examples given an IC file (written from the example's
           own IC formula); ic_file_anchors False (the default) and True

    python benchmarks/anchors_study.py inverse|ic OUT_DIR [--jobs 3] [--seeds 1,2] [--only TEXT] [--smoke]

Each run's folder gets run.py, log.txt and result.json (seconds, the field
error at every reference time and, for Inverse, each parameter's final value
and relative error). OUT_DIR/summary.json collects them.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from study_runner import parse, _smoke  # noqa: E402

IC_EXAMPLES = ["Allen-Cahn 1D – Wight & Zhao (whole interval)", "Heat 2D", "Cahn-Hilliard 2D – Wight & Zhao"]


def _window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox
    app = QApplication.instance() or QApplication(["pinnstudio"])
    QMessageBox.exec = lambda self: 0
    QDialog.exec = lambda self: 0
    for n in ("question", "warning", "information", "critical"):
        setattr(QMessageBox, n, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    app.processEvents()
    return app, w


def _load(app, w, category, name):
    w.quick_examples_category_combo.setCurrentText(category)
    app.processEvents()
    i = w.quick_examples_combo.findText(name)
    if i < 0:
        return None
    w.quick_examples_combo.setCurrentIndex(i)
    for _ in range(5):
        app.processEvents()
    return w._build_config()


def _names(app, w, category):
    w.quick_examples_category_combo.setCurrentText(category)
    app.processEvents()
    c = w.quick_examples_combo
    return [c.itemText(i) for i in range(c.count()) if c.itemText(i) != "None" and not c.itemText(i).startswith("──")]


def _write_ic_file(cfg, path):
    """The example's own IC formula for output 0 on a grid at t_min, in the
    IC file layout (x,t,c / x,y,t,c / x,y,z,t,c)."""
    import numpy as np
    from pinnstudio.core.codegen import _simplify_expr
    is_2d, is_3d = cfg.problem_dim == "2D", cfg.problem_dim == "3D"
    n = 201 if not (is_2d or is_3d) else (101 if is_2d else 31)
    axes = [np.linspace(cfg.x_min, cfg.x_max, n)]
    if is_2d or is_3d:
        axes.append(np.linspace(cfg.y_min, cfg.y_max, n))
    if is_3d:
        axes.append(np.linspace(cfg.z_min, cfg.z_max, n))
    grid = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    x = np.hstack([grid, np.full((len(grid), 1), cfg.t_min)])
    expr = _simplify_expr(cfg.ic_expressions.split("|")[0].strip(), is_2d, is_3d)
    ns = {"np": np, "x": x, "pi": np.pi}
    ns.update({k: getattr(np, k) for k in ("sin", "cos", "exp", "tanh", "sqrt", "abs", "log", "where")})
    vals = np.broadcast_to(np.reshape(eval(expr, ns), (-1,)), (len(x),))
    np.savetxt(path, np.hstack([x, vals[:, None]]))
    return len(x)


def build(study, out, seeds, only, smoke):
    from pinnstudio.core.codegen import generate_script
    app, w = _window()
    runs = []
    if study == "inverse":
        names = [("Inverse", n) for n in _names(app, w, "Inverse")]
        key, values = "inverse_obs_anchors", (True, False)
    else:
        names = [("Forward", n) for n in IC_EXAMPLES]
        key, values = "ic_file_anchors", (False, True)
    for cat, name in names:
        if only and only.lower() not in name.lower():
            continue
        for seed in seeds:
            for val in values:
                cfg = _load(app, w, cat, name)
                if cfg is None:
                    print(f"!! {name}: not found", flush=True)
                    break
                if study == "inverse":
                    obs = json.loads(cfg.inverse_obs_files_json or "[]")
                    if cfg.problem_type != "Inverse" or not obs:
                        break
                    n_pts = 0
                    for o in obs:
                        try:
                            n_pts += sum(1 for ln in open(o["path"]) if ln.strip()
                                         and ln.strip()[0] in "-+.0123456789")
                        except OSError:
                            pass
                elif cfg.steady_state:
                    break
                tag = "".join(ch if ch.isalnum() else "_" for ch in name)[:48]
                d = os.path.join(out, f"{tag}__s{seed}__{'on' if val else 'off'}")
                os.makedirs(d, exist_ok=True)
                cfg.save_dir = d
                cfg.random_seed = seed
                if study == "ic":
                    n_pts = _write_ic_file(cfg, os.path.join(d, "ic.txt"))
                    cfg.forward_ic_from_file, cfg.forward_ic_file = True, os.path.join(d, "ic.txt")
                    ic_h = cfg.ic_hard.split(",")
                    cfg.ic_hard = ",".join(["False"] + ic_h[1:])
                setattr(cfg, key, val)
                if smoke:
                    _smoke(cfg)
                errs = cfg.validate()
                meta = dict(example=name, seed=seed, anchors=val, points=n_pts, validate=errs,
                            true={v["name"]: v.get("true") for v in json.loads(cfg.inverse_variables_json or "[]")})
                json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
                print(f"== {os.path.basename(d)} points={n_pts} validate={errs}", flush=True)
                if errs:
                    continue
                open(os.path.join(d, "run.py"), "w").write(generate_script(cfg))
                runs.append(d)
    w.close()
    return runs


def _params(d, true):
    out = {}
    for name, tv in true.items():
        p = os.path.join(d, f"{name}_convergence.txt")
        try:
            last = [ln for ln in open(p) if ln.strip()][-1].replace(",", " ").split()
            val = float(last[-1])
        except (OSError, IndexError, ValueError):
            continue
        out[name] = dict(value=val, true=tv, rel_err=(abs(val - tv) / abs(tv) if tv else None))
    return out


def run_one(d):
    import subprocess
    meta = json.load(open(os.path.join(d, "meta.json")))
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=5400)
        rc, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, so, se = "timeout", (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), ""
    open(os.path.join(d, "log.txt"), "w").write(so + "\n--- stderr ---\n" + se)
    res = dict(meta, rc=rc, seconds=round(time.time() - t0), l2=parse(so), params=_params(d, meta["true"]))
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
    pe = " ".join(f"{k}={v['value']:.4g}({v['rel_err']:.1%})" for k, v in res["params"].items() if v["rel_err"] is not None)
    print(f"-- {os.path.basename(d)} rc={rc} {res['seconds']} s {pe}", flush=True)
    if rc:
        print(se[-800:], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=("inverse", "ic"))
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--only", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    runs = build(a.study, out, [int(s) for s in a.seeds.split(",")], a.only, a.smoke)
    print(f"{len(runs)} runs", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run_one, runs))
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
