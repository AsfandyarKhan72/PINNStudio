#!/usr/bin/env python3
"""
Round 49 -- codebase cleanup, phase 12 (audit §2e, _build_restore_param_script):
"draw one inferred parameter's convergence curve" was hand-typed three
times with drift: generate_script()'s live param_plot.png (log scale +
title/axis overrides), generate_clean_script()'s exported
param_convergence.png (NO log scale and NO overrides -- the export
silently ignored both settings), and Restore & Visualize's Parameter
Convergence plot (log scale + overrides + subplot-title font size). All
three now emit and call one shared runtime function, _draw_param_ax, from
codegen.py's _param_axis_helper_code.

Checks: the helper's own logic in isolation (log vs linear, the
non-positive fallback, overrides, final/true lines), then real inverse
runs through both generators (log scale + overrides on) and a real
Restore & Visualize parameter-convergence script (static + animated).

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round49_param_axis_shared.py
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("MPLBACKEND", "Agg")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

import numpy as np

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import _param_axis_helper_code, generate_clean_script, generate_script


def _inverse_cfg(save_dir, obs_path):
    cfg = PINNConfig()
    cfg.problem_type = "Inverse"
    cfg.problem_dim = "1D"; cfg.steady_state = False
    cfg.num_outputs = 1; cfg.output_names = "u"
    cfg.x_min, cfg.x_max = 0.0, 1.0
    cfg.t_min, cfg.t_max = 0.0, 1.0
    cfg.num_domain = 30; cfg.num_boundary = 10; cfg.num_initial = 10; cfg.num_test = 20
    cfg.layers = [2, 8, 1]
    cfg.pde_expressions = "du_t - D*du_xx"; cfg.pde_expression = cfg.pde_expressions
    cfg.ic_expressions = "sin(pi*x)"; cfg.ic_expression = cfg.ic_expressions
    cfg.custom_bc_json = "[]"
    cfg.adapt_method = "None"
    cfg.iterations = 20; cfg.optimizer_scheduler = False; cfg.iterations2 = 0
    cfg.inverse_variables_json = json.dumps([{"name": "D", "init": 0.5}])
    cfg.inverse_obs_files_json = json.dumps([{"path": obs_path, "output_idx": 0, "weight": 1.0}])
    cfg.inv_param_log_scale = True
    cfg.plot_title_override = "R49 title"
    cfg.plot_xlabel_override = "R49 x"
    cfg.plot_ylabel_override = "R49 y"
    cfg.plot_type = "Surface"; cfg.plot_resolution = 12
    cfg.save_dir = save_dir
    return cfg


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── 1. helper logic in isolation ──────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = {"np": np}
    exec(_param_axis_helper_code(10.0, 9.0), ns)
    draw = ns["_draw_param_ax"]
    fig, ax = plt.subplots()
    fv = draw(ax, "D", [0, 1, 2], [0.5, 0.3, 0.2], 0.1, True, "", "", "")
    check(ax.get_yscale() == "log" and ax.get_ylabel() == "D", "positive values + log_scale -> log axis (labelled with the name: the axis shows the values)")
    check(abs(fv - 0.2) < 1e-12 and ax.get_title() == "Inferred Parameter: D", "final value / default title")
    check(len(ax.get_lines()) == 3, "curve + final line + true line")
    plt.close(fig)
    fig, ax = plt.subplots()
    draw(ax, "k", [0, 1], [-1.0, 2.0], None, True, "T", "X", "Y")
    check(ax.get_yscale() == "linear", "non-positive values must fall back to linear")
    check((ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("T", "X", "Y"), "overrides should win")
    check(len(ax.get_lines()) == 2, "no true line when true_val is None")
    plt.close(fig)

    try:
        import deepxde as _dde_probe  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures

    # ── 2. real inverse runs through both generators ──────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        xs, ts = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 6))
        obs = np.column_stack([xs.ravel(), ts.ravel(), (np.sin(np.pi * xs) * np.exp(-0.1 * np.pi ** 2 * ts)).ravel()])
        obs_path = os.path.join(tmpdir, "obs.txt")
        np.savetxt(obs_path, obs)
        for gen_name, gen in (("generate_script", generate_script), ("generate_clean_script", generate_clean_script)):
            save_dir = os.path.join(tmpdir, gen_name)
            os.makedirs(save_dir, exist_ok=True)
            script = gen(_inverse_cfg(save_dir, obs_path))
            check("def _draw_param_ax(" in script and "_draw_param_ax(" in script.split("def _draw_param_ax(")[1],
                  f"{gen_name}: should define and call the shared _draw_param_ax")
            check("R49 title" in script, f"{gen_name}: the title override should reach the parameter plot")
            sp = os.path.join(tmpdir, f"{gen_name}.py")
            with open(sp, "w", encoding="utf-8") as f:
                f.write(script)
            proc = subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=300)
            check(proc.returncode == 0, f"{gen_name}: inverse run should exit cleanly, got {proc.returncode}:\n"
                                        f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
            plots = [p for p in glob.glob(os.path.join(save_dir, "**", "*.png"), recursive=True)
                     if "param" in os.path.basename(p)]
            check(bool(plots), f"{gen_name}: a parameter-convergence PNG should be saved")
            check("Final inferred D" in proc.stdout, f"{gen_name}: should report the final inferred D")

        # ── 3. Restore & Visualize parameter-convergence script ──────────
        conv = os.path.join(tmpdir, "D_convergence.txt")
        with open(conv, "w", encoding="utf-8") as f:
            f.write("iteration,D\n" + "".join(f"{i * 100},{0.5 * 0.9 ** i}\n" for i in range(12)))
        win = MainWindow()
        win._restore_viz_settings = {"title": "R49 restore", "subplot_title_fontsize": 9.0}
        for animate in (False, True):
            out_dir = os.path.join(tmpdir, f"restore_{animate}")
            script = win._build_restore_param_script([conv], out_dir, True, True, animate, true_vals=[0.1])
            check("def _draw_param_ax(" in script, "restore param script should embed the shared helper")
            sp = os.path.join(tmpdir, f"restore_{animate}.py")
            with open(sp, "w", encoding="utf-8") as f:
                f.write(script)
            proc = subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=240)
            check(proc.returncode == 0 and "RESTORE_DONE" in proc.stdout,
                  f"restore param script (animate={animate}) should run cleanly:\n{proc.stdout[-1200:]}\n{proc.stderr[-1200:]}")
            want = "param_convergence_animation.gif" if animate else "param_convergence_plot.png"
            check(os.path.exists(os.path.join(out_dir, want)), f"restore param script should save {want}")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 49 PARAMETER-AXIS TESTS PASSED")
    return 0


def test_round49_param_axis_shared():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
