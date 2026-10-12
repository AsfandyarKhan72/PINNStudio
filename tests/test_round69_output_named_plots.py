#!/usr/bin/env python3
"""
Round 69 -- results of multi-output problems say which output they show.

 - With more than one output, every output also gets its own plot,
   solution_plot_<name>.<ext>, besides the selected "Plot output"
   (solution_plot.<ext>), in Solve and in the exported script; the
   "All outputs" checkbox (PINNConfig.plot_all_outputs) turns it off.
 - Error Analysis figure titles name the compared output
   ("PINN vs Reference — p — 2D Heatmaps"); 1D titles and the line
   animation's axis label name the output too.
 - DeepXDE examples use the jet colormap, like PINNStudio's own templates.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round69_output_named_plots.py
"""
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.core import deepxde_examples as dx
from pinnstudio.core.codegen import generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow


def _two_output(d=".", **kw):
    """u'' = q with q a second output (unknown-field style), 1D steady."""
    c = PINNConfig()
    base = dict(problem_dim="1D", steady_state=True, geometry_type="Interval", x_min=-1.0, x_max=1.0, num_initial=0,
                num_outputs=2, output_names="u,q", num_equations=1, pde_expressions="-du_xx + q",
                custom_bc_json=json.dumps([{"type": "dirichlet", "location": "True", "value": "0", "component": 0}]),
                layers=[1, 16, 16, 2], iterations=50, optimizer2="none", iterations2=0, optimizer_scheduler=False,
                num_domain=32, num_boundary=2, num_test=40, ea_ref_mode="formula",
                ea_ref_formula="sin(pi*x)|-pi**2*sin(pi*x)", save_dir=d, plot_resolution=40)
    base.update(kw)
    for k, v in base.items():
        setattr(c, k, v)
    c.pde_expression = c.pde_expressions
    return c


def run():
    failures = []

    def check(cond, msg):
        print(("ok: " if cond else "FAIL: ") + msg)
        if not cond:
            failures.append(msg)

    c = _two_output()
    exp = generate_clean_script(c)
    check('solution_plot_u.png' in exp and 'solution_plot_q.png' in exp, "export saves each output's plot")
    check("_plot_idx = 1" in exp and "solution_path = _main_solution_path" in exp,
          "export switches the plotted output and restores the selected one")
    live = generate_script(c)
    check("_plot_jobs.append" in live and "def _ea_title_txt" in live, "live script: per-output jobs and named EA titles")
    off = _two_output(plot_all_outputs=False)
    check("solution_plot_q.png" not in generate_clean_script(off), "All outputs off -> only the selected output")
    one = _two_output(num_outputs=1, output_names="u", pde_expressions="-du_xx - 1", ea_ref_formula="(1 - x**2)/2",
                      layers=[1, 16, 1])
    check("Every output's own plot" not in generate_clean_script(one), "single output -> nothing extra")

    w = MainWindow()
    check(w.plot_all_outputs_cb.isChecked(), "All outputs is on by default")
    w.plot_all_outputs_cb.setChecked(False)
    cfg = w._build_config()
    check(cfg.plot_all_outputs is False, "checkbox -> config")
    cfg.plot_all_outputs = True
    w._apply_config(cfg)
    check(w.plot_all_outputs_cb.isChecked(), "config -> checkbox")

    check(all(dx.build_config(e, "/r").plot_colormap == "jet" for e in dx.EXAMPLES), "DeepXDE examples use jet")

    try:
        import deepxde  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures
    with tempfile.TemporaryDirectory() as tmp:
        for gen in (generate_script, generate_clean_script):
            d = os.path.join(tmp, gen.__name__)
            os.makedirs(d)
            with open(os.path.join(d, "run.py"), "w") as fh:
                fh.write(gen(_two_output(d)))
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
            check(p.returncode == 0, f"{gen.__name__} ran:\n{p.stderr[-1500:]}")
            files = {f for _r, _d, fs in os.walk(d) for f in fs}
            check({"solution_plot.png", "solution_plot_u.png", "solution_plot_q.png"} <= files,
                  f"{gen.__name__}: named plots for every output")
            check({"line_comparison_u.png", "line_comparison_q.png"} <= files, f"{gen.__name__}: EA per output")
    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 69 OUTPUT-NAMING TESTS PASSED")
    return 0


def test_round69_output_named_plots():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
