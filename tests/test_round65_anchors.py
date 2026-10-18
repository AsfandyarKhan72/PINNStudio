#!/usr/bin/env python3
"""
Round 65 -- extra training points ("anchors", DeepXDE's anchors=).

Collocation Points gets "Extra training points": None, From a file (one
column per input, x[, y[, z]][, t]) or Random in a box (a count plus a
"lo,hi" range per input column, blank = that column's domain range) --
DeepXDE's American-put example concentrates 1000 extra points around the
strike price this way. They're added to the measured-data points an
inverse problem already anchors. Not supported with Time-Adaptive
training (reported clearly). Shared by both generators via
codegen._anchor_points_code; a problem without them generates exactly the
same scripts as before.

Also here: the Domain/Boundary point minimums were 100/10, too high for
DeepXDE's own small examples (16 domain / 2 boundary points for its 1D
Poisson problems), so loading those settings silently changed them.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round65_anchors.py
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

import numpy as np

from pinnstudio.core.codegen import _anchor_points_code, generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow


def _heat(d, it, **kw):
    c = PINNConfig()
    base = dict(problem_dim="1D", steady_state=False, geometry_type="Interval", x_min=0.0, x_max=4.0,
                t_min=0.0, t_max=1.0, pde_expressions="du_t - 0.4*du_xx", ic_expressions="sin(pi*x/4)",
                custom_bc_json=json.dumps([{"type": "dirichlet", "location": "True", "value": "0", "component": 0}]),
                num_domain=200, num_boundary=20, num_initial=20, num_test=100, layers=[2, 16, 16, 1],
                iterations=it, optimizer2="none", iterations2=0, optimizer_scheduler=False,
                anchors_mode="box", anchors_count=500, anchors_box="0.5,1.5;",
                save_dir=d, ea_files="[]", plot_resolution=20)
    base.update(kw)
    for k, v in base.items():
        setattr(c, k, v)
    c.pde_expression = c.pde_expressions
    c.ic_expression = c.ic_expressions
    return c


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── code ──────────────────────────────────────────────────────────────
    code = _anchor_points_code(_heat("/tmp/x", 10))
    check("np.random.uniform(0.5, 1.5, (500, 1))" in code and "np.random.uniform(0.0, 1.0, (500, 1))" in code,
          f"box: given x range, t defaults to the domain range:\n{code}")
    check(_anchor_points_code(_heat("/tmp/x", 10, anchors_mode="none")) == "", "none -> no code")
    check("np.loadtxt('/data/a.txt'" in _anchor_points_code(_heat("/tmp/x", 10, anchors_mode="file", anchors_file="/data/a.txt")),
          "file mode loads the file")
    for gen in (generate_script, generate_clean_script):
        ast.parse(gen(_heat("/tmp/x", 10)))
        try:
            gen(_heat("/tmp/x", 10, time_adaptive=True, adapt_method="Time Adaptive Training"))
            check(False, f"{gen.__name__}: anchors + Time-Adaptive should be rejected")
        except ValueError as e:
            check("anchors" in str(e), str(e))

    # ── GUI ───────────────────────────────────────────────────────────────
    win = MainWindow()
    check(win.num_domain.minimum() <= 10 and win.num_boundary.minimum() == 0, "small point counts allowed")
    win._set_combo_data(win.anchors_mode_combo, "box")
    win.anchors_count_spin.setValue(321)
    win.anchors_box_edit.setText("0.5,1.5;0,1")
    c1 = win._build_config()
    win2 = MainWindow()
    win2._apply_config(c1)
    c2 = win2._build_config()
    check((c2.anchors_mode, c2.anchors_count, c2.anchors_box) == ("box", 321, "0.5,1.5;0,1"), "anchors survive save/load")

    try:
        import deepxde as _dde_probe  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures

    # ── the extra points really are in the training set ───────────────────
    with tempfile.TemporaryDirectory() as tmp:
        counts = {}
        for mode in ("none", "box"):
            for gen in (generate_script, generate_clean_script):
                d = os.path.join(tmp, f"{mode}_{gen.__name__}")
                os.makedirs(d)
                with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as fh:
                    fh.write(gen(_heat(d, 50, anchors_mode=mode)) + "\nprint('NTRAIN', len(data.train_x_all))\n")
                p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
                check(p.returncode == 0, f"{mode}/{gen.__name__} failed:\n{p.stderr[-1200:]}")
                m = re.search(r"NTRAIN (\d+)", p.stdout)
                counts[(mode, gen.__name__)] = int(m.group(1)) if m else None
        for gen in ("generate_script", "generate_clean_script"):
            a, b = counts[("none", gen)], counts[("box", gen)]
            check(a is not None and b is not None and b - a == 500, f"{gen}: 500 extra training points expected: {counts}")
        check(counts[("box", "generate_script")] == counts[("box", "generate_clean_script")], f"live/export parity: {counts}")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 65 ANCHOR TESTS PASSED")
    return 0


def test_round65_anchors():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
