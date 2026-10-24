#!/usr/bin/env python3
"""
Round 59 -- two maintainability changes in main_window.py, plus the two
stale-value bugs the first one surfaced.

1. _on_template_selected applied five Quick Example tables (2D, 2D steady,
   3D, 3D steady, 1D), each repeating the same steps by hand. They now go
   through shared _tpl_* helpers. The hand-kept copies had drifted:
     - only the 1D branch reset the L-BFGS float type, so picking
       "1D Schrödinger" (float64) and then any 2D/3D template kept float64;
     - only the 1D branch ended the fallback Time-Adaptive row at the
       template's own t_max, so "2D Allen-Cahn (Wight & Zhao)" (t to 10) in
       Inverse mode pre-filled a 0..1 row.
   Before the commit, 104 template selections (every template, Forward and
   Inverse, after four different previous templates) produced the same
   config and GUI state as before apart from exactly these two fixes.

2. _build_ui (2,500 lines) is now a short driver plus one method per panel
   (_build_setup_panels, _build_domain_panels, _build_network_panel,
   _build_training_panel, _build_adaptive_and_sweep_panels,
   _build_inverse_panel, _build_solve_buttons, _build_restore_panel,
   _build_right_panel) -- pure code motion: the full widget tree (941
   widgets: type, name, text, visibility, enabled state, tooltips, styles,
   sizes, combo items), the window's attribute set and the default config
   were identical before and after.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round59_templates_and_build_ui.py
"""
import inspect
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    # the templates' own settings (Settings: Default would load a paper's method instead)
    win._apply_example_settings_mode = lambda key: None

    def pick(name):
        win.quick_examples_combo.setCurrentText("📋 Examples")
        win.quick_examples_combo.setCurrentText(name)
        _app.processEvents()
        return win._build_config()

    # Stale float64 from Schrödinger no longer leaks into 2D/3D templates.
    for name in ("2D Heat", "2D Burgers (Mathias)", "2D Poisson (L-Shape)", "3D Heat", "3D Poisson (Sphere)", "1D Heat"):
        check(pick("1D Schrödinger").lbfgs_float_type == "float64", "Schrödinger itself should still use float64")
        check(pick(name).lbfgs_float_type == "float32", f"{name} after Schrödinger should reset to float32")

    # Inverse-mode fallback Time-Adaptive row ends at the template's own t_max.
    win.radio_inverse.setChecked(True)
    cfg = pick("2D Allen-Cahn (Wight & Zhao)")
    groups = json.loads(cfg.ta_step_groups) if isinstance(cfg.ta_step_groups, str) else cfg.ta_step_groups
    check(groups and groups[0]["t_end"] == 10.0, f"W&Z Inverse fallback TA row should end at 10.0: {groups}")
    check(win._ta_suspended_for_inverse, "W&Z's TA default should be suspended (not applied) in Inverse mode")
    cfg = pick("1D Schrödinger")
    groups = json.loads(cfg.ta_step_groups) if isinstance(cfg.ta_step_groups, str) else cfg.ta_step_groups
    check(abs(groups[0]["t_end"] - 1.5707963267948966) < 1e-3, "Schrödinger fallback row should end at pi/2 (4-decimal spinbox)")
    win.radio_forward.setChecked(True)

    # Steady templates still clear any Time-Adaptive state left by a TA template.
    pick("2D Burgers (Mathias)")
    check(win._build_config().time_adaptive, "2D Burgers (Mathias) should enable Time-Adaptive in Forward mode")
    for steady in ("2D Poisson (Disk)", "3D Poisson (Sphere)"):
        pick("2D Burgers (Mathias)")
        c = pick(steady)
        check(not c.time_adaptive and c.steady_state and win._current_ta_cfg is None,
              f"{steady}: Time-Adaptive state should be cleared")

    # 2D Heat / 3D Heat Dirichlet edge weight 100 still survives the scheduler rebuild.
    for name in ("2D Heat", "3D Heat"):
        pick(name)
        dir_keys = [f"bc_{j}" for j, e in enumerate(win.custom_bc_list) if e['type'].currentData() == 'dirichlet']
        check(dir_keys and all(win.weight_widgets[k].value() == 100.0 for k in dir_keys if k in win.weight_widgets),
              f"{name}: Dirichlet BC weights should be 100")

    # _build_ui is a short driver over per-panel builders.
    src = inspect.getsource(MainWindow._build_ui)
    check(len(src.splitlines()) < 250, f"_build_ui should be a short driver, got {len(src.splitlines())} lines")
    for m in ("_build_setup_panels", "_build_domain_panels", "_build_network_panel", "_build_training_panel",
              "_build_adaptive_and_sweep_panels", "_build_inverse_panel", "_build_solve_buttons",
              "_build_restore_panel", "_build_right_panel"):
        check(hasattr(MainWindow, m) and f"self.{m}(" in src, f"_build_ui should call {m}")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 59 TEMPLATE / BUILD_UI TESTS PASSED")
    return 0


def test_round59_templates_and_build_ui():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
