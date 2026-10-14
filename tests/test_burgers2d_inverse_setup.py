#!/usr/bin/env python3
"""Burgers 2D (Mathias) in Inverse mode: the unknown is the viscosity nu
itself (true value 0.01/pi), and U and V are each measured at t = 1 against
their own output. Before, only V's snapshot was loaded and fitted to output
U, and the substitution made the unknown pi*nu while 0.01/pi was reported
as its true value."""
import ast
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication(sys.argv)


def _config(inverse):
    w = MainWindow()
    app.processEvents()
    if inverse:
        w.quick_examples_category_combo.setCurrentText("Inverse")
        app.processEvents()
    w.quick_examples_combo.setCurrentText("2D Burgers (Mathias)")
    app.processEvents()
    return w._build_config()


def test_inverse_unknown_is_nu_and_both_outputs_are_measured():
    cfg = _config(True)
    assert cfg.problem_type == "Inverse"
    assert "(nu)*(du_xx" in cfg.pde_expressions and "(nu)*(dv_xx" in cfg.pde_expressions
    var = json.loads(cfg.inverse_variables_json)[0]
    assert var["name"] == "nu" and abs(var["true"] - 0.01 / 3.141592653589793) < 1e-6
    obs = [(os.path.basename(o["path"]), o["output_idx"]) for o in json.loads(cfg.inverse_obs_files_json)]
    assert obs == [("t_1.0_u.txt", 0), ("t_1.0_v.txt", 1)], obs
    assert not cfg.validate()


def test_forward_keeps_the_numbers_and_reference_files_pair_up():
    cfg = _config(False)
    assert cfg.problem_type == "Forward"
    assert "(0.01/pi)*(du_xx" in cfg.pde_expressions
    ea = ast.literal_eval(cfg.ea_files)
    # one U and one V snapshot at each of t = 0, 0.25, 0.5, 0.75, 1
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert sorted(sel for tt, _p, sel in ea if abs(tt - t) < 1e-9) == [0, 1], (t, ea)


if __name__ == "__main__":
    test_inverse_unknown_is_nu_and_both_outputs_are_measured()
    test_forward_keeps_the_numbers_and_reference_files_pair_up()
    print("PASSED")
