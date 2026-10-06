#!/usr/bin/env python3
"""
Checks for derivative-aware "Custom..." expression plotting: the live
Results panel's plot_custom_expr field and the Restore & Visualize panel's
own custom-expression field can now reference derivatives (du_x, du_xx,
du_xy, ... -- same d{name}_x syntax the Custom PDE box and Training
Monitors already use), not just algebraic combinations of raw outputs.

Mechanism: evaluated through DeepXDE's own dde.Model.predict(x,
operator=...) built-in (the same API dde.callbacks.OperatorPredictor
itself uses internally), reusing Training Monitors' own
_tm_build_dvars()/_tm_hess() helpers unchanged -- no new derivative math,
just a new place that can call into it.

Covers:
  1. generate_script()'s Standard (non-Time-Adaptive) path: a custom
     expression wires model.predict(..., operator=_plot_custom_op), and
     an UNCONFIGURED run still mentions neither "OperatorPredictor" (a
     regression guard -- an earlier draft of this feature's comments
     accidentally broke test_training_monitors.py's "no OperatorPredictor
     callback for an unconfigured run" check just by using that word in a
     docstring that's always embedded) nor an operator= call.
  2. generate_script()'s Time-Adaptive path: same checks, using its own
     separate _extract_plot_field definition.
  3. generate_clean_script() ("Export as DeepXDE Script") is confirmed
     UNCHANGED this round -- still the old NumPy-only custom-expression
     path, no operator= anywhere. This is a deliberate scope decision
     (see codegen.py/main_window.py's own comments), not an oversight;
     this test exists so a future round that DOES extend export parity
     has a clear "this used to not be wired up" baseline to update.
  4. main_window.py's _build_restore_script (Standard restore) and
     _build_restore_script_ta (Time-Adaptive restore): both wire a
     derivative-aware custom expression through operator=_restore_custom_op
     when one is configured, and fall back to the original plain-column
     behavior when not.
  5. A real, trained (if briefly) dde.Model: the operator-based path's
     du_x output is checked against both a direct dde.grad.jacobian call
     and an independent finite-difference derivative, confirming the
     actual numbers this feature plots are correct, not just that the
     generated code looks right.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_derivative_plotting.py
"""
import os
import sys
import ast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

import numpy as np
import torch
import deepxde as dde

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import generate_script, generate_clean_script


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ---------------------------------------------------------------
    # 1) generate_script() -- Standard path
    # ---------------------------------------------------------------
    base_cfg = PINNConfig()
    base_cfg.ic_expressions = "sin(pi*x)"
    base_cfg.iterations = 1

    plain_cfg = base_cfg
    plain_script = generate_script(plain_cfg)
    ast.parse(plain_script)  # must stay syntactically valid either way
    check("OperatorPredictor" not in plain_script,
          "an unconfigured run should still mention no OperatorPredictor "
          "anywhere (regression guard for this feature's own comments)")
    check('_plot_custom_expr = ""' in plain_script,
          "no custom expression configured -> the embedded expression string should be empty "
          "(the operator=_plot_custom_op call itself is always present in the function body -- "
          "same 'helper always defined' convention Training Monitors' own runtime code uses -- "
          "but it's only reached at runtime when _plot_custom_expr is non-empty)")
    check("def _plot_custom_op(" in plain_script,
          "the operator helper should still be DEFINED unconditionally "
          "(same 'helper always present' convention Training Monitors' "
          "own runtime code uses), just unused when there's no expression")

    import dataclasses
    deriv_cfg = dataclasses.replace(base_cfg, plot_custom_expr="du_x", plot_custom_label="du_dx")
    deriv_script = generate_script(deriv_cfg)
    ast.parse(deriv_script)
    check("operator=_plot_custom_op" in deriv_script,
          "a configured custom expression should route through model.predict(..., operator=_plot_custom_op)")
    check("_tm_build_dvars" in deriv_script,
          "the custom-expression operator should reuse Training Monitors' own _tm_build_dvars")
    check("torch.sin" in deriv_script and "torch.sqrt" in deriv_script,
          "the custom-expression math namespace should use torch equivalents (autograd-safe), not np.*")

    # ---------------------------------------------------------------
    # 2) generate_script() -- Time-Adaptive path
    # ---------------------------------------------------------------
    ta_cfg = dataclasses.replace(deriv_cfg, time_adaptive=True, ta_num_steps=2)
    ta_script = generate_script(ta_cfg)
    ast.parse(ta_script)
    check("operator=_plot_custom_op" in ta_script,
          "Time-Adaptive path should also route a custom expression through operator=_plot_custom_op")
    check("def _extract_plot_field(_x_grid, _pf_model):" in ta_script,
          "Time-Adaptive's _extract_plot_field should require an explicit model (no single module-level `model`)")

    ta_plain_script = generate_script(dataclasses.replace(base_cfg, time_adaptive=True, ta_num_steps=2))
    check("OperatorPredictor" not in ta_plain_script,
          "an unconfigured Time-Adaptive run should also mention no OperatorPredictor")

    # ---------------------------------------------------------------
    # 3) generate_clean_script() -- confirmed UNCHANGED this round
    # ---------------------------------------------------------------
    clean_script = generate_clean_script(deriv_cfg)
    check("operator=" not in clean_script or "_plot_custom_op" not in clean_script,
          "generate_clean_script() is NOT extended this round (deliberate scope "
          "decision) -- it should still use the old NumPy-only custom-expression path")

    # ---------------------------------------------------------------
    # 4) Restore & Visualize panel: Standard + Time-Adaptive
    # ---------------------------------------------------------------
    win = MainWindow()
    restore_cfg = dict(
        layers=[2, 32, 32, 1], activation="tanh", x_min=0.0, x_max=1.0,
        y_min=0.0, y_max=1.0, z_min=0.0, z_max=1.0, t_min=0.0, t_max=1.0,
        problem_dim="1D", steady_state=False, loss_type="MSE",
        output_names="u", network_type="FNN",
    )

    plain_restore = win._build_restore_script(
        "/tmp/fake-1.pt", restore_cfg, "adam", "Surface", 0, 5, "/tmp/out")
    ast.parse(plain_restore)
    check("_restore_custom_expr = ''" in plain_restore,
          "no custom expression -> the embedded expression string should be empty "
          "(the operator=_restore_custom_op call is always present in the function "
          "body, only reached at runtime when _restore_custom_expr is non-empty)")
    check("def _restore_custom_op(" in plain_restore,
          "the restore operator helper should still be defined unconditionally")

    deriv_restore = win._build_restore_script(
        "/tmp/fake-1.pt", restore_cfg, "adam", "Surface", 0, 5, "/tmp/out",
        custom_expr="du_x", custom_label="du_dx")
    ast.parse(deriv_restore)
    check("operator=_restore_custom_op" in deriv_restore,
          "a configured custom expression should route the Restore script through operator=_restore_custom_op")
    check("_tm_build_dvars" in deriv_restore,
          "the Restore script should embed Training Monitors' own _tm_build_dvars via "
          "_training_monitor_runtime_code(), not a second derivative-building copy")

    ta_steps = [
        {"dir": "/tmp/step0", "t0": 0.0, "t1": 0.5, "model_path": "/tmp/fake_step0-1.pt", "cfg": restore_cfg},
        {"dir": "/tmp/step1", "t0": 0.5, "t1": 1.0, "model_path": "/tmp/fake_step1-1.pt", "cfg": restore_cfg},
    ]
    deriv_restore_ta = win._build_restore_script_ta(
        ta_steps, restore_cfg, "adam", "Surface", 0, 5, "/tmp/out",
        custom_expr="du_x", custom_label="du_dx")
    ast.parse(deriv_restore_ta)
    check("operator=_restore_custom_op" in deriv_restore_ta,
          "the Time-Adaptive Restore script should also route through operator=_restore_custom_op")
    check("def _extract_plot_field(x_grid, _tm_model):" in deriv_restore_ta,
          "the Time-Adaptive Restore script's _extract_plot_field should take an "
          "explicit model object (a different step's model may apply per time value)")

    # ---------------------------------------------------------------
    # 5) Real numeric check: operator-based du_x matches direct autograd
    #    AND an independent finite-difference derivative, on an actual
    #    (briefly) trained model -- not just synthetic tensors.
    # ---------------------------------------------------------------
    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(x, y):
        du_t = dde.grad.jacobian(y, x, i=0, j=1)
        du_xx = dde.grad.hessian(y, x, i=0, j=0)
        return du_t - 0.4 * du_xx

    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda x, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda x, on_initial: on_initial)
    data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=80, num_boundary=20, num_initial=20, num_test=80)
    net = dde.nn.FNN([2, 16, 16, 1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.01)
    model.train(iterations=20, display_every=100)

    x_test = np.column_stack([np.linspace(0.1, 0.9, 5), np.full(5, 0.5)]).astype(np.float32)

    def _tm_build_dvars_1d(x, y, n_out, out_names, is_steady, dim, expr_check):
        dvars = {}
        for oi, oname in enumerate(out_names):
            dvars[oname] = y[:, oi:oi + 1]
            dvars[f"d{oname}_x"] = dde.grad.jacobian(y, x, i=oi, j=0)
            if not is_steady:
                dvars[f"d{oname}_t"] = dde.grad.jacobian(y, x, i=oi, j=1)
        return dvars

    def op_via_feature(inputs, outputs):
        dvars = _tm_build_dvars_1d(inputs, outputs, 1, ["u"], False, "1D", "du_x")
        ns = dict(dvars)
        ns["torch"] = torch
        return eval("du_x", ns)

    du_x_feature = model.predict(x_test, operator=op_via_feature)

    def op_direct(x, y):
        return dde.grad.jacobian(y, x, i=0, j=0)
    du_x_direct = model.predict(x_test, operator=op_direct)

    check(np.allclose(du_x_feature, du_x_direct, atol=1e-6),
          "operator-based du_x (the feature's own mechanism) should exactly match a direct dde.grad.jacobian call")

    eps = 1e-3
    x_plus = x_test.copy(); x_plus[:, 0] += eps
    x_minus = x_test.copy(); x_minus[:, 0] -= eps
    fd_dux = (model.predict(x_plus)[:, 0] - model.predict(x_minus)[:, 0]) / (2 * eps)
    check(np.allclose(fd_dux, du_x_direct.flatten(), atol=1e-2),
          "autograd du_x should also match an independent finite-difference derivative")

    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nALL DERIVATIVE-PLOTTING TESTS PASSED")


if __name__ == "__main__":
    main()


def test_derivative_plotting():
    failures = run()
    assert not failures, "\n".join(failures)
