#!/usr/bin/env python3
"""
Checks for the Training Monitors feature: an opt-in panel that watches one
or more expressions (plain output names and/or derivative names using the
same d{name}_x / d{name}_xx / d{name}_xy syntax the Custom PDE expression
box already uses) at a fixed set of points, logged every N iterations via
DeepXDE's own dde.callbacks.OperatorPredictor -- pure observability, zero
effect on training (see config.py's training_monitors_enabled/
training_monitors docstring).

Covers:
  1. Config defaults + validate() (JSON syntax, per-entry structure, point
     dimension matching config.problem_dim/steady_state).
  2. _build_training_monitors_json()/_apply_training_monitors_json() UI
     wiring, including the project save/load round-trip through
     dataclasses.asdict()/PINNConfig(**filtered)) and backward
     compatibility for a config saved before this feature existed.
  3. generate_script()/generate_clean_script() both embed the shared
     derivative-building/eval helpers and wire monitors into the shared
     _train_cbs list (every Standard-path .train() call), NOT into the
     Time-Adaptive per-step list.
  4. The actual _tm_build_dvars()/_tm_eval_terms() helpers, executed
     directly (not just inspected as text) against a synthetic torch
     tensor with a known analytic derivative, for 1D/2D/3D and
     single-/multi-output -- confirming the derivative math itself, not
     just that the generated code mentions the right function names.
  5. main_window.py's Results-panel wiring: _display_monitor_plots()/
     _show_monitor_plot() discover and show monitor_*_plot.png files (or
     correctly hide the panel when none exist).

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_training_monitors.py
"""
import os
import sys
import json
import dataclasses
import ast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

import torch
import numpy as np
import deepxde as dde

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import (
    generate_script, generate_clean_script,
    _training_monitor_runtime_code, _build_training_monitors_code,
)


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ---------------------------------------------------------------
    # 1) Config defaults + validate()
    # ---------------------------------------------------------------
    default_cfg = PINNConfig()
    check(default_cfg.training_monitors_enabled is False, "default training_monitors_enabled should be False")
    check(default_cfg.training_monitors == "[]", "default training_monitors should be '[]'")
    check(default_cfg.validate() == [], "default config should validate cleanly")

    bad_json_cfg = dataclasses.replace(default_cfg, training_monitors_enabled=True, training_monitors="{not json")
    check(any("training_monitors" in e for e in bad_json_cfg.validate()), "invalid JSON should be flagged")

    empty_list_cfg = dataclasses.replace(default_cfg, training_monitors_enabled=True, training_monitors="[]")
    check(any("no monitors have" in e for e in empty_list_cfg.validate()),
          "enabled with zero monitors should be flagged")

    missing_fields_cfg = dataclasses.replace(
        default_cfg, training_monitors_enabled=True,
        training_monitors=json.dumps([{"name": "", "points": [], "expr": "", "period": 0}]),
    )
    errs = missing_fields_cfg.validate()
    check(any("needs a name" in e for e in errs), "missing name should be flagged")
    check(any("needs at least one expression" in e for e in errs), "missing expr should be flagged")
    check(any("positive integer period" in e for e in errs), "non-positive period should be flagged")
    check(any("needs at least one point" in e for e in errs), "missing points should be flagged")

    # Point dimension must match problem_dim/steady_state: 1D + time-
    # dependent (default) needs exactly 2 coordinates per point.
    wrong_dim_cfg = dataclasses.replace(
        default_cfg, training_monitors_enabled=True,
        training_monitors=json.dumps([{"name": "m", "points": [[0.3]], "expr": "u", "period": 100}]),
    )
    check(any("coordinate(s)" in e for e in wrong_dim_cfg.validate()),
          "a point with too few coordinates for this problem should be flagged")

    good_cfg = dataclasses.replace(
        default_cfg, training_monitors_enabled=True,
        training_monitors=json.dumps([{"name": "m", "points": [[0.3, 0.0], [0.5, 0.0]], "expr": "u, du_x", "period": 100}]),
    )
    check(good_cfg.validate() == [], "a well-formed monitor should validate cleanly")

    # ---------------------------------------------------------------
    # 2) UI wiring: build/apply + save/load round-trip
    # ---------------------------------------------------------------
    win = MainWindow()
    win.radio_1d.setChecked(True)
    win.quick_examples_combo.setCurrentText("1D Heat")

    config = win._build_config()
    check(config.training_monitors_enabled is False, "_build_config() should default training_monitors_enabled to False")
    check(config.training_monitors == "[]", "_build_config() should default training_monitors to '[]' with no rows")

    win.training_monitors_cb.setChecked(True)
    win._add_training_monitor_row(name="probe1", points_text="(0.3, 0.0); (0.5, 0.0)", expr="u, du_x, du_xx", period=500)
    tm_config = win._build_config()
    check(tm_config.training_monitors_enabled is True, "_build_config() should read the enable checkbox")
    tm_parsed = json.loads(tm_config.training_monitors)
    check(tm_parsed == [{"name": "probe1", "points": [[0.3, 0.0], [0.5, 0.0]], "expr": "u, du_x, du_xx", "period": 500}],
          f"_build_config() should serialize the row correctly; got {tm_parsed!r}")

    # A fully-blank row should be skipped, not written out half-empty.
    win._add_training_monitor_row()
    blank_row_config = win._build_config()
    check(len(json.loads(blank_row_config.training_monitors)) == 1,
          "a blank monitor row should be skipped when building config")

    # Round-trip through the same (de)serialization _save_problem()/
    # _open_problem() use.
    payload = dataclasses.asdict(tm_config)
    known_fields = {f.name for f in dataclasses.fields(PINNConfig)}
    reloaded = PINNConfig(**{k: v for k, v in payload.items() if k in known_fields})
    check(reloaded.training_monitors_enabled is True and reloaded.training_monitors == tm_config.training_monitors,
          "asdict()/PINNConfig(**filtered) round-trip should preserve training monitors")

    # A config saved before this feature existed simply lacks the keys.
    old_payload = {k: v for k, v in payload.items() if k not in ("training_monitors_enabled", "training_monitors")}
    old_reloaded = PINNConfig(**{k: v for k, v in old_payload.items() if k in known_fields})
    check(old_reloaded.training_monitors_enabled is False and old_reloaded.training_monitors == "[]",
          "a saved config missing the training-monitor fields should fall back to off/empty")

    # _apply_config() should rebuild the rows (used by _open_problem()).
    win2 = MainWindow()
    win2._apply_config(tm_config)
    check(win2.training_monitors_cb.isChecked(), "_apply_config() should restore the enable checkbox")
    check(len(win2.training_monitor_list) == 1, "_apply_config() should rebuild exactly one row")
    if win2.training_monitor_list:
        row = win2.training_monitor_list[0]
        check(row['name'].text() == "probe1", "_apply_config() should restore the monitor's name")
        check(row['expr'].text() == "u, du_x, du_xx", "_apply_config() should restore the monitor's expression")
        check(row['period'].value() == 500, "_apply_config() should restore the monitor's period")
    check(win2.train_callbacks_show_cb.isChecked(),
          "loading a config with training monitors enabled should auto-reveal Training Callbacks")

    # ---------------------------------------------------------------
    # 3) generate_script() / generate_clean_script() content
    # ---------------------------------------------------------------
    plain_script = generate_script(config)
    ast.parse(plain_script)
    check("_tm_build_dvars" in plain_script,
          "generate_script() should always embed the shared helper (cheap no-op wiring point)")
    check("OperatorPredictor" not in plain_script, "an unconfigured run should add no OperatorPredictor callbacks")

    tm_script = generate_script(tm_config)
    ast.parse(tm_script)
    check("_tm_build_dvars" in tm_script, "generate_script() should embed the derivative-building helper")
    check("_tm_eval_terms" in tm_script, "generate_script() should embed the expression-evaluation helper")
    check("dde.callbacks.OperatorPredictor(" in tm_script, "generate_script() should construct an OperatorPredictor")
    check("monitor_probe1.txt" in tm_script, "generate_script() should name the log file after the monitor")
    check("monitor_probe1.meta.json" in tm_script, "generate_script() should write a meta sidecar")
    check("monitor_probe1_plot.png" in tm_script, "generate_script() should render a plot PNG after training")
    check("_train_cbs.append(dde.callbacks.OperatorPredictor(" in tm_script,
          "OperatorPredictor should be appended to the shared _train_cbs list")
    check("_train_cbs_ta.append(dde.callbacks.OperatorPredictor(" not in tm_script,
          "OperatorPredictor should NOT be appended to _train_cbs_ta (Time-Adaptive is out of scope -- see docstring)")

    tm_clean = generate_clean_script(tm_config)
    ast.parse(tm_clean)
    check("_tm_build_dvars" in tm_clean, "generate_clean_script() should also embed the derivative-building helper")
    check("dde.callbacks.OperatorPredictor(" in tm_clean, "generate_clean_script() should also construct an OperatorPredictor")
    check("monitor_probe1_plot.png" in tm_clean, "generate_clean_script() should also render a plot PNG")

    plain_clean = generate_clean_script(config)
    ast.parse(plain_clean)
    check("OperatorPredictor" not in plain_clean, "an unconfigured export should add no OperatorPredictor callbacks")

    # ---------------------------------------------------------------
    # 4) _tm_build_dvars() / _tm_eval_terms() executed directly
    # ---------------------------------------------------------------
    ns = {"dde": dde, "torch": torch, "np": np}
    exec(_training_monitor_runtime_code(), ns)
    tm_build_dvars = ns["_tm_build_dvars"]
    tm_eval_terms = ns["_tm_eval_terms"]

    # 1D steady, 2-output: mu = x^3, c = x^2 -> dmu_x=3x^2, dmu_xx=6x
    x1 = torch.tensor([[0.3], [0.5], [0.7]], requires_grad=True)
    mu = x1 ** 3
    c = x1 ** 2
    y1 = torch.cat([mu, c], dim=1)
    dvars1 = tm_build_dvars(x1, y1, 2, ["mu", "c"], True, "1D", "mu, dmu_x, dmu_xx, c")
    x1n = x1.detach().cpu().numpy().flatten()
    check(np.allclose(dvars1["dmu_x"].detach().cpu().numpy().flatten(), 3 * x1n ** 2, atol=1e-5), "1D dmu_x should match 3x^2")
    check(np.allclose(dvars1["dmu_xx"].detach().cpu().numpy().flatten(), 6 * x1n, atol=1e-5), "1D dmu_xx should match 6x")
    result1 = tm_eval_terms("mu, dmu_x, dmu_xx, c", x1, y1, dvars1, ["mu", "c"])
    expected1 = np.stack([x1n ** 3, 3 * x1n ** 2, 6 * x1n, x1n ** 2], axis=1)
    check(np.allclose(result1.detach().cpu().numpy(), expected1, atol=1e-5),
          "_tm_eval_terms() should return a (batch, k) tensor matching each expression")

    # 2D steady, single output, mixed partial: u = x^2 y^3
    xy = torch.tensor([[0.4, 0.6], [0.7, 0.2]], requires_grad=True)
    u2 = (xy[:, 0:1] ** 2) * (xy[:, 1:2] ** 3)
    dvars2 = tm_build_dvars(xy, u2, 1, ["u"], True, "2D", "du_x, du_y, du_xy")
    xn, yn = xy[:, 0].detach().cpu().numpy(), xy[:, 1].detach().cpu().numpy()
    check(np.allclose(dvars2["du_x"].detach().cpu().numpy().flatten(), 2 * xn * yn ** 3, atol=1e-5), "2D du_x mismatch")
    check(np.allclose(dvars2["du_y"].detach().cpu().numpy().flatten(), 3 * xn ** 2 * yn ** 2, atol=1e-5), "2D du_y mismatch")
    check(np.allclose(dvars2["du_xy"].detach().cpu().numpy().flatten(), 6 * xn * yn ** 2, atol=1e-5), "2D mixed du_xy mismatch")

    # 3D steady, 2-output (component-indexed hessian path)
    xyz = torch.tensor([[0.3, 0.5, 0.2], [0.6, 0.1, 0.4]], requires_grad=True)
    a3 = xyz[:, 0:1] ** 2 + xyz[:, 2:3] ** 3
    b3 = xyz[:, 1:2] ** 2
    y3 = torch.cat([a3, b3], dim=1)
    dvars3 = tm_build_dvars(xyz, y3, 2, ["a", "b"], True, "3D", "da_z, da_xz, db_y")
    z3n = xyz[:, 2].detach().cpu().numpy()
    check(np.allclose(dvars3["da_z"].detach().cpu().numpy().flatten(), 3 * z3n ** 2, atol=1e-5), "3D da_z mismatch")
    check(np.allclose(dvars3["da_xz"].detach().cpu().numpy().flatten(), np.zeros_like(z3n), atol=1e-5), "3D mixed da_xz should be ~0")
    check(np.allclose(dvars3["db_y"].detach().cpu().numpy().flatten(), 2 * xyz[:, 1].detach().cpu().numpy(), atol=1e-5), "3D db_y mismatch")

    # ---------------------------------------------------------------
    # 5) Results-panel wiring
    # ---------------------------------------------------------------
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(prefix="tm_results_test_")
    try:
        sol_dir = os.path.join(tmpdir, "solution_results")
        os.makedirs(sol_dir, exist_ok=True)
        # A minimal real PNG (matplotlib) so QPixmap can actually load it.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(); plt.plot([0, 1], [0, 1])
        plt.savefig(os.path.join(sol_dir, "monitor_probe1_plot.png"))
        plt.close()

        win3 = MainWindow()
        check(win3.monitor_container.isHidden(), "the monitor panel should start hidden")
        win3._display_monitor_plots(tmpdir)
        check(not win3.monitor_container.isHidden(), "the monitor panel should reveal itself once a plot is found")
        check(win3.monitor_plot_files == [("probe1", os.path.join(sol_dir, "monitor_probe1_plot.png"))],
              f"should discover exactly the one monitor plot; got {win3.monitor_plot_files!r}")
        check(not win3.monitor_label.pixmap().isNull(), "the monitor label should have a real pixmap loaded")

        win3._display_monitor_plots(os.path.join(tmpdir, "does_not_exist"))
        check(win3.monitor_container.isHidden(), "the monitor panel should hide again when no plot files are found")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL TRAINING-MONITOR TESTS PASSED")
    return 0


def test_training_monitors():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
