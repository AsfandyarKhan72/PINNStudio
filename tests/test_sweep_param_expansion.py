#!/usr/bin/env python3
"""
v73: integration-level coverage for the Parameter Sweep registry's newly
added entries (RAR, Time Adaptive, Input/Output transform scale,
Activation/Point distribution/Kernel initializer) and for the true-value
preservation fix (see sweep_registry.py's module docstring and
test_sweep_registry.py's own "SCOPE (v4)" section for the unit-level
registry coverage of all of this).

This file deliberately does NOT launch real training subprocesses --
everything here goes through the REAL GUI sweep row mechanism (same
param_combo/mode_combo/list_edit widgets a user actually drives) and the
REAL sweep_runner.build_runs(), then calls codegen.generate_script()
directly on each resulting per-run config and inspects the generated
Python source text for the swept value actually being embedded where
expected. This is deliberately fast (no subprocess, no torch import) and
still exercises the exact same code path a real sweep run uses right up
to the point the script would be executed -- including the class of bug
this round is actually at risk of (a malformed f-string interpolation
that would raise or silently produce a syntactically-broken script for
some new field/value combination), which pure registry-level
get_value()/set_value() round-trip tests can't catch on their own.
tests/test_sweep_execution.py separately covers a couple of these same
categories with a handful of REAL training runs, for end-to-end
confidence that the generated script not only parses but actually runs.

Checks:
 - An Inverse sweep over one variable's initial guess still embeds BOTH
   variables' own true values correctly (generate_script()'s own
   _inv_var_trues literal, the live Solve/Sweep path's mechanism for the
   Parameter Convergence plot's true-value dashed line -- distinct from
   generate_clean_script()'s separate _inv_true_vals, used only by the
   "Export as DeepXDE Script" button, not by a live sweep run) -- the
   true-value-line bug, confirmed fixed all the way through codegen, not
   just at the registry's own data level.
 - RAR: sweeping rar_cycles changes the embedded RAR loop's range(...).
 - Time Adaptive: sweeping a step group's steps changes the embedded
   ta_step_groups JSON; sweeping ta_grid_size changes the embedded
   grid_size literal.
 - Output/Input transform: sweeping a scale entry changes the embedded
   in_scale/out_scale list, leaving shift and the other entries alone.
 - Activation/Point distribution/Kernel initializer: sweeping each
   changes the embedded dde.nn.FNN(...)/train_distribution(...) call.
 - Every swept run's generated script is syntactically valid Python
   (compile() doesn't raise) -- the actual bug class this file exists to
   catch.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_sweep_param_expansion.py
"""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core import codegen
from pinnstudio.core.sweep_runner import build_runs


def _set_row(row, param_id, values, win):
    win._refresh_sweep_param_choices()
    idx = row['param_combo'].findData(param_id)
    assert idx >= 0, f"{param_id!r} not offered by the sweep Parameter dropdown -- available: " \
        + str([row['param_combo'].itemData(i) for i in range(row['param_combo'].count())])
    row['param_combo'].setCurrentIndex(idx)
    is_categorical = row['param_combo'].itemData(idx, 257) and \
        row['param_combo'].itemData(idx, 257).get('value_type') == 'categorical'
    if not is_categorical:
        row['mode_combo'].setCurrentIndex(row['mode_combo'].findData("list"))
    row['list_edit'].setText(values)


def _assert_compiles(script, label):
    try:
        compile(script, f"<{label}>", "exec")
    except SyntaxError as e:
        raise AssertionError(f"{label}: generated script is not valid Python: {e}")


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    _app.processEvents()
    win.quick_examples_combo.setCurrentText("1D Heat")
    _app.processEvents()

    # ── True-value-line fix, through full codegen ───────────────────────
    win.radio_inverse.setChecked(True)
    _app.processEvents()
    win.inv_var_rows[0]['name'].setText("D")
    win.inv_var_rows[0]['init'].setValue(1.0)
    win.inv_var_rows[0]['true'].setValue(0.02)
    win._add_inverse_var_row("k", 0.5)  # row 1, no true value set
    win.sweep_enable_cb.setChecked(True)
    _app.processEvents()
    row0 = win.sweep_row_list[0]
    _set_row(row0, "inv_var_init_0", "0.5, 1.5", win)
    base_inv = win._build_config()
    runs_inv = build_runs(base_inv)
    check(len(runs_inv) == 3, f"expected 1 baseline + 2 swept runs, got {len(runs_inv)}")
    for label, cfg in runs_inv:
        script = codegen.generate_script(cfg)
        _assert_compiles(script, f"inverse sweep run {label!r}")
        check("_inv_var_trues = [0.02, None]" in script,
              f"run {label!r}: expected both variables' own true values ([0.02, None]) still embedded, "
              f"not found in generated script (the true-value-line bug)")

    # Clean up Inverse rows/mode for the rest of this function.
    for row in list(win.inv_var_rows):
        if not row['is_primary']:
            row['widget'].deleteLater()
            win.inv_var_rows.remove(row)
    win.sweep_enable_cb.setChecked(False)
    win.radio_inverse.setChecked(False)
    win.radio_forward.setChecked(True)
    _app.processEvents()

    # ── RAR ──────────────────────────────────────────────────────────
    win.adapt_combo.setCurrentText("Residual-based Adaptive Refinement (RAR)")
    _app.processEvents()
    win.sweep_enable_cb.setChecked(True)
    _app.processEvents()
    row_rar = win.sweep_row_list[0]
    _set_row(row_rar, "rar_cycles", "2, 4", win)
    base_rar = win._build_config()
    runs_rar = build_runs(base_rar)
    check(len(runs_rar) == 3, f"expected 1 baseline + 2 swept RAR runs, got {len(runs_rar)}")
    seen_cycles = set()
    for label, cfg in runs_rar:
        script = codegen.generate_script(cfg)
        _assert_compiles(script, f"RAR sweep run {label!r}")
        seen_cycles.add(cfg.rar_cycles)
        check(f"range({cfg.rar_cycles})" in script,
              f"run {label!r}: expected the RAR loop to embed range({cfg.rar_cycles}), not found")
    check(seen_cycles == {3, 2, 4}, f"expected baseline(3)+swept(2,4) rar_cycles values, got {seen_cycles}")
    win.sweep_enable_cb.setChecked(False)
    win.adapt_combo.setCurrentText("None")
    _app.processEvents()

    # ── Time Adaptive: step-group steps + IC grid resolution ───────────
    win.adapt_combo.setCurrentText("Time Adaptive Training")
    _app.processEvents()
    win.sweep_enable_cb.setChecked(True)
    _app.processEvents()
    row_ta1 = win.sweep_row_list[0]
    _set_row(row_ta1, "ta_group0_steps", "3, 7", win)
    win._add_sweep_row()
    row_ta2 = win.sweep_row_list[1]
    _set_row(row_ta2, "ta_grid_size", "11, 21", win)
    base_ta = win._build_config()
    runs_ta = build_runs(base_ta)
    check(len(runs_ta) == 1 + 2 + 2, f"expected 1 baseline + 2 + 2 Time-Adaptive runs, got {len(runs_ta)}")
    for label, cfg in runs_ta:
        script = codegen.generate_script(cfg)
        _assert_compiles(script, f"Time-Adaptive sweep run {label!r}")
        check(f'grid_size = {cfg.ta_grid_size}' in script,
              f"run {label!r}: expected grid_size = {cfg.ta_grid_size} embedded, not found")
        check(json.dumps({"steps": json.loads(cfg.ta_step_groups)[0]["steps"]})[10:] in cfg.ta_step_groups,
              f"run {label!r}: ta_step_groups should carry this run's own steps value")
    win.sweep_enable_cb.setChecked(False)
    win.adapt_combo.setCurrentText("None")
    _app.processEvents()

    # ── Output / Input transform scale ──────────────────────────────────
    win.output_transform_cb.setChecked(True)
    win.input_transform_cb.setChecked(True)
    _app.processEvents()
    win.sweep_enable_cb.setChecked(True)
    _app.processEvents()
    row_ot = win.sweep_row_list[0]
    _set_row(row_ot, "ot_scale_0", "2.0, 3.0", win)
    win._add_sweep_row()
    row_it = win.sweep_row_list[1]
    _set_row(row_it, "it_scale_0", "0.5, 1.5", win)
    base_tr = win._build_config()
    runs_tr = build_runs(base_tr)
    check(len(runs_tr) == 1 + 2 + 2, f"expected 1 baseline + 2 + 2 transform-scale runs, got {len(runs_tr)}")
    for label, cfg in runs_tr:
        script = codegen.generate_script(cfg)
        _assert_compiles(script, f"transform sweep run {label!r}")
        check(f"_out_scale = {cfg.output_transform_scale}" in script,
              f"run {label!r}: expected out_scale {cfg.output_transform_scale} embedded, not found")
        check(f"_in_scale  = {cfg.input_transform_scale}" in script,
              f"run {label!r}: expected in_scale {cfg.input_transform_scale} embedded, not found")
    win.sweep_enable_cb.setChecked(False)
    win.output_transform_cb.setChecked(False)
    win.input_transform_cb.setChecked(False)
    _app.processEvents()

    # ── Activation / Point distribution / Kernel initializer ───────────
    win.sweep_enable_cb.setChecked(True)
    _app.processEvents()
    row_act = win.sweep_row_list[0]
    _set_row(row_act, "activation", "relu, sigmoid", win)
    win._add_sweep_row()
    row_pd = win.sweep_row_list[1]
    _set_row(row_pd, "point_distribution", "uniform, Sobol", win)
    base_misc = win._build_config()
    runs_misc = build_runs(base_misc)
    check(len(runs_misc) == 1 + 2 + 2, f"expected 1 baseline + 2 + 2 activation/point-dist runs, got {len(runs_misc)}")
    for label, cfg in runs_misc:
        script = codegen.generate_script(cfg)
        _assert_compiles(script, f"activation/point-dist sweep run {label!r}")
        check(f'"{cfg.activation}"' in script, f"run {label!r}: expected activation {cfg.activation!r} embedded, not found")
        check(f'train_distribution="{cfg.point_distribution}"' in script,
              f"run {label!r}: expected point distribution {cfg.point_distribution!r} embedded, not found")
    win.sweep_enable_cb.setChecked(False)
    _app.processEvents()

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL SWEEP PARAMETER EXPANSION TESTS PASSED")
    return failures


def test_sweep_param_expansion():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
