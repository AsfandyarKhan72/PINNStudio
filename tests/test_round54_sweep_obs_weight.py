#!/usr/bin/env python3
"""
Round 54 -- found by wiring test_sweep_execution.py into pytest (it used to
run only as a script): a Parameter Sweep over an Inverse problem's
measured-data loss weight ("Obs N: Data loss weight", inv_obs_weight_N)
never reached training. The setter updated inverse_obs_files_json (and the
legacy loss_weight_obs), but the always-on Optimizer Scheduler's phases
each carry their OWN copy of every loss weight -- observation terms
included, appended after the PDE/BC/IC slots in file order -- and
codegen.py compiles each phase with that copy verbatim. So every swept run
trained with the template's original weight (test_sweep_execution.py's
real run showed identical final losses for weight 100 vs 100000).

The setter now re-syncs every phase's weights at the observation slot via
the same helper the loss-weight sweeps already use (_phase_weights_set).
This test checks it without training; test_sweep_execution.py checks it
with real training.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round54_sweep_obs_weight.py
"""
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
from pinnstudio.core import sweep_registry


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    win.radio_2d.setChecked(True)
    win.quick_examples_combo.setCurrentText("2D Poisson (Disk)")
    win.radio_inverse.setChecked(True)
    _app.processEvents()
    cfg = win._build_config()
    params = {p.id: p for p in sweep_registry.available_params(cfg)}
    check("inv_obs_weight_0" in params, "inv_obs_weight_0 should be available for the 2D Poisson (Disk) inverse template")
    if "inv_obs_weight_0" not in params:
        return failures
    n_slots = len(sweep_registry._weight_slot_descriptors(cfg))
    before = [ph["weights"] for ph in json.loads(cfg.scheduler_phases)]
    params["inv_obs_weight_0"].set_value(cfg, 100000.0)
    phases = json.loads(cfg.scheduler_phases)
    check(json.loads(cfg.inverse_obs_files_json)[0]["weight"] == 100000.0, "the obs file's own weight should be set")
    for i, ph in enumerate(phases):
        w = [float(x) for x in ph["weights"].split(",")]
        check(len(w) > n_slots and w[n_slots] == 100000.0,
              f"phase {i + 1}: the observation slot (index {n_slots}) should now be 100000.0, got {ph['weights']} "
              f"(was {before[i]})")
        check(w[:n_slots] == [float(x) for x in before[i].split(",")][:n_slots],
              f"phase {i + 1}: PDE/BC/IC weights should be untouched")
    check(params["inv_obs_weight_0"].get_value(cfg) == 100000.0, "get_value should read the new weight back")
    win.close()
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 54 SWEEP OBS-WEIGHT TESTS PASSED")
    return 0


def test_round54_sweep_obs_weight():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
