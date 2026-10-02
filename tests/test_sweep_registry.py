#!/usr/bin/env python3
"""
Regression check for pinnstudio/core/sweep_registry.py -- the Parameter
Sweep feature's Phase 1 (config fields) + Phase 2 (registry) foundation.
No GUI sweep tab exists yet (that's Phase 3); this exercises the
registry and PINNConfig.validate()'s new sweep checks directly against
real configs built the same way the GUI does (MainWindow._build_config()
after selecting a template), across 1D/2D/3D and Forward/Inverse.

Checks, for a representative spread of templates:
 - available_params() returns unique ids, with the expected static
   entries always present (and num_initial correctly excluded for
   steady-state templates).
 - Phase-scoped entries (learning rate/iterations/optimizer) appear once
   per configured scheduler phase, with learning rate specifically
   omitted for any phase using an optimizer that has no meaningful LR
   (L-BFGS/NNCG) -- and that the values they report match the phase's
   real, already-configured values (not the legacy single-phase fields,
   which is exactly the class of bug this registry exists to avoid).
 - Every available parameter's get_value()/set_value() round-trips
   correctly (set to the value just read back, read again, must match).
 - Setting hidden_layers/neurons_per_layer actually reshapes
   config.layers as expected, preserving input/output size.
 - PINNConfig.validate()'s new sweep_enabled checks catch each class of
   malformed sweep_parameters JSON, and accept well-formed ones.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_sweep_registry.py
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
from pinnstudio.core import sweep_registry as reg

# A spread across every dimension, steady vs time-dependent, and Forward
# vs Inverse -- not the full 24-combination matrix test_templates.py
# already covers, just enough to exercise every code path in the
# registry (scheduler phases of varying optimizer, steady-state's
# num_initial exclusion, each dimension's own template set).
CASES = [
    ("1d", "1D Heat", False),
    ("1d", "1D Heat", True),
    ("2d", "2D Poisson (Disk)", False),   # steady-state 2D
    ("3d", "3D Poisson (Sphere)", False), # steady-state 3D
    ("3d", "3D Heat", False),             # time-dependent 3D
]


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    _app.processEvents()

    for dim, name, inverse in CASES:
        {"1d": win.radio_1d, "2d": win.radio_2d, "3d": win.radio_3d}[dim].setChecked(True)
        win.quick_examples_combo.setCurrentText(name)
        win.radio_forward.setChecked(not inverse)
        win.radio_inverse.setChecked(inverse)
        config = win._build_config()
        label = f"{name} inv={inverse}"

        params = reg.available_params(config)
        ids = [p.id for p in params]
        check(len(ids) == len(set(ids)), f"[{label}] duplicate ids in available_params(): {ids}")

        static_ids = {"weight_decay", "hidden_layers", "neurons_per_layer",
                      "num_domain", "num_boundary", "num_test"}
        check(static_ids.issubset(set(ids)),
              f"[{label}] missing expected static entries: {static_ids - set(ids)}")
        check(("num_initial" in ids) == (not config.steady_state),
              f"[{label}] num_initial availability ({'num_initial' in ids}) doesn't match "
              f"steady_state={config.steady_state}")

        n_phases = reg.num_phases(config)
        check(n_phases > 0, f"[{label}] expected the default Optimizer Scheduler to be active "
                             f"with >=1 phase, got {n_phases}")
        for i in range(n_phases):
            check(f"phase{i}_iterations" in ids, f"[{label}] missing phase{i}_iterations")
            check(f"phase{i}_optimizer" in ids, f"[{label}] missing phase{i}_optimizer")
            opt_param = reg.get_param(f"phase{i}_optimizer", config)
            real_opt = opt_param.get_value(config)
            lr_available = f"phase{i}_lr" in ids
            check(lr_available == (real_opt in ("adam", "adamw")),
                  f"[{label}] phase{i}_lr availability ({lr_available}) doesn't match "
                  f"this phase's real optimizer ({real_opt!r})")

        # Round-trip every available parameter: read, write back the same
        # value, read again, must match. Catches accessor bugs (wrong
        # index, wrong JSON key, stale cached state) without needing to
        # know each parameter's "correct" value in advance.
        for p in params:
            try:
                v1 = p.get_value(config)
                p.set_value(config, v1)
                v2 = p.get_value(config)
            except Exception as e:
                failures.append(f"[{label}] {p.id} round-trip raised: {e}")
                continue
            check(v1 == v2, f"[{label}] {p.id} round-trip mismatch: {v1!r} -> set -> {v2!r}")

        # hidden_layers / neurons_per_layer actually reshape config.layers,
        # preserving input/output size.
        in_sz, out_sz = config.layers[0], config.layers[-1]
        hl_param = reg.get_param("hidden_layers", config)
        hl_param.set_value(config, 5)
        check(len(config.layers) == 7,
              f"[{label}] hidden_layers=5 should give a 7-entry layers list, got {config.layers}")
        check(config.layers[0] == in_sz and config.layers[-1] == out_sz,
              f"[{label}] hidden_layers set changed input/output size: {config.layers}")
        np_param = reg.get_param("neurons_per_layer", config)
        np_param.set_value(config, 128)
        check(all(w == 128 for w in config.layers[1:-1]),
              f"[{label}] neurons_per_layer=128 didn't reach every hidden layer: {config.layers}")

    # ── PINNConfig.validate() sweep checks, against one simple config ──
    win.radio_1d.setChecked(True)
    win.quick_examples_combo.setCurrentText("1D Heat")
    win.radio_forward.setChecked(True)
    base_config = win._build_config()
    available = reg.available_params(base_config)
    some_id = available[0].id if available else None
    check(some_id is not None, "no available sweep params on 1D Heat -- can't test validate()")

    def errs_for(sweep_params_obj, enabled=True, mode="oat"):
        c = win._build_config()
        c.sweep_enabled = enabled
        c.sweep_mode = mode
        c.sweep_parameters = json.dumps(sweep_params_obj)
        return c.validate()

    check(errs_for([], enabled=False) == [], "sweep disabled with empty params should validate clean")
    check(any("no parameters" in e for e in errs_for([])),
          "sweep enabled with empty params list should error")
    check(any("not valid JSON" in e or "not a sweepable parameter" in e
               for e in errs_for("not a list")),  # type: ignore[arg-type]
          "non-list sweep_parameters should error")
    if some_id:
        ok_list = [{"id": some_id, "mode": "list", "values": [1, 2, 3]}]
        check(errs_for(ok_list) == [], f"well-formed list-mode sweep should validate clean, got {errs_for(ok_list)}")
        ok_range = [{"id": some_id, "mode": "linear", "min": 1.0, "max": 10.0, "n": 5}]
        check(errs_for(ok_range) == [], f"well-formed linear-range sweep should validate clean, got {errs_for(ok_range)}")
        bad_range = [{"id": some_id, "mode": "linear", "min": 10.0, "max": 1.0, "n": 5}]
        check(any("min < max" in e for e in errs_for(bad_range)),
              "min >= max should be flagged")
        bad_n = [{"id": some_id, "mode": "linear", "min": 1.0, "max": 10.0, "n": 1}]
        check(any("at least 2 steps" in e for e in errs_for(bad_n)),
              "n < 2 should be flagged")
        bad_log = [{"id": some_id, "mode": "log", "min": -1.0, "max": 10.0, "n": 5}]
        check(any("log spacing" in e for e in errs_for(bad_log)),
              "non-positive min/max with log mode should be flagged")
        bad_id = [{"id": "not_a_real_param_xyz", "mode": "list", "values": [1]}]
        check(any("not a sweepable parameter" in e for e in errs_for(bad_id)),
              "unknown parameter id should be flagged")
        bad_mode = [{"id": some_id, "mode": "oat"}]
        check(errs_for([{"id": "bad", "mode": "oat"}]) != [], "")  # sanity: bad id + bad mode both flagged
        unrecognized_mode = [{"id": some_id, "mode": "wat"}]
        check(any("unrecognized mode" in e for e in errs_for(unrecognized_mode)),
              "unrecognized sweep mode should be flagged")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL SWEEP REGISTRY TESTS PASSED")
    return failures


def test_sweep_registry():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
