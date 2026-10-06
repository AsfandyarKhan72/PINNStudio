#!/usr/bin/env python3
"""
Checks for the network-architecture feature: an independent Network Type
dropdown (FNN / DeepXDE's own built-in PFNN class) in the Neural Network
panel (see config.py's network_type field and MainWindow's
network_type_combo).

Covers:
  1. Config defaults + validate().
  2. _build_config()/_apply_config() UI wiring (including project save/
     load round-trip through dataclasses.asdict()/PINNConfig(**filtered),
     and backward compatibility for a config saved before this field
     existed).
  3. generate_script() / generate_clean_script() both embed the shared
     _make_net() helper (via _net_construction_helper_code()) and dispatch
     correctly for FNN vs PFNN -- and that model_config.json's dict
     literal records network_type so a later Model Restore can read it
     back.
  4. The actual _make_net() helper, executed directly (not just inspected
     as text): PFNN's nested per-branch layer shape (DeepXDE's own
     convention for dde.nn.PFNN) and PFNN silently dropping the
     weight-decay regularization kwarg (DeepXDE's PFNN has no such
     parameter).
  5. Backward compatibility: _net_construction_helper_code() with the
     same "key absent" default the 4 Model Restore / Error Analysis
     builders in main_window.py fall back to for a config saved before
     this feature existed.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_network_architecture.py
"""
import os
import sys
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
import deepxde as dde

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import (
    generate_script, generate_clean_script, _net_construction_helper_code,
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
    check(default_cfg.network_type == "FNN", "default network_type should be FNN")
    check(default_cfg.validate() == [], "default config should validate cleanly")

    # ---------------------------------------------------------------
    # 2) _build_config()/_apply_config() UI wiring
    # ---------------------------------------------------------------
    win = MainWindow()
    win.radio_1d.setChecked(True)
    win.quick_examples_combo.setCurrentText("1D Heat")

    config = win._build_config()
    check(config.network_type == "FNN", "_build_config() should default network_type to FNN")

    win.network_type_combo.setCurrentText("PFNN")
    pfnn_config = win._build_config()
    check(pfnn_config.network_type == "PFNN", "_build_config() should read network_type from the combo")
    # layers must stay a plain flat list -- PFNN's nested per-branch shape
    # is computed later, purely locally, inside _make_net(); never baked
    # into the stored config.
    check(all(isinstance(w, int) for w in pfnn_config.layers),
          "config.layers should stay a flat list of ints even when network_type is PFNN")

    # Round-trip through the same (de)serialization _save_problem()/
    # _open_problem() use (dataclasses.asdict() -> PINNConfig(**filtered)).
    payload = dataclasses.asdict(pfnn_config)
    known_fields = {f.name for f in dataclasses.fields(PINNConfig)}
    reloaded = PINNConfig(**{k: v for k, v in payload.items() if k in known_fields})
    check(reloaded.network_type == "PFNN",
          "asdict()/PINNConfig(**filtered) round-trip should preserve network_type")

    # A config saved before this feature existed simply lacks the key --
    # filtered construction must still default cleanly (not raise).
    old_payload = {k: v for k, v in payload.items() if k != "network_type"}
    old_reloaded = PINNConfig(**{k: v for k, v in old_payload.items() if k in known_fields})
    check(old_reloaded.network_type == "FNN",
          "a saved config missing network_type should fall back to FNN")

    # _apply_config() should push network_type back into the widget (used
    # by _open_problem() / quick-example loading).
    win2 = MainWindow()
    win2._apply_config(pfnn_config)
    check(win2.network_type_combo.currentText() == "PFNN",
          "_apply_config() should restore the Network type combo")

    # ---------------------------------------------------------------
    # 3) generate_script() / generate_clean_script() content
    # ---------------------------------------------------------------
    fnn_script = generate_script(config)
    ast.parse(fnn_script)
    check("_network_type = 'FNN'" in fnn_script, "generate_script() should embed network_type='FNN'")
    check("_make_net(" in fnn_script, "generate_script() should call _make_net(...) instead of a bare dde.nn.FNN(...)")
    check('"network_type": \'FNN\'' in fnn_script or '"network_type": "FNN"' in fnn_script,
          "model_config.json's dict literal should record network_type")

    fnn_clean = generate_clean_script(config)
    ast.parse(fnn_clean)
    check("_make_net(" in fnn_clean, "generate_clean_script() should also call _make_net(...)")

    pfnn_script = generate_script(pfnn_config)
    ast.parse(pfnn_script)
    check("_network_type = 'PFNN'" in pfnn_script, "generate_script() should embed network_type='PFNN'")
    check('"network_type": \'PFNN\'' in pfnn_script or '"network_type": "PFNN"' in pfnn_script,
          "model_config.json's dict literal should record network_type=PFNN")

    pfnn_clean = generate_clean_script(pfnn_config)
    ast.parse(pfnn_clean)
    check("_network_type = 'PFNN'" in pfnn_clean, "generate_clean_script() should also embed network_type='PFNN'")

    # Weight decay is passed through to _make_net(...) as a plain runtime
    # argument (rather than spliced into a `regularization=...` kwarg at
    # codegen time) -- _make_net() itself decides at runtime whether/how
    # to apply it, and silently drops it for PFNN.
    wd_config = dataclasses.replace(config, weight_decay=0.01)
    wd_script = generate_script(wd_config)
    check('_make_net(_layers, "tanh", "Glorot uniform", 0.01)' in wd_script,
          "FNN + weight_decay>0 should pass the weight_decay value through to _make_net(...)")
    wd_pfnn_config = dataclasses.replace(pfnn_config, weight_decay=0.01)
    wd_pfnn_script = generate_script(wd_pfnn_config)
    check('_make_net(_layers, "tanh", "Glorot uniform", 0.01)' in wd_pfnn_script,
          "PFNN + weight_decay>0 should still pass weight_decay through to _make_net(...) "
          "(which must itself drop it for PFNN -- checked directly in section 4 below)")

    # ---------------------------------------------------------------
    # 4) _make_net() executed directly
    # ---------------------------------------------------------------
    def _make_net_ns(network_type):
        ns = {"dde": dde, "torch": torch}
        exec(_net_construction_helper_code(network_type), ns)
        return ns["_make_net"]

    # Plain FNN: layer_sizes pass through unchanged.
    make_net_fnn = _make_net_ns("FNN")
    net = make_net_fnn([2, 32, 32, 3], "tanh", "Glorot uniform")
    check(isinstance(net, dde.nn.FNN), "network_type=FNN should build a dde.nn.FNN")

    # PFNN: hidden layers become per-branch lists of width n_out; input/
    # output entries stay scalars (DeepXDE's own dde.nn.PFNN convention).
    make_net_pfnn = _make_net_ns("PFNN")
    net_p = make_net_pfnn([2, 32, 32, 3], "tanh", "Glorot uniform")
    check(isinstance(net_p, dde.nn.PFNN), "network_type=PFNN should build a dde.nn.PFNN")

    # PFNN silently drops weight_decay (no `regularization` kwarg exists
    # on DeepXDE's PFNN -- confirmed against the installed backend).
    try:
        make_net_pfnn([2, 32, 32, 3], "tanh", "Glorot uniform", 0.01)
        pfnn_wd_ok = True
    except TypeError:
        pfnn_wd_ok = False
    check(pfnn_wd_ok, "_make_net() must not pass regularization to PFNN even when weight_decay>0")

    # ---------------------------------------------------------------
    # 5) Backward-compatible default in the restore-side helper
    # ---------------------------------------------------------------
    old_cfg_like = {}  # simulates cfg.get(...) against a model_config.json saved before this feature
    legacy_helper = _net_construction_helper_code(old_cfg_like.get("network_type", "FNN"))
    check("_network_type = 'FNN'" in legacy_helper,
          "restore-site default (key absent) should reproduce FNN exactly")

    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL NETWORK-ARCHITECTURE TESTS PASSED")
    return 0


def test_network_architecture():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
