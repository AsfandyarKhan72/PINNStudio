#!/usr/bin/env python3
"""
Dev-branch round: standardized, timestamped per-run results folders.

Before this, every run (Solve, Export Script, or Restore) that saved to
disk wrote directly into the exact "Save to:"/restore-save-path folder
the user configured -- so a second run silently overwrote the first
run's model/plots/config. For someone tracking many runs for a paper,
that meant only the latest run of each kind ever survived.

Now every such action gets its own "<label>__<timestamp>" subfolder
underneath that unchanged base path (see MainWindow._timestamped_save_dir,
_run_folder_label, _run_results_dir). The label is the active Quick
Example's name when one is selected, or a short dimension/stationarity/
problem-type label for a custom (non-template) problem; for Restore
(which has no Quick Example of its own) it's a generic "Restore__..."
name, refined to a dimension/stationarity/problem-type label once the
restored model's own config.json is loaded.

Parameter Sweep is NOT exercised here: sweep_runner._apply_run_output_
settings always overwrites config.save_dir with its own per-run folder
right after _build_config() returns, regardless of what this feature
computed, so a sweep run never ends up double-nested -- see that
function's own docstring.
"""
import json
import os
import re
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication(sys.argv)

_STAMP_RE = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"


def test_timestamped_save_dir_basics():
    assert MainWindow._timestamped_save_dir("", "Foo") == ""
    assert MainWindow._timestamped_save_dir("   ", "Foo") == ""

    p = MainWindow._timestamped_save_dir("/tmp/results", "My Example (v2)")
    assert re.match(rf"^/tmp/results/My_Example_v2__{_STAMP_RE}$", p), p


def test_build_config_gives_each_run_its_own_folder():
    w = MainWindow()
    w.save_dir_input.setText("/Users/asykhan/PINNStudio_Results")

    # No Quick Example selected -> dimension/stationarity/problem-type
    # fallback label.
    cfg = w._build_config()
    m = re.match(
        rf"^/Users/asykhan/PINNStudio_Results/(1D|2D|3D)_(Stationary|TimeDependent)_(Forward|Inverse)__{_STAMP_RE}$",
        cfg.save_dir)
    assert m, cfg.save_dir

    # Blank base path -> save-to-disk stays off, untouched.
    w.save_dir_input.setText("")
    cfg2 = w._build_config()
    assert cfg2.save_dir == ""

    # A Quick Example's own name becomes the label.
    w.save_dir_input.setText("/Users/asykhan/PINNStudio_Results")
    if w.quick_examples_combo.count() > 1:
        w.quick_examples_combo.blockSignals(True)
        w.quick_examples_combo.setCurrentIndex(1)
        w.quick_examples_combo.blockSignals(False)
        chosen = w.quick_examples_combo.currentText()
        cfg3 = w._build_config()
        expected_label = MainWindow._sanitize_run_label(chosen)
        assert cfg3.save_dir.split("/")[-1].startswith(expected_label + "__"), cfg3.save_dir


def test_restore_parameter_convergence_branch_gets_own_folder():
    w = MainWindow()
    w.restore_save_path.setText("/tmp/restore_test_base")

    # Flip to Inverse Model (to unlock the Parameter Convergence viz
    # items) without going through the mode-change signal handler --
    # unrelated, pre-existing UI behavior on that signal pops a modal
    # dialog that would block this headless test.
    w.restore_mode_combo.blockSignals(True)
    w.restore_mode_combo.setCurrentText("Inverse Model")
    w.restore_mode_combo.blockSignals(False)
    items = w._restore_forward_viz_items() + list(w._RESTORE_PARAM_VIZ)
    w.restore_viz_combo.blockSignals(True)
    w.restore_viz_combo.clear()
    w.restore_viz_combo.addItems(items)
    w.restore_viz_combo.setCurrentText("Parameter Convergence Plot (PNG)")
    w.restore_viz_combo.blockSignals(False)

    # No convergence files configured -> _on_restore logs an error and
    # returns before starting anything, but only AFTER it has already
    # resolved and stashed this run's own timestamped folder.
    w._on_restore()
    got = getattr(w, "_last_restore_save_dir", None)
    assert got and re.match(
        rf"^/tmp/restore_test_base/ParameterConvergence__{_STAMP_RE}$", got), got

    # _on_restore_done must read that SAME resolved path back, not
    # recompute a fresh (and different) one from the raw widget text.
    w._last_restore_is_param = True
    w._on_restore_done(success=False)  # must not raise


def test_restore_main_branch_relabels_from_loaded_model_config():
    w = MainWindow()
    w.restore_save_path.setText("/tmp/restore_test_base")

    # Deliberately omit "layers" so _build_restore_script raises a
    # KeyError that _on_restore's own try/except catches and returns
    # from BEFORE spawning the real restore subprocess/thread -- this
    # keeps the test fast and side-effect-free while still exercising
    # the cfg-based relabel, which happens just after the config.json
    # is parsed and before that script-building call.
    cfgdict = {"problem_dim": "3D", "steady_state": False, "problem_type": "Forward"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfgdict, f)
        cfg_path = f.name

    w.restore_viz_combo.setCurrentText("Surface")
    w.restore_model_path.setText("/tmp/does_not_exist.ckpt")
    w.restore_config_path.setText(cfg_path)
    w._on_restore()
    got = getattr(w, "_last_restore_save_dir", None)
    assert got and re.match(
        rf"^/tmp/restore_test_base/Restore_3D_TimeDependent_Forward__{_STAMP_RE}$", got), got


if __name__ == "__main__":
    test_timestamped_save_dir_basics()
    test_build_config_gives_each_run_its_own_folder()
    test_restore_parameter_convergence_branch_gets_own_folder()
    test_restore_main_branch_relabels_from_loaded_model_config()
    print("ALL PASSED")
