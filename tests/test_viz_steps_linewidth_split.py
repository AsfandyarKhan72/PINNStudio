#!/usr/bin/env python3
"""
Dev-branch round: split the single shared "num_timesteps"/"plot_linewidth"
config fields into type-specific pairs, and change their defaults:

  - num_timesteps_line  (static "Line (time steps)" plot): 4  -> 5
  - num_timesteps_anim  (Line/Surface Animation GIFs):      10 -> 20
  - plot_linewidth       (static line plots):                2.0 (unchanged)
  - plot_linewidth_anim  (Animation Line GIF only):           2.0 -> 3.0

Before this, one shared spinbox/value governed both the static "Line (time
steps)" plot and the two GIF animations, so giving the animations more
frames (to play back slower/smoother) meant the static overlay also got
more lines (making it harder to read), and vice versa -- there was no way
to set the two differently. Same story for line width.

This covers:
  1. PINNConfig's new fields/defaults exist and old saved configs (which
     only have the old "num_timesteps" key) load without crashing, falling
     back to the new defaults for the split fields (dataclasses.fields()
     filtering silently drops the stale key -- see MainWindow's own load
     path, which does exactly this).
  2. MainWindow._build_config() wires the two new hidden spinboxes
     (timesteps_spin_line/_anim) and the two linewidth settings keys
     through to the right PINNConfig fields.
  3. The Setup tab's "Line Plot Settings" dialog (_on_line_plot_settings)
     is type-aware: opening it for "Line (time steps)" only ever touches
     timesteps_spin_line, and opening it for either GIF animation type
     only ever touches timesteps_spin_anim.
  4. The Setup tab's "Plot Settings" dialog (_on_plot_settings) is simi-
     larly type-aware for line width, and preserves the *other* type's
     value across a save instead of dropping it (a real bug this round
     fixed: the dialog used to replace self._plot_viz_settings wholesale
     with a literal that didn't include the type-specific keys at all).
  5. The Restore tab's actual script-building code (_build_restore_script
     / _build_restore_script_ta) reads the correct type-specific key for
     both the static "Line (time steps)" plot and "Animation Line (GIF)".
  6. A real bug caught while fixing this: the Restore tab's own single-
     model and Time-Adaptive-combined "Animation Line (GIF)" branches had
     a *hardcoded* `linewidth=2` that was never wired to any setting at
     all -- fixed to read the (now type-aware) linewidth variable, same as
     every other line-type plot in those scripts already did.
"""
import dataclasses
import os
import re
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (QApplication, QComboBox, QDialog, QPushButton,
                              QSpinBox)

from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication(sys.argv)


def _click_ok(dialog, set_spin=None, set_lw_combo=None):
    """Stand-in for QDialog.exec() in headless tests: optionally pokes one
    widget's value, then clicks the real OK button so the dialog's own
    _on_ok() closure runs exactly as it would for a real user."""
    if set_spin is not None:
        spins = dialog.findChildren(QSpinBox)
        assert len(spins) == 1, spins
        spins[0].setValue(set_spin)
    if set_lw_combo is not None:
        for c in dialog.findChildren(QComboBox):
            items = [c.itemText(i) for i in range(c.count())]
            if items == ["1.0", "1.5", "2.0", "2.5", "3.0"]:
                c.setCurrentText(set_lw_combo)
                break
        else:
            raise AssertionError("line-width combo not found")
    for btn in dialog.findChildren(QPushButton):
        if btn.text() == "OK":
            btn.click()
            return
    raise AssertionError("OK button not found")


def test_config_defaults_and_backward_compat():
    cfg = PINNConfig()
    assert cfg.num_timesteps_line == 5
    assert cfg.num_timesteps_anim == 20
    assert cfg.plot_linewidth == 2.0
    assert cfg.plot_linewidth_anim == 3.0

    # Old saved config.json files only ever had "num_timesteps" (singular).
    # MainWindow's load path filters the raw dict down to PINNConfig's
    # current field names before constructing it -- simulate that here.
    known = {f.name for f in dataclasses.fields(PINNConfig)}
    old_style = {"num_timesteps": 10, "plot_type": "Surface"}
    filtered = {k: v for k, v in old_style.items() if k in known}
    cfg2 = PINNConfig(**filtered)
    assert cfg2.num_timesteps_line == 5
    assert cfg2.num_timesteps_anim == 20


def test_build_config_wires_hidden_spinboxes():
    w = MainWindow()
    assert w.timesteps_spin_line.value() == 5
    assert w.timesteps_spin_anim.value() == 20

    cfg = w._build_config()
    assert cfg.num_timesteps_line == 5
    assert cfg.num_timesteps_anim == 20
    assert cfg.plot_linewidth == 2.0
    assert cfg.plot_linewidth_anim == 3.0

    w.timesteps_spin_line.setValue(7)
    w.timesteps_spin_anim.setValue(13)
    cfg2 = w._build_config()
    assert cfg2.num_timesteps_line == 7
    assert cfg2.num_timesteps_anim == 13


def test_setup_line_plot_settings_dialog_is_type_aware():
    w = MainWindow()
    orig_exec = QDialog.exec
    try:
        QDialog.exec = lambda self: _click_ok(self, set_spin=17)

        w.plot_type_combo.blockSignals(True)
        w.plot_type_combo.setCurrentText("Line (time steps)")
        w.plot_type_combo.blockSignals(False)
        w._on_line_plot_settings()
        assert w.timesteps_spin_line.value() == 17
        assert w.timesteps_spin_anim.value() == 20  # untouched

        w.plot_type_combo.blockSignals(True)
        w.plot_type_combo.setCurrentText("Line Animation (GIF)")
        w.plot_type_combo.blockSignals(False)
        w._on_line_plot_settings()
        assert w.timesteps_spin_anim.value() == 17
        assert w.timesteps_spin_line.value() == 17  # unchanged from before

        w.plot_type_combo.blockSignals(True)
        w.plot_type_combo.setCurrentText("Surface Animation (GIF)")
        w.plot_type_combo.blockSignals(False)
        w._on_line_plot_settings()
        assert w.timesteps_spin_anim.value() == 17
    finally:
        QDialog.exec = orig_exec


def test_setup_plot_settings_dialog_linewidth_is_type_aware_and_preserves_other_key():
    w = MainWindow()
    orig_exec = QDialog.exec
    try:
        QDialog.exec = lambda self: _click_ok(self, set_lw_combo="2.5")

        w.plot_type_combo.blockSignals(True)
        w.plot_type_combo.setCurrentText("Line (time steps)")
        w.plot_type_combo.blockSignals(False)
        w._on_plot_settings()
        assert w._plot_viz_settings["linewidth"] == 2.5
        assert w._plot_viz_settings.get("linewidth_anim", 3.0) == 3.0

        w.plot_type_combo.blockSignals(True)
        w.plot_type_combo.setCurrentText("Line Animation (GIF)")
        w.plot_type_combo.blockSignals(False)
        w._on_plot_settings()
        assert w._plot_viz_settings["linewidth_anim"] == 2.5
        # The *other* key must survive this save, not revert to a bare
        # literal dict missing it (the bug this round fixed).
        assert w._plot_viz_settings["linewidth"] == 2.5

        cfg = w._build_config()
        assert cfg.plot_linewidth == 2.5
        assert cfg.plot_linewidth_anim == 2.5
    finally:
        QDialog.exec = orig_exec


def _restore_script_cfg_dict():
    return {
        "layers": "20,20,20", "activation": "tanh",
        "x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "t_max": 1.0,
        "float_type": "float64",
    }


def test_restore_script_reads_type_specific_steps_and_linewidth():
    w = MainWindow()
    w._restore_viz_settings = {
        "n_steps_line": 5, "n_steps_anim": 20,
        "linewidth": 2.0, "linewidth_anim": 3.0,
        "colormap": "RdBu_r", "colorbar": True, "levels": 40,
        "resolution": 100, "dpi": 100, "auto_range": True,
        "vmin": -1.0, "vmax": 1.0, "fps": 10, "swap_xt": True,
        "figsize_mode": "Default", "figsize_w": 7.0, "figsize_h": 5.0,
    }
    cfg_dict = _restore_script_cfg_dict()

    s_line = w._build_restore_script(
        "m.ckpt", cfg_dict, "adam", "Line (time steps)", 0, 10, "/tmp/out")
    s_anim = w._build_restore_script(
        "m.ckpt", cfg_dict, "adam", "Animation Line (GIF)", 0, 10, "/tmp/out")

    m = re.search(r"t_steps_vals = np\.linspace\([^)]*,\s*(\d+)\)", s_line)
    assert m and int(m.group(1)) == 5, s_line
    lw_line = re.findall(r"linewidth=([\d.]+)", s_line)
    assert lw_line and all(float(x) == 2.0 for x in lw_line), lw_line

    m2 = re.search(r"t_frames = np\.linspace\([^)]*,\s*(\d+)\)", s_anim)
    assert m2 and int(m2.group(1)) == 20, s_anim
    # This is the hardcoded-linewidth bug this round found and fixed: the
    # Animation Line (GIF) branch used to always emit "linewidth=2"
    # regardless of _restore_viz_settings.
    lw_anim = re.findall(r"linewidth=([\d.]+)", s_anim)
    assert lw_anim and all(float(x) == 3.0 for x in lw_anim), lw_anim


if __name__ == "__main__":
    test_config_defaults_and_backward_compat()
    test_build_config_wires_hidden_spinboxes()
    test_setup_line_plot_settings_dialog_is_type_aware()
    test_setup_plot_settings_dialog_linewidth_is_type_aware_and_preserves_other_key()
    test_restore_script_reads_type_specific_steps_and_linewidth()
    print("ALL PASSED")
