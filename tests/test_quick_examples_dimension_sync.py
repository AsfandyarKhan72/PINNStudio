#!/usr/bin/env python3
"""
Dev-branch round: panel reorg -- Quick Examples moved above Problem
Dimension, and a new Time Dependent/Stationary radio pair added to
Problem Type (before Forward/Inverse), replacing the old "Steady-state"
checkbox that used to sit in the Domain section.

Before this, the Quick Examples list was rebuilt and FILTERED down to
only the current dimension's templates every time the dimension radio
changed -- so picking an example required first picking the right
dimension by hand, backwards from what a "quick" example should be.
Now the combo always lists every template across all three dimensions;
picking one sets the matching dimension (and Forward/Inverse via the
existing per-template fields, and Time Dependent/Stationary) itself,
and changing dimension BY HAND instead resets the combo back to "None"
since whatever template was active no longer necessarily applies.

The old steady_state_check QCheckBox still exists and still drives
everything it always did (_on_steady_state_changed, the t-domain row,
every other isChecked()/setChecked() call site in main_window.py) --
it's just hidden now, two-way synced with the new visible
radio_time_dependent/radio_stationary pair instead of being the visible
control itself.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication(sys.argv)


def test_quick_examples_combo_lists_every_dimension_up_front():
    w = MainWindow()
    items = [w.quick_examples_combo.itemText(i) for i in range(w.quick_examples_combo.count())]
    assert items[0] == "None"
    for name in ("1D Heat", "1D Allen-Cahn", "1D Burgers", "1D Schrödinger",
                 "2D Heat", "2D Poisson (L-Shape)", "2D Poisson (Disk)",
                 "3D Heat", "3D Poisson (Sphere)"):
        assert name in items, f"{name!r} missing from the combo's full list: {items}"


def test_defaults():
    w = MainWindow()
    assert w.radio_1d.isChecked()
    assert w.radio_time_dependent.isChecked()
    assert not w.radio_stationary.isChecked()
    assert not w.steady_state_check.isChecked()
    assert w.quick_examples_combo.currentText() == "None"


def test_selecting_template_sets_matching_dimension_and_keeps_combo_text():
    w = MainWindow()
    assert not w.radio_2d.isChecked()
    w.quick_examples_combo.setCurrentText("2D Heat")
    assert w.radio_2d.isChecked()
    assert w.quick_examples_combo.currentText() == "2D Heat"

    assert not w.radio_3d.isChecked()
    w.quick_examples_combo.setCurrentText("3D Heat")
    assert w.radio_3d.isChecked()
    assert w.quick_examples_combo.currentText() == "3D Heat"

    assert not w.radio_1d.isChecked()
    w.quick_examples_combo.setCurrentText("1D Burgers")
    assert w.radio_1d.isChecked()
    assert w.quick_examples_combo.currentText() == "1D Burgers"


def test_steady_template_sets_stationary_radio_and_resets_on_time_dependent_template():
    w = MainWindow()
    w.quick_examples_combo.setCurrentText("2D Poisson (Disk)")
    assert w.steady_state_check.isChecked()
    assert w.radio_stationary.isChecked()
    assert not w.radio_time_dependent.isChecked()

    # A time-dependent template picked afterward must flip it back off.
    w.quick_examples_combo.setCurrentText("2D Heat")
    assert not w.steady_state_check.isChecked()
    assert w.radio_time_dependent.isChecked()
    assert not w.radio_stationary.isChecked()


def test_manual_dimension_change_resets_quick_examples_to_none():
    w = MainWindow()
    w.quick_examples_combo.setCurrentText("2D Heat")
    assert w.quick_examples_combo.currentText() == "2D Heat"

    # Switching dimension BY HAND (not via a template) invalidates the
    # active template.
    w.radio_1d.setChecked(True)
    assert w.quick_examples_combo.currentText() == "None"


def test_manual_radio_toggle_drives_steady_state_check_both_ways():
    w = MainWindow()
    w.radio_stationary.setChecked(True)
    assert w.steady_state_check.isChecked()

    w.radio_time_dependent.setChecked(True)
    assert not w.steady_state_check.isChecked()

    # And the reverse direction: driving the (hidden) checkbox directly,
    # the way _build_config()/config-load/template code still does,
    # must update the visible radios too.
    w.steady_state_check.setChecked(True)
    assert w.radio_stationary.isChecked()
    assert not w.radio_time_dependent.isChecked()

    w.steady_state_check.setChecked(False)
    assert w.radio_time_dependent.isChecked()
    assert not w.radio_stationary.isChecked()


def test_steady_state_check_never_added_to_a_visible_layout():
    w = MainWindow()
    assert w.steady_state_check.isVisible() is False


if __name__ == "__main__":
    test_quick_examples_combo_lists_every_dimension_up_front()
    test_defaults()
    test_selecting_template_sets_matching_dimension_and_keeps_combo_text()
    test_steady_template_sets_stationary_radio_and_resets_on_time_dependent_template()
    test_manual_dimension_change_resets_quick_examples_to_none()
    test_manual_radio_toggle_drives_steady_state_check_both_ways()
    test_steady_state_check_never_added_to_a_visible_layout()
    print("ALL PASSED")
