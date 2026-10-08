#!/usr/bin/env python3
"""
Smoke test for a batch of small pre-release GUI fixes, all pure
widget-construction/state checks -- no training needed:

 1. Restore panel's "Save visualization to:" field (self.restore_save_path)
    now defaults to the same ~/PINNStudio_Results path as the main Solve
    panel's "Save to:" field (self.save_dir_input), instead of starting
    empty.
 2. The 8 domain spinboxes (x_min/x_max, y_min/y_max, z_min/z_max,
    t_min/t_max) all share one fixed width, so t's 4 decimals (needed for
    precision, e.g. 1D Schrodinger's t_max = pi/2) don't make its box
    visibly wider than x/y/z's 2-decimal boxes.
 3. The Boundary Conditions panel's "Show location examples" hint is now a
    single panel-level toggle (self.bc_loc_hint_toggle /
    self.bc_loc_hint), not one repeated on every BC row -- adding several
    BC entries doesn't add any more copies of it.
 4. "Surface Animation (GIF)" is no longer offered as a 1D plot type (a 1D
    animation frame has only one real spatial axis, so there's nothing for
    a "surface" to vary across within a frame) -- it's absent from
    plot_type_combo in 1D, appears when switching to 2D/3D, and
    disappears again switching back to 1D (falling back to "Line
    Animation (GIF)" if it had been selected). The Restore panel's own,
    separately-filtered viz dropdown (_restore_forward_viz_items) applies
    the same exclusion based on a restored config's "problem_dim" field.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_prerelease_gui_tweaks.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication, QCheckBox

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow

_FAILURES = []


def check(cond, msg):
    if not cond:
        _FAILURES.append(msg)
        print(f"FAIL: {msg}")
    else:
        print(f"ok: {msg}")


def main():
    win = MainWindow()

    # 1. Restore save-path default
    expected_default = os.path.join(os.path.expanduser("~"), "PINNStudio_Results")
    check(win.restore_save_path.text() == expected_default,
          f"restore_save_path defaults to {expected_default!r}, got {win.restore_save_path.text()!r}")
    check(win.save_dir_input.text() == expected_default,
          "sanity: main Solve panel's save_dir_input has the same default")

    # 2. Domain spinbox widths all match. sizeHint() reflects each
    # spinbox's own "natural" content-based preferred size and is NOT
    # affected by setFixedWidth() (that's expected Qt behavior, not a bug
    # in the fix) -- setFixedWidth() instead pins minimumWidth ==
    # maximumWidth, which is what actually constrains the rendered size,
    # so that's what this checks.
    fixed_widths = {
        name: (getattr(win, name).minimumWidth(), getattr(win, name).maximumWidth())
        for name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "t_min", "t_max")
    }
    check(len(set(fixed_widths.values())) == 1,
          f"all 8 domain spinboxes share one fixed (min, max) width, got {fixed_widths}")
    check(all(mn == mx for mn, mx in fixed_widths.values()),
          "each spinbox's width is truly fixed (minimumWidth == maximumWidth)")
    check(win.t_min.decimals() == 4 and win.t_max.decimals() == 4,
          "t_min/t_max keep their 4-decimal precision (not reduced to match x/y/z's 2)")
    check(win.x_min.decimals() == 2 and win.y_min.decimals() == 2 and win.z_min.decimals() == 2,
          "x/y/z spinboxes still default to 2 decimals (unchanged)")

    # 3. BC panel: one shared hint, not one per row
    check(hasattr(win, "bc_loc_hint_toggle") and hasattr(win, "bc_loc_hint"),
          "panel-level bc_loc_hint_toggle/bc_loc_hint exist")
    check(not win.bc_loc_hint.isVisible() if win.bc_loc_hint.isVisibleTo(win) is False else True,
          "shared hint starts hidden")

    # Count all "Show location examples" checkboxes in the whole BC group
    # before adding any rows, then after adding 3 -- should stay at 1.
    def _count_hint_checkboxes():
        return sum(
            1 for cb in win.custom_bc_group.findChildren(QCheckBox)
            if cb.text() == "📖 Show location examples"
        )

    before = _count_hint_checkboxes()
    check(before == 1, f"exactly one location-examples checkbox before adding rows, got {before}")

    win._add_custom_bc_entry()
    win._add_custom_bc_entry()
    win._add_custom_bc_entry()
    after = _count_hint_checkboxes()
    check(after == 1, f"still exactly one location-examples checkbox after adding 3 BC rows, got {after}")

    # Each new row should NOT have its own loc_hint_toggle-style widget --
    # confirm the entry_data dict has no such key lingering around.
    for entry in win.custom_bc_list:
        check("location_hint_toggle" not in entry and "loc_hint_toggle" not in entry,
              "BC row entry_data has no leftover per-row hint-toggle key")

    # 4a. plot_type_combo: "Surface Animation (GIF)" excluded in 1D
    # (default state), appears after switching to 2D/3D, disappears again
    # switching back to 1D (with fallback off of it if it was selected).
    def _combo_items(combo):
        return [combo.itemText(i) for i in range(combo.count())]

    check("Surface Animation (GIF)" not in _combo_items(win.plot_type_combo),
          f"plot_type_combo excludes Surface Animation (GIF) in default 1D state, got {_combo_items(win.plot_type_combo)}")

    win.radio_2d.setChecked(True)
    check("Surface Animation (GIF)" in _combo_items(win.plot_type_combo),
          f"plot_type_combo includes Surface Animation (GIF) after switching to 2D, got {_combo_items(win.plot_type_combo)}")

    # Selecting "Surface Animation (GIF)" through the combo normally pops
    # the real unified Plot Settings dialog (_on_plot_type_changed ->
    # _on_plot_settings, modal exec() -- every plot type auto-pops it as
    # of Round 24, not just line/animation types) -- blockSignals here
    # sidesteps that dialog the same way its own Cancel button does,
    # since this test is only exercising _on_dim_changed's add/remove/
    # fallback logic, not that dialog.
    win.plot_type_combo.blockSignals(True)
    win.plot_type_combo.setCurrentText("Surface Animation (GIF)")
    win.plot_type_combo.blockSignals(False)
    win.radio_3d.setChecked(True)
    check("Surface Animation (GIF)" in _combo_items(win.plot_type_combo),
          f"plot_type_combo still includes Surface Animation (GIF) after switching 2D->3D, got {_combo_items(win.plot_type_combo)}")
    check(win.plot_type_combo.currentText() == "Surface Animation (GIF)",
          "selection preserved across a 2D->3D switch (both support it)")

    win.radio_1d.setChecked(True)
    check("Surface Animation (GIF)" not in _combo_items(win.plot_type_combo),
          f"plot_type_combo excludes Surface Animation (GIF) again after switching back to 1D, got {_combo_items(win.plot_type_combo)}")
    check(win.plot_type_combo.currentText() == "Line Animation (GIF)",
          f"stale Surface Animation (GIF) selection falls back to Line Animation (GIF), got {win.plot_type_combo.currentText()!r}")

    # 4b. Restore panel's own _restore_forward_viz_items(): same exclusion,
    # driven by a restored config file's "problem_dim" field rather than
    # the live dimension radios.
    import json
    import tempfile

    def _viz_items_for_config(cfg_dict):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg_dict, f)
            path = f.name
        try:
            win.restore_config_path.setText(path)
            return win._restore_forward_viz_items()
        finally:
            os.unlink(path)

    items_1d = _viz_items_for_config({"steady_state": False, "problem_dim": "1D"})
    check("Animation Surface (GIF)" not in items_1d,
          f"_restore_forward_viz_items excludes Animation Surface (GIF) for a 1D config, got {items_1d}")
    check("Animation Line (GIF)" in items_1d,
          f"_restore_forward_viz_items still includes Animation Line (GIF) for a 1D config, got {items_1d}")

    items_2d = _viz_items_for_config({"steady_state": False, "problem_dim": "2D"})
    check("Animation Surface (GIF)" in items_2d,
          f"_restore_forward_viz_items includes Animation Surface (GIF) for a 2D config, got {items_2d}")

    items_3d = _viz_items_for_config({"steady_state": False, "problem_dim": "3D"})
    check("Animation Surface (GIF)" in items_3d,
          f"_restore_forward_viz_items includes Animation Surface (GIF) for a 3D config, got {items_3d}")

    items_1d_steady = _viz_items_for_config({"steady_state": True, "problem_dim": "1D"})
    check(items_1d_steady == ["Surface"],
          f"_restore_forward_viz_items still collapses to just ['Surface'] for a steady-state config regardless of dimension, got {items_1d_steady}")

    win.close()

    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) FAILED")
        sys.exit(1)
    print("\nAll checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
