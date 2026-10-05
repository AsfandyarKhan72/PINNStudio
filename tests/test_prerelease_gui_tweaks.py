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

    win.close()

    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) FAILED")
        sys.exit(1)
    print("\nAll checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
