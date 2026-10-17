#!/usr/bin/env python3
"""
Round 57 -- the two Plot Settings dialogs (Setup tab: _on_plot_settings;
Restore & Visualize: _on_restore_viz_settings) built ~300 lines of fields
each, independently, and had drifted: same fields in a different order,
different labels ("Animation FPS:" vs "Frame rate (fps):", "Time steps:"
vs "Time steps to show:") and different frame-rate choices. Both now build
their fields through _build_plot_settings_fields() and read them back
through _plot_settings_values(); each keeps only what is genuinely its own
(the Setup dialog's write-back to its hidden steps/slice widgets and its
Cancel revert; Restore's info text and per-viz-type steps key).

Before the commit, 216 open/OK scenarios (every viz type, 1D/2D/3D,
steady and time-dependent, empty and fully-populated settings, with and
without editing every field) saved identical settings dicts and showed
the same fields as before, apart from the unified labels, title and fps
choices. This file checks the shared shape directly.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round57_plot_settings_shared.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QLabel, QPushButton

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    seen = {}
    real_exec = QDialog.exec

    def fake_exec(dlg, _edit=[False]):
        labels = [w.text() for w in dlg.findChildren(QLabel) if w.isVisibleTo(dlg) and w.text().endswith(":")]
        fps = [c for c in dlg.findChildren(QComboBox) if c.isVisibleTo(dlg) and c.itemText(0) == "5"]
        seen["labels"] = labels
        seen["fps"] = [fps[0].itemText(i) for i in range(fps[0].count())] if fps else None
        seen["title"] = dlg.windowTitle()
        if seen.get("edit"):
            fps[0].setCurrentText("24")
        for b in dlg.findChildren(QPushButton):
            if b.text() == "OK":
                b.click()
                break
        return 1

    QDialog.exec = fake_exec
    try:
        win = MainWindow()
        with tempfile.TemporaryDirectory() as td:
            cp = os.path.join(td, "cfg.json")
            with open(cp, "w", encoding="utf-8") as f:
                json.dump({"problem_dim": "2D", "steady_state": False}, f)
            win.restore_config_path.setText(cp)
            win.radio_2d.setChecked(True)

            # Same viz type in both dialogs -> same fields, same order, same labels.
            for setup_vt, restore_vt in (("Surface", "Surface"),
                                         ("Line (time steps)", "Line (time steps)"),
                                         ("Line Animation (GIF)", "Animation Line (GIF)")):
                win._plot_viz_settings = {}
                win._on_plot_settings(setup_vt)
                a = dict(seen)
                win._restore_viz_settings = {}
                win._on_restore_viz_settings(restore_vt)
                b = dict(seen)
                check(a["labels"] == b["labels"], f"{setup_vt}: dialogs show different fields:\n{a['labels']}\n{b['labels']}")
                check(a["title"] == f"Plot Settings — {setup_vt}" and b["title"] == f"Plot Settings — {restore_vt}",
                      f"{setup_vt}: unexpected titles {a['title']!r} / {b['title']!r}")
                if "Animation" in setup_vt:
                    check("Frame rate (fps):" in a["labels"] and "Frames to show:" in a["labels"],
                          "animation types should show 'Frame rate (fps):' and 'Frames to show:'")
                    check(a["fps"] == b["fps"] == ["5", "8", "10", "15", "20", "24", "30"],
                          f"fps choices should match in both dialogs: {a['fps']} / {b['fps']}")
                if setup_vt == "Line (time steps)":
                    check("Time steps to show:" in a["labels"], "static Line type should say 'Time steps to show:'")

            # OK saves through _plot_settings_values in both, plus each dialog's own extras.
            seen["edit"] = True
            win._restore_viz_settings = {"extra_key": 1}
            win._on_restore_viz_settings("Animation Line (GIF)")
            s = win._restore_viz_settings
            check(s.get("fps") == 24 and s.get("extra_key") == 1 and "n_steps_anim" in s,
                  f"restore dialog should save fps, keep unknown keys, save its steps key: {s}")
            win._plot_viz_settings = {}
            win._on_plot_settings("Line Animation (GIF)")
            check(win._plot_viz_settings.get("fps") == 24, "setup dialog should save the fps choice")
    finally:
        QDialog.exec = real_exec
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 57 PLOT-SETTINGS TESTS PASSED")
    return 0


def test_round57_plot_settings_shared():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
