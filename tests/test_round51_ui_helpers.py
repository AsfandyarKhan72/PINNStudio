#!/usr/bin/env python3
"""
Round 51 -- codebase cleanup, phase 14 (audit §4): two pieces of UI
boilerplate in main_window.py that were hand-typed over and over now go
through one helper each:
  - _register_hint(widget, color, extra=""): the "hint" text-style
    registration (58 call sites of the same _register_style lambda);
  - _add_ok_cancel_row(layout, dialog=None): the right-aligned OK/Cancel
    footer of nine settings dialogs.
Behavior-preserving: before the commit, every registered widget's final
stylesheet and every dialog's button set / Cancel behavior were compared
against the pre-refactor build and found identical. This file re-checks the
dialogs directly and the hint helper's stylesheet format.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round51_ui_helpers.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QPushButton

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow

_DIALOGS = [("_on_float_settings", ()), ("_on_gpu_settings", ()), ("_on_seed_settings", ()),
            ("_on_lbfgs_settings", ()), ("_on_optimizer_settings", ()), ("_on_export_settings", ()),
            ("_on_loss_plot_settings", ()), ("_on_restore_viz_settings", ("Surface",)),
            ("_on_plot_settings", ("Surface",))]


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    lbl = QLabel("x")
    win._register_hint(lbl, "#123456", "margin-top: 2px; ")
    css = win._cat_css("hint")
    check(lbl.styleSheet() == f"color: #123456; margin-top: 2px; {css}",
          f"_register_hint should produce 'color: <c>; <extra><hint css>', got {lbl.styleSheet()!r}")

    seen = {}
    real_exec = QDialog.exec

    def fake_exec(self):
        btns = self.findChildren(QPushButton)
        rejected = []
        self.rejected.connect(lambda: rejected.append(1))
        cancel = [b for b in btns if b.text() == "Cancel"]
        if cancel:
            cancel[-1].click()
        seen[cur] = ([b.text() for b in btns][-2:], bool(rejected))
        return 0

    QDialog.exec = fake_exec
    try:
        for cur, args in _DIALOGS:
            getattr(win, cur)(*args)
            texts, rejected = seen.get(cur, ([], False))
            check(texts == ["OK", "Cancel"], f"{cur}: footer should end with OK, Cancel (got {texts})")
            check(rejected, f"{cur}: Cancel should still close (reject) the dialog")
    finally:
        QDialog.exec = real_exec
    win.close()
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 51 UI-HELPER TESTS PASSED")
    return 0


def test_round51_ui_helpers():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
