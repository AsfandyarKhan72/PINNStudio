#!/usr/bin/env python3
"""
Round 60 -- Adaptive Training's single "📖 Show reference" toggle, whose
text followed the Method dropdown, is replaced by one toggle per method,
inside that method's own section ("Show Time-Adaptive reference" in the
Time-Adaptive section, "Show RAR reference" in the RAR section), so each
only appears while its method is selected -- the same pattern as every
other panel's reference toggle. Both are built by _add_reference_toggle().

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round60_adapt_reference_toggles.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow

_TA = "Time Adaptive Training"
_RAR = "Residual-based Adaptive Refinement (RAR)"


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    win.show()
    _app.processEvents()

    check(not hasattr(win, "adapt_ref_toggle"), "the old shared toggle should be gone")
    check(win.ta_ref_toggle.text() == "📖 Show Time-Adaptive reference", win.ta_ref_toggle.text())
    check(win.rar_ref_toggle.text() == "📖 Show RAR reference", win.rar_ref_toggle.text())
    check("Splits the time domain" in win.ta_ref_hint.text(), "TA toggle should carry the Time-Adaptive text")
    check("worst-residual points" in win.rar_ref_hint.text(), "RAR toggle should carry the RAR text")

    for method, ta_vis, rar_vis in (("None", False, False), (_TA, True, False), (_RAR, False, True)):
        win.adapt_combo.setCurrentText(method)
        _app.processEvents()
        check(win.ta_ref_toggle.isVisible() == ta_vis and win.rar_ref_toggle.isVisible() == rar_vis,
              f"{method}: TA toggle visible={win.ta_ref_toggle.isVisible()}, RAR toggle visible={win.rar_ref_toggle.isVisible()}")

    # Each toggle shows and hides only its own text, starting hidden.
    win.adapt_combo.setCurrentText(_RAR)
    _app.processEvents()
    check(not win.rar_ref_hint.isVisible(), "reference text should start hidden")
    win.rar_ref_toggle.setChecked(True)
    _app.processEvents()
    check(win.rar_ref_hint.isVisible() and not win.ta_ref_hint.isVisible(), "RAR toggle should show only the RAR text")
    win.rar_ref_toggle.setChecked(False)
    _app.processEvents()
    check(not win.rar_ref_hint.isVisible(), "unchecking should hide the RAR text again")
    win.adapt_combo.setCurrentText(_TA)
    win.ta_ref_toggle.setChecked(True)
    _app.processEvents()
    check(win.ta_ref_hint.isVisible(), "TA toggle should show the Time-Adaptive text")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 60 ADAPTIVE-TRAINING REFERENCE TOGGLE TESTS PASSED")
    return 0


def test_round60_adapt_reference_toggles():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
