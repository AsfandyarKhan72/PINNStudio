#!/usr/bin/env python3
"""
Training Callbacks (Early Stopping / Point Resampling / Model Checkpoint
/ Training Timer) used to always render in the left Training panel, even
fully unchecked and unused -- paying for 4 group boxes' worth of space
regardless. They're now tucked inside a hidden-by-default container
behind a single "Show Training Callbacks" switch (see main_window.py's
_build_ui, around the "Training Callbacks" divider), and the divider no
longer says "(optional)".

Checks:
 - The divider label no longer contains "(optional)".
 - The container (and the 4 group boxes inside it) is hidden by default,
   and the switch itself starts unchecked.
 - Checking the switch reveals the container; unchecking hides it again.
 - The 4 individual callback checkboxes/fields inside the container are
   completely unaffected by the container's own visibility -- their
   checked state and _build_config() output are identical whether the
   container is shown or hidden (this is purely a visibility toggle, not
   a second enable/disable mechanism).
 - Restoring a saved config that already has any one of the four
   callbacks enabled auto-ticks the switch (so a previously-configured
   callback is never hidden from view on load); restoring one with all
   four off leaves the switch unticked.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_training_callbacks_visibility.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication, QGroupBox

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    _app.processEvents()

    # ── Divider text / default state ──────────────────────────────────
    from PyQt6.QtWidgets import QLabel
    divider_labels = [w.text() for w in win.findChildren(QLabel) if "Training Callbacks" in w.text()]
    check(len(divider_labels) == 1, f"expected exactly one 'Training Callbacks' divider label, found {divider_labels}")
    if divider_labels:
        check("(optional)" not in divider_labels[0],
              f"divider label should no longer say '(optional)', got {divider_labels[0]!r}")

    check(hasattr(win, 'train_callbacks_show_cb'), "expected a train_callbacks_show_cb switch")
    check(hasattr(win, 'train_callbacks_container'), "expected a train_callbacks_container widget")
    check(not win.train_callbacks_show_cb.isChecked(), "Show Training Callbacks should be unchecked by default")
    check(win.train_callbacks_container.isHidden(), "the callbacks container should be hidden by default")

    # The 4 group boxes should live INSIDE the container, not directly in
    # the Training panel -- otherwise hiding the container wouldn't hide
    # them.
    inner_groups = {g.title() for g in win.train_callbacks_container.findChildren(QGroupBox)}
    expected_groups = {"Early Stopping", "Point Resampling", "Model Checkpoint", "Training Timer"}
    check(expected_groups.issubset(inner_groups),
          f"expected all 4 callback groups inside train_callbacks_container, found {inner_groups}")

    # ── Toggling the switch shows/hides the container ─────────────────
    win.train_callbacks_show_cb.setChecked(True)
    _app.processEvents()
    check(not win.train_callbacks_container.isHidden(), "checking the switch should reveal the container")
    win.train_callbacks_show_cb.setChecked(False)
    _app.processEvents()
    check(win.train_callbacks_container.isHidden(), "unchecking the switch should hide the container again")

    # ── Visibility is purely cosmetic -- doesn't affect the callbacks'
    # own checked state or what _build_config() reports ────────────────
    win.cb_early_stopping_cb.setChecked(True)
    win.cb_es_patience.setValue(500)
    config_hidden = win._build_config()
    check(config_hidden.cb_early_stopping is True,
          "a callback's own checked state must not depend on the container being visible")
    check(config_hidden.cb_early_stopping_patience == 500, "a callback's own field values must not depend on container visibility")

    win.train_callbacks_show_cb.setChecked(True)
    _app.processEvents()
    config_shown = win._build_config()
    check(config_shown.cb_early_stopping is True and config_shown.cb_early_stopping_patience == 500,
          "revealing the container must not change any callback's own configured values")
    win.cb_early_stopping_cb.setChecked(False)
    win.train_callbacks_show_cb.setChecked(False)
    _app.processEvents()

    # ── Auto-reveal on restoring a config that already has a callback
    # enabled ──────────────────────────────────────────────────────────
    cfg_with_cb = win._build_config()
    cfg_with_cb.cb_model_checkpoint = True
    win._apply_config(cfg_with_cb)
    _app.processEvents()
    check(win.train_callbacks_show_cb.isChecked(),
          "restoring a config with Model Checkpoint enabled should auto-tick Show Training Callbacks")
    check(not win.train_callbacks_container.isHidden(),
          "auto-ticking the switch should also reveal the container")

    cfg_without_cb = win._build_config()
    cfg_without_cb.cb_model_checkpoint = False
    cfg_without_cb.cb_early_stopping = False
    cfg_without_cb.cb_point_resampler = False
    cfg_without_cb.cb_timer = False
    win._apply_config(cfg_without_cb)
    _app.processEvents()
    check(not win.train_callbacks_show_cb.isChecked(),
          "restoring a config with all 4 callbacks off should leave Show Training Callbacks unticked")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL TRAINING-CALLBACKS-VISIBILITY TESTS PASSED")
    return failures


def test_training_callbacks_visibility():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
