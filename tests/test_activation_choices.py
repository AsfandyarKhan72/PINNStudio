#!/usr/bin/env python3
"""
Activation function audit (user-requested, against DeepXDE's own pasted
`deepxde.nn.activations` documentation/source): the Activation combo
previously only offered ["tanh", "relu", "sigmoid", "swish"] -- 4 of the
9 identifiers DeepXDE's own `deepxde.nn.activations.get()` dict actually
supports (`elu, gelu, relu, selu, sigmoid, silu, sin, swish, tanh`), and
in lowercase rather than DeepXDE's own documented capitalization (its
get() docstring: "ELU, GELU, ReLU, SELU, Sigmoid, SiLU, sin, Swish,
tanh").

Checks:
 - MainWindow._ACTIVATION_CHOICES / the live Activation combo offer all
   9 of DeepXDE's supported identifiers, spelled exactly per DeepXDE's
   own docstring capitalization -- not the old 4-item lowercase subset.
 - Every one of those 9 labels is something DeepXDE's REAL, installed
   `deepxde.nn.activations.get()` actually accepts and resolves to a
   callable (skipped if deepxde/torch aren't installed in this
   environment) -- confirming the capitalization change is purely
   cosmetic and doesn't change what reaches DeepXDE.
 - "swish" and "SiLU" resolve to the identical underlying function
   (DeepXDE documents both spellings for the same op).
 - Backward compatibility: _canonical_activation_label() maps an old,
   pre-existing lowercase value (as a problem saved before this change
   would have stored, e.g. "relu") back onto its new canonical label
   ("ReLU"), case-insensitively, so _apply_config() (Open Problem) can
   select the matching combo item instead of silently leaving whatever
   was already selected (which would otherwise change a reopened
   problem's actual activation out from under the user, since
   QComboBox.setCurrentText() only matches exact, case-sensitive text).
 - Opening a saved problem whose stored config.activation is the OLD
   lowercase "relu" actually leaves the live Activation combo showing
   "ReLU" selected (end-to-end through _apply_config()), not silently
   stuck on whatever the combo's default item was.
 - An unrecognized/garbage activation string is passed through
   unchanged by _canonical_activation_label() rather than silently
   dropped.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_activation_choices.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow

_EXPECTED = ["tanh", "sin", "Sigmoid", "ReLU", "SiLU", "Swish", "ELU", "GELU", "SELU"]


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    check(MainWindow._ACTIVATION_CHOICES == _EXPECTED,
          f"expected the full DeepXDE-supported, canonically-capitalized activation list "
          f"{_EXPECTED}, got {MainWindow._ACTIVATION_CHOICES}")

    win = MainWindow()
    _app.processEvents()
    combo_items = [win.activation_combo.itemText(i) for i in range(win.activation_combo.count())]
    check(combo_items == _EXPECTED,
          f"live Activation combo should offer {_EXPECTED}, got {combo_items}")

    # Every canonical label must be something DeepXDE's REAL installed
    # activations.get() accepts (skip gracefully if not installed).
    try:
        from deepxde.nn import activations as _dde_activations
        _have_dde = True
    except ImportError:
        _have_dde = False

    if _have_dde:
        resolved = {}
        for label in _EXPECTED:
            try:
                fn = _dde_activations.get(label)
            except Exception as e:
                fn = None
                failures.append(f"DeepXDE's activations.get({label!r}) raised: {e}")
            else:
                resolved[label] = fn
            check(fn is not None, f"DeepXDE's activations.get({label!r}) should resolve to a callable")
        if "Swish" in resolved and "SiLU" in resolved:
            check(resolved["Swish"] is resolved["SiLU"],
                  "DeepXDE documents 'swish' and 'silu' as the same underlying function -- "
                  "activations.get('Swish') and activations.get('SiLU') should be identical")
    else:
        print("NOTE: deepxde not installed -- skipping the real activations.get() resolution checks.")

    # ── Backward compatibility: old lowercase values still resolve ──────
    for old, expected_new in [("relu", "ReLU"), ("sigmoid", "Sigmoid"), ("swish", "Swish"),
                               ("tanh", "tanh"), ("SiLU", "SiLU"), ("ELU", "ELU"),
                               ("gelu", "GELU"), ("selu", "SELU"), ("SIN", "sin")]:
        got = MainWindow._canonical_activation_label(old)
        check(got == expected_new,
              f"_canonical_activation_label({old!r}) should normalize to {expected_new!r}, got {got!r}")

    # An unrecognized string must be passed through unchanged, not dropped.
    check(MainWindow._canonical_activation_label("some_custom_thing") == "some_custom_thing",
          "an unrecognized activation string should be returned unchanged, not dropped/blanked")
    check(MainWindow._canonical_activation_label("") == "",
          "a blank activation string should be returned unchanged")

    # ── End-to-end: opening a problem saved with the OLD lowercase name
    # must select the matching NEW canonically-capitalized combo item,
    # not silently leave the combo on its default ("tanh"). ────────────
    config = win._build_config()
    config.activation = "relu"  # as an old, pre-existing saved .json problem would have it
    win.activation_combo.setCurrentIndex(0)  # force away from ReLU first
    win._apply_config(config)
    _app.processEvents()
    check(win.activation_combo.currentText() == "ReLU",
          f"opening a saved problem with the old stored 'relu' should select the 'ReLU' combo item "
          f"(BUG if not: the problem would silently reopen with a different activation than it was "
          f"actually trained with), got {win.activation_combo.currentText()!r}")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("OK: Activation combo offers DeepXDE's full supported set with DeepXDE's own "
              "canonical capitalization, and old saved problems still restore correctly.")
    return failures


def test_activation_choices_match_deepxde():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
