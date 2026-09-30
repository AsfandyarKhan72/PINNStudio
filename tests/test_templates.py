#!/usr/bin/env python3
"""
Regression check across every Quick Example template, in both Forward and
Inverse mode: for each combination, build a config the same way the GUI
does (MainWindow._build_config()), validate it, and generate both the
internal Solve-path script (generate_script()) and the standalone "Export
as DeepXDE Script" output (generate_clean_script()) -- checking that each
one is at least syntactically valid Python (ast.parse).

This does NOT execute any training -- it's a fast, no-GPU-required check
that config building and code generation still work for every template
after a code change, not a check that the resulting PDE/training is
numerically correct. It has caught real regressions before (e.g. a
template losing its default Inverse trainable-variable values, or a
geometry-type selector not resetting between templates) well before a
release, and is meant to be run on every push/PR via
.github/workflows/smoke-test.yml, not just by hand.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_templates.py

Exits 0 if every combination passes, 1 otherwise (with each failure
printed). No test framework required -- this can also be discovered by
pytest (see test_all_templates() below) if pytest happens to be
installed, but nothing here depends on it.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

# Make sure the repo root (this file's parent's parent) is importable when
# run directly, e.g. `python3 tests/test_templates.py` from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import ast

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.codegen import generate_script, generate_clean_script

TEMPLATES_BY_DIM = {
    "1d": ["1D Heat", "1D Allen-Cahn", "1D Burgers", "1D Schrödinger"],
    "2d": [
        "2D Heat",
        "2D Allen-Cahn (Mattey & Ghosh)",
        "2D Allen-Cahn (Wight & Zhao)",
        "2D Burgers (Mathias)",
        "2D Poisson (L-Shape)",
        "2D Poisson (Disk)",
    ],
    "3d": ["3D Heat", "3D Poisson (Sphere)"],
}

# Combinations that are EXPECTED to fail config.validate() in their fresh,
# just-selected state -- not a regression. 1D Schrödinger's Inverse mode
# has no bundled ground-truth file by design (see INVERSE_AUTO_OBS in
# main_window.py), so it genuinely has no observed-data file until the
# user browses to one; validate() catching that pre-emptively (instead of
# crashing deep in np.loadtxt("")) is correct, expected behavior.
EXPECTED_VALIDATE_FAILURES = {
    ("1D Schrödinger", True),
}


def run(win=None):
    """Runs the full matrix. Returns (total, failures) where failures is a
    list of human-readable failure strings. Pass an existing MainWindow to
    reuse it (e.g. from another test in the same process); otherwise one is
    created."""
    win = win or MainWindow()
    failures = []
    total = 0

    for dim, names in TEMPLATES_BY_DIM.items():
        {"1d": win.radio_1d, "2d": win.radio_2d, "3d": win.radio_3d}[dim].setChecked(True)
        for name in names:
            win.quick_examples_combo.setCurrentText(name)
            for inv in (False, True):
                win.radio_forward.setChecked(not inv)
                win.radio_inverse.setChecked(inv)
                total += 1
                config = win._build_config()
                errs = config.validate()
                expected_fail = (name, inv) in EXPECTED_VALIDATE_FAILURES
                if errs and not expected_fail:
                    failures.append(f"[validate] {name} inv={inv}: {errs}")
                    continue
                if errs:
                    continue  # expected failure -- nothing further to check
                try:
                    s1 = generate_script(config)
                    ast.parse(s1)
                except Exception as e:
                    failures.append(f"[generate_script] {name} inv={inv}: {e}")
                    continue
                try:
                    s2 = generate_clean_script(config)
                    ast.parse(s2)
                except Exception as e:
                    failures.append(f"[generate_clean_script] {name} inv={inv}: {e}")
                    continue

    return total, failures


def main():
    total, failures = run()
    passed = total - len(failures)
    for f in failures:
        print("FAIL:", f)
    print(f"\n{passed}/{total} template x mode combinations passed")
    if failures:
        print(f"{len(failures)} FAILURE(S)")
        return 1
    print("ALL REGRESSION TESTS PASSED")
    return 0


def test_all_templates():
    """pytest-discoverable entry point, if pytest is ever added to this
    project -- asserts the same thing main() checks by exit code."""
    total, failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
