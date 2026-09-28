#!/usr/bin/env python3
"""
Focused checks for the GPU device/memory and random-seed settings (see
config.py's gpu_device_index/gpu_memory_fraction/use_random_seed/
random_seed fields, and MainWindow's "GPU / Hardware..." / "Random
Seed..." Settings-menu dialogs): defaults match documented behavior,
validate() rejects bad values, and the generated script actually reflects
what's configured.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_config_options.py
"""
import os
import sys
import dataclasses
import ast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.codegen import generate_script, generate_clean_script


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    win.radio_1d.setChecked(True)
    win.quick_examples_combo.setCurrentText("1D Heat")
    config = win._build_config()

    # Defaults
    check(config.gpu_device_index == 0, "default gpu_device_index should be 0")
    check(config.gpu_memory_fraction == 0.95, "default gpu_memory_fraction should be 0.95")
    check(config.use_random_seed is True, "random seeding should default on")
    check(config.random_seed == 2026, "default random_seed should be 2026")
    check(config.validate() == [], "default config should validate cleanly")

    script = generate_script(config)
    ast.parse(script)
    check("dde.config.set_random_seed(2026)" in script,
          "generate_script() should call set_random_seed with the configured seed")
    check("_gpu_device_index = 0" in script and "_gpu_memory_fraction = 0.95" in script,
          "generate_script() should use the configured GPU device/memory")

    clean = generate_clean_script(config)
    ast.parse(clean)
    check("dde.config.set_random_seed(2026)" in clean,
          "generate_clean_script() should also seed when use_random_seed is on")

    # Turning seeding off removes the call entirely from both generators
    off_config = dataclasses.replace(config, use_random_seed=False)
    check(off_config.validate() == [], "seeding off should still validate cleanly")
    off_script = generate_script(off_config)
    ast.parse(off_script)
    check("set_random_seed" not in off_script,
          "generate_script() should not call set_random_seed when seeding is off")
    off_clean = generate_clean_script(off_config)
    ast.parse(off_clean)
    check("set_random_seed" not in off_clean,
          "generate_clean_script() should not call set_random_seed when seeding is off")

    # A custom seed value flows through
    custom_config = dataclasses.replace(config, random_seed=42)
    custom_script = generate_script(custom_config)
    check("dde.config.set_random_seed(42)" in custom_script,
          "a custom seed value should appear in the generated script")

    # validate() rejects bad values
    bad_gpu1 = dataclasses.replace(config, gpu_device_index=-1)
    check(any("device index" in e.lower() for e in bad_gpu1.validate()),
          "validate() should reject a negative GPU device index")
    bad_gpu2 = dataclasses.replace(config, gpu_memory_fraction=0.0)
    check(any("memory fraction" in e.lower() for e in bad_gpu2.validate()),
          "validate() should reject a 0 GPU memory fraction")
    bad_seed = dataclasses.replace(config, random_seed="not-an-int")
    check(any("seed" in e.lower() for e in bad_seed.validate()),
          "validate() should reject a non-integer random seed")

    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL CONFIG-OPTION TESTS PASSED")
    return 0


def test_config_options():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
