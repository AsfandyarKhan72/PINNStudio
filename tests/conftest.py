"""Test setup for Windows.

* PyTorch must be loaded before PyQt6 or it fails with "[WinError 1114] ...
  c10.dll" (see pinnstudio/main.py); test modules import PyQt6 first, so load
  PyTorch here, before any test module is collected.
* Scripts the tests start themselves run in Python's UTF-8 mode, as the app
  runs its own (pinnstudio/core/pyexe.py), and print in this process's
  encoding so their output can be read back. This process keeps Windows'
  default encoding, so the app code under test still meets it.
"""
import locale
import os
import sys

if sys.platform == "win32":
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = locale.getpreferredencoding(False) + ":replace"


# Tests that train real models in their own processes (minutes each on a CI
# machine). The platform tests run everything; the platform matrix -- the
# same suite on every other system and Python version -- leaves these out
# (pytest -m "not slow"): they test PINNStudio's own logic, which does not
# change between those systems, and the window check there trains for real.
SLOW_FILES = {
    "test_sweep_execution.py", "test_ta_loss_history.py", "test_training_methods.py",
    "test_round77_fractional.py", "test_restore_steady_state.py", "test_sketch_3d.py",
    "test_rar_round_diagnostics.py", "test_round76_integral.py", "test_round71_pointset_points.py",
    "test_round78_example_phases.py", "test_steady_state_rar_gating.py", "test_sketch_edges.py",
    "test_round46_anim_surface_masking.py", "test_round74_ode.py", "test_round73_msffn.py",
    "test_restore_geometry_masking.py", "test_round38_line_steps_masking.py",
    "test_round45_restore_surface_refactor.py", "test_round47_ic_pretrain_shared_block.py",
    "test_round44_ta_surface_masking.py",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: trains real models; left out of the platform matrix")


def pytest_collection_modifyitems(config, items):
    import pytest
    for item in items:
        if os.path.basename(str(item.fspath)) in SLOW_FILES:
            item.add_marker(pytest.mark.slow)
