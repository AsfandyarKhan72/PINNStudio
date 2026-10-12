"""Shared pytest setup for PINNStudio's tests.

Quick vs full runs: test files listed in SLOW_TEST_FILES get the `slow`
marker, so

    pytest tests -m "not slow"     # quick run, for routine changes
    pytest tests                   # full run, before merging

The list is kept here (rather than as a marker inside each file) so every
test file stays runnable as a plain script without pytest installed, which
is how CI runs them.
"""
import os

import pytest

# Measured on an RTX 4090: test_sweep_execution.py runs real multi-run
# parameter sweeps and takes ~4 of the suite's ~10.5 minutes there (~23 of
# ~60 minutes on a 2-CPU machine).
SLOW_TEST_FILES = {
    "test_sweep_execution.py",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test, skipped by -m \"not slow\"")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if os.path.basename(str(item.fspath)) in SLOW_TEST_FILES:
            item.add_marker(pytest.mark.slow)
