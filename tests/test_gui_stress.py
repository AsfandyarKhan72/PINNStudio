"""Opt-in stress test of the window (PINNSTUDIO_STRESS=1; about 10 s per round on one CPU core).

Loads two examples back and forth in a child process, so a native crash (heap corruption, segfault)
shows as a failed test instead of taking pytest down. PySide6 6.12.0 crashed in about 7 % of rounds of
this loop (see PYSIDE6_TRIAL.md), so 60 rounds detect it with about 99 % probability.
"""
import os
import signal
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUNDS = int(os.environ.get("PINNSTUDIO_STRESS_ROUNDS", "60"))

pytestmark = pytest.mark.skipif(os.environ.get("PINNSTUDIO_STRESS") != "1",
                                reason="stress test: set PINNSTUDIO_STRESS=1 to run")


def test_loading_examples_back_and_forth_does_not_crash():
    env = dict(os.environ, QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    p = subprocess.run([sys.executable, "-X", "faulthandler", os.path.join(REPO, "tools", "gui_stress.py"),
                        "--rounds", str(ROUNDS), "--pages", "--tabs"],
                       capture_output=True, text=True, env=env, timeout=ROUNDS * 120)
    if p.returncode < 0:
        pytest.fail(f"the window crashed ({signal.Signals(-p.returncode).name}) after "
                    f"{p.stdout.count('round ')} rounds:\n{p.stderr[-3000:]}")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "STRESS DONE" in p.stdout
