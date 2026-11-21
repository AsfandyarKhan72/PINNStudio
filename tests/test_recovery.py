"""Crash recovery: an autosave left by a PINNStudio that has gone is offered back, one of a running
PINNStudio is not; the window keeps unsaved work, forgets it when saved, and recovers it as unsaved."""

import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

from pinnstudio.core import recovery as RC  # noqa: E402
from pinnstudio.core.config import PINNConfig  # noqa: E402


def _orphan(home, name="crashed problem", pde="du_t - 0.7*du_xx"):
    """An autosave written by another process, which then exits (as a crash would leave it)."""
    code = ("import sys, dataclasses; sys.path.insert(0, %r)\n"
            "from pinnstudio.core.config import PINNConfig\n"
            "from pinnstudio.core import recovery\n"
            "c = dataclasses.replace(PINNConfig(), pde_expressions=%r, pde_expression=%r)\n"
            "recovery.write(c, '', %r)\n") % (ROOT, pde, pde, name)
    env = dict(os.environ, PINNSTUDIO_HOME=str(home))
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=120)


def test_orphans_are_from_processes_that_have_gone(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path))
    _orphan(tmp_path)
    RC.write(PINNConfig(), None, "mine")              # this process: running, not an orphan
    found = RC.orphans()
    assert len(found) == 1 and found[0]["problem_name"] == "crashed problem"
    assert os.path.isfile(RC.own_path())
    RC.discard()
    assert not os.path.exists(RC.own_path())
    RC.discard(found[0]["path"])
    assert RC.orphans() == []


class Answer:
    def __init__(self, answer):
        self.answer = answer
        self.asked = []

    def ask_recover(self, item, count):
        self.asked.append((item["problem_name"], count))
        return self.answer

    def inform(self, *a):
        pass


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("PINNSTUDIO_RECOVERY", "1")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    for _ in range(3):
        app.processEvents()
    yield w, app
    w.close()
    w.deleteLater()
    app.processEvents()


def test_the_window_keeps_unsaved_work_and_forgets_it_when_saved(window, tmp_path):
    w, app = window
    w.pde_inputs[0].setText("du_t - 0.9*du_xx")
    w._check_problems()
    assert w._dirty
    w._autosave()
    assert os.path.isfile(RC.own_path())
    w._save_problem(str(tmp_path / "p.pinn.json"))
    w._check_problems()
    assert not w._dirty and not os.path.exists(RC.own_path())


def test_a_crashed_session_is_recovered_as_unsaved_work(window, tmp_path):
    w, app = window
    _orphan(tmp_path, "heat study", "du_t - 0.33*du_xx")
    w.recovery.prompter = Answer("recover")
    w._offer_recovery()
    assert w.recovery.prompter.asked == [("heat study", 1)]
    assert w.pde_inputs[0].text() == "du_t - 0.33*du_xx"
    assert w._dirty and w._current_problem_name == "heat study"
    assert RC.orphans() == []
    _orphan(tmp_path)
    w.recovery.prompter = Answer("later")
    w._offer_recovery()
    assert len(RC.orphans()) == 1, "Later keeps it for the next start"
    w.recovery.prompter = Answer("discard")
    w._offer_recovery()
    assert RC.orphans() == []

