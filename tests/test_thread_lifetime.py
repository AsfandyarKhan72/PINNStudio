"""A QThread must stay alive until its thread has ended, even when the window
replaces its only reference from the slot of the thread's own last signal (the
3D view's point sampler starts its next sample there). Without
main_window._keep_until_finished Qt aborts the process ("QThread: Destroyed
while thread is still running"), so the check runs in a child process."""
import os
import subprocess
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = textwrap.dedent('''
    import os, sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, {repo!r})
    from PyQt6.QtCore import QCoreApplication, QThread, QTimer, pyqtSignal
    from pinnstudio.ui.main_window import _keep_until_finished, _LIVE_THREADS
    app = QCoreApplication([])
    KEEP = {keep}
    state = {{"n": 0, "th": None}}

    def start():
        class T(QThread):
            done = pyqtSignal()
            def run(self):
                self.done.emit()
                sum(i * i for i in range(3000))   # still running after its last signal
        t = T()
        t.done.connect(on_done)
        state["th"] = t                          # drops the previous thread's last reference
        (_keep_until_finished(t) if KEEP else t).start()

    def on_done():
        state["n"] += 1
        if state["n"] >= 1500:
            QTimer.singleShot(200, app.quit)
            return
        start()

    QTimer.singleShot(0, start)
    app.exec()
    print("OK", state["n"], len(_LIVE_THREADS))
''')


def _run(keep):
    code = SCRIPT.format(repo=REPO, keep=keep)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)


def test_replaced_thread_is_kept_until_it_ends():
    p = _run(True)
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.split()[:2] == ["OK", "1500"], p.stdout


def test_without_the_keeper_the_pattern_aborts():
    """The check above detects the failure: without the keeper the same loop aborts."""
    p = _run(False)
    assert p.returncode != 0 and "Destroyed while thread" in p.stderr, (p.returncode, p.stderr[-500:])
