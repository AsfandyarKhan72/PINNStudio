"""Closing PINNStudio while the startup update check is still waiting on the
network must exit cleanly. With the check in a QThread, Qt aborted the process
("QThread: Destroyed while thread '' is still running")."""
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = textwrap.dedent(f"""
    import os, sys, time
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.insert(0, {ROOT!r})
    import urllib.request
    def slow(*a, **k):
        time.sleep(3)
        raise OSError("offline")
    urllib.request.urlopen = slow
    from PyQt6.QtWidgets import QApplication
    app = QApplication(["pinnstudio"])
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w._get_installed_version = lambda: "0.0.1"
    w._check_for_updates()
    w.show()
    app.processEvents()
    assert w._update_thread.isRunning()
    w.close()
    app.processEvents()
""")


def test_quit_during_update_check():
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Destroyed while thread" not in r.stderr
