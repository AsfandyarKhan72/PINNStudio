"""Errors in the window are logged and shown instead of ending the program
(errors.py). Simulates the Windows shortcut, where pythonw.exe has no
sys.stdout / sys.stderr."""
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = textwrap.dedent("""
    import os, sys
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.insert(0, ROOT)
    sys.stdout = None          # as under pythonw.exe
    sys.stderr = None
    INSTALL = {install}
    if INSTALL:
        from pinnstudio import errors
        errors.install()
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.exec = lambda self: 0     # do not wait for a click
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication
    app = QApplication(["pinnstudio"])
    def boom():
        print("a library printing with no console")
        raise ValueError("restore went wrong")
    QTimer.singleShot(0, boom)
    QTimer.singleShot(200, app.quit)
    app.exec()
    open(os.path.join(os.environ["HOME"], "survived"), "w").close()
""")


def _run(tmp_path, install):
    env = dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path))
    code = "ROOT = %r\n" % ROOT + SCRIPT.format(install=install)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)


def test_error_in_slot_is_logged_and_app_survives(tmp_path):
    r = _run(tmp_path, True)
    assert r.returncode == 0, r.stderr[-1500:]
    assert (tmp_path / "survived").exists()
    log = (tmp_path / ".pinnstudio" / "pinnstudio.log").read_text()
    assert "ValueError: restore went wrong" in log and "a library printing with no console" in log


def test_without_handler_the_app_dies(tmp_path):
    """What happened before: PyQt ends the program on an uncaught error."""
    r = _run(tmp_path, False)
    assert r.returncode != 0 and not (tmp_path / "survived").exists()
