"""Start PINNStudio the way the `pinnstudio` command does, close the window
after a few seconds as a user would, and check that the process ends with
exit code 0 (no crash during Qt's or Python's teardown).

    python tools/quit_check.py [SECONDS]

Used by the platform workflow after the one-command install; prints the exit
code of the child process and exits 1 if it is not 0.
"""
import os
import subprocess
import sys
import textwrap

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
CHILD = textwrap.dedent(f"""
    import sys
    from pinnstudio import main as m           # the same imports and preload as the command
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from pinnstudio import errors
    errors.install()
    from pinnstudio.ui.main_window import MainWindow
    app = QApplication(["pinnstudio"])
    app.setApplicationName("PINNStudio")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    QTimer.singleShot(int({secs} * 1000), window.close)   # the last window closed: app.exec() returns
    code = app.exec()
    print("app.exec() returned", code, flush=True)
    sys.exit(code)
""")
env = dict(os.environ, QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"), PYTHONFAULTHANDLER="1")
r = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=600)
out = (r.stdout + r.stderr)[-4000:]
print(out)
print(f"exit code after closing the window: {r.returncode}")
if r.returncode != 0 or "app.exec() returned 0" not in r.stdout:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        msg = f"exit code {r.returncode} after the window was closed:\n{out}"
        print("::error title=quit check::" + msg.replace("%", "%25").replace("\n", "%0A"))
    sys.exit(1)
print("PINNStudio closed cleanly")
