"""The Python that runs PINNStudio's generated scripts.

When PINNStudio is started from its desktop shortcut on Windows it runs
under pythonw.exe, which has no console. Training scripts are run with the
matching python.exe instead (so their output reaches the Training Log as
usual), without opening a console window of their own.
"""
import os
import subprocess
import sys


def script_python():
    exe = sys.executable
    d, base = os.path.split(exe)
    if base.lower() == "pythonw.exe" and os.path.isfile(os.path.join(d, "python.exe")):
        return os.path.join(d, "python.exe")
    return exe


# creationflags for subprocess.Popen: no console window on Windows
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def script_env():
    """Environment for a generated script: Python's UTF-8 mode, so files it
    writes and what it prints are UTF-8 on every system. Windows otherwise
    uses its legacy code page (cp1252), which cannot hold the symbols the
    scripts print and save (L2 subscripts, check marks, ...)."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# Popen arguments that go with script_env(): read the script's output as UTF-8
TEXT_UTF8 = dict(text=True, encoding="utf-8", errors="replace")
