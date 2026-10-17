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
