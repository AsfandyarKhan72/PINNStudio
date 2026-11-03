"""Windows: a training process ends when PINNStudio's process ends, even when
PINNStudio is ended without stopping it (its terminal closed, Task Manager,
a crash). Before, the training script kept running on its own."""
import ctypes
import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio.core import pyexe  # noqa: E402

PARENT = r"""
import subprocess, sys, time
sys.path.insert(0, sys.argv[1])
from pinnstudio.core import pyexe
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], creationflags=pyexe.NO_WINDOW)
print(child.pid, pyexe.end_with_pinnstudio(child), flush=True)
time.sleep(120)
"""


def _alive(pid):
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return code.value == 259                          # STILL_ACTIVE


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_training_process_ends_when_pinnstudio_is_killed():
    parent = subprocess.Popen([sys.executable, "-c", PARENT, ROOT], stdout=subprocess.PIPE, text=True)
    pid, attached = parent.stdout.readline().split()
    pid = int(pid)
    try:
        assert attached == "True"
        assert _alive(pid)
        parent.kill()                                 # TerminateProcess, no clean-up in the parent
        parent.wait(10)
        for _ in range(50):
            if not _alive(pid):
                break
            time.sleep(0.1)
        assert not _alive(pid), "the child kept running after its parent was killed"
    finally:
        if _alive(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)


@pytest.mark.skipif(sys.platform == "win32", reason="the job object is Windows only")
def test_nothing_to_do_elsewhere():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    assert pyexe.end_with_pinnstudio(p) is False
    p.wait()
