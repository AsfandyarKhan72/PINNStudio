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


# ── Windows: training processes end with PINNStudio ──────────────────────
# On Windows a child process keeps running when its parent ends without
# stopping it first: PINNStudio's terminal closed, the process ended in Task
# Manager, or a crash. The training script then went on using the CPU or GPU
# with nothing left to show or stop it. A job object with "kill on close"
# holds the training processes; Windows closes PINNStudio's handle to it when
# PINNStudio's process ends, for any reason, and then ends them too.
_JOB = None


def _windows_job():
    global _JOB
    if _JOB is None:
        import ctypes
        from ctypes import wintypes

        class _Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

        class _Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _Basic), ("IoInfo", _IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
        info = _Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):   # extended limits
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        _JOB = (k32, job)
    return _JOB


def end_with_pinnstudio(proc):
    """Windows: put a training process started with subprocess.Popen into the
    job above, so it ends when PINNStudio's process ends. Does nothing on
    other systems, and never stops a run from starting if it fails."""
    if sys.platform != "win32":
        return False
    try:
        k32, job = _windows_job()
        return bool(k32.AssignProcessToJobObject(job, int(proc._handle)))
    except Exception:
        return False


# ── Windows: results folders too long for Windows' 260-character limit ──
# Unless "long paths" is turned on, Windows refuses paths of 260 characters
# or more. PINNStudio saves files up to about 45 characters below a run's
# folder (e.g. "\solution_results\..."), and a sweep adds its own folder per
# run. A results folder of 227 characters failed after training, while
# saving; 210 still worked (benchmarks/platform/windows_speed.md).
_MAX_PATH = 259
_BELOW_RUN = 60        # deepest file below a run's folder, with some margin
_BELOW_SWEEP = 40      # a sweep run's own folder, on top of that


def _long_paths_enabled():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return bool(winreg.QueryValueEx(k, "LongPathsEnabled")[0])
    except Exception:
        return False


def results_path_problem(folder, sweep=False, platform=None, long_paths=None):
    """A message when `folder` is too long for the files a run saves in it
    (Windows without long paths only), else None."""
    platform = sys.platform if platform is None else platform
    if platform != "win32" or not folder:
        return None
    if long_paths is None:
        long_paths = _long_paths_enabled()
    if long_paths:
        return None
    n = len(os.path.abspath(folder)) if platform == sys.platform else len(folder)
    limit = _MAX_PATH - _BELOW_RUN - (_BELOW_SWEEP if sweep else 0)
    if n <= limit:
        return None
    return (f"❌ Cannot start -- the results folder's path is too long for Windows ({n} characters; "
            f"up to {limit} work here). The files PINNStudio saves inside it would pass Windows' "
            "260-character limit. Choose a shorter results folder (Save to), or turn on Windows' "
            "long paths (LongPathsEnabled).")
