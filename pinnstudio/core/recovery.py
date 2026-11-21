"""Unsaved work kept against a crash: one autosave file per running PINNStudio, offered back at the next
start when the PINNStudio that wrote it is no longer running.

Each window writes ``~/.pinnstudio/recovery/autosave-<process id>.pinn.json`` (``$PINNSTUDIO_HOME``
overrides the folder) while its problem has unsaved changes, and removes it when the problem is saved,
replaced or closed normally. Two PINNStudios at once each have their own file. A file whose process has
gone was left by a crash (or a power cut) and is an orphan: the next start offers to recover it.
"""
from __future__ import annotations

import glob
import os
import re
import time

from pinnstudio.core import problem_file as _pf

PREFIX = "autosave-"
SUFFIX = ".pinn.json"


def recovery_dir() -> str:
    from pinnstudio.core.netdesign.trust import home
    return os.path.join(home(), "recovery")


def own_path() -> str:
    return os.path.join(recovery_dir(), f"{PREFIX}{os.getpid()}{SUFFIX}")


def write(config, original_path: str | None, problem_name: str, version: str = "") -> str:
    """Keep the current problem in this process's autosave file (atomically)."""
    os.makedirs(recovery_dir(), exist_ok=True)
    path = own_path()
    _pf.write(config, path, version, extra={"recovery": {
        "pid": os.getpid(), "original_path": original_path or "", "problem_name": problem_name or "",
        "time": time.time()}})
    return path


def discard(path: str | None = None) -> None:
    """Remove an autosave file (this process's own by default)."""
    try:
        os.remove(path or own_path())
    except OSError:
        pass


def _alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259                              # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def orphans() -> list:
    """[{path, pid, original_path, problem_name, time}] of autosave files whose PINNStudio has gone,
    newest first."""
    out = []
    for path in glob.glob(os.path.join(recovery_dir(), f"{PREFIX}*{SUFFIX}")):
        m = re.search(rf"{PREFIX}(\d+){re.escape(SUFFIX)}$", os.path.basename(path))
        if not m or _alive(int(m.group(1))):
            continue
        try:
            _cfg, _name, data = _pf.read(path)
        except _pf.ProblemFileError:
            continue
        meta = dict(data.get("recovery") or {})
        meta.setdefault("time", os.path.getmtime(path))
        meta.update(path=path, pid=int(m.group(1)))
        out.append(meta)
    out.sort(key=lambda m: -float(m.get("time") or 0))
    return out
