"""Problem files (``*.pinn.json``) without the window: read, write, and the reference_data folder.

The window and the command line read the same files the same way: keys PINNConfig does not know are
left out (a newer PINNStudio's file opens), reference_data paths saved as ``@REF/...`` point into this
installation, and a designed network's saved-module paths, kept relative to the file, become absolute.
"""
from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime

from pinnstudio.core import portable_paths as _portable
from pinnstudio.core.config import PINNConfig

REFERENCE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  "reference_data")
FORMAT = 1


class ProblemFileError(ValueError):
    """A file that is not a PINNStudio problem, or cannot be read."""


def read(path: str) -> tuple:
    """(PINNConfig, problem name, the file's whole JSON) of a problem file."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ProblemFileError(f"{path} cannot be read: {exc}")
    raw = data.get("config", data) if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        raise ProblemFileError(f"{path} is not a PINNStudio problem file")
    known = {f.name for f in dataclasses.fields(PINNConfig)}
    filtered = _portable.from_portable({k: v for k, v in raw.items() if k in known}, REFERENCE_DATA_DIR)
    if filtered.get("network_design"):
        from pinnstudio.core.netdesign.library import relocate_paths
        filtered["network_design"] = relocate_paths(filtered["network_design"],
                                                    os.path.dirname(os.path.abspath(path)), relative=False)
    try:
        config = PINNConfig(**filtered)
    except TypeError as exc:
        raise ProblemFileError(f"{path}: {exc}")
    name = data.get("problem_name") if isinstance(data, dict) else None
    if not name:
        name = os.path.basename(path)
        for ext in (".json", ".pinn"):
            if name.endswith(ext):
                name = name[:-len(ext)]
    return config, name, data


def payload(config: PINNConfig, path: str, version: str = "", extra: dict | None = None) -> dict:
    """What a problem file holds for `config` saved at `path`."""
    name = os.path.splitext(os.path.basename(path))[0]
    if name.endswith(".pinn"):
        name = name[:-5]
    out = {
        "pinnstudio_problem_format": FORMAT,
        "problem_name": name,
        "saved_with_pinnstudio_version": version,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "config": _portable.to_portable(dataclasses.asdict(config), REFERENCE_DATA_DIR),
    }
    if out["config"].get("network_design"):
        from pinnstudio.core.netdesign.library import relocate_paths
        out["config"]["network_design"] = relocate_paths(out["config"]["network_design"],
                                                         os.path.dirname(os.path.abspath(path)), relative=True)
    out.update(extra or {})
    return out


def write(config: PINNConfig, path: str, version: str = "", extra: dict | None = None) -> None:
    """Save atomically: a crash or a full disk leaves the old file or the new one, never half of one."""
    text = json.dumps(payload(config, path, version, extra), indent=2)
    folder = os.path.dirname(os.path.abspath(path))
    import tempfile
    fd, tmp = tempfile.mkstemp(prefix=".pinnstudio-", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            mode = os.stat(path).st_mode & 0o777
        except OSError:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
