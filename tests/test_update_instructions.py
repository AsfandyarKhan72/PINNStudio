"""The update notice gives the command that updates this copy of PINNStudio:
the installer command for a copy the one-command installer made (a uv tool
environment), and the pip of the running Python otherwise. The installer
script names another PINNStudio found first on the PATH, its version, and
how to remove it."""
import os
import shutil
import stat
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio.core.update_info import INSTALL_PS1, INSTALL_SH, is_uv_tool_install, update_command  # noqa: E402


def test_installer_copy_gets_the_installer_command():
    assert update_command("/home/a/.local/share/uv/tools/pinnstudio", "/x/python", "linux", {}) == INSTALL_SH
    assert update_command("/Users/a/.local/share/uv/tools/pinnstudio", "/x/python", "darwin", {}) == INSTALL_SH
    assert update_command(r"C:\Users\a\AppData\Roaming\uv\tools\pinnstudio", r"C:\x\python.exe", "win32", {}) == INSTALL_PS1


def test_custom_uv_tool_dir():
    env = {"UV_TOOL_DIR": "/opt/tools"}
    assert is_uv_tool_install("/opt/tools/pinnstudio", env)
    assert not is_uv_tool_install("/opt/tools/other", env)


def test_pip_copy_gets_its_own_pip():
    cmd = update_command("/home/a/miniconda3", "/home/a/miniconda3/bin/python", "linux", {})
    assert cmd == '"/home/a/miniconda3/bin/python" -m pip install --upgrade pinnstudio'
    cmd = update_command(r"C:\venvs\ps", r"C:\venvs\ps\Scripts\python.exe", "win32", {})
    assert cmd == r'"C:\venvs\ps\Scripts\python.exe" -m pip install --upgrade pinnstudio'


def test_installer_script_syntax():
    if not shutil.which("sh"):
        pytest.skip("no sh")
    r = subprocess.run(["sh", "-n", os.path.join(ROOT, "get-pinnstudio.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _fake_copy(d, version):
    """A pip-style pinnstudio command whose Python reports `version`."""
    py = os.path.join(d, "python")
    with open(py, "w") as f:
        f.write(f"#!/bin/sh\necho {version}\n")
    exe = os.path.join(d, "pinnstudio")
    with open(exe, "w") as f:
        f.write(f"#!{py}\nimport sys\n")
    for p in (py, exe):
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    return exe, py


@pytest.mark.skipif(sys.platform.startswith("win") or not shutil.which("sh"), reason="POSIX sh installer")
def test_installer_names_the_other_copy():
    d = tempfile.mkdtemp()
    exe, py = _fake_copy(d, "1.2.2")
    script = os.path.join(ROOT, "get-pinnstudio.sh")
    cmd = (f'PINNSTUDIO_INSTALLER_FUNCTIONS_ONLY=1; . "{script}"; '
           f'report_other_copy "{exe}" /home/a/.local/bin/pinnstudio; '
           f'echo ---same; report_other_copy /home/a/.local/bin/pinnstudio /home/a/.local/bin/pinnstudio; '
           f'echo ---none; report_other_copy "" /home/a/.local/bin/pinnstudio')
    r = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out, rest = r.stdout.split("---same")
    assert "Another PINNStudio (1.2.2) is installed" in out
    assert exe in out
    assert f'"{py}" -m pip uninstall pinnstudio' in out
    assert "Another" not in rest       # the same copy, or none: nothing to say
