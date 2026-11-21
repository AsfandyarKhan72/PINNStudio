"""The Windows program (packaging/windows): inside it, PINNStudio runs its training scripts with the program
itself, tells the user to update with the installer, leaves shortcuts to the installer, and the program's
entry point runs a script it is given as Python would."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio import shortcut  # noqa: E402
from pinnstudio.core import pyexe, torch_index, update_info  # noqa: E402


def test_inside_the_program_scripts_run_with_the_program(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\Users\u\AppData\Local\Programs\PINNStudio\PINNStudio.exe")
    assert pyexe.frozen()
    assert pyexe.script_python() == sys.executable
    assert pyexe.script_env()["PYINSTALLER_SUPPRESS_SPLASH_SCREEN"] == "1", "no splash screen per training run"
    assert "setup.exe" in update_info.update_command() and "pip" not in update_info.update_command()
    assert "pip" not in " ".join(torch_index.gpu_fix_lines("cu128"))
    assert shortcut.main(["--create-shortcut"]) == 0


def test_outside_the_program_nothing_changes(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert not pyexe.frozen()
    assert "PYINSTALLER_SUPPRESS_SPLASH_SCREEN" not in pyexe.script_env()
    assert "pip" in update_info.update_command(prefix="/usr", executable="/usr/bin/python3", env={})


def test_the_entry_point_runs_a_script_as_python_would(tmp_path):
    script = tmp_path / "hello.py"
    script.write_text("import sys\nprint('ran', sys.argv[1:], __name__)\nsys.exit(3)\n", encoding="utf-8")
    entry = os.path.join(ROOT, "packaging", "windows", "pinnstudio_windows.py")
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, entry, str(script), "a", "b"], capture_output=True, text=True, env=env,
                       timeout=120)
    assert r.returncode == 3 and "ran ['a', 'b'] __main__" in r.stdout, (r.stdout, r.stderr)
    bad = tmp_path / "bad.py"
    bad.write_text("raise ValueError('boom')\n", encoding="utf-8")
    r = subprocess.run([sys.executable, entry, str(bad)], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 1 and "ValueError: boom" in r.stderr


def test_the_entry_point_passes_commands_to_the_command_line(tmp_path):
    entry = os.path.join(ROOT, "packaging", "windows", "pinnstudio_windows.py")
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, entry, "--version"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and r.stdout.startswith("PINNStudio "), (r.stdout, r.stderr)


def test_the_windows_version_is_found_without_running_ver(monkeypatch):
    """PyTorch's import asks platform.machine(); when WMI fails, platform ran `cmd /c ver`, a console window
    flashing up before PINNStudio's window. The start replaces that fallback."""
    import platform
    import subprocess as sp
    from pinnstudio import main as M
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "_syscmd_ver", platform._syscmd_ver)     # restored after the test
    ran = []
    monkeypatch.setattr(sp, "check_output", lambda *a, **k: ran.append(a) or "")
    M._no_console_flashes()
    assert platform._syscmd_ver() == ("", "", "") and ran == []
