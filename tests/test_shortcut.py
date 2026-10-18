"""Desktop shortcuts (pinnstudio --create-shortcut / --remove-shortcut) and
the Python used for generated scripts when PINNStudio runs under pythonw."""
import os
import plistlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnstudio import shortcut
from pinnstudio.core import pyexe


def _home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    (tmp_path / "Desktop").mkdir()


def test_icons_are_packaged():
    for f in ("pinnstudio.png", "pinnstudio.ico", "pinnstudio.icns"):
        assert os.path.getsize(os.path.join(shortcut.RES, f)) > 1000


def test_linux(monkeypatch, tmp_path):
    _home(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    (tmp_path / "Desktop/pinnstudio.desktop").write_text("[Desktop Entry]\n")   # from an earlier version
    made = shortcut.create()
    entry = tmp_path / ".local/share/applications/pinnstudio.desktop"
    # the app menu only: Ubuntu's desktop blocks launcher files there
    assert made == [str(entry)] and not (tmp_path / "Desktop/pinnstudio.desktop").exists()
    text = entry.read_text()
    assert f'Exec="{sys.executable}" -m pinnstudio' in text and "Terminal=false" in text
    assert "StartupWMClass=pinnstudio" in text and os.path.isfile(text.split("Icon=")[1].split("\n")[0])
    assert sorted(shortcut.remove()) == sorted(made) and not entry.exists()


def test_macos(monkeypatch, tmp_path):
    _home(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "platform", "darwin")
    app = tmp_path / "Applications/PINNStudio.app"
    assert shortcut.create() == [str(app)]
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "PINNStudio" and info["CFBundleIconFile"] == "pinnstudio"
    exe = app / "Contents/MacOS/PINNStudio"
    assert os.access(exe, os.X_OK) and f'"{sys.executable}" -m pinnstudio' in exe.read_text()
    assert (app / "Contents/Resources/pinnstudio.icns").is_file()
    assert shortcut.remove() == [str(app)] and not app.exists()


def test_windows(monkeypatch, tmp_path):
    _home(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")
    start = tmp_path / "Programs"
    start.mkdir()
    scripts = []

    class R:
        returncode, stderr = 0, ""

    def fake_ps(script):
        r = R()
        if "GetFolderPath" in script:
            r.stdout = f"{start}\n{tmp_path / 'Desktop'}\n"
        else:
            scripts.append(script)
            lnk = script.split("CreateShortcut('")[1].split("')")[0]
            open(lnk, "w", encoding="utf-8").close()
        return r

    monkeypatch.setattr(shortcut, "_ps", fake_ps)
    made = shortcut.create()
    assert made == [str(start / "PINNStudio.lnk"), str(tmp_path / "Desktop/PINNStudio.lnk")]
    assert all("$s.Arguments = '-m pinnstudio'" in s and "pinnstudio.ico" in s for s in scripts)
    assert sorted(shortcut.remove()) == sorted(made)


def test_cli(monkeypatch, tmp_path, capsys):
    _home(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    assert shortcut.main(["--create-shortcut"]) == 0 and "shortcut created" in capsys.readouterr().out
    assert shortcut.main(["--remove-shortcut"]) == 0 and "Removed" in capsys.readouterr().out
    assert shortcut.main([]) is None


def test_script_python(monkeypatch, tmp_path):
    for n in ("python.exe", "pythonw.exe"):
        (tmp_path / n).write_text("")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "pythonw.exe"))
    assert pyexe.script_python() == str(tmp_path / "python.exe")
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    assert pyexe.script_python() == "/usr/bin/python3"
