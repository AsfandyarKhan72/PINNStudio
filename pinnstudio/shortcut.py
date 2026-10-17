"""Desktop shortcuts for PINNStudio, so it opens from an icon like any other app.

    pinnstudio --create-shortcut    add PINNStudio to the Start menu and desktop (Windows),
                                    ~/Applications (macOS) or the app menu (Linux)
    pinnstudio --remove-shortcut    remove them again

The shortcut starts the Python that PINNStudio is installed in, with
``-m pinnstudio``, so it keeps working after an update. On Windows it uses
pythonw.exe, so no console window opens next to the app.
"""
import os
import shutil
import stat
import subprocess
import sys

NAME = "PINNStudio"
COMMENT = "Physics-informed neural networks for forward and inverse problems"
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def _home():
    return os.path.expanduser("~")


# ── Windows ──────────────────────────────────────────────────────────────
def _pythonw():
    exe = sys.executable
    d, base = os.path.split(exe)
    if base.lower() == "python.exe" and os.path.isfile(os.path.join(d, "pythonw.exe")):
        return os.path.join(d, "pythonw.exe")
    return exe


def _ps(script):
    return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                          capture_output=True, text=True)


def _windows_dirs():
    r = _ps("[Environment]::GetFolderPath('Programs'); [Environment]::GetFolderPath('Desktop')")
    dirs = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    if len(dirs) != 2:   # fall back to the usual locations
        appdata = os.environ.get("APPDATA", os.path.join(_home(), "AppData", "Roaming"))
        dirs = [os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs"),
                os.path.join(_home(), "Desktop")]
    return dirs


def _windows_create():
    made = []
    for d in _windows_dirs():
        if not os.path.isdir(d):
            continue
        lnk = os.path.join(d, NAME + ".lnk")
        q = lambda s: s.replace("'", "''")
        r = _ps("$s = (New-Object -ComObject WScript.Shell).CreateShortcut('%s'); "
                "$s.TargetPath = '%s'; $s.Arguments = '-m pinnstudio'; "
                "$s.IconLocation = '%s'; $s.WorkingDirectory = '%s'; $s.Description = '%s'; $s.Save()"
                % (q(lnk), q(_pythonw()), q(os.path.join(RES, "pinnstudio.ico")), q(_home()), q(COMMENT)))
        if r.returncode == 0 and os.path.isfile(lnk):
            made.append(lnk)
        else:
            raise RuntimeError(f"could not create {lnk}: {r.stderr.strip()}")
    return made


def _windows_paths():
    return [os.path.join(d, NAME + ".lnk") for d in _windows_dirs()]


# ── macOS ────────────────────────────────────────────────────────────────
def _mac_app():
    return os.path.join(_home(), "Applications", NAME + ".app")


def _mac_create():
    app = _mac_app()
    if os.path.isdir(app):
        shutil.rmtree(app)
    macos, res = os.path.join(app, "Contents", "MacOS"), os.path.join(app, "Contents", "Resources")
    os.makedirs(macos)
    os.makedirs(res)
    shutil.copy(os.path.join(RES, "pinnstudio.icns"), os.path.join(res, "pinnstudio.icns"))
    exe = os.path.join(macos, NAME)
    with open(exe, "w") as f:
        f.write('#!/bin/sh\ncd "$HOME"\nexec "%s" -m pinnstudio "$@"\n' % sys.executable)
    os.chmod(exe, os.stat(exe).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    with open(os.path.join(app, "Contents", "Info.plist"), "w") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>{NAME}</string>
  <key>CFBundleDisplayName</key><string>{NAME}</string>
  <key>CFBundleIdentifier</key><string>io.github.asfandyarkhan72.pinnstudio</string>
  <key>CFBundleExecutable</key><string>{NAME}</string>
  <key>CFBundleIconFile</key><string>pinnstudio</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
""")
    return [app]


# ── Linux ────────────────────────────────────────────────────────────────
def _linux_entry():
    data = os.environ.get("XDG_DATA_HOME") or os.path.join(_home(), ".local", "share")
    return os.path.join(data, "applications", "pinnstudio.desktop")


def _linux_desktop_copy():
    d = os.path.join(_home(), "Desktop")
    return os.path.join(d, "pinnstudio.desktop") if os.path.isdir(d) else None


def _linux_create():
    entry = _linux_entry()
    os.makedirs(os.path.dirname(entry), exist_ok=True)
    text = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={NAME}\n"
        f"Comment={COMMENT}\n"
        f'Exec="{sys.executable}" -m pinnstudio\n'
        f"Path={_home()}\n"
        f"Icon={os.path.join(RES, 'pinnstudio.png')}\n"
        "Terminal=false\n"
        "Categories=Science;Education;Development;\n"
        "StartupWMClass=pinnstudio\n"
    )
    made = []
    for path in filter(None, (entry, _linux_desktop_copy())):
        with open(path, "w") as f:
            f.write(text)
        os.chmod(path, 0o755)
        made.append(path)
        if path != entry and shutil.which("gio"):   # GNOME: let the desktop icon launch without a prompt
            subprocess.run(["gio", "set", path, "metadata::trusted", "true"], capture_output=True)
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", os.path.dirname(entry)], capture_output=True)
    return made


# ── public ───────────────────────────────────────────────────────────────
def paths():
    """Where the shortcuts go on this computer."""
    if sys.platform == "win32":
        return _windows_paths()
    if sys.platform == "darwin":
        return [_mac_app()]
    return [p for p in (_linux_entry(), _linux_desktop_copy()) if p]


def create():
    """Create the shortcuts; returns the paths written."""
    if sys.platform == "win32":
        return _windows_create()
    if sys.platform == "darwin":
        return _mac_create()
    return _linux_create()


def remove():
    """Remove the shortcuts; returns the paths removed."""
    gone = []
    for p in paths():
        if os.path.isdir(p):
            shutil.rmtree(p)
            gone.append(p)
        elif os.path.isfile(p):
            os.remove(p)
            gone.append(p)
    return gone


def main(argv):
    """Handle --create-shortcut / --remove-shortcut; returns an exit code, or None for other arguments."""
    if "--create-shortcut" in argv:
        try:
            made = create()
        except Exception as e:     # say what went wrong, without a traceback
            print(f"Could not create the PINNStudio shortcut: {e}")
            return 1
        print("PINNStudio shortcut created:")
        for p in made:
            print("  " + p)
        return 0
    if "--remove-shortcut" in argv:
        gone = remove()
        print("Removed:" if gone else "No PINNStudio shortcut found.")
        for p in gone:
            print("  " + p)
        return 0
    return None
