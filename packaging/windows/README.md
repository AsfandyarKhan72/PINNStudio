# PINNStudio for Windows

PINNStudio is a Python package (`pip install pinnstudio`). For Windows it is also built as a program that needs
no Python: PyInstaller makes the program folder, Inno Setup the installer.

| File | What it is |
|---|---|
| `build_windows.bat` | The build, on Windows: libraries, PyInstaller, the build check, the launch check, the installer |
| `build_windows_wsl.sh` | The same from WSL: copies the source to a Windows drive and runs the batch file there |
| `pinnstudio.spec` | PyInstaller's recipe for `PINNStudio.exe` and its `_internal\` folder |
| `pinnstudio_windows.py` | The program's entry point: the window, `PINNStudio.exe script.py` (a training run), or the command line |
| `check_build.py` | The build check, run inside the program |
| `pinnstudio.iss` | The installer (Inno Setup 6) |
| `make_splash.py` | The splash screen shown while the program loads (the icon is `pinnstudio/resources/pinnstudio.ico`) |

## What a build makes

In `dist\`:

- `PINNStudio\PINNStudio.exe` with `_internal\` beside it (Python, Qt, PyTorch's CPU build, DeepXDE, the
  reference data and the licences);
- `PINNStudio-VERSION-setup.exe` and `PINNStudio-VERSION-setup.exe.sha256`, when Inno Setup 6 is installed.

The installer installs for the current user by default, with no administrator needed; the one installing may
choose every user instead. It adds a Start-menu entry, optionally a desktop shortcut and PINNStudio in the
"Open with" menu of `.pinn.json` problem files, and an uninstaller in Windows' list of programs. It is not
signed: SmartScreen warns the first time it runs ("More info", then "Run anyway"); the `.sha256` file says what
the file must hash to.

The program trains with PyTorch's CPU build. An NVIDIA build would make it several GB; GPU users install with
the one-command installer or pip, which pick the build that fits the driver.

## How the program runs training

Solve writes a Python script and runs it in a second process. A Python installation runs it with `python.exe`;
the program has none, so it runs the script itself: `pinnstudio.core.pyexe.script_python()` returns
`PINNStudio.exe` when `sys.frozen` is set, and `pinnstudio_windows.py` runs a `.py` file it is given as Python
would (its output reaches the Training Log as usual, without the splash screen).

## Building

On Windows, with Python 3.12 and Inno Setup 6:

```bat
winget install Python.Python.3.12 JRSoftware.InnoSetup
packaging\windows\build_windows.bat
```

From WSL, with the Windows Python and Inno Setup installed as above:

```bash
packaging/windows/build_windows_wsl.sh                  # BUILD_DIR_WIN (default D:\pinnstudio_build), REF
```

The libraries are pinned to `constraints.txt`, the versions the tests passed with. `PYTHON` names another
Python, `ISCC` another Inno Setup compiler. `PINNSTUDIO_INSTALL_CHECK=1` (from WSL also
`WSLENV=PINNSTUDIO_INSTALL_CHECK`) adds the install check: the installer runs silently into `build\windows\`, the
installed program starts off screen for 8 seconds, and the uninstaller removes it again. It uses the program's
own AppId, so it refuses (exit code 3) where PINNStudio is installed with this installer.

For a release, `.github/workflows/release.yml` builds it on a Windows runner from the version tag, with the
install check, and attaches the installer and its `.sha256` to the draft release.

Silently:

```bat
PINNStudio-1.6.0-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER
"%LOCALAPPDATA%\Programs\PINNStudio\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES
```
