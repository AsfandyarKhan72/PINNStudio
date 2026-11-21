# PyInstaller recipe for the Windows program (packaging/windows/build_windows.bat runs it).
#
# One folder, dist\PINNStudio: PINNStudio.exe beside _internal\, which holds Python, Qt, PyTorch, DeepXDE and
# the rest. A folder rather than one file: one file would unpack all of it (about 1 GB) to a temporary
# folder at every start, and again for every training run.

import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata
from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                 VarFileInfo, VarStruct, VSVersionInfo)

HERE = Path(SPECPATH)                       # noqa: F821 (PyInstaller supplies SPECPATH: this file's folder)
ROOT = HERE.parent.parent
ICON = ROOT / "pinnstudio" / "resources" / "pinnstudio.ico"
SPLASH = HERE / "splash.png"                 # kept in the repository (make_splash.py draws it)
VERSION = re.search(r'__version__\s*=\s*"([^"]+)"', (ROOT / "pinnstudio" / "__init__.py").read_text()).group(1)

# PINNStudio: its modules (the window imports some by name) and resources
datas, binaries, hiddenimports = collect_all("pinnstudio")
# the examples' reference data, beside the package as in a source checkout (pinnstudio/core/problem_file.py)
datas += [(str(p), str(Path("reference_data") / p.parent.relative_to(ROOT / "reference_data")))
          for p in (ROOT / "reference_data").rglob("*") if p.is_file() and "local" not in p.relative_to(ROOT).parts]
# the licences: PINNStudio's, and those of what the program holds (Qt is LGPL)
datas += [(str(ROOT / "LICENSE"), "."), (str(ROOT / "THIRD_PARTY_LICENSES.md"), ".")]
datas += [(str(p), "LICENSES") for p in (ROOT / "LICENSES").glob("*.txt")]

# The training scripts PINNStudio writes run inside this program (PINNStudio.exe script.py), so it holds what
# they import, and DeepXDE picks its backend by name at run time
for name in ("deepxde",):
    d, b, h = collect_all(name)
    datas, binaries, hiddenimports = datas + d, binaries + b, hiddenimports + h
hiddenimports += collect_submodules("deepxde.backend.pytorch")
hiddenimports += ["matplotlib.backends.backend_agg", "matplotlib.backends.backend_qtagg",
                  "PIL.Image", "PIL.PngImagePlugin", "PIL.GifImagePlugin", "scipy.special", "scipy.interpolate",
                  "scipy.spatial", "scipy.integrate", "scipy.optimize", "pandas"]
datas += collect_data_files("scipy", excludes=["**/tests/**"])
for name in ("torch", "deepxde", "numpy", "scipy", "matplotlib", "pandas", "PySide6_Essentials"):
    try:
        datas += copy_metadata(name)      # versions the program reports (environment.json, About)
    except Exception:
        pass

numbers = [int(n) for n in re.findall(r"\d+", VERSION)[:3]]
numbers = tuple(numbers + [0] * (4 - len(numbers)))
version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=numbers, prodvers=numbers),
    kids=[StringFileInfo([StringTable("040904B0", [
              StringStruct("ProductName", "PINNStudio"),
              StringStruct("FileDescription", "PINNStudio: physics-informed neural networks"),
              StringStruct("ProductVersion", VERSION),
              StringStruct("FileVersion", VERSION),
              StringStruct("LegalCopyright", "MIT License"),
              StringStruct("OriginalFilename", "PINNStudio.exe")])]),
          VarFileInfo([VarStruct("Translation", [0x0409, 1200])])],
)

a = Analysis(                               # noqa: F821
    [str(HERE / "pinnstudio_windows.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "IPython", "PyQt5", "PyQt6", "PySide2", "tensorflow", "jax", "paddle", "keras"],
    noarchive=False,
)
pyz = PYZ(a.pure)                           # noqa: F821

# drawn by the bootloader the moment the program starts, before Python, PyTorch and Qt load; the window
# closes it once it is shown (pinnstudio_windows.py), and training runs suppress it (pinnstudio.core.pyexe)
splash = Splash(                            # noqa: F821
    str(SPLASH),
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(24, 226),
    text_size=10,
    text_color="#5b6b8a",
    text_default="Starting…",
    always_on_top=False,
)

exe = EXE(                                  # noqa: F821
    pyz, a.scripts, splash, [],
    exclude_binaries=True,
    name="PINNStudio",
    console=False,                          # no console window; output still reaches a redirected stdout
    upx=False,                              # UPX-packed Qt and PyTorch DLLs fail to load on some machines
    strip=False,
    icon=str(ICON),
    version=version_info,
)
coll = COLLECT(exe, splash.binaries, a.binaries, a.datas, strip=False, upx=False, name="PINNStudio")  # noqa: F821
