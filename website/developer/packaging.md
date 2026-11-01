# Building and packaging

## Package metadata

PINNStudio is a setuptools package. `pyproject.toml` only declares the build system (setuptools ≥ 61 and wheel);
the metadata is in `setup.py`:

- `name="pinnstudio"`, `version` (the version of the latest release; see [Releases](releases.md)),
  `python_requires=">=3.9"`;
- `packages=find_packages()`, which finds `pinnstudio` and its subpackages and `reference_data` (it has an
  `__init__.py`), with `include_package_data=True` and `package_data={"pinnstudio": ["resources/*"]}`;
- the console script `pinnstudio=pinnstudio.main:main`;
- `install_requires`: `deepxde>=1.10.0`, `torch>=2.0`, `PyQt6>=6.4`, `matplotlib`, `pandas`, `scipy`, and
  NumPy limits: below 2.5 (DeepXDE's 2D polygon geometry uses a NumPy call removed in 2.5), and below 2 on
  Intel Macs (PyTorch's last Intel-Mac build needs NumPy 1). On Windows, PyQt6 and PyQt6-Qt6 are pinned to
  6.6.1 because newer builds fail to load on some Windows versions.

`requirements.txt` lists the dependencies for a plain `pip install -r requirements.txt` development setup.

## Source distribution and wheel

`MANIFEST.in` adds to the source distribution the reference data (`reference_data/**/*.txt`), the license files
(`THIRD_PARTY_LICENSES.md`, `LICENSES/*.txt`), `requirements.txt`, `CHANGELOG.md`, `CITATION.cff`,
`select_torch_index.py`, `install.sh` and `install.bat`. With `include_package_data`, the reference-data text
files are also installed with the `reference_data` package, so the examples find their data in an installed
copy.

```bash
python -m pip install build
python -m build                 # writes dist/pinnstudio-<version>.tar.gz and dist/pinnstudio-<version>-py3-none-any.whl
python -m pip install dist/pinnstudio-*.whl     # check the wheel in a fresh environment
```

The wheel is pure Python. PyTorch is a dependency like any other, so `pip` installs the default PyTorch build
from PyPI; a CUDA build matching the machine is chosen by the installers below.

## Installers

| File | What it does |
|---|---|
| `get-pinnstudio.sh` (macOS, Linux), `get-pinnstudio.ps1` (Windows) | the one-command installers of the README: install [uv](https://docs.astral.sh/uv/) if needed, then `uv tool install` PINNStudio with its own Python and the PyTorch build that matches the NVIDIA driver (CPU-only otherwise), put `pinnstudio` on the PATH and add a menu entry or shortcut. Running them again updates. `PINNSTUDIO_SOURCE` installs from a local checkout or a wheel instead of PyPI (the CI workflows use this) |
| `install.sh`, `install.bat` | from a checkout: create `venv/`, `pip install .`, then reinstall the PyTorch build chosen by `select_torch_index.py` and check that it landed |
| `select_torch_index.py` | prints the PyTorch wheel tag (`cu126`, ..., or `cpu`) for this machine's driver; the table is in `pinnstudio/core/torch_index.py` |
| `pinnstudio/shortcut.py` | `python -m pinnstudio --create-shortcut` / `--remove-shortcut`: desktop and menu shortcuts |

The installers install from PyPI by default, that is the latest release (v1.5.0), not the development version.
To try the development version, install from a checkout (`pip install -e .`, `install.sh`, or
`PINNSTUDIO_SOURCE=$PWD sh get-pinnstudio.sh`).

## Checks of a build

- `tools/launch_check.py` starts the installed `pinnstudio` command and checks that it keeps running.
- `tools/smoke_gui.py` runs the window end to end on the current machine (train, results, restore).
- The `installer` jobs of `platform-tests.yml` and `platform-matrix.yml` run the one-command installers from the
  checkout and then the window check ([Tests](tests.md#continuous-integration)).
