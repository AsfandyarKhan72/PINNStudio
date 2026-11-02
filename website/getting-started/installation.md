# Installation

PINNStudio is a Python package. It runs on Linux, Windows and macOS, trains on the CPU or on an NVIDIA GPU, and
needs an internet connection while it is installed (to download PyTorch, DeepXDE, PySide6 and the other packages).

There are two versions you can install:

| Version | What it is | How to install |
|---|---|---|
| **Current release, v1.5.0** | The latest version on [PyPI](https://pypi.org/project/pinnstudio/) | [One-command installer](#one-command-installer-recommended) or [`pip install pinnstudio`](#with-pip) |
| **Development version, v1.6.0 (in development)** | The version this documentation describes; not released yet | [From source](#development-version-from-source) |

Features marked *New in v1.6.0 (in development)* on this site are only in the development version. If you install
the release, the [CHANGELOG](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/CHANGELOG.md) (section
*Unreleased*) lists what it does not have yet.

## Supported systems

The test suite and an end-to-end check of the window (train a model, show the results and the error analysis,
restore the model) run on GitHub Actions on these systems:

| | Tested on |
|---|---|
| **Linux** | Ubuntu 22.04 and 24.04 (x64), Ubuntu 24.04 (ARM) |
| **Windows** | Windows Server 2022 and 2025 (x64), Windows 11 on ARM |
| **macOS** | macOS 14 and 15 (Apple silicon), macOS 15 (Intel) |
| **Python** | 3.9 to 3.14 |
| **One-command installer** | Ubuntu 22.04 and 24.04, Windows Server 2022, Windows 11 on ARM, macOS 15 (Intel) |

The package itself requires Python 3.9 or newer. Training on an NVIDIA GPU is tested on Linux (RTX 4090,
CUDA 12.4); the test machines above train on the CPU. Other versions of these systems may work but are not tested.

Two platforms have restrictions that come from PyTorch:

- **Intel Macs.** PyTorch's last release for Intel Macs is 2.2, which needs Python 3.12 or older and NumPy below 2.
  The installer and `pip` choose these versions automatically (PINNStudio's dependencies require `numpy<2` on Intel
  Macs).
- **Windows on ARM** (for example Snapdragon PCs). PyTorch has no build for it. The one-command installer uses the
  x64 Python, which Windows runs under emulation. With `pip`, install the x64 Python from python.org first.

On a Mac, PINNStudio trains on the CPU. Apple's GPU (MPS) was not faster for PINNs in the project's tests on an
M1 MacBook Air, and it has no float64.

## Open a terminal

Every way of installing starts with a command typed in a terminal:

- **Windows:** open the Start menu, type `PowerShell` and open **Windows PowerShell**.
- **macOS:** press Cmd+Space to open Spotlight, type `Terminal` and press Enter (or open
  **Applications › Utilities › Terminal**).
- **Linux:** open your terminal application (often Ctrl+Alt+T, or search for "Terminal" in the application menu).

For a `pip` or source installation, check Python (3.9 or newer) and, for a source installation, git:

```bash
git --version
python3 --version
```

On Windows type `python --version` instead of `python3 --version`. If a command is not found, see
[Common installation problems](#common-installation-problems). After installing Python or git, close the terminal
and open a new one, so that it finds them.

## One-command installer (recommended)

The installer needs neither Python nor git. It installs the current release (v1.5.0) from PyPI.

=== "Linux / macOS"

    In a terminal:

    ```bash
    curl -LsSf https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/main/get-pinnstudio.sh | sh
    ```

=== "Windows"

    In PowerShell:

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/main/get-pinnstudio.ps1 | iex"
    ```

What the installer does (`get-pinnstudio.sh`, `get-pinnstudio.ps1`):

1. Installs [uv](https://docs.astral.sh/uv/), a Python package manager, if it is not there yet, or updates it if it
   is older than version 0.9.20.
2. Runs `uv tool install` for `pinnstudio` with its own Python 3.12 (the x64 Python on Windows on ARM) and
   `--torch-backend auto`: uv picks the PyTorch build that fits the NVIDIA driver, or a CPU-only build when there
   is no NVIDIA GPU. PINNStudio gets an environment of its own; nothing is installed system-wide, and
   `uv tool dir` shows where it is.
3. Puts the `pinnstudio` command on the PATH and adds a PINNStudio icon: Start menu and desktop on Windows,
   `~/Applications` on macOS (found by Spotlight and Launchpad), the application menu on Linux.
4. Prints the installed version (for example `PINNStudio 1.5.0 installed.`) and where training will run, for
   example `PyTorch will train on the GPU: <name of the GPU>`, or
   `No NVIDIA GPU found: PINNStudio will train on the CPU.`
5. If another PINNStudio (an older `pip`, venv or conda installation) comes first on the PATH, it says where that
   copy is, which version it is, and the command that removes it. It does not remove anything itself.

The installation takes a few minutes. The `pinnstudio` command works in new terminal windows; the installer prints
the full path to use in the current one.

**Updating.** Run the same command again. It replaces the installed version and keeps your settings and results.

**Uninstalling.** These two commands remove the icon, then PINNStudio and its Python (on Windows, in PowerShell):

```bash
pinnstudio --remove-shortcut
uv tool uninstall pinnstudio
```

Your results in `PINNStudio_Results` are not deleted.

!!! warning "Installed with pip before?"
    If you installed PINNStudio with `pip` earlier, run `pip uninstall pinnstudio` in that Python or conda
    environment first. Otherwise typing `pinnstudio` may still start the old copy. The icon the installer adds
    always opens the new one.

## With pip

To install the current release into a Python environment of your own:

```bash
pip install pinnstudio
pinnstudio
```

This installs PINNStudio and its dependencies: DeepXDE (1.10.0 or newer), PyTorch (2.0 or newer), PySide6 (Qt for Python),
NumPy (below 2.5), SciPy, Matplotlib and pandas. NumPy is kept below 2.5 because DeepXDE's 2D `Polygon` and
`Triangle` geometries call `np.cross` on two-component vectors, which NumPy 2.5 no longer allows (see
[Missing or incompatible packages](../help/troubleshooting.md#missing-or-incompatible-packages)). Install it into a virtual environment, so that these packages do
not interfere with other projects:

=== "venv (Linux / macOS)"

    ```bash
    python3 -m venv pinnstudio-env
    source pinnstudio-env/bin/activate
    pip install --upgrade pip
    pip install pinnstudio
    pinnstudio
    ```

=== "venv (Windows)"

    ```powershell
    python -m venv pinnstudio-env
    .\pinnstudio-env\Scripts\activate
    python -m pip install --upgrade pip
    pip install pinnstudio
    pinnstudio
    ```

=== "conda"

    ```bash
    conda create -n pinnstudio python=3.12
    conda activate pinnstudio
    pip install pinnstudio
    pinnstudio
    ```

    PINNStudio is not packaged for conda; `pip` inside the conda environment installs it from PyPI.

To add the desktop icon for a `pip` installation, run `pinnstudio --create-shortcut` (and
`pinnstudio --remove-shortcut` to remove it).

### CPU-only or NVIDIA GPU PyTorch

`pip install pinnstudio` installs whatever PyTorch build PyPI offers for your system; it does not check whether that
build can use your GPU. The one-command installer and the source installers do check (see above and
[below](#development-version-from-source)).

When PINNStudio starts and finds an NVIDIA GPU (the `nvidia-smi` command is available) but the installed PyTorch
cannot use it, it prints a message in the terminal it was started from, beginning with

```text
An NVIDIA GPU was detected, but PyTorch cannot use it (PINNStudio runs on the CPU for now).
```

followed by the commands that install the PyTorch build matching your driver, in the Python PINNStudio runs in. For a
`pip` installation they have this form (the CUDA tag, here `cu126`, is chosen from the CUDA version the driver
reports):

```bash
python -m pip install --upgrade pip
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Upgrading `pip` first matters: an old `pip` can fail on PyTorch's package index with a `flit_core` or
"Could not find a version that satisfies" error. PINNStudio chooses `cu128` for drivers that report CUDA 12.8 or
newer, `cu126` for CUDA 12.0 to 12.7, and `cu118` for CUDA 11.8 to 11.x; with an older driver it tells you to
update the driver. For an installer copy, the message tells you to run the installer again instead.

A GPU is optional. It pays off for larger 2D and 3D problems and inverse runs; small problems (a few hundred to a few
thousand points) often train as fast or faster on the CPU. See
[Seeds, precision and hardware](../reproducibility/environment.md) and the
[Settings menu](../guide/settings.md) (**Settings › GPU / Hardware...**).

### DeepXDE backend

DeepXDE supports several backends; PINNStudio uses **PyTorch** only. Every script it writes, and the exported
script, sets `DDE_BACKEND` to `pytorch` before importing DeepXDE, so a different default backend in your own
DeepXDE configuration does not affect PINNStudio. TensorFlow, JAX or PaddlePaddle do not need to be installed.

## Development version (from source)

The development version, which will be released as v1.6.0, is installed from the
[GitHub repository](https://github.com/AsfandyarKhan72/PINNStudio). You need git and Python 3.9 or newer.

!!! note "Which branch"
    The commands below clone the repository's default branch. If the development version you want is on another
    branch, switch to it with `git checkout <branch>` after cloning, before installing.

```bash
git clone https://github.com/AsfandyarKhan72/PINNStudio.git
cd PINNStudio
```

Then run the source installer. It creates a virtual environment `venv` inside the `PINNStudio` folder, installs
PINNStudio and its dependencies into it, and installs the PyTorch build that fits the NVIDIA driver: it reads the
driver's CUDA version from `nvidia-smi` (`select_torch_index.py`), installs the matching CUDA build of PyTorch last,
checks that this build is the one installed, and falls back to the CPU-only build if not (or if there is no NVIDIA
GPU).

=== "Linux / macOS"

    ```bash
    bash install.sh
    ./venv/bin/pinnstudio
    ```

=== "Windows"

    ```bat
    install.bat
    .\venv\Scripts\pinnstudio.exe
    ```

Nothing is installed system-wide; deleting the `PINNStudio` folder removes everything. Start PINNStudio again later
with the same second command, from inside the `PINNStudio` folder.

The source installer runs `pip install .`, a regular (not editable) installation: after `git pull`, run it again to
install the new code. If you change the code yourself, use one of these instead, inside an activated virtual
environment in the `PINNStudio` folder:

```bash
pip install -e .              # editable install: the pinnstudio command runs the code in this folder
pinnstudio
```

or, as in [CONTRIBUTING.md](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/CONTRIBUTING.md), run it
without installing the package:

```bash
pip install -r requirements.txt
python -m pinnstudio.main
```

Neither of these two picks a PyTorch build for your GPU; see [CPU-only or NVIDIA GPU PyTorch](#cpu-only-or-nvidia-gpu-pytorch).

The one-command installer can also install from a local checkout: set the environment variable
`PINNSTUDIO_SOURCE` to the path of the `PINNStudio` folder (or of a wheel file) and run `get-pinnstudio.sh` or
`get-pinnstudio.ps1` from the checkout. It then installs that code, with its own Python, the matching PyTorch build
and the desktop icon, in place of the release.

!!! warning "Version number of a source installation"
    Until v1.6.0 is released, the package metadata of the development version still says `1.5.0`. `pip show
    pinnstudio` therefore cannot tell a source installation from the release. When you report a problem with the
    development version, give the commit instead: `git rev-parse --short HEAD` in the `PINNStudio` folder.

## Starting PINNStudio

| Installed with | Start it with |
|---|---|
| One-command installer | The PINNStudio icon (Start menu or desktop, Applications, application menu), or `pinnstudio` in a terminal |
| `pip` | `pinnstudio` in the activated environment (`pinnstudio --create-shortcut` adds an icon) |
| `install.sh` / `install.bat` | `./venv/bin/pinnstudio` or `.\venv\Scripts\pinnstudio.exe` in the `PINNStudio` folder |
| Development, without installing | `python -m pinnstudio.main` in the `PINNStudio` folder |

The `pinnstudio` command is the package's entry point (`pinnstudio.main:main`); `python -m pinnstudio` does the
same and is what the desktop icon runs. PINNStudio defines no command-line options besides `--create-shortcut` and
`--remove-shortcut`.

On Windows, PINNStudio started from its icon has no console window. Its output and any unexpected error then go to
the log file `~/.pinnstudio/pinnstudio.log` (in your home folder).

## Verifying the installation

1. Start PINNStudio. The window opens on the **Start** tab, and the status bar at the bottom says **Ready**, the
   floating-point precision (`float32` by default) and the **Save to** folder.
2. Open **Settings › GPU / Hardware...**. The first line of the dialog says what PyTorch finds: either
   `Detected 1 GPU(s) -- 0: <name of the GPU>`, or
   `No CUDA GPU detected -- training will run on CPU regardless of these settings.`
3. Train a small example, as described in [First run](first-run.md). A run that finishes with
   `✅ Training complete!` in the Training Log shows that PyTorch, DeepXDE and the generated scripts work.

`pinnstudio` has no command-line options (`pinnstudio --version` or `--help` simply opens the window). To see
the version, and to check the installed packages, run these in a terminal, in the Python PINNStudio runs in:

```bash
python -c "import importlib.metadata as m; print(m.version('pinnstudio'))"
python -c "import torch, deepxde; print(torch.__version__, torch.cuda.is_available(), deepxde.__version__)"
```

For an installer copy, that Python is inside the folder `uv tool dir` prints (`pinnstudio/bin/python`, or
`pinnstudio\Scripts\python.exe` on Windows).

## Common installation problems

**`pip: command not found`.** Use `pip3`, or `python3 -m pip`. Many systems, macOS in particular, only have `pip3`.

**`pinnstudio: command not found` after `pip install`.** `pip` installed the command into a per-user folder that is
not on your PATH. The install output names it, in a line such as
`WARNING: The script pinnstudio is installed in '.../bin' which is not on PATH.` Add that folder to your PATH
(for example in `~/.zshrc` or `~/.bashrc`), open a new terminal, or install into a virtual environment instead.
After the one-command installer, open a new terminal window: the PATH change applies to new terminals.

**Python missing, or older than 3.9** (source or `pip` installation). Install Python from
[python.org](https://www.python.org/downloads/). On Windows, tick *Add python.exe to PATH* in the installer, then
open a new terminal. `install.bat` stops with
`Could not create the virtual environment. Is Python 3.9 or newer installed and on PATH?` when it cannot find
Python.

**`git` missing** (source installation). Install it from [git-scm.com](https://git-scm.com/downloads). On macOS,
the first `git --version` may offer to install Apple's Command Line Tools.

**The installer finds an NVIDIA GPU but PyTorch cannot use it.** It prints
`An NVIDIA GPU was found, but PyTorch cannot use it; PINNStudio will train on the CPU.` Usually the NVIDIA driver is
too old for the available PyTorch builds: update the driver, restart, and run the installer again. For a source
installation, `install.sh`/`install.bat` say
`Installed PyTorch build doesn't match the requested CUDA version` and fall back to the CPU-only build; after
updating the driver, run the installer again.

**Typing `pinnstudio` starts an old version.** Another copy comes first on the PATH. The one-command installer
reports where it is; remove it with the `pip` of the environment it was installed in
(`pip uninstall pinnstudio`).

**The window does not open** (Qt platform plugin errors on Linux, `DLL load failed` on Windows). See
[PINNStudio does not start](../help/troubleshooting.md#pinnstudio-does-not-start) in Troubleshooting.
