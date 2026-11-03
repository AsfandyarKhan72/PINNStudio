# License

PINNStudio is released under the **MIT License**. Copyright (c) 2026 Asfandyar Khan. The repository's
[`LICENSE`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/LICENSE) file is the authoritative statement;
this page only summarizes it.

| Version | License |
|---|---|
| Releases up to and including **v1.5.0** (PyPI and tag `v1.5.0`) | MIT License, as stated in those releases' own `LICENSE` file |
| **v1.6.0**, the version these pages describe (the source in the repository) | MIT License |

## Generated scripts

PINNStudio writes a Python script for each problem: the script *Export script* saves, and the script *Solve* runs.
You may use, change and share a generated script, including the parts of PINNStudio's code templates it contains,
under terms of your choice. The libraries it imports (such as DeepXDE, PyTorch, NumPy and Matplotlib) keep their own
licenses.

## Your input and your results

What you enter into PINNStudio (equations, constants, settings, geometry, data) and what you produce by running it or
a generated script (data files, figures, trained models, logs) are your own work; PINNStudio's copyright holder
claims no rights in them.

## Third-party software

PINNStudio's interface is built on Qt for Python (PySide6-Essentials, with shiboken6 and the Qt 6 libraries), used
under the GNU Lesser General Public License version 3. PINNStudio also runs on DeepXDE (LGPL-2.1), PyTorch, NumPy,
SciPy, Matplotlib and pandas. All of these are installed as separate packages and keep their own licenses; some of
the interface's icons are derived from Lucide and Feather. See [Dependencies and attribution](dependencies.md) and the
repository's [`THIRD_PARTY_LICENSES.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/THIRD_PARTY_LICENSES.md)
and [`LICENSES/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/LICENSES) folder.

## This documentation

The documentation is part of the PINNStudio repository and is distributed with it, under the same license.
