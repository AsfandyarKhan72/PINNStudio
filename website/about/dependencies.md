# Dependencies and attribution

PINNStudio is built on other free software. This page lists the packages it needs, what it uses them for and their
licenses, and the third-party material included in PINNStudio's own files. The authoritative list is the repository's
[THIRD_PARTY_LICENSES.md](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/THIRD_PARTY_LICENSES.md); the
version requirements are in `setup.py` and `requirements.txt`. PINNStudio's own license is described on
[License](license.md).

PINNStudio is an independent project. It is not affiliated with, sponsored by or endorsed by the developers of
DeepXDE, PyTorch, The Qt Company (Qt, Qt for Python) or the other projects listed here.

## Packages PINNStudio runs on

These packages are installed separately, as their own packages, by `pip`, `uv` or PINNStudio's installers. PINNStudio
does not copy their code. Each keeps its own license, which comes with the installed package. The licenses below are
those stated by the packages' metadata for the versions PINNStudio is tested with; check the installed package for the
version you use.

| Package | Required version | Used for | License |
|---|---|---|---|
| [DeepXDE](https://github.com/lululxvi/deepxde) | 1.10.0 or newer | Defining and training the physics-informed neural networks; every script PINNStudio writes runs on it | LGPL-2.1 |
| [PyTorch](https://pytorch.org) | 2.0 or newer | The computational backend (CPU, CUDA) | BSD-3-Clause for PyTorch itself; its package also contains components under Apache-2.0, BSD-2-Clause, BSL-1.0 and MIT |
| [PySide6-Essentials](https://doc.qt.io/qtforpython-6/) (Qt for Python) | 6.5 or newer; below 6.12 on Python 3.11 | The graphical interface | LGPL-3.0-only, GPL-2.0-only or GPL-3.0-only, or a commercial license from The Qt Company; PINNStudio uses it under the LGPL-3.0 |
| Qt 6 libraries | installed with PySide6-Essentials | The Qt libraries PySide6 binds to (PINNStudio uses Qt Core, GUI, Widgets and SVG, and Qt Test in its tests; none of the Addons) | LGPL-3.0-only, GPL-2.0-only or GPL-3.0-only; used under the LGPL-3.0 (no GPL-only Qt module is used) |
| shiboken6 | installed with PySide6-Essentials | PySide6's binding support | LGPL-3.0-only, GPL-2.0-only or GPL-3.0-only; used under the LGPL-3.0 |
| [NumPy](https://numpy.org) | below 2.5; below 2 on Intel Macs | Arrays | BSD-3-Clause |
| [SciPy](https://scipy.org) | any | Interpolation and spatial and special functions, in the window and in the generated scripts | BSD-3-Clause |
| [Matplotlib](https://matplotlib.org) | any | Figures and animations | Matplotlib License (PSF-based, BSD-style) |
| [pandas](https://pandas.pydata.org) | any | Reading and writing tables | BSD-3-Clause |
| NVIDIA CUDA libraries (`nvidia-*` packages) | installed with the CUDA builds of PyTorch | GPU computation | NVIDIA's own license terms (proprietary, redistributable under their terms) |
| [scikit-learn](https://scikit-learn.org), [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) | installed with DeepXDE | Used by DeepXDE; PINNStudio's Bayesian search is its own code | BSD-3-Clause |

Why some versions are restricted:

- **PySide6 below 6.12 on Python 3.11.** PySide6 6.12.0 loses references to `None` and `True`, which ends
  the program on these Python versions (harmless from Python 3.12 on).
- **NumPy below 2.5.** DeepXDE 1.15's 2D `Polygon` and `Triangle` geometries call `np.cross` on two-component
  vectors, which NumPy deprecated in 2.0 and removed in 2.5.
- **NumPy below 2 on Intel Macs.** PyTorch's last Intel-Mac release (2.2.2) was built against NumPy 1 and cannot
  use NumPy 2.


The one-command installers (`get-pinnstudio.sh`, `get-pinnstudio.ps1`) download [uv](https://docs.astral.sh/uv/)
(MIT or Apache-2.0) from Astral, and uv downloads a Python build. Neither is part of PINNStudio.

PINNStudio is released under the MIT License. It uses PySide6, shiboken6 and Qt unchanged, as separately installed
libraries, under the LGPL-3.0; the license texts (LGPL-3.0 and the GPL-3.0 it refers to) are in the repository's
`LICENSES/` folder, and THIRD_PARTY_LICENSES.md lists what a bundled distribution of PINNStudio would have to add.

The scripts PINNStudio writes import DeepXDE, PyTorch, NumPy, SciPy and Matplotlib, not PySide6; these libraries keep
their own licenses when you use or share a script.

## Third-party material included in PINNStudio

| Component | Where | Source | License |
|---|---|---|---|
| Path data of some of the interface's line icons: *log* (Lucide/Feather `terminal`), *fit* (`maximize`), *delete* (`x`), *add* (`plus`), *code* (`code`), *down* (`chevron-down`), and part of *sketch* (Feather `edit-3`) | `pinnstudio/ui/icons.py` | [Lucide](https://lucide.dev), derived for these icons from [Feather](https://feathericons.com) | ISC (Lucide) and MIT (Feather) |

The other icons were drawn for PINNStudio in the same style. The full license texts are in the repository's
[`LICENSES/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/LICENSES) folder
(`Lucide-ISC.txt`, `Feather-MIT.txt`) and in THIRD_PARTY_LICENSES.md; their notices must stay with the copies.

Not included from other projects:

- **DeepXDE source code.** PINNStudio imports DeepXDE as an installed library. Its *DeepXDE examples*
  (`pinnstudio/core/deepxde_examples.py`) are PINNStudio settings that reproduce the problems of DeepXDE's documented
  examples; each links to DeepXDE's original script, and any difference is stated in the example. The multi-scale
  Fourier feature networks follow the method of
  [Wang, Wang & Perdikaris (2021)](references.md#wang2021fourier) and DeepXDE's `MsFFN` interface, and were written for
  PINNStudio.
- **Datasets.** The reference and measured data in `reference_data/` were computed for PINNStudio; none was copied
  from DeepXDE or from the published papers whose problems the examples follow. See
  [Reference data](../reproducibility/reference-data.md).
- **Fonts.** None are bundled; the interface uses the system's fonts.

## This documentation site

The site is built with [MkDocs](https://www.mkdocs.org) (BSD-2-Clause) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) (MIT), with the
[PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/) (MIT). Equations are typeset in the browser
by [MathJax](https://www.mathjax.org) (Apache-2.0), loaded from the jsDelivr CDN. These are needed only to build or view
the documentation, not to run PINNStudio.
