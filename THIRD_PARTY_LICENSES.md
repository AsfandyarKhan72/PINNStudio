# Third-party software and licenses

PINNStudio's own code is licensed under the MIT License; see [`LICENSE`](LICENSE) and the
[License section of the README](README.md#license). The one exception is
[`pinnstudio/core/msffn_lgpl.py`](pinnstudio/core/msffn_lgpl.py), which is under the LGPL-2.1 (group 1 below).
PINNStudio uses other free software, which keeps its own license. This file lists it in two groups:

1. **Included in this repository** (and in the PINNStudio package on PyPI): material from other projects that is
   part of PINNStudio's own files. Its notices are reproduced here and in [`LICENSES/`](LICENSES).
2. **Installed separately**: the libraries PINNStudio runs on. `pip`, `uv` or the PINNStudio installers download
   them from PyPI (and PyTorch's own package index) as separate packages; PINNStudio does not copy their code into
   this repository or into its own package. Each library's license comes with its own package.

PINNStudio is an independent project. It is not affiliated with, sponsored by or endorsed by the developers of
DeepXDE, PyTorch, The Qt Company (Qt, Qt for Python) or the other projects listed here.

## 1. Included in this repository

| Component | Where | Source | License |
|---|---|---|---|
| Icon path data for some of the interface's line icons: *log* (Lucide/Feather `terminal`), *fit* (`maximize`), *delete* (`x`), *add* (`plus`), *code* (`code`), *down* (`chevron-down`), and part of *sketch* (Feather `edit-3`); the other icons were drawn for PINNStudio in the same style | `pinnstudio/ui/icons.py` | [Lucide](https://lucide.dev) ([repository](https://github.com/lucide-icons/lucide)), derived for these icons from [Feather](https://feathericons.com) ([repository](https://github.com/feathericons/feather)) | ISC (Lucide) and MIT (Feather); full texts in [`LICENSES/Lucide-ISC.txt`](LICENSES/Lucide-ISC.txt) and [`LICENSES/Feather-MIT.txt`](LICENSES/Feather-MIT.txt), reproduced below. Their notices must stay with the copies. |
| The multi-scale Fourier feature network class (MsFFN, STMsFFN) that PINNStudio writes into generated scripts using these networks | [`pinnstudio/core/msffn_lgpl.py`](pinnstudio/core/msffn_lgpl.py) (`MSFFN_CODE`), and the generated scripts that contain it | Method: Wang, Wang & Perdikaris (2021), *Comput. Methods Appl. Mech. Eng.* 384, 113938. Implementation: follows the interface and structure of DeepXDE's `MsFFN`/`STMsFFN` ([`deepxde/nn/pytorch/msffn.py`](https://github.com/lululxvi/deepxde/blob/master/deepxde/nn/pytorch/msffn.py), the DeepXDE developers) and may be derived from it (see below) | **LGPL-2.1**, not MIT; full text in [`LICENSES/LGPL-2.1.txt`](LICENSES/LGPL-2.1.txt). The file and every generated script that contains the class carry its notice, which must stay with the class. |

Not included from other projects:

- **DeepXDE's source files are not included in PINNStudio**; PINNStudio imports DeepXDE as a separately installed
  library. The
  *DeepXDE examples* in PINNStudio (`pinnstudio/core/deepxde_examples.py`) are written as PINNStudio settings that
  reproduce the problems of DeepXDE's documented examples (equations, domains, conditions, network sizes, training
  schedules); each one links to DeepXDE's original script, and any difference is stated in the example. The
  multi-scale Fourier feature network class is the exception listed in group 1: it was generated with an AI
  assistant during PINNStudio's development to match DeepXDE's `MsFFN`/`STMsFFN`, and it is not known whether
  DeepXDE's source was consulted. It is therefore treated as possibly derived from DeepXDE's
  `deepxde/nn/pytorch/msffn.py` and distributed under DeepXDE's license, the LGPL-2.1. This is a precaution; it does
  not state that DeepXDE code was copied. It differs from DeepXDE's code in how the fixed Fourier matrices are
  stored (buffers), in the size of the output layer (any number of scales) and in having no dropout.
- **No dataset is copied from DeepXDE's repository or from published papers.** The reference and measured data
  under `reference_data/` were computed for PINNStudio by the scripts in `tools/`: `make_exact_reference_data.py`,
  `make_deepxde_reference_data.py`, `make_phase_field_reference_data.py`, `make_poisson_reference_data.py`
  (Poisson disk, L-shape, sphere), `make_burgers2d_reference_data.py` (Burgers 2D of Mathias et al.),
  `make_heat_reference_data.py` (Heat 2D and 3D), `make_allen_cahn_mattey_reference_data.py` and
  `make_allen_cahn_wight_reference_data.py` (2D Allen-Cahn of Mattey & Ghosh and of Wight & Zhao). Earlier
  versions of eight of these datasets were finite-element exports (four of them documented as computed by
  PINNStudio's author with COMSOL Multiphysics); they were replaced in October 2026, see
  `benchmarks/reference_regeneration/README.md`. The
  spectral solvers use the ETDRK4 scheme as published by Kassam & Trefethen (SIAM J. Sci. Comput. 26, 2005) and
  the Chebyshev differentiation matrix of Trefethen (Spectral Methods in MATLAB, SIAM, 2000); the L-shape solver
  uses the algebraic multigrid of PyAMG (MIT license), an optional dependency of that script only.
- **The application icon** (`pinnstudio/resources/pinnstudio.png`, `.ico`, `.icns`, and the window icon drawn in
  `pinnstudio/ui/icons.py`) was created for PINNStudio, with the assistance of Claude (Anthropic), in October 2026,
  replacing an earlier icon. It is part of PINNStudio and is licensed with it; `tools/make_app_icon.py` renders the
  icon files from its SVG source.
- **No fonts are bundled.** The interface uses the system's fonts.

## 2. Installed separately

These are PINNStudio's dependencies (`setup.py`, `requirements.txt`) and the main packages they bring in. The
licenses are as stated by each package's own metadata for the versions PINNStudio is tested with; check the
installed package for the version you use.

| Package | What PINNStudio uses it for | Project | License |
|---|---|---|---|
| **PySide6-Essentials** (Qt for Python) | the graphical interface | [The Qt Company](https://doc.qt.io/qtforpython-6/) ([source](https://code.qt.io/cgit/pyside/pyside-setup.git/)) | LGPL-3.0-only, GPL-2.0-only or GPL-3.0-only (the user's choice); PINNStudio uses it under the **LGPL-3.0** |
| Qt 6 libraries (inside the PySide6-Essentials package) | the Qt libraries PySide6 binds to; PINNStudio uses Qt Core, Qt GUI, Qt Widgets and Qt SVG (and Qt Test in its tests), all of them Qt Essentials modules available under the LGPL-3.0 | [The Qt Company](https://www.qt.io/licensing/open-source-lgpl-obligations) ([source](https://code.qt.io/cgit/qt/qt5.git/)) | LGPL-3.0-only, GPL-2.0-only or GPL-3.0-only; used under the **LGPL-3.0** |
| **shiboken6** | PySide6's binding support | The Qt Company | LGPL-3.0-only, GPL-2.0-only or GPL-3.0-only; used under the **LGPL-3.0** |
| **DeepXDE** | defining and training the physics-informed neural networks (every script PINNStudio writes runs on it) | [github.com/lululxvi/deepxde](https://github.com/lululxvi/deepxde) | LGPL-2.1 |
| **PyTorch** | the computational backend (CPU, CUDA, Apple MPS) | [pytorch.org](https://pytorch.org), [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD-3-Clause for PyTorch itself; its package also contains components under Apache-2.0, BSD-2-Clause, BSL-1.0 and MIT |
| NVIDIA CUDA libraries (installed with the CUDA builds of PyTorch, `nvidia-*` packages) | GPU computation | [NVIDIA](https://docs.nvidia.com/cuda/eula/) | NVIDIA's own license terms (proprietary, redistributable under their terms) |
| **NumPy** | arrays | [numpy.org](https://numpy.org) | BSD-3-Clause |
| **SciPy** (with DeepXDE) | interpolation and reference-data tools | [scipy.org](https://scipy.org) | BSD-3-Clause |
| **Matplotlib** | the figures and animations | [matplotlib.org](https://matplotlib.org) | Matplotlib License (PSF-based, BSD-style) |
| **pandas** | reading and writing tables | [pandas.pydata.org](https://pandas.pydata.org) | BSD-3-Clause |
| scikit-learn, scikit-optimize (with DeepXDE) | used by DeepXDE; PINNStudio's Bayesian search is its own code | [scikit-learn.org](https://scikit-learn.org), [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) | BSD-3-Clause |

The one-command installers (`get-pinnstudio.sh`, `get-pinnstudio.ps1`) download [uv](https://docs.astral.sh/uv/)
(MIT or Apache-2.0) from Astral, and uv downloads a Python build; neither is part of this repository.

### What these licenses mean for PINNStudio

*This is a summary for orientation, not legal advice.*

- **PySide6, shiboken6 and Qt (LGPL-3.0)** may be used by a program under its own license, here the MIT License,
  as long as the LGPL's conditions are met for the libraries themselves. PINNStudio uses them unchanged, as
  separately installed packages that `import PySide6` loads as shared libraries, so a user can replace them with
  another build or version; PINNStudio contains no Qt or PySide6 code. The LGPL-3.0 and the GPL-3.0 it builds on
  are reproduced in [`LICENSES/LGPL-3.0.txt`](LICENSES/LGPL-3.0.txt) and
  [`LICENSES/GPL-3.0.txt`](LICENSES/GPL-3.0.txt). The modules PINNStudio uses are all available under the LGPL;
  no GPL-only Qt module (for example Qt Charts or Qt Data Visualization) is used.
- **DeepXDE (LGPL-2.1)** is used the same way: unchanged, as a separately installed package, and attributed
  (README, *Acknowledgment* and *References*).
- **The multi-scale Fourier feature network class (LGPL-2.1)** is kept in its own file,
  `pinnstudio/core/msffn_lgpl.py`, under the LGPL-2.1, with a notice that names the method, DeepXDE's implementation
  and PINNStudio's changes. It is kept separate so that the rest of PINNStudio's code can stay under the MIT
  License: `codegen.py` only reads the class's source text and writes it, unchanged and with its notice, into the
  scripts that use MsFFN or STMsFFN.
- **The BSD, MIT, Apache-2.0 and PSF-style licenses** of the other packages ask for their notices to be kept with
  copies of those packages, which their own packages do.
- **The scripts PINNStudio writes** import DeepXDE, PyTorch, NumPy and Matplotlib, not PySide6. They contain code
  from PINNStudio's own templates, which PINNStudio's copyright holder allows to be used under any terms (README,
  *License*); the libraries they import keep their own licenses. A script that uses an MsFFN or STMsFFN network
  also contains the multi-scale Fourier feature network class, marked by an LGPL-2.1 notice at its start and an end
  line. That class stays under the LGPL-2.1: keep its notice, and give the LGPL-2.1 with the script when you share
  it. Scripts with other networks contain no LGPL code.

### If PINNStudio is ever distributed as a bundled application

PINNStudio is distributed today as source code and as a Python package (PyPI) that contains no third-party
libraries; every dependency is installed separately. A self-contained bundle (for example a PyInstaller build or an
installer that ships Python with the libraries) would redistribute the libraries above, which brings obligations
that do not arise today:

- include the license texts and copyright notices of every bundled library (the LGPL-3.0 with the GPL-3.0 text it
  refers to, the LGPL-2.1, the BSD/MIT/Apache notices, PyTorch's `NOTICE`, NVIDIA's license for the CUDA
  libraries), and state that PySide6, shiboken6 and Qt are used under the LGPL-3.0;
- give the Qt and PySide6 libraries' corresponding source code, or a written offer valid for three years, or a
  link to the exact source release used (for example `code.qt.io`, tag `v<version>`), and the same for DeepXDE;
- keep the LGPL libraries replaceable by the user: ship them as separate, dynamically loaded files (as
  PyInstaller does by default), unmodified, and say in the bundle's notice how to replace them;
- not restrict reverse engineering for debugging such modifications (LGPL-3.0 section 4) in the bundle's terms;
- check NVIDIA's terms for redistributing the CUDA libraries at the time.

A notice template for such a bundle is in `notes/licensing-proposal/drafts/LGPL-NOTICE.txt`.

---

## License texts of the material in group 1

### Multi-scale Fourier feature networks (LGPL-2.1)

The GNU Lesser General Public License version 2.1 is in [`LICENSES/LGPL-2.1.txt`](LICENSES/LGPL-2.1.txt). It
applies to `pinnstudio/core/msffn_lgpl.py` and to the class it writes into generated scripts.

### Lucide (ISC)

```text
ISC License

Copyright (c) 2026 Lucide Icons and Contributors

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```

Lucide's license adds that the Lucide icons derived from Feather (among them `terminal`, `maximize`, `x`, `plus`,
`code` and `chevron-down`) are under Feather's MIT license, below. The complete file is
[`LICENSES/Lucide-ISC.txt`](LICENSES/Lucide-ISC.txt).

### Feather (MIT)

```text
The MIT License (MIT)

Copyright (c) 2013-2023 Cole Bemis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
