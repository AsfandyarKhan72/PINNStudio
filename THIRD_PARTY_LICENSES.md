# Third-party software and licenses

PINNStudio's own code is licensed under the GNU General Public License version 3 only (GPL-3.0-only); see
[`LICENSE`](LICENSE) and the [License section of the README](README.md#license). PINNStudio uses other free
software, which keeps its own license. This file lists it in two groups:

1. **Included in this repository** (and in the PINNStudio package on PyPI): material from other projects that is
   part of PINNStudio's own files. Its notices are reproduced here and in [`LICENSES/`](LICENSES).
2. **Installed separately**: the libraries PINNStudio runs on. `pip`, `uv` or the PINNStudio installers download
   them from PyPI (and PyTorch's own package index) as separate packages; PINNStudio does not copy their code into
   this repository or into its own package. Each library's license comes with its own package.

PINNStudio is an independent project. It is not affiliated with, sponsored by or endorsed by the developers of
DeepXDE, PyTorch, Riverbank Computing (PyQt), The Qt Company or the other projects listed here.

## 1. Included in this repository

| Component | Where | Source | License |
|---|---|---|---|
| Icon path data for some of the interface's line icons: *log* (Lucide/Feather `terminal`), *fit* (`maximize`), *delete* (`x`), *add* (`plus`), *code* (`code`), *down* (`chevron-down`), and part of *sketch* (Feather `edit-3`); the other icons were drawn for PINNStudio in the same style | `pinnstudio/ui/icons.py` | [Lucide](https://lucide.dev) ([repository](https://github.com/lucide-icons/lucide)), derived for these icons from [Feather](https://feathericons.com) ([repository](https://github.com/feathericons/feather)) | ISC (Lucide) and MIT (Feather); full texts in [`LICENSES/Lucide-ISC.txt`](LICENSES/Lucide-ISC.txt) and [`LICENSES/Feather-MIT.txt`](LICENSES/Feather-MIT.txt), reproduced below. Both are compatible with the GPL; their notices must stay with the copies. |

Not included from other projects:

- **No DeepXDE source code is copied into PINNStudio.** PINNStudio imports DeepXDE as an installed library. The
  *DeepXDE examples* in PINNStudio (`pinnstudio/core/deepxde_examples.py`) are written as PINNStudio settings that
  reproduce the problems of DeepXDE's documented examples (equations, domains, conditions, network sizes, training
  schedules); each one links to DeepXDE's original script, and any difference is stated in the example. The
  multi-scale Fourier feature networks in `pinnstudio/core/codegen.py` follow the method of Wang, Wang & Perdikaris
  (2021) and DeepXDE's `MsFFN` interface, and were written for PINNStudio.
- **No dataset is copied from DeepXDE's repository.** The reference and measured data under `reference_data/` were
  computed for PINNStudio: by the scripts in `tools/` (`make_exact_reference_data.py`,
  `make_deepxde_reference_data.py`, `make_phase_field_reference_data.py`), except the data of four templates
  (`reference_data/2D/burgers_mathias`, `2D/poisson_disk`, `2D/poisson_lshape`, `3D/poisson_sphere`). These four
  were computed by PINNStudio's author with COMSOL Multiphysics on a computer of the Computational Materials Design
  Lab at Boise State University; they were not copied from DeepXDE or from the published papers whose problems the
  templates follow.
- **The application icon** (`pinnstudio/resources/pinnstudio.png`, `.ico`, `.icns`, and the window icon drawn in
  `pinnstudio/ui/icons.py`) was created for PINNStudio, with the assistance of Claude (Anthropic), in October 2026,
  replacing an earlier icon. It is part of PINNStudio and is licensed with it; `tools/make_app_icon.py` renders the
  icon files from its SVG source.
- **No fonts are bundled.** The interface uses the system's fonts. The project page (`docs/index.html`) loads its
  web fonts from Google Fonts at viewing time.

## 2. Installed separately

These are PINNStudio's dependencies (`setup.py`, `requirements.txt`) and the main packages they bring in. The
licenses are as stated by each package's own metadata for the versions PINNStudio is tested with; check the
installed package for the version you use.

| Package | What PINNStudio uses it for | Project | License |
|---|---|---|---|
| **PyQt6** | the graphical interface | [Riverbank Computing](https://www.riverbankcomputing.com/software/pyqt/) | GPL-3.0-only, or a commercial license from Riverbank |
| **PyQt6-Qt6** (Qt 6 libraries, installed with PyQt6) | the Qt libraries PyQt6 binds to | [The Qt Company](https://www.qt.io/licensing/) | LGPL-3.0 (some Qt modules are GPL-3.0; PINNStudio uses Qt Core, GUI, Widgets and SVG) |
| **PyQt6-sip** | PyQt6's binding support | Riverbank Computing | BSD-2-Clause |
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

- **PyQt6 (GPL-3.0)** is the reason PINNStudio is distributed under the GPL version 3: Riverbank Computing licenses
  PyQt6 under the GPL version 3 or a commercial license, and a program built on the GPL version is distributed under
  GPL-compatible terms. Anyone who distributes PINNStudio, changed or not, does so under the GPL version 3, with its
  corresponding source code.
- **DeepXDE (LGPL-2.1)** is a library that programs may use and be distributed with under their own terms, subject
  to the conditions of the LGPL. PINNStudio uses DeepXDE unchanged, as a separately installed package, and attributes
  it (README, *Acknowledgment* and *References*). The LGPL-2.1 is compatible with the GPL version 3.
- **Qt (LGPL-3.0)** and the BSD, MIT, Apache-2.0 and PSF-style licenses of the other packages are compatible with
  the GPL version 3.
- **The scripts PINNStudio writes** import DeepXDE, PyTorch, NumPy and Matplotlib, not PyQt6. They contain code
  from PINNStudio's own templates, which PINNStudio's copyright holders allow to be used under any terms (the
  additional permission in the README's License section); the libraries they import keep their own licenses.

### If PINNStudio is ever distributed as a bundled application

PINNStudio is distributed today as source code and as a Python package (PyPI), with every dependency installed
separately. A self-contained bundle (for example a PyInstaller build or an installer that ships Python with the
libraries) would redistribute the libraries above, which brings obligations that do not arise today:

- include the full license text and copyright notices of every bundled library (GPL-3.0, LGPL-3.0 and LGPL-2.1
  texts, the BSD/MIT/Apache notices, PyTorch's `NOTICE`, NVIDIA's license for the CUDA libraries);
- provide, or offer in writing, the complete corresponding source code of PINNStudio and of the GPL/LGPL libraries
  in the bundle (PyQt6, Qt, DeepXDE);
- keep the LGPL libraries (Qt, DeepXDE) replaceable by the user, e.g. as separate, dynamically loaded files, as
  PyInstaller does by default;
- check whether NVIDIA's CUDA libraries may be bundled together with a GPL program under the terms in force at the
  time (the GPL's system-library exception does not obviously cover libraries shipped inside the bundle).

---

## License texts of the material in group 1

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
