# Licensing proposal for PINNStudio v1.6.0 with PySide6 (draft, not adopted)

**Status: proposal only.** Nothing in this folder changes PINNStudio's license. `LICENSE`, the README's License
section and `THIRD_PARTY_LICENSES.md` are unchanged on this branch. Who may choose PINNStudio's license (ownership
and distribution authorization at Boise State University) has to be settled first; that question is not answered
here, and Boise State's silence is not taken as permission. This is a technical reading of the licenses and of the
licensors' own documentation, not legal advice.

## 1. What the switch to PySide6 changes

| | PyQt6 (current release candidate) | PySide6 (`experiment/pyside6-migration`) |
|---|---|---|
| Binding license | GPL v3 or Riverbank commercial | LGPLv3 / GPLv2 / GPLv3 or Qt commercial |
| Qt libraries used | Qt Core, Gui, Widgets, Svg (LGPLv3) | the same |
| Package PINNStudio depends on | `PyQt6` | `PySide6-Essentials` (not the `PySide6` meta-package, which also installs `PySide6-Addons`) |
| What forces the GPL on the combined program | PyQt6 itself | nothing, as long as only LGPL Qt modules are used |

With PyQt6 the program as distributed is GPL v3 whatever PINNStudio's own files are licensed under. With
PySide6 the only copyleft components are LGPL libraries (PySide6, shiboken6, Qt, and DeepXDE under LGPL-2.1), and
the LGPL allows the application that uses them to be under terms of its own choice.

### Checked: no GPL-only Qt component is needed

- PINNStudio imports `PySide6.QtCore`, `QtGui`, `QtWidgets`, `QtSvg` (and `QtTest` in the tests only); no other
  Qt module (inventory of `pinnstudio/**/*.py`, `tests/`, `tools/`).
- Qt's licensing page lists the modules that are **not** available under the LGPLv3 in the open-source edition:
  Qt Canvas Painter, Qt CoAP, Qt Graphs, Qt GRPC, Qt HTTP Server, Qt Lottie Animation, Qt MQTT, Qt Network
  Authorization, Qt Qml Compiler, Qt Quick 3D, Qt Quick 3D Physics, Qt Quick Timeline, Qt Virtual Keyboard and
  Qt Wayland Compositor. None of them is used. (Some of these ship in `PySide6-Addons`, which is why the
  dependency is `PySide6-Essentials`.)
- The installed wheels declare `License: LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only` (PySide6-Essentials
  6.12.0, shiboken6 6.12.0).
- A clean virtual environment with only `PySide6-Essentials` (no Addons) installs PINNStudio, starts it and runs
  the test suite (see the release-readiness report).

## 2. MIT or Apache-2.0 for PINNStudio's own files

Both are permissive, both are compatible with the LGPLv3/GPLv3 of the libraries PINNStudio uses (Apache-2.0 is not
compatible with GPLv2-only code; no dependency is GPLv2-only), and both let anyone, including companies, reuse
PINNStudio's code. They differ in the following:

| | MIT (releases up to 1.5.0) | Apache-2.0 |
|---|---|---|
| Length and familiarity | One paragraph; the most common license in scientific Python | ~10 pages; common in larger projects and companies |
| Patent grant from contributors | None stated (implied at most) | Explicit patent license from every contributor (§3) |
| Patent retaliation | None | A user who sues over a patent in the work loses the patent license (§3) |
| Notices | Keep the copyright and permission notice | Keep notices; modified files must say they were changed (§4b); a `NOTICE` file, if present, travels with redistributions (§4d) |
| Trademarks | Not mentioned | No trademark rights granted (§6), which protects the name "PINNStudio" |
| Contributions | Not addressed | Contributions are under the same license unless stated otherwise (§5) |
| Continuity | Same as 1.0–1.5.0; nothing to explain to users | A change of license between 1.5.0 and 1.6.0 to describe in the changelog and README |
| Compatibility with GPLv2-only code | Yes | No (irrelevant today) |

**Suggested choice: Apache-2.0**, for the patent grant and contribution clause (useful if others contribute or a
company builds on PINNStudio) and the trademark clause (protects the name). It suits the stated preference for
long-term flexibility, and costs little: a `NOTICE` file and short per-file headers, which are optional. **MIT
remains a sound choice** if continuity with 1.0–1.5.0 and the shortest possible license matter more; the practical
difference for academic users is small. Either way, earlier releases keep the license they were published under.

Before either is adopted, the copyright holder must be confirmed (the author and/or Boise State University, per
the university's intellectual-property policy and any funding or program terms). The drafts
below leave the holder as a placeholder.

## 3. LGPL obligations and how PINNStudio meets them

PINNStudio is distributed as a Python package on PyPI and from this repository. It does **not** contain or
redistribute Qt or PySide6: `pip`/`uv` install `PySide6-Essentials` and `shiboken6` from PyPI as separate packages,
published by The Qt Company with their licenses and source. Under the LGPLv3 the obligations then are:

| Obligation (LGPLv3 §4, Qt's "Obligations of the GPL and LGPL" page) | How it is met |
|---|---|
| Prominent notice that the library is used and is under the LGPL; copies of the LGPLv3 and GPLv3 texts | `THIRD_PARTY_LICENSES.md` section for PySide6/shiboken6/Qt (draft below) and the README; the license texts are installed with the PySide6 wheels and linked; for a bundled build they are copied in (`drafts/LGPL-NOTICE.txt`) |
| Qt's copyright notice among the notices the program displays | **About** dialog lists PySide6/Qt with their license (to add in the integration step) |
| The user can replace the library with a modified version | Qt and PySide6 are separately installed shared libraries; a user can install their own build in the environment. No static linking, no modification of Qt/PySide6 |
| Source of the library | PySide6 and Qt source are published by the Qt Project (code.qt.io); the notice points there. PINNStudio distributes no Qt binaries, so it has no source to supply itself |
| No terms that restrict the LGPL rights (no ban on reverse engineering for debugging the modified library) | Neither MIT nor Apache-2.0 contains such terms |

**If a bundled installer is ever made** (PyInstaller, Nuitka, briefcase, a conda constructor): keep Qt and
PySide6 as separate shared libraries (no static linking), include the LGPLv3 and GPLv3 texts and Qt's notices in
the bundle, include or offer the corresponding source of the exact Qt/PySide6 version shipped (a written offer or
a link to the tagged source), and do not encrypt or sign the bundle in a way that prevents replacing those
libraries. The current installers (`install.sh`, `install.bat`, `get-pinnstudio.*`) create an environment and
install packages from PyPI; they do not bundle Qt, so these extra conditions do not apply to them.

DeepXDE (LGPL-2.1) is used the same way: imported as a separately installed package, not copied.

## 4. Files in `drafts/`

- `LICENSE.Apache-2.0.header.txt`: what `LICENSE` would contain under Apache-2.0 (the official text, to be copied
  verbatim from https://www.apache.org/licenses/LICENSE-2.0.txt at adoption; not reproduced here, to avoid an
  inexact copy), plus the copyright line.
- `NOTICE`: the Apache-2.0 `NOTICE` file.
- `LICENSE.MIT`: the MIT alternative.
- `THIRD_PARTY_PySide6.md`: the PySide6/shiboken6/Qt section for `THIRD_PARTY_LICENSES.md`, replacing the PyQt6
  entry.
- `LGPL-NOTICE.txt`: the notice for a bundled distribution.
- `README-license-section.md`: the README License section for each choice.

## 5. Sources

- Qt, "Licensing" (Qt 6 documentation), list of modules not available under the LGPLv3:
  https://doc.qt.io/qt-6/licensing.html
- Qt, "Obligations of the GPL and LGPL": https://www.qt.io/development/open-source-lgpl-obligations
- Qt for Python, "Commercial Use" (licensing of Qt for Python): https://doc.qt.io/qtforpython-6/commercial/index.html
- PySide6-Essentials / shiboken6 6.12.0 package metadata (`pip show -v`).
- Apache Software Foundation, "GPL compatibility": https://www.apache.org/licenses/GPL-compatibility
- GNU, "Various Licenses and Comments about Them": https://www.gnu.org/licenses/license-list.html
- GNU Lesser General Public License v3: https://www.gnu.org/licenses/lgpl-3.0.html
- Riverbank Computing, PyQt License FAQ (for the PyQt6 comparison).
