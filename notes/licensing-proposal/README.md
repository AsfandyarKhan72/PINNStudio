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

**Stated preference (2026-11-01): MIT**, for continuity with the published 1.0-1.5.0 releases, provided the final
dependency audit and Boise State's authorization support it. The audit supports it: with PySide6-Essentials only
LGPL components are combined with PINNStudio's files (section 1), and MIT is compatible with them. What MIT does not
give, compared with Apache-2.0: an explicit contributor patent grant, a contribution clause and a trademark
clause; for the name "PINNStudio" a short trademark sentence in the README can cover the last point. Apache-2.0
remains the alternative and its drafts are kept. Either way, earlier releases keep the license they were
published under.

### Where the branches stand today (checked 2026-11-02)

| Ref | `LICENSE` | `setup.py` license / classifier | GUI binding |
|---|---|---|---|
| tag `v1.5.0`, `main`, PyPI 1.5.0 | MIT | `MIT` (no classifier) | PyQt6 |
| `audit/v1.6.0` | GPL-3.0 text | `GPL-3.0-only`, GPLv3 classifier | PyQt6 |
| `experiment/pyside6-migration` | GPL-3.0 text (inherited from the audit branch) | `GPL-3.0-only`, GPLv3 classifier | PySide6 |

The GPL-3.0-only files came from `licensing/gplv3-compliance` because PyQt6 makes the distributed program GPL.
They are consistent with the PyQt6 branch and **inconsistent with a permissive PySide6 release**; the PySide6
branch keeps them unchanged only so that no license change is made before approval. Releasing PySide6 with the
GPL files would also be valid (stricter than needed). Mixing the two, a permissive `LICENSE` with PyQt6 or GPL
metadata with an MIT `LICENSE`, is what must not happen.

### Exact changes at adoption of MIT for the PySide6 release (each needs the rights holder's approval)

| File | Change |
|---|---|
| `LICENSE` | replace the GPL-3.0 text with `drafts/LICENSE.MIT` (copyright holder confirmed first) |
| `setup.py` | `license="MIT"`; classifier `License :: OSI Approved :: MIT License` instead of GPLv3; `license_files` keeps `THIRD_PARTY_LICENSES.md` and `LICENSES/*` |
| `THIRD_PARTY_LICENSES.md` | replace the PyQt6 entry with `drafts/THIRD_PARTY_PySide6.md` (PySide6, shiboken6, Qt under LGPLv3) |
| `LICENSES/` | add `LGPL-3.0.txt` and `GPL-3.0.txt` texts (LGPLv3 is a set of additional permissions on the GPLv3) |
| `README.md` License section | `drafts/README-license-section.md` (MIT variant) |
| `pinnstudio/ui` About dialog | list PySide6/Qt with their license and a link to the source (Qt's notice requirement) |
| `CHANGELOG.md` | one entry: binding PyQt6 -> PySide6, license of 1.6.0, earlier releases unchanged |
| `CONTRIBUTING.md`, bug-report template | "PyQt6" -> "PySide6" where they name the binding (not a license change; no approval needed) |
| docs site (`docs/` license and install pages) | the same wording as the README |
| `CITATION.cff` | `license: MIT` and its comment line (now: GPL-3.0-only) |

Not changed: the `v1.5.0` tag and every earlier tag, and the PyQt6 fallback branch (which stays GPL-consistent).

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

## 6. Final comparison for the v1.6.0 decision (checked on `integration/v1.6.0-final`, 2026-11-02)

**Component check (unchanged by the integration):** PINNStudio imports `PySide6.QtCore` (46 places), `QtWidgets`
(166), `QtGui` (16), `QtSvg` (2) and, in the tests only, `QtTest` (5); no other Qt module and nothing from
`PySide6-Addons`. The wheel depends on `PySide6-Essentials` only: below 6.12 on Python 3.11 (6.11.3 is what the
resolver installs there), 6.12.0 from Python 3.12 on (checked with `uv pip compile` for 3.11 to 3.14). Both are
offered under LGPL-3.0-only (or GPL-2.0/3.0). PINNStudio does not bundle, modify or statically link Qt or PySide6.

| | Keep MIT (stated preference) | Move to Apache-2.0 |
|---|---|---|
| Compatible with PySide6-Essentials (LGPLv3), Qt (LGPLv3), DeepXDE (LGPL-2.1), PyTorch (BSD) | yes | yes |
| Continuity with 1.0-1.5.0 (all published MIT, PyPI 1.5.0 metadata `License: MIT`) | same license, nothing to explain | a license change to announce |
| Patent grant / retaliation, contribution clause, trademark clause | none | yes (sections 3, 5, 6) |
| Extra files | none beyond `LICENSE` | `NOTICE`; changed-file notices on redistribution |
| Generated scripts (the GPL section-7 permission now in the README) | not needed: MIT already lets users do anything with generated code | not needed either |
| What the LGPL asks of PINNStudio in both cases | notice that PySide6/Qt are used and are LGPLv3, the license texts or links, Qt's copyright notice reachable from the program (About), no terms that forbid replacing or reverse-engineering the libraries, and for any future bundled installer: shared libraries, license texts and the corresponding source or a written offer | same |

**Metadata at adoption (either license):** `license="MIT"` (or `"Apache-2.0"`) as an SPDX expression and **no**
`License ::` classifier (setuptools now warns that license classifiers are deprecated; the current build prints
that warning for the GPLv3 classifier); `license_files` keeps `LICENSE`, `THIRD_PARTY_LICENSES.md` and
`LICENSES/*`; the version becomes `1.6.0`.

**Needs the rights holder's approval (Boise State ownership and authorization), nothing done yet:**
the license itself; `LICENSE`; `setup.py` license and classifier; `CITATION.cff`; the README and website License
sections (including the GPL section-7 paragraph, which only exists because of the GPL); `THIRD_PARTY_LICENSES.md`
(PyQt6 entry -> PySide6/shiboken6/Qt, factual but part of the license set); the About dialog notice; the copyright
line. **Does not need approval:** the technical PySide6 wording already changed in the README, CONTRIBUTING and
the website (done on the candidate), and the dependency facts in `website/about/dependencies.md`.

**State of the candidate today:** consistent but not final: GPL-3.0-only files and metadata (valid with
PySide6, stricter than needed), PySide6 dependency, README License section still giving PyQt6's GPL as the
reason. Before a release either the GPL text must say why it is kept, or the approved permissive license replaces it.
