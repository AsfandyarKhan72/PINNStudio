# PINNStudio v1.6.0 with PySide6 — release-readiness report

Branch `experiment/pyside6-migration` (not merged). Baseline for every comparison: `audit/v1.6.0` (PyQt6 6.11.0,
Qt 6.11.0). Machine: Linux x86_64, Python 3.13, 2 CPU cores; offscreen and real X server (Xvfb + xcb).

## 1. The memory-corruption crash: root cause and resolution

### Symptom
Aborts (`malloc_consolidate(): unaligned fastbin chunk detected`, `free(): invalid pointer`) or segmentation
faults while Qt processes events after an example is loaded. Never in PyQt6.

### How often (identical stress loop: load Heat 3D ↔ Allen–Cahn 1D, visit every page, paint the views)

| Build | Rounds | Crashes |
|---|---|---|
| PyQt6 6.11.0 (audited) | 150 | **0** |
| PySide6 6.12.0, before the fix | 150 | **12** (8 %) |
| PySide6 6.11.3 (Qt 6.11, same Qt as PyQt6), before the fix | 40 | **4** |
| PySide6 6.12.0, load examples only (no page switching) | 93 | 0 |
| PySide6 6.12.0, switch pages only, no painting | 42 | 3 |
| PySide6 6.12.0 with `align_forms` disabled | 100 | **0** |
| PySide6 6.12.0, `align_forms` read-only walk (no layout changes) | 81 | 6 |
| PySide6 6.12.0 + fix A (`align_forms` without `itemAt`) — full loop | 150 | **0** |
| PySide6 6.12.0 + fix A — page-switch loop | 150 | **0** |
| PySide6 6.12.0 + final fix (no `itemAt`/`takeAt` anywhere) — full loop | 150 | **0** |
| PySide6 6.12.0 + final fix — all 9 walkthrough examples, every page and view (270 loads) | 30 | **0** |

Ruled out on the way (each with a test): QThread lifetimes (no thread runs in the loop), the application-wide
event filter, nested `processEvents()`, QImage buffers, tree-item ownership, QPainter lifetimes, deleted
Qt-internal widgets, `takeAt` ownership of spacers, Qt 6.12 itself (6.11.3 also crashes).

### Root cause
PySide6 keeps the Python wrapper of every `QLayoutItem` returned by `QLayout.itemAt()` alive, tied to the
layout's wrapper (PyQt6 keeps none). Qt deletes such items itself — when a widget in a layout, or the widget that
owns a layout, is destroyed — and, `QLayoutItem` not being a `QObject`, PySide6 is not told. PINNStudio's
`workspace.align_forms` walked every row of a page with `itemAt()` on every page switch, and each example load
deletes and rebuilds hundreds of rows: after a few loads ~1 200 such wrappers were alive (PyQt6: 0). When Qt then
deletes the widgets, shiboken's own bookkeeping for those wrappers is corrupted:

**valgrind (standalone reproducer, no PINNStudio code):** 2 046 errors, among them

```
Invalid write of size 1 / size 8
   at Shiboken::Object::destroy(SbkObject*, void*)              (libshiboken6)
   by PySide6.QtWidgets wrapper destructor
   by QObjectPrivate::deleteChildren()  ← QWidget::~QWidget()  ← QGroupBox deferred delete
 Address ... is 0 bytes inside a block of size 32 free'd
   at operator delete
   by ... libshiboken6
   by Shiboken::Object::destroy(SbkObject*, void*)              ← freed in the same call
```

i.e. a use-after-free *inside shiboken* during widget destruction. glibc notices the damaged heap later, in an
unrelated allocation (text shaping, painter state), which is why every crash trace pointed elsewhere.

**Standalone reproducer** (`pyside_layoutitem_repro.py`, ~150 lines, PySide6 only):

| Variant | Result |
|---|---|
| `itemAt()` walk + rows deleted with `deleteLater` | crashes (3 of 3 runs, < 400 iterations) |
| `itemAt()` only in the walk | crashes (2 of 2) |
| `itemAt()` only to find the rows to delete | crashes (3 of 3) |
| no `itemAt()` at all | 3 × 400 iterations, no crash; valgrind: **0 errors** (vs 2 046 with `itemAt`) |
| same script on PyQt6 | 3 × 400 iterations, no crash |

This matches the stale-wrapper / address-reuse family reported on the Qt forum (PySide6 6.10.2 and 6.11.1,
`QTableWidgetItem`). A draft upstream report is ready (`UPSTREAM_REPORT.md`, not filed).

### Correction (on the branch)
1. `workspace.align_forms` finds rows, labels and controls through the QObject tree (`findChildren(QHBoxLayout)`,
   `QLayout.indexOf(widget)`) and never calls `itemAt`; the stretch that used to be taken out of a row is kept
   with stretch factor 0 (no deletion through Python).
2. Every other `itemAt`/`takeAt` in the interface (11 places) uses the same helpers (`layout_widgets`,
   `layout_sublayouts`); the Results figure list keeps one trailing stretch instead of removing it.
3. Tests: `tests/test_layout_item_wrappers.py` (live layout-item wrappers after switching examples and pages:
   843 before → 18 after, fails on the old code; `align_forms` idempotent and aligned; no `itemAt`/`takeAt` in
   `pinnstudio/ui`), `tests/test_gui_stress.py` (opt-in 60-round stress run in a child process),
   `tools/gui_stress.py`.

Behaviour preserved: all 113 model-tree pages of 7 examples render pixel-identical before and after (the only
pixel differences are temporary folder names in path fields).

## 2. Other fixes made during the investigation (also on `audit/v1.6.0`, so PyQt6 has them)
- Restore & Visualize with a model of a different problem than the one in the window failed in matplotlib; the
  window's Error Analysis is now skipped (with a log note) when it does not fit the model. Test added.
- Background QThreads are kept alive until their thread has ended (replacing one still finishing could abort Qt in
  either binding). Subprocess regression test added.
- Portable-paths test: PyQt6 segfaulted in pytest's final garbage collection; windows are now deleted explicitly.

## 3. Migration completeness
- No PyQt6 import, `pyqtSignal`, `sip` left in `pinnstudio/`, `tests/`, `tools/`, `benchmarks/` (only docs/CI text,
  listed under integration steps).
- Binding differences found and handled: uncaught slot exceptions (PyQt6 aborts, PySide6 continues — `errors.py`
  keeps the dialog/log), `takeAt` ownership, `itemAt` wrapper retention (the crash).
- Checked without change needed: enums (`.value`, flags), `QMessageBox` return values, `disconnect` behaviour,
  `QTimer.singleShot`, signals across threads, QSettings (values written by either binding read identically by the
  other), Matplotlib QtAgg backend (picks PySide6), `shiboken6.isValid` on Qt-internal widgets.

## 4. Test results (final PySide6 code)

| Check | PySide6 (final) | PyQt6 (audited) |
|---|---|---|
| Full test suite, file by file | 136 files: 383 passed, 6 skipped, 0 failed | running (earlier today: 377 passed, 0 failed) |
| Crash stress loop (Heat 3D ↔ Allen–Cahn 1D, every page and view) | 150 rounds, 0 crashes | 150 rounds, 0 crashes |
| All 9 walkthrough examples, every page and view, 30 rounds (270 loads) | 0 crashes; widgets constant (1 978); RSS 298 → 380 MB | 0 crashes; widgets constant; RSS 256 → 320 MB |
| 40 separate launches (load an example, open pages, close normally) | 40 clean exits | 40 clean exits |
| Full GUI walkthrough on a real X server (Xvfb, xcb) | 91 passed / 9 failed | 91 passed / 9 failed (the same 9, all pre-existing: 5 run-folder names that are new on every save by design, AC1D Default not time-adaptive, 3 Restore failures fixed since) |
| Screens at 1× and 2× scaling | pixel-identical to PyQt6 except a temporary path | — |
| All 113 pages of 7 examples, before vs after the fix | pixel-identical (paths aside) | — |
| Live layout-item wrappers after switching examples/pages | 18 (before the fix 843–1 200) | 0 |
| Clean install with only PySide6-Essentials | installs, `pinnstudio` starts on X; Qt modules loaded: Core, Gui, Widgets, Svg, DBus, XcbQpa | — |

## 5. Numerical equivalence (PyQt6 audited baseline vs PySide6, same seeds, CPU)
- 600 generated files (Solve script, exported script, configuration for 100 examples × Default/Tuned): identical.
- Training from the window (Burgers 1D, Diffusion 1D inverse, Poisson 2D disk, Allen–Cahn 1D time-adaptive with
  4 windows): loss histories, **model weights bit-identical**, Error Analysis files identical, inferred parameter
  identical (C = 1.00144), `model_config.json` identical.
- Saved problems written by PyQt6 reopen identically in PySide6 (5 files, every setting).

## 6. Packaging and platforms
- Wheel requires `PySide6-Essentials>=6.5`; a clean environment with only Essentials (no Addons) installs and starts
  the `pinnstudio` command on a real X server; only Qt Core/Gui/Widgets/Svg/DBus/XcbQpa are loaded (all LGPLv3).
- Linux: offscreen and Xvfb/xcb at 1× and 2× scaling: window and pages pixel-identical to PyQt6.
- **Not tested:** Windows, macOS, a physical high-DPI monitor, Wayland. (PyTorch's own package index is not
  reachable from this machine, so the clean install linked the system PyTorch.)

## 7. Licensing
See `licensing-proposal/README.md` on the branch: MIT vs Apache-2.0 compared (Apache-2.0 suggested for its patent
and trademark clauses; MIT remains sound), LGPLv3 obligations and how a PyPI distribution meets them, draft
`NOTICE`, LGPL notice for bundled installers, third-party entry. No GPL-only Qt module is used. The public license
is unchanged; ownership/authorization at Boise State must be settled first.

## 8. Remaining defects

**Release blockers (for a PySide6 v1.6.0)**
1. Windows and macOS are untested with PySide6 (installers, Qt DLL loading on Windows, the PyQt6-era Windows pin,
   Apple Silicon and Intel Macs, real high-DPI screens). The CI workflows on the branch now install PySide6; the
   *Platform matrix* workflow can be run by hand on `experiment/pyside6-migration` (13 OS/Python combinations,
   tests + window check + installers). It could not be started from here (no GitHub API access).
2. The licence decision (MIT or Apache-2.0, ownership/authorization at Boise State) and the matching updates to
   `LICENSE`, `setup.py` metadata (`license="GPL-3.0-only"`, `version="1.5.0"`), README and THIRD_PARTY_LICENSES.

**Important defects (both bindings unless noted)**
- Initial-condition rows are not deleted when an example is loaded: about 8 hidden widgets per load stay alive
  (33 after 4 loads, 1 in use). Pre-existing in PyQt6; not a stability problem in the stress tests.
- Allen–Cahn 1D (Wight & Zhao) "(time-adaptive)" loads with time windows off under *Default* settings (on under
  *Tuned*).
- Known from the audit: ODE problems labelled "1D" in run folders and script headers; Error Analysis summary has
  no output column; status bar silent during L-BFGS.
- PySide6 only: memory grows ~2.7 MB per round of loading all 9 walkthrough examples (PyQt6 ~2.1 MB); widget count
  stays constant, so this is caches, not leaked widgets.

**Optional improvements**
- PySide6 only: closing the window leaves its Python object and 18 never-shown settings controls alive until the
  program exits (one window per process, so ~64 widgets once).
- File the upstream PySide6 report (draft ready).
- Window construction ~20-50 % slower under PySide6 (training unaffected).
- Duplicate-key risk for BC pages keyed by `id()` (no collision observed in the stress loops).

## 9. Integration steps (none done; each needs your approval)
1. Run *Platform matrix* by hand on `experiment/pyside6-migration`; fix what Windows/macOS show; test the Windows
   installer and a real high-DPI display.
2. Wait for GPU rerun 5, finish the benchmark corrections on `audit/v1.6.0`, merge `audit/v1.6.0` into the PySide6
   branch again, and repeat: script equivalence (600 files), the matched training runs, the stress loops.
3. Settle the licence; then update `LICENSE`, `NOTICE` (Apache), `setup.py` (licence, version 1.6.0),
   README/CONTRIBUTING/CHANGELOG/bug-report template/THIRD_PARTY_LICENSES (drafts in `licensing-proposal/`).
4. Decide whether `pyside6-notes/` and `PYSIDE6_TRIAL.md` ship or move to the developer docs; file the upstream
   report.
5. Merge into a release candidate branch (not `main`), build the wheel and sdist, clean-install test, then the
   website/README wording for PySide6. Keep `audit/v1.6.0` (PyQt6) as the fallback until the release is out.
