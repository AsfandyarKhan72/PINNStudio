# PySide6 migration experiment -- change log

Branch `experiment/pyside6-migration` (worktree /home/claude/pyside_wt), base a0f620d (audit/v1.6.0).
PySide6 6.12.0 / shiboken6 6.12.0 (venv_pyside, Python 3.13.16); baseline PyQt6 6.11.0 (Qt 6.11.0), system python3.

## Commit 1 -- mechanical import/API changes (no behaviour change intended)

Done with one sed over the 125 .py files that mention PyQt6 (pinnstudio/ 14, tests/ 97, tools/ 9, benchmarks/ 5):

| Rule | Why |
|---|---|
| `from PyQt6.X import ...` -> `from PySide6.X import ...` (QtCore, QtGui, QtWidgets, QtSvg, QtTest) | same module and class names in PySide6 |
| `pyqtSignal` -> `Signal` (incl. `pyqtSignal as _sig` -> `Signal as _sig`) | PySide6 name; all signal signatures used (`()`, `int`, `str`, `float`, `bool`, `list`, `dict`) are supported |
| `from PyQt6 import sip` -> `import shiboken6` | PySide6's wrapper-introspection module |
| `sip.isdeleted(w)` -> `not shiboken6.isValid(w)` | same meaning: the C++ object is gone |
| `PyQt6` -> `PySide6` in comments/docstrings naming the binding | text only |

Per-file locations in pinnstudio/ (line numbers after the change):
- pinnstudio/main.py:17 import; :5 docstring reworded to "once Qt (seen with PyQt6) is loaded" (the Windows DLL clash was observed with PyQt6 only; not verified for PySide6)
- pinnstudio/errors.py:65-66 imports (inside `_report`)
- pinnstudio/ui/main_window.py:6,14,15 imports; :65,339,340,390-394,452 Signal; :226,1985,11452,11920,12057 local imports; :13002,13378,16541,16868 `Signal as _sig` (signals of QThread subclasses defined inside methods)
- pinnstudio/ui/theme.py:21-23,541,558,573,590,603,613,627 imports; :592-593 sip -> shiboken6.isValid (deferred `_fit_height` guard)
- pinnstudio/ui/icons.py:12-14 imports (incl. QtSvg.QSvgRenderer); :118-126 `_prune()` uses `shiboken6.isValid(w)`; :164 import
- pinnstudio/ui/sketch_canvas.py:33-36 imports; :165-169,735-739 Signal; :972 QShortcut/QKeySequence (QtGui in both bindings); :1457 QFileDialog/QInputDialog/QMessageBox
- pinnstudio/ui/results.py:20-22,757,873-874 imports; :753 Signal
- pinnstudio/ui/workspace.py:18-20,89,158,218,250-251,257,424,435,498 imports; :49 Signal
- pinnstudio/ui/start_page.py:5-6 imports; :59-62 Signal
- pinnstudio/ui/equation.py:12-14, export_tools.py:10-12, figure_style.py:11, sci_spinbox.py:2,29, view3d.py:19-20 imports
- tests/*.py (97 files), tools/*.py (9), benchmarks/*.py + benchmarks/platform/win_checks.py (5): imports (QtWidgets/QtCore/QtTest) and comments only

## Commit 2 (f7ecaa1) -- behavioural difference found by testing

- tests/test_errors.py:50-56 `test_without_handler_the_app_dies` -> `test_without_handler_the_error_is_lost`.
  Why: PyQt6 aborts the process on an uncaught exception in a slot; PySide6 calls sys.excepthook / prints and keeps
  running. Under PySide6 the old test failed (`assert (0 != 0)`, seen in the first PySide6 run). The new test checks
  the PySide6 behaviour: the app survives, and without errors.install() nothing is logged.
  `test_error_in_slot_is_logged_and_app_survives` passes unchanged (PySide6 routes slot exceptions to sys.excepthook,
  so errors.py's message box + log keep working).
- pinnstudio/errors.py:3-7 docstring only: describes PySide6's behaviour (the error is lost with no console) instead of
  "PyQt ends the program". errors.install() is still needed (pythonw has no stderr) -- no code change.

Looked at and left unchanged (no behavioural difference hit):
- main_window.py:12137 `_vth.done_sig.disconnect()` guarded by `except TypeError`; :13084/:13090/:13370
  disconnect-then-connect guarded by `except Exception`. PySide6 6.12 raises nothing when there is nothing to
  disconnect (returns False + RuntimeWarning "Failed to disconnect"), PyQt6 raises TypeError. Same outcome; at
  most a RuntimeWarning line on stderr. No such warning appeared in any PySide6 run.
- `box.exec()` return compared with `QMessageBox.StandardButton.Save` (main_window.py:1769): both bindings return int and
  StandardButton is an IntFlag in both (probe: enumprobe.py) -> same result.
- QDialog.DialogCode, QEvent.Type, Qt.Key, Qt.ItemDataRole: IntEnum in both; `.value` works in both.
- `shiboken6.isValid` on a deleted widget: replaces sip.isdeleted (icons.py, theme.py).
- QTimer.singleShot(ms, callable), lambdas with default args in connect(), Signal(dict)/Signal(list) across threads
  (SweepThread.run_done_signal), QThread subclasses defined inside methods: all work unchanged.
- Matplotlib: backend_qtagg picks PySide6 automatically (already imported); verified with QT_API unset and PyQt6
  also installed (qt_compat.QT_API == 'PySide6', PyQt6 not imported).

## Commit 3 (8a4d626) -- packaging

- setup.py:24-29 `PyQt6>=6.4` + Windows pins `PyQt6==6.6.1`, `PyQt6-Qt6==6.6.1` -> `PySide6-Essentials>=6.5`
  (Essentials has QtCore/QtGui/QtWidgets/QtSvg/QtTest; Addons not needed). No Windows pin: untested.
- requirements.txt:3-6 same.
- Not changed (listed in the report): .github/workflows/smoke-test.yml:27,35, platform-matrix.yml:64,
  .github/ISSUE_TEMPLATE/bug_report.md:23, README.md (5373, 5477, 5502, 5508), CONTRIBUTING.md:25,68, CHANGELOG,
  THIRD_PARTY_LICENSES.md (licence statements -- left alone as instructed), benchmarks/platform/windows_speed.md.

## Commit 4 (c94da60) -- second difference found by testing: takeAt() ownership

Found by tests/test_layout_boxes.py (passes on PyQt6, failed on PySide6: "inverse: QLineEdit 'lambda_': 18 of 28 px
shown", every row on the Inverse page). Diagnosis (dbg_inv.py / dbg2.py in scratch): the page layout's item for the
"Unknown parameters" group kept heightForWidth = 271 px while the group needed 303 px; updateGeometry()/invalidate()
did not refresh it. Cause: `QLayout.takeAt()` returns an item that PySide6 does not own (shiboken6.ownedByPython ->
False) and never deletes; PyQt6 transfers it back to Python and deletes it. Qt registers only the first live
QWidgetItemV2 of a widget for size-cache invalidation, so while the old (leaked) item lives, the widget's item in its
new layout never hears that the widget's size changed.

- pinnstudio/ui/main_window.py:1101-1110 (`_build_workspace`): every panel moved from the old left layout onto the
  model-tree pages -- `left_layout.takeAt(0)` -> `left_layout.removeWidget(w)` (deletes the item in C++), takeAt only
  for non-widget items (spacers).
- main_window.py:1138-1139: sketch selection box moved to the top of the BCs page -- removeWidget + insertWidget.
- main_window.py:6646-6651 (custom-geometry shape rows reordered and re-added): same change.
- Left unchanged (only leak a small item, the widget is deleted anyway): main_window.py:5690/5734/5852 and
  results.py:479/501 (`takeAt(0)` then `widget().deleteLater()`), workspace.py:456 (spacer).
After the fix: item heightForWidth = 303 (as PyQt6); test_layout_boxes, test_custom_geometry, test_sketch_geometry,
test_line_slice_and_sketch_bcs, test_workspace_layout, test_form_alignment: 11 passed on PySide6.

## Verification summary (2026-11-01)

- PyQt6 never loaded: a `sitecustomize.py` guard on PYTHONPATH refused any `PyQt6`/`sip` import (inherited by the
  training subprocesses); `sys.modules` showed only PySide6 in every driver.
- Test suite, per file with `-n 1`, 379 tests: PyQt6 baseline (a0f620d) 372 passed / 5 skipped / 2 failed;
  PySide6 identical apart from the renamed errors test. The two failures were the same on both bindings and are
  fixed on audit/v1.6.0 (fresh ODE problem, anchors test); after merging those fixes the affected files
  (test_studio_extras, test_round65_anchors, test_audit_v160, test_audit_v160_gui, test_layout_boxes,
  test_errors: 65 tests) pass under PySide6.
- Generated scripts: `generate_script` and `generate_clean_script` for all 100 examples x Default/Tuned (400 scripts)
  are byte-identical between the two bindings (sha256).
- Solve: the launched script is identical after path normalisation; Burgers 1D and Diffusion 1D inverse (CPU, seed
  2026) print identical losses, Error Analysis values and inferred parameter (C = 1.22237).
- GUI workflows offscreen with the real MainWindow: launch/exit, examples (ODE, 1D, 2D polygon/disk, 3D, inverse,
  time-adaptive), every model-tree page, equation preview, Sketch, BC/IC/phase rows, 11 dialogs, Save/reopen,
  Export script, Solve (forward and inverse) with live loss and Results tabs, Restore & visualize, sweep (2 runs),
  Bayesian search (3 runs), `tools/smoke_gui.py`: all worked.
- Packaging: wheel requires `PySide6-Essentials>=6.5`; installs and launches in a fresh venv; Essentials contains
  every Qt module PINNStudio imports (QtCore, QtGui, QtWidgets, QtSvg; QtTest for tests).

## Open

- The heap-corruption crash: found and fixed, see the section below.
- Windows (the PyQt6 6.6.1 DLL pin), macOS and real high-DPI displays untested.
- GUI construction 20-50 % slower (training unaffected: separate process).
- CI workflows, README/CONTRIBUTING/CHANGELOG and license wording not changed.

- tests/test_portable_paths.py: PyQt6 crashed (segfault) in pytest's final garbage collection after all 8 tests passed;
  fixed on audit/v1.6.0 (5dc6559, windows deleted explicitly). Under PySide6 the file passes 3 of 3 runs, rc 0.

## The heap-corruption crash (root cause and fix, 2026-11-01)

- **Rate** (load Heat 3D <-> Allen-Cahn 1D, visit every page): PySide6 6.12.0 12 crashes in 150 rounds,
  PySide6 6.11.3 4 in 40, PyQt6 0 in 150. After the fix: PySide6 0 in 150 (and 0 in 150 + 150 with the first part
  of the fix alone).
- **Cause:** PySide6 keeps the wrapper of every `QLayoutItem` that `QLayout.itemAt()` returns (PyQt6 keeps none).
  Qt deletes those items itself when a row widget or layout is destroyed and PySide6 is not told; when the widgets
  are destroyed, `Shiboken::Object::destroy` writes into a 32-byte block it freed earlier in the same call
  (valgrind, `pyside6-notes/valgrind_invalid_write.txt`). `workspace.align_forms` walked every row with `itemAt()`
  on each page switch: ~1200 such wrappers after a few example loads.
- **Standalone reproducer:** `pyside6-notes/pyside_layoutitem_repro.py` crashes PySide6 in < 400 iterations;
  without `itemAt()` it runs clean (valgrind: 0 errors vs 2046); the PyQt6 version runs clean.
- **Fix:** no `QLayout.itemAt`/`takeAt` in `pinnstudio/ui` (`workspace.layout_widgets`, `layout_sublayouts`,
  `_row_slots`: QObject tree + `indexOf`). Pages render pixel-identical; generated scripts identical.
- **Tests:** `tests/test_layout_item_wrappers.py` (fails on the old code: 843 wrappers), `tests/test_gui_stress.py`
  (`PINNSTUDIO_STRESS=1`), `tools/gui_stress.py`.
- Upstream report drafted (`pyside6-notes/UPSTREAM_REPORT.md`), not filed.
