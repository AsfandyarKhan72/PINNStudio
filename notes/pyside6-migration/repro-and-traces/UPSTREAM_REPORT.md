# Draft bug report for Qt for Python (bugreports.qt.io, project PYSIDE) — not filed

**Title:** Heap corruption: QLayout.itemAt() wrappers outlive layout items that Qt deletes implicitly

**Versions:** PySide6 6.12.0 (Qt 6.12.0) and PySide6 6.11.3 (Qt 6.11.3); Python 3.13.16; Linux x86_64 (Ubuntu 24.04,
glibc 2.39); QPA offscreen and xcb (Xvfb). Not reproduced with PyQt6 6.11.0 on the same Qt 6.11 and machine.

**Summary.** The Python wrapper of every `QLayoutItem` returned by `QLayout.itemAt()` stays alive after the last
Python reference is dropped (it appears to be kept as a child of the layout's wrapper). When Qt itself deletes
such an item — e.g. a widget in a layout is destroyed with `deleteLater()` and its `QWidgetItem` is deleted by the
layout — PySide6 is not notified (`QLayoutItem` is not a `QObject`), so the wrapper stays registered at the freed
address and `shiboken6.isValid()` still returns `True`. Code that walks layouts with `itemAt()` and rebuilds
forms accumulates such wrappers; after the addresses are reused the process aborts with glibc heap-corruption
errors or segfaults during event processing.

**Minimal example:** `pyside_layoutitem_repro.py` (attached; ~150 lines, PySide6 only; `pyqt6_layoutitem_repro.py` is the same script for PyQt6). `WALK=0` drops the read-only walk; `NOITEMAT=1` finds the rows to delete from a Python list instead of `itemAt()`.
Pages of group boxes with nested `QHBoxLayout` rows; each iteration deletes the rows with `deleteLater()`,
builds new ones, shows each page and walks its layouts read-only with `itemAt()`.

**Observed** (see results below): abort within a few hundred iterations, e.g.

```
malloc_consolidate(): unaligned fastbin chunk detected
Fatal Python error: Aborted
  File "pyside_layoutitem_repro.py", line 84 in settle   (QCoreApplication.sendPostedEvents / processEvents)
```

or `free(): invalid pointer`, or `Segmentation fault`. The native backtrace (gdb) shows the corruption being
detected later, in an unrelated allocation (e.g. `QTextEngine::shapeTextWithHarfbuzzNG` →
`malloc_consolidate`), i.e. the heap was corrupted earlier.

**Controls:**
- `WALK=0` (itemAt only in the deletion loop): still crashes (3 of 3);
- `WALK=0 NOITEMAT=1` (no `itemAt()` at all): 3 × 400 iterations, no crash;
- the same script with PyQt6: no crash, and no `QLayoutItem` wrappers are kept alive (`gc.get_objects()`);
- valgrind memcheck, 120 iterations: **2 046 errors** with `itemAt()`, **0** without it. The first invalid
  write (`valgrind_invalid_write.txt`) is inside `Shiboken::Object::destroy`, called from the PySide6 QtWidgets
  wrapper destructor during `QObjectPrivate::deleteChildren()` of a deferred-deleted `QGroupBox`; the 32-byte block
  written to was freed earlier in the same `Shiboken::Object::destroy` call (also `Invalid read` in
  `Shiboken::BindingManager::unregisterWrapper`). Live `QWidgetItem`/`QSpacerItem` wrappers grow 588 → 2075 → 3486.
- AddressSanitizer (LD_PRELOAD, uninstrumented Qt/shiboken) reports nothing: the write lands in memory its
  quarantine keeps, so the heap is not damaged under ASan.

**Isolated observations** (`itemat_keep.py`, `itemat_stale.py`):
- after `for i in range(lay.count()): lay.itemAt(i)` and dropping all references, `shiboken6.getAllValidWrappers()`
  still lists the items (PyQt6: none);
- `lay.removeWidget(w)` invalidates the wrapper of `w`'s item, but deleting the widget (or its parent) does not:
  the wrapper of the item that Qt deleted remains registered and reports valid.

**Workaround used in our application:** do not call `itemAt()` on layouts whose items Qt may delete; walk the
QObject tree instead (`findChildren(QLayout)`, `QLayout.indexOf(widget)`), which never creates `QLayoutItem`
wrappers.

**Expected:** a wrapper for a `QLayoutItem` owned by a layout should not keep a registration (address→wrapper)
that outlives the item, or should not be kept alive once Python drops it, as with PyQt6.

---

# Draft 2 (not filed): PySide6 6.12.0 drops references to None (QTimer.singleShot) and True (Signal.emit)

**Found by:** the Smoke test workflow (ubuntu-latest, Python 3.11) aborting in a PINNStudio GUI test with
`Fatal Python error: none_dealloc: deallocating None`; reproduced locally 10 of 10 times on Python 3.11.17.

**Versions:** PySide6-Essentials / shiboken6 6.12.0 on Python 3.10/3.11 (Linux x86-64). Not seen with 6.11.3,
6.10.2 or 6.9.3 on the same Python. On Python 3.12+ None/True are immortal, so the lost references are invisible.

**Reproducer:** `pyside612_refcount_repro.py` (prints the change in `sys.getrefcount(None)` / `(True)`).
Each `QTimer.singleShot(0, callable)` lowers None's count by one, whatever the callable returns; each
Python-side `signal.emit()` lowers True's count by one (emit's bool result is released once too often).

**PINNStudio's handling:** setup.py / requirements.txt require PySide6 < 6.12 on Python < 3.12;
tests/test_binding_refcounts.py fails on any installed binding with the defect (it runs in the Smoke test).

---

# Draft 3 (not filed): Windows, PySide6 6.12.0 - access violation at process exit while a widget wrapper is alive

**Native stack** (`windows_exit_crash_cdb.txt`): `ExitProcess` -> `LdrShutdownProcess` -> shiboken6's DLL
`atexit` table -> `Shiboken::BindingManager::~BindingManager` -> `Shiboken::Object::destroy` ->
`SbkDeallocWrapper` deletes the still-alive, Python-owned `QMainWindow` -> `QWidget::~QWidget` sends an event ->
`QCoreApplicationPrivate::sendThroughApplicationEventFilters` -> PySide calls the application's Python event
filter -> `Shiboken::Conversions::copyToPython` with a null converter (`mov rax,[rcx+18h]`, rcx = 0): the type
tables are already gone while the binding manager still deletes C++ objects and calls back into Python.

**When:** only when the process ends without Python's own finalization (`os._exit`, i.e. `_exit` -> `ExitProcess`
on Windows, which still runs DLL static destructors; Linux `_exit` does not). With `sys.exit` or the app's normal
quit, Python finalization releases the window first and nothing is left for the binding manager.

**PINNStudio's handling:** the program itself never calls `os._exit`; closing the window exits with code 0
(tools/quit_check.py on Windows, macOS and Linux). Tools that end with `os._exit` delete their windows first
(tools/smoke_gui.py: close, deleteLater, send the posted DeferredDelete events).
