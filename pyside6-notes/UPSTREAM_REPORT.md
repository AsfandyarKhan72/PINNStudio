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
