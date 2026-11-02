"""Standalone reproducer (PySide6 only): PySide6 6.12.0 loses references to None and True.

Every QTimer.singleShot(ms, callable) drops one reference to None, and every Python-side signal emit()
drops one reference to True. From Python 3.12 on these objects are immortal and nothing visible happens;
on Python 3.10 and 3.11 the counts reach zero and the interpreter aborts:
    Fatal Python error: none_dealloc: deallocating None: bug likely caused by a refcount error in a C extension
    Fatal Python error: bool_dealloc: deallocating True or False: ...

    python3.11 pyside612_refcount_repro.py

Measured (1000 calls each, Linux x86-64):
    PySide6 6.12.0 / Python 3.11.17: singleShot None -1004 .. -1010, emit True -1000 (then abort)
    PySide6 6.11.3, 6.10.2, 6.9.3 / Python 3.11.17: 0 or +1 (no defect)
    PySide6 6.12.0 / Python 3.13.16: 0 (immortal objects)
"""
import gc
import sys
from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal

app = QCoreApplication([])


class O(QObject):
    sig = Signal()


o = O()
o.sig.connect(lambda: None)
for name, step in (("QTimer.singleShot(0, lambda: None)", lambda: QTimer.singleShot(0, lambda: None)),
                   ("Signal.emit() connected to a lambda", o.sig.emit)):
    gc.collect()
    n0, t0 = sys.getrefcount(None), sys.getrefcount(True)
    for _ in range(50):               # few enough not to abort on 3.11
        step()
    for _ in range(3):
        app.processEvents()
    gc.collect()
    print(f"{name:38s} None {sys.getrefcount(None) - n0:+d}   True {sys.getrefcount(True) - t0:+d}  (50 calls)")
import os
os._exit(0)
