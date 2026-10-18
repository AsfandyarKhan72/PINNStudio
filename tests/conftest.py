"""Test setup for Windows.

* PyTorch must be loaded before PyQt6 or it fails with "[WinError 1114] ...
  c10.dll" (see pinnstudio/main.py); test modules import PyQt6 first, so load
  PyTorch here, before any test module is collected.
* Scripts the tests start themselves run in Python's UTF-8 mode, as the app
  runs its own (pinnstudio/core/pyexe.py), and print in this process's
  encoding so their output can be read back. This process keeps Windows'
  default encoding, so the app code under test still meets it.
"""
import locale
import os
import sys

if sys.platform == "win32":
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = locale.getpreferredencoding(False) + ":replace"
