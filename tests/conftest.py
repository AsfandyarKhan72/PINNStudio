"""Test setup. On Windows, PyTorch must be loaded before PyQt6 or it fails with
"[WinError 1114] ... c10.dll" (see pinnstudio/main.py); test modules import
PyQt6 first, so load PyTorch here, before any test module is collected."""
import sys

if sys.platform == "win32":
    try:
        import torch  # noqa: F401
    except Exception:
        pass
