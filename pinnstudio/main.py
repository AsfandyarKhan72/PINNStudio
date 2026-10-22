import sys


def _preload_torch_on_windows():
    """On Windows, PyTorch fails to load once PyQt6 is loaded ("[WinError
    1114] A dynamic link library (DLL) initialization routine failed ...
    c10.dll"): both bring a Microsoft C++ runtime, and the first one loaded
    wins. Loading PyTorch first avoids it. Other systems are unaffected."""
    if sys.platform == "win32":
        try:
            import torch  # noqa: F401
        except Exception:
            pass


_preload_torch_on_windows()
from PyQt6.QtWidgets import QApplication  # noqa: E402  (after the preload above)


def _check_gpu_torch_mismatch():
    """Warn if an NVIDIA GPU is present but the installed PyTorch build cannot use it."""
    import shutil
    if not shutil.which("nvidia-smi"):
        return
    try:
        import torch
        if torch.cuda.is_available():
            return
    except Exception:
        pass
    from pinnstudio.core import torch_index
    print("=" * 70)
    print("An NVIDIA GPU was detected, but PyTorch cannot use it (PINNStudio runs on the CPU for now).")
    print("This happens with a CPU-only PyTorch, or one built for a newer CUDA than the driver supports.")
    for line in torch_index.gpu_fix_lines(torch_index.select_tag()):
        print(line)
    print("=" * 70)

def main():
    # Create the QApplication (and set its style) before importing PyTorch
    # (which happens inside _check_gpu_torch_mismatch()). Importing PyTorch
    # first has been observed to leave some Qt widgets built-but-not-painted
    # until an unrelated event nudges a repaint -- creating the QApplication
    # first avoids that ordering issue entirely.
    from pinnstudio import shortcut
    code = shortcut.main(sys.argv[1:])
    if code is not None:              # --create-shortcut / --remove-shortcut
        sys.exit(code)
    # errors are shown and logged instead of closing the window (see errors.py)
    from pinnstudio import errors
    errors.install()
    from pinnstudio.ui.main_window import MainWindow
    # "pinnstudio" as the program name, so the desktop matches the window to
    # its shortcut and icon however PINNStudio was started
    app = QApplication(["pinnstudio"] + sys.argv[1:])
    app.setApplicationName("PINNStudio")
    app.setDesktopFileName("pinnstudio")
    app.setStyle("Fusion")
    _check_gpu_torch_mismatch()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
