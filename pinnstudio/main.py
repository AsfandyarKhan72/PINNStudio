import sys
from PyQt6.QtWidgets import QApplication


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
    print("=" * 70)
    print("An NVIDIA GPU was detected, but PyTorch cannot use it.")
    print("This usually happens when PyTorch was installed with pip install pinnstudio directly,")
    print("which can pull a CUDA build newer than what your GPU driver supports.")
    print("Fix it with these two commands:")
    print("  pip install --upgrade pip")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall")
    print("(PINNStudio will still run on CPU in the meantime.)")
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
