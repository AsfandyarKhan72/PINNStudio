import sys
from PyQt6.QtWidgets import QApplication
from pinnstudio.ui.main_window import MainWindow


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
    _check_gpu_torch_mismatch()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
