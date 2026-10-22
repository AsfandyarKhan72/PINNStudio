"""The PyTorch build chosen for an NVIDIA driver, and the GPU advice printed at
start-up (pinnstudio/core/torch_index.py, select_torch_index.py)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio.core import torch_index as ti  # noqa: E402


def test_tag_for_driver():
    assert ti.select_tag((13, 0)) == "cu128"
    assert ti.select_tag((12, 9)) == "cu128"
    assert ti.select_tag((12, 8)) == "cu128"      # first with the RTX 50 series
    assert ti.select_tag((12, 3)) == "cu126"      # e.g. driver 546: newer 12.x builds run on it
    assert ti.select_tag((12, 0)) == "cu126"
    assert ti.select_tag((11, 8)) == "cu118"
    assert ti.select_tag((11, 4)) == "cpu"
    assert ti.select_tag(None) in {"cpu", "cu118", "cu126", "cu128"}   # this computer, whatever it has


def test_select_torch_index_script_runs():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "select_torch_index.py")],
                         capture_output=True, text=True, timeout=60, cwd=os.path.expanduser("~"))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() in {"cpu", "cu118", "cu126", "cu128"}


def test_advice_for_a_pip_install():
    lines = ti.gpu_fix_lines("cu126", prefix="/home/u/venv", executable="/home/u/venv/bin/python",
                             platform="linux", env={})
    text = "\n".join(lines)
    assert '"/home/u/venv/bin/python" -m pip uninstall -y torch' in text
    assert "https://download.pytorch.org/whl/cu126" in text
    assert "cu121" not in text and "--force-reinstall" not in text


def test_advice_for_the_installer():
    lines = ti.gpu_fix_lines("cu126", prefix=r"C:\Users\A B\AppData\Roaming\uv\tools\pinnstudio",
                             executable=r"C:\Users\A B\AppData\Roaming\uv\tools\pinnstudio\Scripts\python.exe",
                             platform="win32", env={})
    text = "\n".join(lines)
    assert "get-pinnstudio.ps1" in text and "pip" not in text


def test_advice_for_an_old_driver():
    assert "driver" in "\n".join(ti.gpu_fix_lines("cpu")).lower()
