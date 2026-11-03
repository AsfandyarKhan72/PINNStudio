"""Which PyTorch CUDA build fits this computer's NVIDIA driver, and how to
install it when the PyTorch PINNStudio runs with cannot use the GPU.

Used by select_torch_index.py (install.sh / install.bat) and by the message
pinnstudio prints at start-up when it finds an NVIDIA GPU PyTorch can't use.
"""
import re
import subprocess
import sys

from pinnstudio.core import update_info

# PyTorch CUDA wheel tags, newest first, with the CUDA version the driver must
# report (`nvidia-smi`, "CUDA Version"). A build for CUDA 12.x runs on every
# driver that reports 12.0 or newer (CUDA's minor-version compatibility): on
# a laptop with driver 546.29 (CUDA 12.3) the cu126 and cu129 builds both ran
# normally (benchmarks/platform/windows_speed.md), and uv's --torch-backend
# auto chooses such a build there too. cu121 and cu124 only exist for PyTorch
# 2.5 and older. cu128 is the first with the RTX 50 series (Blackwell).
_TAGS = [
    (12, 8, "cu128"),
    (12, 0, "cu126"),
    (11, 8, "cu118"),
]


def driver_cuda_version():
    """The CUDA version the NVIDIA driver supports, e.g. (12, 3), or None."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def select_tag(version=None):
    """The wheel tag for the driver ("cu126", ...), or "cpu" without a usable
    NVIDIA driver (none, or older than any CUDA build PyTorch still has)."""
    version = driver_cuda_version() if version is None else version
    if version is None:
        return "cpu"
    for major, minor, tag in _TAGS:
        if tuple(version) >= (major, minor):
            return tag
    return "cpu"


def gpu_fix_lines(tag, prefix=None, executable=None, platform=None, env=None):
    """What to tell the user when PyTorch can't use their NVIDIA GPU."""
    prefix = sys.prefix if prefix is None else prefix
    executable = sys.executable if executable is None else executable
    platform = sys.platform if platform is None else platform
    if tag == "cpu":
        return ["The NVIDIA driver is too old for the PyTorch builds available now.",
                "Update the NVIDIA driver, restart, and start PINNStudio again."]
    if update_info.is_uv_tool_install(prefix, env):
        return ["Run PINNStudio's installer again; it installs the PyTorch build that fits the driver:",
                "  " + update_info.update_command(prefix, executable, platform, env)]
    return ["Install the PyTorch build that fits the driver, in the Python PINNStudio runs in:",
            f'  "{executable}" -m pip uninstall -y torch',
            f'  "{executable}" -m pip install torch --index-url https://download.pytorch.org/whl/{tag}']
