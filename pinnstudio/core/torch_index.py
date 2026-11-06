"""Which PyTorch CUDA build fits this computer's NVIDIA driver, and how to
install it when the PyTorch PINNStudio runs with cannot use the GPU.

Used by select_torch_index.py (install.sh / install.bat) and by the message
pinnstudio prints at start-up when it finds an NVIDIA GPU PyTorch can't use.
"""
import re
import subprocess
import sys

from pinnstudio.core import update_info

# Which PyTorch CUDA build to install depends on two things: the CUDA version
# the NVIDIA driver reports (`nvidia-smi`, "CUDA Version"), and the GPU's
# architecture (compute capability). A build for CUDA 12.x runs on every driver
# that reports 12.0 or newer (CUDA's minor-version compatibility): on a laptop
# with driver 546.29 (CUDA 12.3) the cu126 and cu129 builds both ran normally
# (benchmarks/platform/windows_speed.md). A 13.x build needs a driver that
# reports 13.0 or newer. But each build only has kernels for some GPUs
# (PyTorch 2.14, November 2026):
#
#   cu126  Maxwell (5.0) to Hopper (9.0), no Blackwell. Current torch (2.14),
#          the last release with CUDA 12.x wheels.
#   cu128  Volta (7.0) to Blackwell (12.0), no Maxwell/Pascal since torch 2.8.
#          Its newest torch is 2.11; later releases are not built for it.
#   cu130  Turing (7.5) to Blackwell (12.0), no Maxwell/Pascal/Volta.
#   cu118  Older drivers (11.8 to 11.x); torch 2.7 at the newest.
#
# So: Blackwell gets cu130 (or cu128 with a 12.8/12.9 driver); Turing to Hopper
# get cu130 with a 13.x driver and cu126 otherwise; Maxwell, Pascal and Volta
# get cu126. When the architecture can't be read, cu126 is chosen, because it
# runs on the widest range of GPUs (every NVIDIA GPU except Blackwell, which
# always reports its architecture). With several GPUs the newest one decides,
# since no single build covers both Blackwell and Pascal/Volta. The installers
# then run a small calculation on the GPU and fall back to the CPU build if it
# fails (check_cuda_build below).
_BLACKWELL = (10, 0)
_TURING = (7, 5)
_MAXWELL = (5, 0)


def _nvidia_smi(*args):
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                              timeout=10).stdout
    except Exception:
        return None


def driver_cuda_version(output=None):
    """The CUDA version the NVIDIA driver supports, e.g. (12, 3), or None.
    `output` is the text of `nvidia-smi` (read from the system if None)."""
    out = _nvidia_smi() if output is None else output
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def gpu_compute_capabilities(output=None):
    """The compute capability of each NVIDIA GPU, e.g. [(6, 1)], or None if it
    can't be read. `output` is the text of
    `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`."""
    out = (_nvidia_smi("--query-gpu=compute_cap", "--format=csv,noheader")
           if output is None else output)
    caps = []
    for line in (out or "").splitlines():
        m = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", line)
        if m:
            caps.append((int(m.group(1)), int(m.group(2))))
        elif line.strip():
            return None          # "[N/A]", an error message, ...: unknown
    return caps or None


_DETECT = object()


def select_tag(version=None, caps=_DETECT):
    """The wheel tag for this computer ("cu126", ...), or "cpu" without a usable
    NVIDIA GPU and driver. `version` is the driver's CUDA version and `caps` the
    GPUs' compute capabilities (None: unknown); both are read from the system
    when not given."""
    version = driver_cuda_version() if version is None else tuple(version)
    if version is None:
        return "cpu"
    if caps is _DETECT:
        caps = gpu_compute_capabilities()
    newest = max(caps) if caps else None
    if newest is not None and newest < _MAXWELL:
        return "cpu"             # Kepler and older: no current PyTorch build
    if newest is not None and newest >= _BLACKWELL:
        if version >= (13, 0):
            return "cu130"
        return "cu128" if version >= (12, 8) else "cpu"
    if newest is not None and newest >= _TURING and version >= (13, 0):
        return "cu130"
    if version >= (12, 0):
        return "cu126"
    return "cu118" if version >= (11, 8) else "cpu"


# Run by the installers after installing a CUDA build: imports torch, checks the
# build tag, and does a small calculation on the GPU. A build without kernels
# for the GPU still reports torch.cuda.is_available() == True; only running a
# kernel shows the problem ("no kernel image is available").
def check_cuda_build(tag):
    """True if the installed PyTorch is the `tag` build and can compute on the GPU."""
    try:
        import torch
        if f"+{tag}" not in torch.__version__ or not torch.cuda.is_available():
            return False
        x = torch.arange(8, dtype=torch.float32, device="cuda")
        return float((x * 2).sum().cpu()) == 56.0
    except Exception:
        return False


def gpu_fix_lines(tag, prefix=None, executable=None, platform=None, env=None):
    """What to tell the user when PyTorch can't use their NVIDIA GPU."""
    prefix = sys.prefix if prefix is None else prefix
    executable = sys.executable if executable is None else executable
    platform = sys.platform if platform is None else platform
    if tag == "cpu":
        return ["The NVIDIA driver, or the GPU, is too old for the PyTorch builds available now.",
                "Update the NVIDIA driver, restart, and start PINNStudio again;"
                " a GPU older than Maxwell (GTX 9xx) can only train on the CPU."]
    if update_info.is_uv_tool_install(prefix, env):
        return ["Run PINNStudio's installer again; it installs the PyTorch build that fits the driver:",
                "  " + update_info.update_command(prefix, executable, platform, env)]
    return ["Install the PyTorch build that fits the driver, in the Python PINNStudio runs in:",
            f'  "{executable}" -m pip uninstall -y torch',
            f'  "{executable}" -m pip install torch --index-url https://download.pytorch.org/whl/{tag}']
