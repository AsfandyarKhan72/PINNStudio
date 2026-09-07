#!/usr/bin/env python3
"""
Detects the NVIDIA driver's maximum supported CUDA version (via `nvidia-smi`)
and prints the best-matching PyTorch CUDA wheel tag (e.g. "cu126") to stdout,
or "cpu" if no compatible GPU/driver is found. Used by install.sh/install.bat
to pick the right --extra-index-url before installing torch, instead of
hardcoding one CUDA version that may not match every machine's driver.
"""
import re
import subprocess

# Known-live PyTorch CUDA wheel tags, newest first. A driver that supports
# CUDA X can run any build compiled for CUDA <= X, so we pick the newest tag
# the driver's reported version can support. Verified live as of 2026-09-07;
# revisit this list occasionally as PyTorch adds/retires CUDA tags.
_KNOWN_TAGS = [
    (12, 8, "cu128"),
    (12, 6, "cu126"),
    (12, 4, "cu124"),
    (12, 1, "cu121"),
    (11, 8, "cu118"),
]


def _driver_cuda_version():
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def select_tag():
    version = _driver_cuda_version()
    if version is None:
        return "cpu"
    for major, minor, tag in _KNOWN_TAGS:
        if version >= (major, minor):
            return tag
    # A GPU/driver is present but older than any known-good tag here --
    # CPU is safer than guessing at a CUDA build that may not work.
    return "cpu"


if __name__ == "__main__":
    print(select_tag())
