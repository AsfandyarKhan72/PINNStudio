#!/usr/bin/env python3
"""
Prints the PyTorch CUDA wheel tag that fits this computer's NVIDIA GPU and
driver (e.g. "cu126"), or "cpu" if no usable GPU/driver is found. Used by
install.sh/install.bat to pick the right --index-url before installing torch.
With `--check TAG` it instead exits with 0 if the installed PyTorch is the TAG
build and can compute on the GPU, and 1 if not.
The rules and the reasons behind them are in pinnstudio/core/torch_index.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinnstudio.core.torch_index import check_cuda_build, select_tag  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        sys.exit(0 if check_cuda_build(sys.argv[2]) else 1)
    print(select_tag())
