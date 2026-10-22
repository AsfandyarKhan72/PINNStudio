#!/usr/bin/env python3
"""
Prints the PyTorch CUDA wheel tag that fits this computer's NVIDIA driver
(e.g. "cu126"), or "cpu" if no usable GPU/driver is found. Used by
install.sh/install.bat to pick the right --index-url before installing torch.
The table and the reasons behind it are in pinnstudio/core/torch_index.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinnstudio.core.torch_index import select_tag  # noqa: E402

if __name__ == "__main__":
    print(select_tag())
