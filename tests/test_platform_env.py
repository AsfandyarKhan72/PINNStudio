"""Environment settings that behave differently per platform.

- Train on: CPU hides the GPU with CUDA_VISIBLE_DEVICES="-1". An empty value
  does not hide it on Windows: CUDA treats it as unset while PyTorch's Python
  side sees 0 devices, and Solve's script stopped with "Invalid device id".
- A GPU run with the allocator setting PINNStudio passes on prints no warning
  about it (expandable_segments on Windows, the old variable name on torch
  2.9). tests/test_alloc_env.py checks the name and value it picks.
The GPU checks only mean something on a computer with an NVIDIA GPU; on others
they pass trivially or are skipped.
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnstudio.core import codegen as cg  # noqa: E402  (sets the allocator variable, as the window does)
from pinnstudio.core.config import PINNConfig  # noqa: E402


def _run(code, env=None):
    e = dict(os.environ, DDE_BACKEND="pytorch")
    e.pop("CUDA_VISIBLE_DEVICES", None)
    e.update(env or {})
    return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=300)


def test_cpu_line_hides_the_gpu():
    c = PINNConfig()
    c.train_device = "cpu"
    line = [ln for ln in cg.generate_script(c).splitlines() if "CUDA_VISIBLE_DEVICES" in ln][0]
    p = _run("import os\n" + line + "\nimport torch\n"
             "print(torch.cuda.is_available(), torch.cuda.device_count())")
    assert p.returncode == 0, p.stderr[-1500:]
    assert p.stdout.split()[-2:] == ["False", "0"], (line, p.stdout, p.stderr[-800:])


def test_gpu_run_prints_no_allocator_warnings():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    p = _run("import torch; x = torch.ones(1000, device='cuda'); print(float((x * 2).sum()))")
    assert p.returncode == 0, p.stderr[-1500:]
    assert "expandable_segments not supported" not in p.stderr, p.stderr[-800:]
    assert "ALLOC_CONF is deprecated" not in p.stderr, p.stderr[-800:]
