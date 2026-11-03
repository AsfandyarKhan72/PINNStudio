"""The CUDA allocator setting PINNStudio passes to its training runs: no
expandable segments on Windows (PyTorch warns there on every GPU run), and
PyTorch 2.9's new variable name when that torch is installed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnstudio.core import codegen  # noqa: E402


def _env(monkeypatch, platform, torch_version):
    monkeypatch.setattr(sys, "platform", platform)
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: torch_version)
    return codegen._torch_alloc_env()


def test_windows_has_no_expandable_segments(monkeypatch):
    name, value = _env(monkeypatch, "win32", "2.9.0+cu129")
    assert name == "PYTORCH_ALLOC_CONF"
    assert "expandable_segments" not in value and "max_split_size_mb:512" in value


def test_linux_keeps_expandable_segments(monkeypatch):
    name, value = _env(monkeypatch, "linux", "2.14.1+cu130")
    assert name == "PYTORCH_ALLOC_CONF" and "expandable_segments:True" in value


def test_older_torch_uses_the_old_name(monkeypatch):
    name, _ = _env(monkeypatch, "linux", "2.6.0+cu124")
    assert name == "PYTORCH_CUDA_ALLOC_CONF"
