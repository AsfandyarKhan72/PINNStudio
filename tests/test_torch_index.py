"""The PyTorch build chosen for an NVIDIA driver, and the GPU advice printed at
start-up (pinnstudio/core/torch_index.py, select_torch_index.py)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio.core import torch_index as ti  # noqa: E402


TAGS = {"cpu", "cu118", "cu126", "cu128", "cu130"}

# Representative GPUs (compute capability) and what each build has kernels for
# (PyTorch 2.14): cu126 5.0-9.0, cu128 7.0-12.0, cu130 7.5-12.0.
MAXWELL, PASCAL, P100, VOLTA = (5, 2), (6, 1), (6, 0), (7, 0)
TURING, AMPERE, ADA, HOPPER = (7, 5), (8, 6), (8, 9), (9, 0)
BLACKWELL_DC, BLACKWELL = (10, 0), (12, 0)
KEPLER = (3, 7)
RUNS_ON = {"cu126": ((5, 0), (9, 0)), "cu128": ((7, 0), (12, 0)), "cu130": ((7, 5), (12, 0)),
           "cu118": ((5, 0), (9, 0))}
NEEDS_DRIVER = {"cu118": (11, 8), "cu126": (12, 0), "cu128": (12, 8), "cu130": (13, 0)}


def test_tag_for_gpu_and_driver():
    cases = [
        # GPU, driver CUDA version, expected build
        (PASCAL, (13, 0), "cu126"),      # GTX 10xx with a 580 driver: not cu128/cu130
        (PASCAL, (12, 9), "cu126"),
        (P100, (12, 8), "cu126"),
        (MAXWELL, (12, 8), "cu126"),
        (MAXWELL, (12, 2), "cu126"),
        (VOLTA, (13, 0), "cu126"),       # cu130 has no Volta kernels
        (VOLTA, (12, 8), "cu126"),
        (PASCAL, (11, 8), "cu118"),
        (TURING, (13, 0), "cu130"),
        (TURING, (12, 8), "cu126"),
        (AMPERE, (13, 2), "cu130"),
        (AMPERE, (12, 3), "cu126"),      # e.g. driver 546
        (ADA, (12, 9), "cu126"),
        (HOPPER, (13, 0), "cu130"),
        (BLACKWELL, (13, 0), "cu130"),   # RTX 50 series
        (BLACKWELL, (12, 9), "cu128"),   # a 575 driver: only cu128 has Blackwell kernels
        (BLACKWELL, (12, 8), "cu128"),
        (BLACKWELL_DC, (13, 1), "cu130"),
        (KEPLER, (12, 4), "cpu"),        # no current PyTorch build
        (AMPERE, (11, 4), "cpu"),        # driver too old for any build
    ]
    for gpu, driver, expected in cases:
        tag = ti.select_tag(driver, [gpu])
        assert tag == expected, (gpu, driver, tag)
        if tag != "cpu":                 # the build runs on this GPU and driver
            lo, hi = RUNS_ON[tag]
            assert lo <= gpu <= hi and driver >= NEEDS_DRIVER[tag], (gpu, driver, tag)


def test_every_gpu_and_driver_gets_a_build_that_runs():
    gpus = [MAXWELL, P100, PASCAL, VOLTA, TURING, AMPERE, ADA, HOPPER, BLACKWELL_DC, BLACKWELL]
    drivers = [(11, 8), (12, 0), (12, 4), (12, 8), (12, 9), (13, 0), (13, 2)]
    for gpu in gpus:
        for driver in drivers:
            tag = ti.select_tag(driver, [gpu])
            if gpu >= (10, 0) and driver < (12, 8):
                assert tag == "cpu"      # not a real combination: Blackwell needs a 12.8 driver
                continue
            lo, hi = RUNS_ON[tag]
            assert lo <= gpu <= hi and driver >= NEEDS_DRIVER[tag], (gpu, driver, tag)


def test_unknown_architecture_is_handled_conservatively():
    # Can't read the architecture: cu126 (every GPU except Blackwell), never cu128/cu130.
    for driver in [(12, 0), (12, 8), (12, 9), (13, 0), (13, 2)]:
        assert ti.select_tag(driver, None) == "cu126"
    assert ti.select_tag((11, 8), None) == "cu118"
    assert ti.select_tag((11, 4), None) == "cpu"
    assert ti.select_tag(None, None) == "cpu"   # no driver


def test_several_gpus_the_newest_decides():
    assert ti.select_tag((13, 0), [PASCAL, AMPERE]) == "cu130"
    assert ti.select_tag((13, 0), [PASCAL, BLACKWELL]) == "cu130"
    assert ti.select_tag((13, 0), [PASCAL, PASCAL]) == "cu126"


def test_reading_nvidia_smi_output():
    smi = ("| NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0     |\n")
    assert ti.driver_cuda_version(smi) == (13, 0)
    assert ti.driver_cuda_version("NVIDIA-SMI has failed because it couldn't communicate") is None
    assert ti.driver_cuda_version("") is None
    assert ti.gpu_compute_capabilities("6.1\n") == [(6, 1)]
    assert ti.gpu_compute_capabilities("12.0\n8.6\n") == [(12, 0), (8, 6)]
    assert ti.gpu_compute_capabilities("[N/A]\n") is None
    assert ti.gpu_compute_capabilities("Field \"compute_cap\" is not a valid field to query.\n") is None
    assert ti.gpu_compute_capabilities("") is None


def test_detection_without_nvidia_smi(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(ti.subprocess, "run", missing)
    assert ti.driver_cuda_version() is None
    assert ti.gpu_compute_capabilities() is None
    assert ti.select_tag() == "cpu"


def test_detection_from_mocked_nvidia_smi(monkeypatch):
    class Out:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake(gpu_line, driver_line):
        def run(cmd, **k):
            return Out(gpu_line if "--query-gpu=compute_cap" in cmd else driver_line)
        return run
    driver13 = "Driver Version: 580.95.05   CUDA Version: 13.0"
    monkeypatch.setattr(ti.subprocess, "run", fake("6.1\n", driver13))
    assert ti.select_tag() == "cu126"
    monkeypatch.setattr(ti.subprocess, "run", fake("12.0\n", driver13))
    assert ti.select_tag() == "cu130"
    monkeypatch.setattr(ti.subprocess, "run", fake("[N/A]\n", driver13))
    assert ti.select_tag() == "cu126"


def test_check_cuda_build_needs_a_working_gpu():
    # This computer has no GPU PyTorch can use, so no CUDA build passes the check.
    import torch
    if torch.cuda.is_available():
        return
    for tag in ["cu126", "cu128", "cu130"]:
        assert ti.check_cuda_build(tag) is False


def test_check_cuda_build_catches_missing_kernels(monkeypatch):
    # A build without kernels for the GPU: cuda.is_available() is True and the
    # version tag matches, but the first calculation fails.
    import torch
    monkeypatch.setattr(torch, "__version__", "2.14.1+cu128")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def no_kernel(*a, **k):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
    monkeypatch.setattr(torch, "arange", no_kernel)
    assert ti.check_cuda_build("cu128") is False
    assert ti.check_cuda_build("cu126") is False   # wrong tag


def test_check_cuda_build_passes_when_the_calculation_works(monkeypatch):
    import torch
    real_arange = torch.arange
    monkeypatch.setattr(torch, "__version__", "2.14.1+cu130")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "arange", lambda *a, device=None, **k: real_arange(*a, **k))
    assert ti.check_cuda_build("cu130") is True


def test_select_torch_index_script_runs():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "select_torch_index.py")],
                         capture_output=True, text=True, timeout=60, cwd=os.path.expanduser("~"))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() in TAGS


def test_select_torch_index_check_mode():
    import torch
    out = subprocess.run([sys.executable, os.path.join(ROOT, "select_torch_index.py"), "--check", "cu126"],
                         capture_output=True, text=True, timeout=120, cwd=os.path.expanduser("~"))
    expected = 0 if ti.check_cuda_build("cu126") else 1
    assert out.returncode == expected, out.stderr
    if not torch.cuda.is_available():
        assert out.returncode == 1


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
