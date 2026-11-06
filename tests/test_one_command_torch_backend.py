"""The PyTorch build the one-command installers (get-pinnstudio.sh / .ps1)
install, and their check that PyTorch can compute on the GPU.

uv's `--torch-backend auto` chooses from the NVIDIA driver alone, so for
Maxwell, Pascal and Volta GPUs the scripts choose the build themselves
(cu126 / cu118 by the driver's CUDA version, or CPU). Everything else keeps
`auto`; UV_TORCH_BACKEND overrides. After installing, a calculation on cuda:0
decides between GPU, CPU fallback (reinstall with the CPU build) and CPU.

No GPU is needed: nvidia-smi, torch and uv are replaced by small fakes."""
import os
import shutil
import stat
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SH = os.path.join(ROOT, "get-pinnstudio.sh")
PS1 = os.path.join(ROOT, "get-pinnstudio.ps1")
WIN = sys.platform.startswith("win")
POSIX_SH = (not WIN) and shutil.which("sh")
PWSH = os.environ.get("PWSH") or shutil.which("pwsh")

# (compute capabilities, driver CUDA version, expected --torch-backend)
CASES = [
    ("6.1", "13.0", "cu126"),       # GTX 10xx, 580 driver: not uv's cu130/cu132
    ("6.1", "12.9", "cu126"),       # 575 driver: not uv's cu129/cu128
    ("6.0", "12.8", "cu126"),       # P100
    ("5.2", "12.4", "cu126"),       # Maxwell
    ("7.0", "13.0", "cu126"),       # V100: cu130 has no Volta kernels
    ("6.1", "11.8", "cu118"),       # older driver: only CUDA 11.8
    ("6.1", "11.4", "cpu"),         # driver too old for any current build
    ("3.7", "12.4", "cpu"),         # Kepler: no current build
    ("6.1", "", "cpu"),             # older GPU, driver version unreadable
    ("7.5", "13.0", "auto"),        # Turing and newer: uv's choice is fine
    ("8.9", "12.8", "auto"),
    ("12.0", "12.9", "auto"),       # Blackwell
    ("12.0\n8.6", "13.0", "auto"),
    ("8.6\n6.1", "13.0", "cu126"),  # mixed: the build must run on both
    ("12.0\n6.1", "13.0", "cpu"),   # Blackwell + Pascal: no common build
    ("12.0\n7.0", "12.8", "cpu"),
    ("[N/A]", "13.0", "auto"),      # GPU information can't be read: unchanged
    ("Field \"compute_cap\" is not a valid field to query.", "12.0", "auto"),
    ("", "13.0", "auto"),
]

SMI = """#!/bin/sh
case "$*" in
  *compute_cap*) printf '%s\\n' "$FAKE_CAPS" ;;
  *) echo "| NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: $FAKE_CUDA |" ;;
esac
"""
# cmd.exe splits "--query-gpu=compute_cap" at the "=", so %1 is "--query-gpu".
SMI_CMD = """@echo off
if "%~1"=="--query-gpu" goto caps
echo NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: %FAKE_CUDA%
exit /b 0
:caps
echo %FAKE_CAPS%
"""

# A stand-in torch: FAKE_TORCH=none (no GPU), failed (sees a GPU, no kernels), ok.
FAKE_TORCH = '''import os
_state = os.environ.get("FAKE_TORCH", "none")
float32 = "float32"
class _T:
    def __mul__(self, k): return self
    def sum(self): return self
    def cpu(self): return self
    def __float__(self): return 56.0
def arange(*a, device=None, **k):
    if _state == "failed":
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
    return _T()
class cuda:
    @staticmethod
    def is_available(): return _state != "none"
    @staticmethod
    def get_device_name(i): return "NVIDIA Fake GPU"
'''


def _exe(path, text):
    with open(path, "w", newline="\n") as f:
        f.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def _fake_smi_dir():
    d = tempfile.mkdtemp()
    if WIN:
        with open(os.path.join(d, "nvidia-smi.cmd"), "w") as f:
            f.write(SMI_CMD)
    else:
        _exe(os.path.join(d, "nvidia-smi"), SMI)
    return d


def _fake_torch_dir():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "torch"))
    with open(os.path.join(d, "torch", "__init__.py"), "w") as f:
        f.write(FAKE_TORCH)
    return d


def _bin_without_nvidia_smi():
    """A PATH with the shell tools the script needs, and no nvidia-smi."""
    d = tempfile.mkdtemp()
    for tool in ["sh", "awk", "sed", "head", "sort", "printf", "cat"]:
        p = shutil.which(tool)
        if p:
            os.symlink(p, os.path.join(d, tool))
    return d


def _env(env, **more):
    """This environment without a UV_TORCH_BACKEND of its own, plus `env`."""
    full = {k: v for k, v in os.environ.items() if k != "UV_TORCH_BACKEND"}
    full.update(env or {})
    full.update(more)
    return full


def _sh(cmd, env=None):
    full = _env(env)
    r = subprocess.run(["sh", "-c", f'PINNSTUDIO_INSTALLER_FUNCTIONS_ONLY=1; . "{SH}"; {cmd}'],
                       capture_output=True, text=True, env=full, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _ps(cmd, env=None):
    full = _env(env, PINNSTUDIO_INSTALLER_FUNCTIONS_ONLY="1")
    r = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-Command", f'. "{PS1}"; {cmd}'],
                       capture_output=True, text=True, env=full, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip()


def _ps_str(s):
    """s as a double-quoted PowerShell string."""
    return '"' + s.replace("`", "``").replace('"', '`"').replace("$", "`$").replace("\n", "`n") + '"'


# ---- get-pinnstudio.sh ---------------------------------------------------

@pytest.mark.skipif(not POSIX_SH, reason="POSIX sh installer")
def test_sh_choice_for_gpu_and_driver():
    for caps, cuda, expected in CASES:
        caps_sh = caps.replace("'", "'\\''")
        got = _sh(f"choose_torch_backend '{caps_sh}' '{cuda}'")
        assert got == expected, (caps, cuda, got)


@pytest.mark.skipif(not POSIX_SH, reason="POSIX sh installer")
def test_sh_detection_from_nvidia_smi():
    d = _fake_smi_dir()
    path = d + os.pathsep + os.environ["PATH"]
    assert _sh("torch_backend", {"PATH": path, "FAKE_CAPS": "6.1", "FAKE_CUDA": "13.0"}) == "cu126"
    assert _sh("torch_backend", {"PATH": path, "FAKE_CAPS": "6.1", "FAKE_CUDA": "11.8"}) == "cu118"
    assert _sh("torch_backend", {"PATH": path, "FAKE_CAPS": "8.9", "FAKE_CUDA": "13.0"}) == "auto"
    assert _sh("torch_backend", {"PATH": path, "FAKE_CAPS": "[N/A]", "FAKE_CUDA": "13.0"}) == "auto"
    # an explicit UV_TORCH_BACKEND wins over the detection
    assert _sh("torch_backend", {"PATH": path, "FAKE_CAPS": "6.1", "FAKE_CUDA": "13.0",
                                 "UV_TORCH_BACKEND": "cu130"}) == "cu130"


@pytest.mark.skipif(not POSIX_SH, reason="POSIX sh installer")
def test_sh_without_nvidia_smi_keeps_auto():
    assert _sh("torch_backend", {"PATH": _bin_without_nvidia_smi()}) == "auto"


@pytest.mark.skipif(not POSIX_SH, reason="POSIX sh installer")
def test_sh_gpu_check_computes_on_cuda0():
    pyp = _fake_torch_dir()
    for state, expected in [("none", "none"), ("failed", "failed"), ("ok", "ok NVIDIA Fake GPU")]:
        got = _sh(f'gpu_state "{sys.executable}"', {"PYTHONPATH": pyp, "FAKE_TORCH": state})
        assert got == expected, (state, got)
    # a Python that can't import torch at all reports no GPU, not a failure
    assert _sh('gpu_state "/nonexistent/python"') == "none"


FAKE_UV = """#!/bin/sh
echo "uv $*" >> "$UV_LOG"
case "$1 $2" in
  "--version ") echo "uv 0.13.7" ;;
  "tool install")
      for a in "$@"; do case "$prev" in --torch-backend) echo "$a" > "$UV_STATE" ;; esac; prev=$a; done ;;
  "tool dir")
      if [ "${3:-}" = "--bin" ]; then echo "$UV_HOME/bin"; else echo "$UV_HOME/tools"; fi ;;
esac
exit 0
"""

# The tool environment's python: answers the version and shortcut questions,
# and runs the GPU check with the fake torch -- which fails only while the
# installed build is not the CPU one.
FAKE_TOOL_PY = """#!/bin/sh
case "$*" in
  *importlib.metadata*) echo 1.6.0 ;;
  *pinnstudio.shortcut*) exit 1 ;;
  *torch*)
      st=$FAKE_TORCH
      [ "$(cat "$UV_STATE" 2>/dev/null)" = cpu ] && st=none
      FAKE_TORCH=$st PYTHONPATH="$FAKE_TORCH_DIR" exec "$REAL_PY" "$@" ;;
esac
"""


def _run_installer(fake_torch, caps="6.1", cuda="13.0", extra_env=None):
    home = tempfile.mkdtemp()
    os.makedirs(os.path.join(home, "bin"))
    tool_bin = os.path.join(home, "tools", "pinnstudio", "bin")
    os.makedirs(tool_bin)
    fakes = _fake_smi_dir()
    _exe(os.path.join(fakes, "uv"), FAKE_UV)
    _exe(os.path.join(tool_bin, "python"), FAKE_TOOL_PY)
    env = {**os.environ, "PATH": fakes + os.pathsep + os.environ["PATH"], "HOME": home,
           "UV_HOME": home, "UV_LOG": os.path.join(home, "uv.log"), "UV_STATE": os.path.join(home, "state"),
           "FAKE_CAPS": caps, "FAKE_CUDA": cuda, "FAKE_TORCH": fake_torch,
           "FAKE_TORCH_DIR": _fake_torch_dir(), "REAL_PY": sys.executable,
           "PINNSTUDIO_SOURCE": "pinnstudio"}
    env.pop("UV_TORCH_BACKEND", None)
    env.update(extra_env or {})
    r = subprocess.run(["sh", SH], capture_output=True, text=True, env=env, timeout=120)
    log = open(env["UV_LOG"]).read()
    return r, [line for line in log.splitlines() if line.startswith("uv tool install") or
               line.startswith("uv tool uninstall")]


@pytest.mark.skipif(not POSIX_SH, reason="POSIX sh installer")
def test_sh_installer_older_gpu_that_computes():
    r, calls = _run_installer("ok")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(calls) == 1 and "--torch-backend cu126" in calls[0]
    assert "installing the PyTorch build that supports it (cu126)" in r.stdout
    assert "PyTorch will train on the GPU: NVIDIA Fake GPU" in r.stdout


@pytest.mark.skipif(not POSIX_SH, reason="POSIX sh installer")
def test_sh_installer_falls_back_to_cpu_when_the_gpu_cannot_compute():
    r, calls = _run_installer("failed", caps="8.9")     # e.g. a build without kernels: auto, then fails
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--torch-backend auto" in calls[0]
    assert calls[1].startswith("uv tool uninstall pinnstudio")
    assert "--torch-backend cpu" in calls[2] and len(calls) == 3
    assert "cannot compute on this computer's GPU" in r.stdout
    assert "PINNStudio will train on the CPU" in r.stdout
    assert "PyTorch will train on the GPU" not in r.stdout


@pytest.mark.skipif(not POSIX_SH, reason="POSIX sh installer")
def test_sh_installer_respects_uv_torch_backend():
    r, calls = _run_installer("ok", extra_env={"UV_TORCH_BACKEND": "cu130"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--torch-backend cu130" in calls[0] and len(calls) == 1
    assert "you set with UV_TORCH_BACKEND: cu130" in r.stdout


@pytest.mark.skipif(not POSIX_SH, reason="POSIX sh installer")
def test_sh_installer_no_gpu_keeps_auto_and_does_not_reinstall():
    r, calls = _run_installer("none", caps="8.6")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(calls) == 1 and "--torch-backend auto" in calls[0]
    assert "cannot compute" not in r.stdout


# ---- get-pinnstudio.ps1 --------------------------------------------------

@pytest.mark.skipif(not PWSH, reason="PowerShell (pwsh) not installed")
def test_ps1_parses():
    out = _ps('$e = $null; $null = [System.Management.Automation.Language.Parser]::ParseFile('
              f'"{PS1}", [ref]$null, [ref]$e); "errors=" + @($e).Count')
    assert out.splitlines()[-1] == "errors=0", out


@pytest.mark.skipif(not PWSH, reason="PowerShell (pwsh) not installed")
def test_ps1_choice_for_gpu_and_driver():
    calls = "; ".join(f"Select-TorchBackend {_ps_str(c)} {_ps_str(d)}" for c, d, _ in CASES)
    got = _ps(calls).splitlines()
    assert got == [e for _, _, e in CASES], list(zip(CASES, got))


@pytest.mark.skipif(not PWSH, reason="PowerShell (pwsh) not installed")
def test_ps1_detection_from_nvidia_smi():
    d = _fake_smi_dir()
    path = d + os.pathsep + os.environ["PATH"]
    assert _ps("Get-TorchBackend", {"PATH": path, "FAKE_CAPS": "6.1", "FAKE_CUDA": "13.0"}) == "cu126"
    assert _ps("Get-TorchBackend", {"PATH": path, "FAKE_CAPS": "6.1", "FAKE_CUDA": "11.8"}) == "cu118"
    assert _ps("Get-TorchBackend", {"PATH": path, "FAKE_CAPS": "8.9", "FAKE_CUDA": "13.0"}) == "auto"
    assert _ps("Get-TorchBackend", {"PATH": path, "FAKE_CAPS": "6.1", "FAKE_CUDA": "13.0",
                                    "UV_TORCH_BACKEND": "cu130"}) == "cu130"
    assert _ps("Get-TorchBackend", {"PATH": tempfile.mkdtemp()}) == "auto"     # no nvidia-smi


@pytest.mark.skipif(not PWSH, reason="PowerShell (pwsh) not installed")
def test_ps1_gpu_check_computes_on_cuda0():
    pyp = _fake_torch_dir()
    for state, expected in [("none", "none"), ("failed", "failed"), ("ok", "ok NVIDIA Fake GPU")]:
        got = _ps(f'Get-GpuState "{sys.executable}"', {"PYTHONPATH": pyp, "FAKE_TORCH": state})
        assert got == expected, (state, got)
