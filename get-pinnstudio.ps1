# Install PINNStudio on Windows with one command (in PowerShell):
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/main/get-pinnstudio.ps1 | iex"
#
# Python and git do not need to be installed. The script uses uv
# (https://docs.astral.sh/uv/) to install PINNStudio with its own Python and
# the PyTorch build that matches this computer's GPU driver (CPU-only if there
# is no NVIDIA GPU), puts the `pinnstudio` command on the PATH, and adds
# PINNStudio to the Start menu and the desktop.
# Running it again updates PINNStudio. To uninstall:
#   pinnstudio --remove-shortcut; uv tool uninstall pinnstudio
#
# $env:PINNSTUDIO_SOURCE installs from somewhere else than PyPI, e.g. a local
# checkout or a wheel file.

$ErrorActionPreference = "Continue"
$pkg = if ($env:PINNSTUDIO_SOURCE) { $env:PINNSTUDIO_SOURCE } else { "pinnstudio" }
$uvMin = [version]"0.9.20"   # first uv release with --torch-backend for `uv tool install`

# throw, not exit: run as `irm ... | iex`, exit would close the user's PowerShell window
function Fail($msg) { Write-Host $msg -ForegroundColor Red; throw $msg }

# Which PyTorch build to install. uv's `--torch-backend auto` chooses it from
# the NVIDIA driver alone, so with a current driver it installs a CUDA 12.8+
# or 13 build, which has no kernels for Maxwell, Pascal or Volta GPUs
# (GTX 9xx/10xx, P100, V100). For those GPUs the build is chosen here from the
# driver's CUDA version: cu126 (CUDA 12.x or newer) or cu118 (CUDA 11.8); with
# an older driver, a GPU older than Maxwell, or a Blackwell GPU next to an
# older one (no build runs on both), the CPU build. Everything else -- Turing
# and newer, no NVIDIA GPU, GPU information that can't be read -- keeps
# `auto`. A UV_TORCH_BACKEND you set yourself is used as it is.
# (pinnstudio/core/torch_index.py has the same rules for install.bat.)
function Select-TorchBackend([string]$caps, [string]$cuda) {
    $list = @($caps -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    if ($list.Count -eq 0) { return "auto" }
    $v = @()
    foreach ($c in $list) {
        if ($c -notmatch '^\d+\.\d+$') { return "auto" }   # "[N/A]", an error message, ...: leave it to uv
        $v += [version]$c
    }
    $sorted = @($v | Sort-Object)
    $old = $sorted[0]; $new = $sorted[-1]
    if ($old -ge [version]"7.5") { return "auto" }
    if (($old -lt [version]"5.0") -or ($new -ge [version]"10.0")) { return "cpu" }
    if ($cuda -notmatch '^\d+\.\d+$') { return "cpu" }
    $d = [version]$cuda
    if ($d -ge [version]"12.0") { return "cu126" }
    if ($d -ge [version]"11.8") { return "cu118" }
    return "cpu"
}

function Get-TorchBackend {
    if ($env:UV_TORCH_BACKEND) { return $env:UV_TORCH_BACKEND }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return "auto" }
    $caps = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null) -join "`n"
    if ($LASTEXITCODE -ne 0) { $caps = "" }
    $cuda = ""
    $out = (& nvidia-smi 2>$null) -join "`n"
    if ($out -match 'CUDA (UMD )?Version:\s*(\d+\.\d+)') { $cuda = $Matches[2] }
    return (Select-TorchBackend $caps $cuda)
}

# "ok <GPU name>" if PyTorch computes on the default GPU (cuda:0), which
# PINNStudio trains on; "failed" if it sees the GPU but can't compute on it
# (a build without kernels for it); "none" if it sees no GPU.
function Get-GpuState($py) {
    $check = Join-Path ([IO.Path]::GetTempPath()) "pinnstudio_gpu_check.py"
    Set-Content -Path $check -Encoding ascii -Value @'
import torch
if not torch.cuda.is_available():
    print('none')
else:
    try:
        x = torch.arange(8, dtype=torch.float32, device='cuda:0')
        ok = float((x * 2).sum().cpu()) == 56.0
    except Exception:
        ok = False
    print('ok ' + torch.cuda.get_device_name(0) if ok else 'failed')
'@
    $state = (& $py $check 2>$null | Select-Object -Last 1)
    Remove-Item $check -ErrorAction SilentlyContinue
    if ($state) { return "$state".Trim() } else { return "none" }
}

# tests dot-source the functions above without installing anything
if ($env:PINNSTUDIO_INSTALLER_FUNCTIONS_ONLY -eq "1") { return }

if (Get-Command uv -ErrorAction SilentlyContinue) {
    $uvVer = [version]((uv --version).Split(" ")[1])
    if ($uvVer -lt $uvMin) {
        Write-Host "Updating uv ($uvVer is older than $uvMin)..."
        uv self update
        if ($LASTEXITCODE -ne 0) { Fail "Could not update uv. Update it (see https://docs.astral.sh/uv/) and run this again." }
    }
} else {
    Write-Host "Installing uv, which installs PINNStudio and its own Python..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "uv could not be installed." }
}

$torchBackend = Get-TorchBackend
Write-Host ""
if ($env:UV_TORCH_BACKEND) {
    Write-Host "Using the PyTorch build you set with UV_TORCH_BACKEND: $torchBackend"
} elseif ($torchBackend -ne "auto") {
    Write-Host "An older NVIDIA GPU was found: installing the PyTorch build that supports it ($torchBackend)."
}
Write-Host "Installing PINNStudio and the PyTorch build for this computer (this takes a few minutes)..."
# PyTorch has no build for ARM Windows: on an ARM PC, use the x64 Python,
# which Windows runs under emulation.
$py = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "cpython-3.12-windows-x86_64-none" } else { "3.12" }
uv tool install --compile-bytecode --upgrade --reinstall-package pinnstudio --refresh-package pinnstudio --python $py --torch-backend $torchBackend $pkg
if ($LASTEXITCODE -ne 0) { Fail "The installation failed (see the messages above)." }

uv tool update-shell *> $null
$bin = (uv tool dir --bin).Trim()
$toolPy = Join-Path (uv tool dir).Trim() "pinnstudio\Scripts\python.exe"

# A PyTorch that sees the GPU but can't compute on it would stop at the first
# training step, so install the CPU build instead.
$gpuState = Get-GpuState $toolPy
$cpuFallback = $false
if ($gpuState -eq "failed") {
    Write-Host ""
    Write-Host "The PyTorch build that was installed ($torchBackend) cannot compute on this computer's GPU."
    Write-Host "Installing the CPU-only build instead; PINNStudio will train on the CPU..."
    uv tool uninstall pinnstudio *> $null
    uv tool install --compile-bytecode --refresh-package pinnstudio --python $py --torch-backend cpu $pkg
    if ($LASTEXITCODE -ne 0) { Fail "The installation failed (see the messages above)." }
    $cpuFallback = $true
    $gpuState = "none"
}
$version = & $toolPy -c "import importlib.metadata as m; print(m.version('pinnstudio'))" 2>$null

Write-Host ""
& $toolPy -c "import pinnstudio.shortcut" 2>$null
$hasShortcut = ($LASTEXITCODE -eq 0)
if ($hasShortcut) {
    & (Join-Path $bin "pinnstudio.exe") --create-shortcut
    if ($LASTEXITCODE -ne 0) { Write-Host "(The shortcut could not be created; PINNStudio still starts with the pinnstudio command.)" }
} else {   # older versions (1.4.0 on PyPI) have no --create-shortcut and would open the window instead
    Write-Host "(This PINNStudio version cannot create a desktop shortcut; start it with the pinnstudio command.)"
}

$gpu = if ($gpuState -like "ok *") { $gpuState.Substring(3) } else { "" }
Write-Host ""
if ($gpu) {
    Write-Host "PyTorch will train on the GPU: $gpu"
} elseif ($cpuFallback) {
    Write-Host "No PyTorch build that was tried can compute on this GPU, so PINNStudio will train on the CPU."
    Write-Host "See 'Common installation problems' in the installation guide: https://asfandyarkhan72.github.io/PINNStudio/getting-started/installation/#common-installation-problems"
} elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "An NVIDIA GPU was found, but PyTorch cannot use it; PINNStudio will train on the CPU."
    Write-Host "See 'Common installation problems' in the installation guide: https://asfandyarkhan72.github.io/PINNStudio/getting-started/installation/#common-installation-problems"
} else {
    Write-Host "No NVIDIA GPU found: PINNStudio will train on the CPU."
}
Write-Host ""
if ($version) { Write-Host "PINNStudio $version installed." }
if ($hasShortcut) { Write-Host "Done. Open PINNStudio from the Start menu or the desktop icon, or run: pinnstudio" }
else { Write-Host "Done. Start PINNStudio with: pinnstudio" }
Write-Host "(The pinnstudio command works in new PowerShell windows. In this one, run: & '$(Join-Path $bin 'pinnstudio.exe')')"
# Another PINNStudio (an older pip, venv or conda install) found first on the
# PATH would start instead of this one. Say where it is, which version, and
# how to remove it; nothing is removed here, it belongs to the user's own
# Python environment.
function Get-OtherPython($exe) {   # the python.exe next to a pinnstudio.exe (a venv) or one folder up (Scripts\)
    $dir = Split-Path $exe -Parent
    foreach ($c in @((Join-Path $dir "python.exe"), (Join-Path (Split-Path $dir -Parent) "python.exe"))) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Show-OtherCopy($otherPath, $thisPath) {
    if (-not $otherPath -or ($otherPath -eq $thisPath)) { return }
    $opy = Get-OtherPython $otherPath
    $over = ""
    if ($opy) { $over = (& $opy -c "import importlib.metadata as m; print(m.version('pinnstudio'))" 2>$null) }
    $label = if ($over) { " ($over)" } else { "" }
    Write-Host ""
    Write-Host "------------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "Another PINNStudio$label is installed, and typing pinnstudio starts it:" -ForegroundColor Yellow
    Write-Host "  $otherPath"
    if ($opy) {
        Write-Host "To remove it (recommended), run:"
        Write-Host "  & '$opy' -m pip uninstall pinnstudio"
    } else {
        Write-Host "Remove it with the pip of the Python it was installed in (pip uninstall pinnstudio)."
    }
    Write-Host "The Start menu and desktop icons already open the version installed now; so does: & '$thisPath'"
    Write-Host "------------------------------------------------------------------" -ForegroundColor Yellow
}

$other = Get-Command pinnstudio -ErrorAction SilentlyContinue | Select-Object -First 1
if ($other) { Show-OtherCopy $other.Source (Join-Path $bin "pinnstudio.exe") }
