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

function Fail($msg) { Write-Host $msg -ForegroundColor Red; exit 1 }

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

Write-Host ""
Write-Host "Installing PINNStudio and the PyTorch build for this computer (this takes a few minutes)..."
uv tool install --upgrade --reinstall-package pinnstudio --python 3.12 --torch-backend auto $pkg
if ($LASTEXITCODE -ne 0) { Fail "The installation failed (see the messages above)." }

uv tool update-shell *> $null
$bin = (uv tool dir --bin).Trim()
$toolPy = Join-Path (uv tool dir).Trim() "pinnstudio\Scripts\python.exe"

Write-Host ""
& (Join-Path $bin "pinnstudio.exe") --create-shortcut
if ($LASTEXITCODE -ne 0) { Write-Host "(The shortcut could not be created; PINNStudio still starts with the pinnstudio command.)" }

$gpu = & $toolPy -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>$null
Write-Host ""
if ($gpu) {
    Write-Host "PyTorch will train on the GPU: $gpu"
} elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "An NVIDIA GPU was found, but PyTorch cannot use it; PINNStudio will train on the CPU."
    Write-Host "See 'GPU troubleshooting' in the README: https://github.com/AsfandyarKhan72/PINNStudio#installation"
} else {
    Write-Host "No NVIDIA GPU found: PINNStudio will train on the CPU."
}
Write-Host ""
Write-Host "Done. Open PINNStudio from the Start menu or the desktop icon, or run: pinnstudio"
Write-Host "(The pinnstudio command works in new PowerShell windows. In this one, run: & '$(Join-Path $bin 'pinnstudio.exe')')"
