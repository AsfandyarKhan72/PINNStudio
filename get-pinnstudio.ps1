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
# PyTorch has no build for ARM Windows: on an ARM PC, use the x64 Python,
# which Windows runs under emulation.
$py = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "cpython-3.12-windows-x86_64-none" } else { "3.12" }
uv tool install --compile-bytecode --upgrade --reinstall-package pinnstudio --refresh-package pinnstudio --python $py --torch-backend auto $pkg
if ($LASTEXITCODE -ne 0) { Fail "The installation failed (see the messages above)." }

uv tool update-shell *> $null
$bin = (uv tool dir --bin).Trim()
$toolPy = Join-Path (uv tool dir).Trim() "pinnstudio\Scripts\python.exe"
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

$gpu = & $toolPy -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>$null
Write-Host ""
if ($gpu) {
    Write-Host "PyTorch will train on the GPU: $gpu"
} elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "An NVIDIA GPU was found, but PyTorch cannot use it; PINNStudio will train on the CPU."
    Write-Host "See 'Troubleshooting' under Installation in the README: https://github.com/AsfandyarKhan72/PINNStudio#installation"
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
