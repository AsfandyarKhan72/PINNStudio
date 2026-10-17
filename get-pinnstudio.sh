#!/bin/sh
# Install PINNStudio on macOS or Linux with one command:
#
#   curl -LsSf https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/main/get-pinnstudio.sh | sh
#
# Python and git do not need to be installed. The script uses uv
# (https://docs.astral.sh/uv/) to install PINNStudio with its own Python and
# the PyTorch build that matches this computer's GPU driver (CPU-only if there
# is no NVIDIA GPU), puts the `pinnstudio` command on the PATH, and adds
# PINNStudio to the app menu (Linux) or ~/Applications (macOS).
# Running it again updates PINNStudio. To uninstall:
#   pinnstudio --remove-shortcut && uv tool uninstall pinnstudio
#
# PINNSTUDIO_SOURCE installs from somewhere else than PyPI, e.g. a local
# checkout or a wheel file.
set -eu

PKG="${PINNSTUDIO_SOURCE:-pinnstudio}"
UV_MIN="0.9.20"     # first uv release with --torch-backend for `uv tool install`

say() { printf '%s\n' "$*"; }

version_ge() {      # version_ge A B: is A >= B?
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -t. -k1,1n -k2,2n -k3,3n | head -n1)" = "$2" ]
}

if command -v uv >/dev/null 2>&1; then
    UV_VER=$(uv --version | awk '{print $2}')
    if ! version_ge "$UV_VER" "$UV_MIN"; then
        say "Updating uv ($UV_VER is older than $UV_MIN)..."
        uv self update || { say "Could not update uv. Update it (see https://docs.astral.sh/uv/) and run this again."; exit 1; }
    fi
else
    say "Installing uv, which installs PINNStudio and its own Python..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
    export PATH
fi

say ""
say "Installing PINNStudio and the PyTorch build for this computer (this takes a few minutes)..."
uv tool install --upgrade --python 3.12 --torch-backend auto "$PKG"

uv tool update-shell >/dev/null 2>&1 || true
BIN="$(uv tool dir --bin)"
TOOL_PY="$(uv tool dir)/pinnstudio/bin/python"

say ""
"$BIN/pinnstudio" --create-shortcut || say "(The shortcut could not be created; PINNStudio still starts with the pinnstudio command.)"

GPU=$("$TOOL_PY" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>/dev/null || true)
say ""
if [ -n "$GPU" ]; then
    say "PyTorch will train on the GPU: $GPU"
elif command -v nvidia-smi >/dev/null 2>&1; then
    say "An NVIDIA GPU was found, but PyTorch cannot use it; PINNStudio will train on the CPU."
    say "See 'GPU troubleshooting' in the README: https://github.com/AsfandyarKhan72/PINNStudio#installation"
else
    say "No NVIDIA GPU found: PINNStudio will train on the CPU."
fi
say ""
say "Done. Open PINNStudio from its icon, or run: pinnstudio"
say "(If the pinnstudio command is not found, open a new terminal first.)"
