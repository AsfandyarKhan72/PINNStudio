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

other_python() {    # the Python a pinnstudio command runs with
    _py=$(head -n 1 "$1" 2>/dev/null | sed -n 's/^#![[:space:]]*//p' | awk '{print $1}')
    case "$_py" in
        */python*) [ -x "$_py" ] && { printf '%s\n' "$_py"; return; } ;;
    esac
    for _c in "$(dirname "$1")/python" "$(dirname "$1")/python3"; do
        [ -x "$_c" ] && { printf '%s\n' "$_c"; return; }
    done
}

report_other_copy() {   # report_other_copy <path of the pinnstudio command found first> <this install's command>
    [ -n "$1" ] && [ "$1" != "$2" ] || return 0
    _opy=$(other_python "$1")
    _over=""
    [ -n "$_opy" ] && _over=$("$_opy" -c "import importlib.metadata as m; print(m.version('pinnstudio'))" 2>/dev/null || true)
    say ""
    say "------------------------------------------------------------------"
    say "Another PINNStudio${_over:+ ($_over)} is installed, and typing pinnstudio starts it:"
    say "  $1"
    if [ -n "$_opy" ]; then
        say "To remove it (recommended), run:"
        say "  \"$_opy\" -m pip uninstall pinnstudio"
    else
        say "Remove it with the pip of the Python it was installed in (pip uninstall pinnstudio)."
    fi
    say "The PINNStudio icon already opens the version installed now; so does $2"
    say "------------------------------------------------------------------"
}

# tests source the functions above without installing anything
[ "${PINNSTUDIO_INSTALLER_FUNCTIONS_ONLY:-}" = 1 ] && return 0 2>/dev/null

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
uv tool install --upgrade --reinstall-package pinnstudio --refresh-package pinnstudio --python 3.12 --torch-backend auto "$PKG"

uv tool update-shell >/dev/null 2>&1 || true
BIN="$(uv tool dir --bin)"
TOOL_PY="$(uv tool dir)/pinnstudio/bin/python"
VERSION=$("$TOOL_PY" -c "import importlib.metadata as m; print(m.version('pinnstudio'))" 2>/dev/null || true)

say ""
WHERE="its icon in the app menu (Linux) or ~/Applications (macOS)"
if "$TOOL_PY" -c "import pinnstudio.shortcut" 2>/dev/null; then
  "$BIN/pinnstudio" --create-shortcut || say "(The shortcut could not be created; PINNStudio still starts with the pinnstudio command.)"
else   # older versions (1.4.0 on PyPI) have no --create-shortcut and would open the window instead
  say "(This PINNStudio version cannot create a desktop shortcut; start it with the pinnstudio command.)"
  WHERE=""
fi

GPU=$("$TOOL_PY" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>/dev/null || true)
say ""
if [ -n "$GPU" ]; then
    say "PyTorch will train on the GPU: $GPU"
elif command -v nvidia-smi >/dev/null 2>&1; then
    say "An NVIDIA GPU was found, but PyTorch cannot use it; PINNStudio will train on the CPU."
    say "See 'GPU troubleshooting' in the README: https://github.com/AsfandyarKhan72/PINNStudio#installation"
elif [ "$(uname -s)" = "Darwin" ]; then
    say "PINNStudio trains on the CPU on a Mac (Apple's GPU was no faster for PINNs in our tests)."
else
    say "No NVIDIA GPU found: PINNStudio will train on the CPU."
fi
say ""
[ -n "$VERSION" ] && say "PINNStudio $VERSION installed."
if [ -n "$WHERE" ]; then say "Done. Open PINNStudio from $WHERE, or run: pinnstudio"; else say "Done. Start PINNStudio with: pinnstudio"; fi
say "(The pinnstudio command works in new terminals. In this one, run: $BIN/pinnstudio)"

# Another PINNStudio (an older pip, venv or conda install) found first on the
# PATH would start instead of this one. Say where it is, which version, and
# how to remove it; nothing is removed here, it belongs to the user's own
# Python environment.
report_other_copy "$(command -v pinnstudio 2>/dev/null || true)" "$BIN/pinnstudio"
