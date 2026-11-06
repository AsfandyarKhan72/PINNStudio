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

# Which PyTorch build to install. uv's `--torch-backend auto` chooses it from
# the NVIDIA driver alone, so with a current driver it installs a CUDA 12.8+
# or 13 build, which has no kernels for Maxwell, Pascal or Volta GPUs
# (GTX 9xx/10xx, P100, V100). For those GPUs the build is chosen here from the
# driver's CUDA version: cu126 (CUDA 12.x or newer) or cu118 (CUDA 11.8); with
# an older driver, a GPU older than Maxwell, or a Blackwell GPU next to an
# older one (no build runs on both), the CPU build. Everything else -- Turing
# and newer, no NVIDIA GPU, GPU information that can't be read -- keeps
# `auto`. A UV_TORCH_BACKEND you set yourself is used as it is.
# (pinnstudio/core/torch_index.py has the same rules for install.sh.)
ver_lt() {          # ver_lt A B: is version A (major.minor) lower than B?
    awk -v a="$1" -v b="$2" 'BEGIN { split(a, x, "."); split(b, y, ".");
        exit !((x[1] + 0 < y[1] + 0) || (x[1] + 0 == y[1] + 0 && x[2] + 0 < y[2] + 0)) }'
}

choose_torch_backend() {    # choose_torch_backend <compute capabilities, one per line> <driver CUDA version>
    _old=""; _new=""
    for _c in $1; do
        case "$_c" in
            [0-9]*.[0-9]*) ;;
            *) echo auto; return ;;      # "[N/A]", an error message, ...: leave it to uv
        esac
        if [ -z "$_old" ] || ver_lt "$_c" "$_old"; then _old=$_c; fi
        if [ -z "$_new" ] || ver_lt "$_new" "$_c"; then _new=$_c; fi
    done
    if [ -z "$_old" ] || ! ver_lt "$_old" 7.5; then echo auto; return; fi
    if ver_lt "$_old" 5.0 || ! ver_lt "$_new" 10.0; then echo cpu; return; fi
    case "$2" in
        [0-9]*.[0-9]*) ;;
        *) echo cpu; return ;;
    esac
    if ! ver_lt "$2" 12.0; then echo cu126
    elif ! ver_lt "$2" 11.8; then echo cu118
    else echo cpu
    fi
}

torch_backend() {   # the --torch-backend value for this computer
    if [ -n "${UV_TORCH_BACKEND:-}" ]; then echo "$UV_TORCH_BACKEND"; return; fi
    command -v nvidia-smi >/dev/null 2>&1 || { echo auto; return; }
    _caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
    _cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\2/p' | head -n 1)
    choose_torch_backend "$_caps" "$_cuda"
}

# "ok <GPU name>" if PyTorch computes on the default GPU (cuda:0), which
# PINNStudio trains on; "failed" if it sees the GPU but can't compute on it
# (a build without kernels for it); "none" if it sees no GPU.
gpu_state() {
    "$1" -c "
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
" 2>/dev/null || echo none
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

TORCH_BACKEND=$(torch_backend)
say ""
if [ -n "${UV_TORCH_BACKEND:-}" ]; then
    say "Using the PyTorch build you set with UV_TORCH_BACKEND: $TORCH_BACKEND"
elif [ "$TORCH_BACKEND" != auto ]; then
    say "An older NVIDIA GPU was found: installing the PyTorch build that supports it ($TORCH_BACKEND)."
fi
say "Installing PINNStudio and the PyTorch build for this computer (this takes a few minutes)..."
uv tool install --compile-bytecode --upgrade --reinstall-package pinnstudio --refresh-package pinnstudio --python 3.12 --torch-backend "$TORCH_BACKEND" "$PKG"

uv tool update-shell >/dev/null 2>&1 || true
BIN="$(uv tool dir --bin)"
TOOL_PY="$(uv tool dir)/pinnstudio/bin/python"

# A PyTorch that sees the GPU but can't compute on it would stop at the first
# training step, so install the CPU build instead.
GPU_STATE=$(gpu_state "$TOOL_PY")
CPU_FALLBACK=""
if [ "$GPU_STATE" = failed ]; then
    say ""
    say "The PyTorch build that was installed ($TORCH_BACKEND) cannot compute on this computer's GPU."
    say "Installing the CPU-only build instead; PINNStudio will train on the CPU..."
    uv tool uninstall pinnstudio >/dev/null 2>&1 || true
    uv tool install --compile-bytecode --refresh-package pinnstudio --python 3.12 --torch-backend cpu "$PKG"
    CPU_FALLBACK=1
    GPU_STATE=none
fi
VERSION=$("$TOOL_PY" -c "import importlib.metadata as m; print(m.version('pinnstudio'))" 2>/dev/null || true)

say ""
WHERE="its icon in the app menu (Linux) or ~/Applications (macOS)"
if "$TOOL_PY" -c "import pinnstudio.shortcut" 2>/dev/null; then
  "$BIN/pinnstudio" --create-shortcut || say "(The shortcut could not be created; PINNStudio still starts with the pinnstudio command.)"
else   # older versions (1.4.0 on PyPI) have no --create-shortcut and would open the window instead
  say "(This PINNStudio version cannot create a desktop shortcut; start it with the pinnstudio command.)"
  WHERE=""
fi

say ""
case "$GPU_STATE" in
    ok\ *) GPU=${GPU_STATE#ok } ;;
    *) GPU="" ;;
esac
if [ -n "$GPU" ]; then
    say "PyTorch will train on the GPU: $GPU"
elif [ -n "$CPU_FALLBACK" ]; then
    say "No PyTorch build that was tried can compute on this GPU, so PINNStudio will train on the CPU."
    say "See 'Common installation problems' in the installation guide: https://asfandyarkhan72.github.io/PINNStudio/getting-started/installation/#common-installation-problems"
elif command -v nvidia-smi >/dev/null 2>&1; then
    say "An NVIDIA GPU was found, but PyTorch cannot use it; PINNStudio will train on the CPU."
    say "See 'Common installation problems' in the installation guide: https://asfandyarkhan72.github.io/PINNStudio/getting-started/installation/#common-installation-problems"
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
