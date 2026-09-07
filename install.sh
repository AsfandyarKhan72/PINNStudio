#!/usr/bin/env bash
set -e

echo "Setting up PINNStudio..."
python3 -m venv venv
./venv/bin/pip install --upgrade pip --quiet

echo "Installing PINNStudio and its dependencies..."
./venv/bin/pip install . --quiet

echo "Checking for a compatible GPU..."
TORCH_TAG=$(./venv/bin/python "$(dirname "$0")/select_torch_index.py")

# Install the matching PyTorch build (a specific CUDA tag, or CPU-only) LAST,
# with --force-reinstall. The dependency resolution above may already have
# pulled in a default PyPI PyTorch build that doesn't match this machine's GPU
# driver (or unnecessarily bundles CUDA runtime packages on a CPU-only
# machine) -- doing this last, and verifying what actually landed, means
# nothing installed afterward can silently leave the wrong build in place.
echo "Installing the matching PyTorch build (${TORCH_TAG})..."
_torch_ok=0
if ./venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_TAG}" --force-reinstall --quiet; then
    if [ "$TORCH_TAG" = "cpu" ]; then
        _torch_ok=1
    elif ./venv/bin/python -c "import torch; assert '+${TORCH_TAG}' in torch.__version__" 2>/dev/null; then
        _torch_ok=1
    fi
fi
if [ "$_torch_ok" != "1" ]; then
    if [ "$TORCH_TAG" != "cpu" ]; then
        echo "Installed PyTorch build doesn't match the requested CUDA version — falling back to CPU..."
        echo "(You can retry GPU setup later — see the README's GPU troubleshooting section.)"
    fi
    ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall --quiet
fi

echo ""
echo "Done! Launch PINNStudio with:"
echo "  ./venv/bin/pinnstudio"
