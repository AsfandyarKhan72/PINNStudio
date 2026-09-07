#!/usr/bin/env bash
set -e

echo "Setting up PINNStudio..."
python3 -m venv venv
./venv/bin/pip install --upgrade pip --quiet

echo "Checking for a compatible GPU..."
TORCH_TAG=$(./venv/bin/python "$(dirname "$0")/select_torch_index.py")

if [ "$TORCH_TAG" = "cpu" ]; then
    echo "No compatible NVIDIA GPU/driver detected — installing CPU-only PyTorch..."
    ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
else
    echo "NVIDIA GPU detected — installing a PyTorch build matching your driver (${TORCH_TAG})..."
    if ! ./venv/bin/pip install torch --extra-index-url "https://download.pytorch.org/whl/${TORCH_TAG}" --quiet; then
        echo "GPU-matched PyTorch install failed — falling back to the default CPU build so setup can still finish..."
        echo "(You can retry GPU setup later — see the README's GPU troubleshooting section.)"
        ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
    fi
fi

echo "Installing PINNStudio and remaining dependencies..."
./venv/bin/pip install . --quiet

echo ""
echo "Done! Launch PINNStudio with:"
echo "  ./venv/bin/pinnstudio"
