#!/usr/bin/env bash
set -e

echo "Setting up PINNStudio..."
python3 -m venv venv
./venv/bin/pip install --upgrade pip --quiet

echo "Checking for a compatible GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected — installing a broadly compatible PyTorch build (CUDA 12.1) first..."
    ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
else
    echo "No NVIDIA GPU detected — installing CPU-only PyTorch..."
    ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
fi

echo "Installing PINNStudio and remaining dependencies..."
./venv/bin/pip install . --quiet

echo ""
echo "Done! Launch PINNStudio with:"
echo "  ./venv/bin/pinnstudio"
