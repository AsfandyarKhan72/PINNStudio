#!/usr/bin/env bash
set -e

echo "Setting up PINNStudio..."
python3 -m venv venv
./venv/bin/pip install --upgrade pip --quiet
./venv/bin/pip install . --quiet

echo "Checking GPU support..."
GPU_OK=$(./venv/bin/python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$GPU_OK" = "False" ] && command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU found but not usable with the default PyTorch build — installing a broadly compatible version instead..."
    ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
fi

echo ""
echo "Done! Launch PINNStudio with:"
echo "  ./venv/bin/pinnstudio"
