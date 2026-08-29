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

if [ "$(uname -s)" = "Linux" ]; then
    echo "Creating a desktop launcher..."
    INSTALL_DIR="$(pwd)"
    DESKTOP_FILE="$HOME/.local/share/applications/pinnstudio.desktop"
    mkdir -p "$HOME/.local/share/applications"
    cat > "$DESKTOP_FILE" << DESKTOPEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=PINNStudio
Comment=No-code GUI for Physics-Informed Neural Networks
Exec=$INSTALL_DIR/venv/bin/pinnstudio
Path=$INSTALL_DIR
Icon=applications-science
Terminal=false
Categories=Science;Education;
DESKTOPEOF
    chmod +x "$DESKTOP_FILE"
    if [ -d "$HOME/Desktop" ]; then
        cp "$DESKTOP_FILE" "$HOME/Desktop/PINNStudio.desktop"
        chmod +x "$HOME/Desktop/PINNStudio.desktop"
    fi
    echo "Desktop launcher created — look for 'PINNStudio' in your applications menu (and on your Desktop, if you have one)."
fi

echo ""
echo "Done! Launch PINNStudio with:"
echo "  ./venv/bin/pinnstudio"
