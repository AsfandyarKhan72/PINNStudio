#!/usr/bin/env bash
# Build the Windows program and installer from WSL. PyInstaller builds only for the system it runs on, so
# this copies the source to a folder on a Windows drive and runs packaging\windows\build_windows.bat there
# with the Windows Python (3.12) and Inno Setup 6.
#
#   packaging/windows/build_windows_wsl.sh
#
#   BUILD_DIR_WIN   the Windows build folder (default D:\pinnstudio_build); a venv and pip's cache stay there
#   REF             the commit to build (default HEAD), or worktree: the checkout's tracked files as they are
#
# PINNSTUDIO_INSTALL_CHECK=1 (with WSLENV=PINNSTUDIO_INSTALL_CHECK) adds the install check.
set -euo pipefail

BUILD_DIR_WIN="${BUILD_DIR_WIN:-D:\\pinnstudio_build}"
REF="${REF:-HEAD}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR_WSL="$(wslpath -u "$BUILD_DIR_WIN")"
# a folder on a Windows drive, never a drive's root: its old source is deleted below
[[ "$BUILD_DIR_WSL" =~ ^/mnt/[a-z]/[^/]+ ]] || { echo "BUILD_DIR_WIN must be a folder on a Windows drive" >&2; exit 2; }

# What the build uses, nothing more: the benchmark records hold symbolic links, which a Windows drive cannot
# take, and the docs and benchmarks are not part of the program
PARTS=(pinnstudio reference_data packaging LICENSE LICENSES THIRD_PARTY_LICENSES.md README.md setup.py
       pyproject.toml MANIFEST.in constraints.txt requirements.txt)

mkdir -p "$BUILD_DIR_WSL/build"
# the source only (git leaves out what it ignores: local examples, results, caches); build\ keeps the venv,
# pip's cache and PyInstaller's work folder between builds
find "$BUILD_DIR_WSL" -mindepth 1 -maxdepth 1 ! -name build ! -name dist -exec rm -rf {} +
if [[ "$REF" == worktree ]]; then
  echo "[source] the checkout, $(git -C "$REPO" describe --tags --always --dirty) -> $BUILD_DIR_WIN"
  git -C "$REPO" ls-files -z -- "${PARTS[@]}" | tar -C "$REPO" --null -T - --ignore-failed-read -c 2>/dev/null \
    | tar -x -C "$BUILD_DIR_WSL"
else
  echo "[source] $(git -C "$REPO" describe --tags --always "$REF") -> $BUILD_DIR_WIN"
  git -C "$REPO" archive "$REF" -- "${PARTS[@]}" | tar -x -C "$BUILD_DIR_WSL"
fi
sed -i 's/\r*$/\r/' "$BUILD_DIR_WSL/packaging/windows/build_windows.bat"     # cmd.exe reads CRLF

cd "$BUILD_DIR_WSL"     # cmd.exe cannot start in a WSL folder
cmd.exe /c "packaging\\windows\\build_windows.bat"
