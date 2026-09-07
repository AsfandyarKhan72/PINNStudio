@echo off
setlocal EnableDelayedExpansion

echo Setting up PINNStudio...
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo Installing PINNStudio and its dependencies...
venv\Scripts\pip.exe install . --quiet

echo Checking for a compatible GPU...
for /f "delims=" %%T in ('venv\Scripts\python.exe "%~dp0select_torch_index.py"') do set TORCH_TAG=%%T

echo Installing the matching PyTorch build ^(!TORCH_TAG!^)...
set _TORCH_OK=0
venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/!TORCH_TAG! --force-reinstall --quiet
if not errorlevel 1 (
    if "!TORCH_TAG!"=="cpu" (
        set _TORCH_OK=1
    ) else (
        venv\Scripts\python.exe -c "import torch; assert '+!TORCH_TAG!' in torch.__version__" 2>nul
        if not errorlevel 1 set _TORCH_OK=1
    )
)
if "!_TORCH_OK!"=="0" (
    if not "!TORCH_TAG!"=="cpu" (
        echo Installed PyTorch build doesn't match the requested CUDA version - falling back to CPU...
        echo ^(You can retry GPU setup later - see the README's GPU troubleshooting section.^)
    )
    venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall --quiet
)

echo.
echo Done! Launch PINNStudio with:
echo   .\venv\Scripts\pinnstudio.exe
pause
