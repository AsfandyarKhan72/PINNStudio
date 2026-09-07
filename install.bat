@echo off
echo Setting up PINNStudio...
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip --quiet
echo Checking for a compatible GPU...
for /f "delims=" %%T in ('venv\Scripts\python.exe "%~dp0select_torch_index.py"') do set TORCH_TAG=%%T
if "%TORCH_TAG%"=="cpu" (
    echo No compatible NVIDIA GPU/driver detected - installing CPU-only PyTorch...
    venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu --quiet
) else (
    echo NVIDIA GPU detected - installing a PyTorch build matching your driver ^(%TORCH_TAG%^)...
    venv\Scripts\pip.exe install torch --extra-index-url https://download.pytorch.org/whl/%TORCH_TAG% --quiet
    if errorlevel 1 (
        echo GPU-matched PyTorch install failed - falling back to the default CPU build so setup can still finish...
        echo ^(You can retry GPU setup later - see the README's GPU troubleshooting section.^)
        venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu --quiet
    )
)
echo Installing PINNStudio and remaining dependencies...
venv\Scripts\pip.exe install . --quiet
echo.
echo Done! Launch PINNStudio with:
echo   .\venv\Scripts\pinnstudio.exe
pause
