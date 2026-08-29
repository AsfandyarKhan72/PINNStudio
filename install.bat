@echo off
echo Setting up PINNStudio...
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip --quiet
echo Checking for a compatible GPU...
where nvidia-smi >nul 2>nul
if %ERRORLEVEL%==0 (
    echo NVIDIA GPU detected - installing a broadly compatible PyTorch build ^(CUDA 12.1^) first...
    venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo No NVIDIA GPU detected - installing CPU-only PyTorch...
    venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu --quiet
)
echo Installing PINNStudio and remaining dependencies...
venv\Scripts\pip.exe install . --quiet
echo.
echo Done! Launch PINNStudio with:
echo   .\venv\Scripts\pinnstudio.exe
pause
