@echo off
echo Setting up PINNStudio...
python -m venv venv
venv\Scripts\pip.exe install --upgrade pip --quiet
venv\Scripts\pip.exe install . --quiet

echo Checking GPU support...
venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())" > gpu_check.tmp
set /p GPU_OK=<gpu_check.tmp
del gpu_check.tmp

where nvidia-smi >nul 2>nul
if %ERRORLEVEL%==0 if "%GPU_OK%"=="False" (
    echo NVIDIA GPU found but not usable with the default PyTorch build - installing a broadly compatible version instead...
    venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
)

echo.
echo Done! Launch PINNStudio with:
echo   venv\Scripts\pinnstudio.exe
pause
