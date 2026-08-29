@echo off
echo Setting up PINNStudio...
python -m venv venv
venv\Scripts\pip.exe install --upgrade pip --quiet

echo Checking for a compatible GPU...
where nvidia-smi >nul 2>nul
if %ERRORLEVEL%==0 (
    echo NVIDIA GPU detected - installing a broadly compatible PyTorch build (CUDA 12.1) first...
    venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo No NVIDIA GPU detected - installing CPU-only PyTorch...
    venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu --quiet
)

echo Installing PINNStudio and remaining dependencies...
venv\Scripts\pip.exe install . --quiet

echo Creating a desktop shortcut...
set SCRIPT="%TEMP%\PINNStudioShortcut.vbs"
echo Set oWS = WScript.CreateObject("WScript.Shell") > %SCRIPT%
echo sLinkFile = oWS.SpecialFolders("Desktop") ^& "\PINNStudio.lnk" >> %SCRIPT%
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> %SCRIPT%
echo oLink.TargetPath = "%CD%\venv\Scripts\pinnstudio.exe" >> %SCRIPT%
echo oLink.WorkingDirectory = "%CD%" >> %SCRIPT%
echo oLink.Description = "PINNStudio" >> %SCRIPT%
echo oLink.Save >> %SCRIPT%
cscript /nologo %SCRIPT%
del %SCRIPT%
echo Desktop shortcut created.

echo.
echo Done! Launch PINNStudio with:
echo   venv\Scripts\pinnstudio.exe
echo (or just double-click the new PINNStudio shortcut on your Desktop)
pause
