@echo off
rem PINNStudio as a Windows program and installer. Run on Windows from the source folder's root, or from WSL
rem with packaging/windows/build_windows_wsl.sh. Makes, in dist\:
rem
rem   PINNStudio\PINNStudio.exe (with _internal\ beside it: Python, Qt, PyTorch, DeepXDE ...)
rem   PINNStudio-VERSION-setup.exe and PINNStudio-VERSION-setup.exe.sha256, when Inno Setup 6 is installed
rem
rem PYTHON  the Python 3.12 to build with (default: py -3.12, else %LOCALAPPDATA%\Programs\Python\Python312)
rem ISCC    Inno Setup's compiler (default: where winget or its installer puts it)
rem PINNSTUDIO_INSTALL_CHECK=1  also install the setup.exe silently into build\, start the installed program
rem                             off screen and uninstall it again
rem
rem The libraries are pinned to constraints.txt (the versions the tests passed with); PyTorch is the CPU build.
setlocal
cd /d %~dp0..\..
if not defined PYTHON (
  where py >nul 2>nul && (set "PYTHON=py -3.12") || (set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe")
)
rem pip's cache and every temporary file stay beside the build, not on C:
set "TEMP=%CD%\build\windows\tmp"
set "TMP=%TEMP%"
if not exist "%TEMP%" mkdir "%TEMP%"
if not defined PIP_CACHE_DIR set "PIP_CACHE_DIR=%CD%\build\windows\pip-cache"
set VENV=build\windows\venv
set PY=%VENV%\Scripts\python.exe
set PIP=%PY% -m pip --disable-pip-version-check -q
set DIST=dist\PINNStudio
set CONSTRAINTS=
if exist constraints.txt set "CONSTRAINTS=-c constraints.txt"

if not exist %PY% (
  %PYTHON% -m venv %VENV% || (
    echo Cannot make a venv with "%PYTHON%": set PYTHON to a Python 3.12, e.g. set PYTHON=C:\Python312\python.exe
    exit /b 1
  )
)
echo [1/5] Libraries
%PIP% install --upgrade pip || exit /b 1
rem PyTorch's CPU build first, from PyTorch's own index; pip then keeps it for PINNStudio
%PIP% install %CONSTRAINTS% torch --index-url https://download.pytorch.org/whl/cpu || exit /b 1
%PIP% install %CONSTRAINTS% . "pyinstaller>=6.10,<7" || exit /b 1
rem pip keeps an installed pinnstudio of the same version: a reused venv would bundle the old source
%PIP% install --force-reinstall --no-deps . || exit /b 1
%PY% packaging\windows\make_splash.py build\windows\splash.png || exit /b 1

echo [2/5] PyInstaller
rem --clean removes the last dist\PINNStudio, which fails while that program runs: close it first
%PY% -m PyInstaller --noconfirm --clean --log-level=WARN --workpath build\windows\work --distpath dist ^
  packaging\windows\pinnstudio.spec || exit /b 1

rem the checks show no splash screen and no window
set PYINSTALLER_SUPPRESS_SPLASH_SCREEN=1
echo [3/5] Build check: a training run, the scripts and the window, inside the program
%DIST%\PINNStudio.exe packaging\windows\check_build.py > build\windows\check.log 2>&1
if errorlevel 1 (
  type build\windows\check.log
  echo %DIST%\PINNStudio.exe failed packaging\windows\check_build.py
  exit /b 1
)
findstr /c:"build check passed" build\windows\check.log || exit /b 1

echo [4/5] Launch check: the window starts and runs for 8 s, off screen
set PINNSTUDIO_LAUNCH_CHECK=8
set QT_QPA_PLATFORM=offscreen
%DIST%\PINNStudio.exe > build\windows\launch.log 2>&1
set RC=%errorlevel%
set PINNSTUDIO_LAUNCH_CHECK=
set QT_QPA_PLATFORM=
if not "%RC%"=="0" (
  type build\windows\launch.log
  echo %DIST%\PINNStudio.exe did not start
  exit /b 1
)
findstr /c:"launch check passed" build\windows\launch.log || exit /b 1
echo %CD%\%DIST%\PINNStudio.exe

echo [5/5] Installer
if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC (
  echo No Inno Setup 6, so no installer: dist\PINNStudio\ only. winget install JRSoftware.InnoSetup, or set ISCC.
  exit /b 0
)
%PY% -c "import pinnstudio; print(pinnstudio.__version__)" > build\windows\version.txt || exit /b 1
set /p VERSION=<build\windows\version.txt
set SETUP=dist\PINNStudio-%VERSION%-setup.exe
rem up to three tries: a virus scanner can hold the new setup.exe while ISCC writes its resources
set TRIES=0
:compile
set /a TRIES+=1
"%ISCC%" /Q /DAppVersion=%VERSION% packaging\windows\pinnstudio.iss && goto compiled
if %TRIES% geq 3 exit /b 1
echo ISCC failed: trying again in 10 s
ping -n 11 127.0.0.1 >nul
goto compile
:compiled
rem its SHA-256 as sha256sum writes it (certutil's lines with a colon are its header and footer)
certutil -hashfile %SETUP% SHA256 | findstr /v ":" > build\windows\hash.txt || exit /b 1
set /p HASH=<build\windows\hash.txt
set "HASH=%HASH: =%"
(echo %HASH%  PINNStudio-%VERSION%-setup.exe)> %SETUP%.sha256
echo %CD%\%SETUP%
type %SETUP%.sha256
if not "%PINNSTUDIO_INSTALL_CHECK%"=="1" exit /b 0

rem The install check uses the program's own AppId: where PINNStudio is installed, it would take over that
rem installation's uninstall entry and then remove it. It refuses there (exit code 3).
set "UNINST=Software\Microsoft\Windows\CurrentVersion\Uninstall\{B993415D-A9F2-4A53-8B10-C30E7CA0E736}_is1"
reg query "HKCU\%UNINST%" >nul 2>nul && goto installed_here
reg query "HKLM\%UNINST%" /reg:64 >nul 2>nul && goto installed_here
goto install_check
:installed_here
echo PINNStudio is installed on this machine: the install check would take over its uninstall entry.
echo Uninstall it first, or install %SETUP% over it by hand instead.
exit /b 3
:install_check
echo Install check: install, start, uninstall
set "TARGET=%CD%\build\windows\installcheck"
if exist "%TARGET%" rmdir /s /q "%TARGET%"
%SETUP% /VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER /NORESTART /TASKS="" /DIR="%TARGET%" ^
  /LOG="%CD%\build\windows\install.log" || (
  echo the installer failed: build\windows\install.log
  exit /b 1
)
set PINNSTUDIO_LAUNCH_CHECK=8
set QT_QPA_PLATFORM=offscreen
"%TARGET%\PINNStudio.exe" > build\windows\installed.log 2>&1
set RC=%errorlevel%
set PINNSTUDIO_LAUNCH_CHECK=
set QT_QPA_PLATFORM=
"%TARGET%\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
rem the uninstaller copies itself to TEMP and returns at once: wait up to two minutes for it
set WAITED=0
:uninstalling
if not exist "%TARGET%\PINNStudio.exe" goto uninstalled
if %WAITED% geq 120 (
  echo the uninstaller did not remove %TARGET%\PINNStudio.exe
  exit /b 1
)
ping -n 2 127.0.0.1 >nul
set /a WAITED+=1
goto uninstalling
:uninstalled
if not "%RC%"=="0" (
  type build\windows\installed.log
  echo the installed PINNStudio.exe did not start
  exit /b 1
)
findstr /c:"launch check passed" build\windows\installed.log || exit /b 1
echo install check passed: installed, started, uninstalled (in %WAITED% s)
