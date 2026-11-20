"""The entry point of the Windows program (PyInstaller, packaging/windows/pinnstudio.spec).

    PINNStudio.exe [problem.pinn.json]          the window
    PINNStudio.exe script.py [args]             run a Python script inside the program
    PINNStudio.exe check|export|run ...         the command line (pinnstudio.main)

The program holds its own Python, so it runs the training scripts PINNStudio writes itself:
pinnstudio.core.pyexe.script_python() is the program, and Solve starts `PINNStudio.exe script.py`.

PINNSTUDIO_LAUNCH_CHECK=SECONDS starts the window as usual and ends the program that many seconds after its
event loop starts, with exit code 0, or 1 and the traceback on stderr. A windowed program would otherwise show
a failed start in a dialog that waits for a click (packaging/windows/build_windows.bat)."""
import multiprocessing
import os
import sys

COMMANDS = ("check", "export", "run", "--version", "-h", "--help")


def _end_now(code):
    """End the process at once, without Python's clean-up or Qt's close events (so the launch check
    writes no settings). Windows: TerminateProcess, which unloads no libraries while threads may still run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except Exception:
            pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE          # a pseudo handle of -1: keep all its bits
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.TerminateProcess(k32.GetCurrentProcess(), code)
    os._exit(code)


def _close_splash():
    try:
        import pyi_splash      # only in a build with a splash screen
        pyi_splash.close()
    except Exception:
        pass


def run_script(argv):
    """Run argv[0], a Python script, as `python script.py args` would."""
    import runpy
    _close_splash()
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
            except Exception:
                pass
    sys.argv = list(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(argv[0])))
    try:
        runpy.run_path(argv[0], run_name="__main__")
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
    raise SystemExit(0)


def run_window(argv):
    from pinnstudio import main as _main
    check = os.environ.get("PINNSTUDIO_LAUNCH_CHECK")
    if check and sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    def started(app, window):
        _close_splash()
        if check:
            from PySide6.QtCore import QTimer

            def failed(exc_type, exc, tb):     # not the error dialog, which would wait for a click
                import traceback
                traceback.print_exception(exc_type, exc, tb)
                _end_now(1)
            sys.excepthook = failed

            def passed():
                sys.stderr.write(f"launch check passed: {type(window).__name__} shown\n")
                _end_now(0)
            QTimer.singleShot(int(float(check) * 1000), passed)

    try:
        return _main.run_gui(argv, on_started=started)
    except BaseException:
        if not check:
            raise
        import traceback
        traceback.print_exc()
        _end_now(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = sys.argv[1:]
    if args and args[0].lower().endswith(".py"):
        run_script(args)
    if args and args[0] in COMMANDS:
        _close_splash()
        from pinnstudio import main as _main
        _main.main(args)                    # exits
    raise SystemExit(run_window(args))
