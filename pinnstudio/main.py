"""PINNStudio's entry point: the window, or a command that needs no display.

    pinnstudio                               the window
    pinnstudio problem.pinn.json             the window, with that problem open
    pinnstudio --version                     the version
    pinnstudio export problem.pinn.json [-o script.py] [--full]
                                             write the standalone script (Export script), or Solve's own
    pinnstudio run problem.pinn.json [--save-dir DIR]
                                             train it as Solve does, printing the Training Log
    pinnstudio check problem.pinn.json       the settings that would stop Solve, or "OK"
    pinnstudio --create-shortcut | --remove-shortcut

The commands import no Qt, so they work on a server or a cluster node without a display.
"""
import sys


def _preload_torch_on_windows():
    """On Windows, PyTorch fails to load once Qt (seen with PyQt6) is loaded ("[WinError
    1114] A dynamic link library (DLL) initialization routine failed ...
    c10.dll"): both bring a Microsoft C++ runtime, and the first one loaded
    wins. Loading PyTorch first avoids it. Other systems are unaffected."""
    if sys.platform == "win32":
        try:
            import torch  # noqa: F401
        except Exception:
            pass


def _check_gpu_torch_mismatch():
    """Warn if an NVIDIA GPU is present but the installed PyTorch build cannot use it."""
    import shutil
    if not shutil.which("nvidia-smi"):
        return
    try:
        import torch
        if torch.cuda.is_available():
            return
    except Exception:
        pass
    from pinnstudio.core import torch_index
    print("=" * 70)
    print("An NVIDIA GPU was detected, but PyTorch cannot use it (PINNStudio runs on the CPU for now).")
    print("This happens with a CPU-only PyTorch, or one built for a newer CUDA than the driver supports.")
    for line in torch_index.gpu_fix_lines(torch_index.select_tag()):
        print(line)
    print("=" * 70)


def _version():
    from pinnstudio import __version__
    return __version__


def _parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="pinnstudio",
        description="PINNStudio: physics-informed neural networks. Without a command, the window opens.")
    p.add_argument("--version", action="version", version=f"PINNStudio {_version()}")
    sub = p.add_subparsers(dest="command")
    e = sub.add_parser("export", help="write the standalone DeepXDE script of a problem file")
    e.add_argument("problem", help="a .pinn.json problem file")
    e.add_argument("-o", "--output", help="the script to write (default: next to the problem, .py)")
    e.add_argument("--full", action="store_true",
                   help="Solve's own script (with PINNStudio's run folders and plots) instead of Export script's")
    r = sub.add_parser("run", help="train a problem file as Solve does, printing the Training Log")
    r.add_argument("problem", help="a .pinn.json problem file")
    r.add_argument("--save-dir", help="where the run folder goes (default: the problem's Save to folder)")
    r.add_argument("--trust-code", action="store_true",
                   help="run a designed network's own code and saved files without asking (this run only)")
    c = sub.add_parser("check", help="list the settings that would stop Solve")
    c.add_argument("problem", help="a .pinn.json problem file")
    return p


def _load(path, with_name=False):
    from pinnstudio.core.problem_file import ProblemFileError, read
    try:
        config, name, _data = read(path)
        return (config, name) if with_name else config
    except ProblemFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


def _errors(config):
    try:
        return [str(e) for e in (config.validate() or [])]
    except Exception as exc:
        return [f"the settings cannot be checked: {exc}"]


def cmd_check(args):
    errors = _errors(_load(args.problem))
    if errors:
        for e in errors:
            print(f"• {e}")
        return 1
    print("OK")
    return 0


def cmd_export(args):
    import os
    config = _load(args.problem)
    errors = _errors(config)
    if errors:
        print("The problem has settings that would stop Solve:", file=sys.stderr)
        for e in errors:
            print(f"• {e}", file=sys.stderr)
        return 1
    from pinnstudio.core.codegen import generate_clean_script, generate_script
    out = args.output or os.path.splitext(os.path.splitext(args.problem)[0])[0] + ".py"
    script = (generate_script if args.full else generate_clean_script)(config)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(script)
    print(f"Wrote {out}")
    return 0


def cmd_run(args):
    import dataclasses
    import os
    import tempfile
    import re
    from datetime import datetime
    config, name = _load(args.problem, with_name=True)
    # a folder of its own, named as the window names a run's: <problem>__<date>_<time>
    base = os.path.abspath(args.save_dir or config.save_dir or os.path.join(os.path.expanduser("~"),
                                                                            "PINNStudio_Results"))
    stem = re.sub(r"[^A-Za-z0-9.+-]+", "_", name).strip("_") or "problem"
    config = dataclasses.replace(config, save_dir=os.path.join(
        base, f"{stem}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"))
    errors = _errors(config)
    if errors:
        print("The problem has settings that would stop Solve:", file=sys.stderr)
        for e in errors:
            print(f"• {e}", file=sys.stderr)
        return 1
    if config.network_type == "Designed":
        from pinnstudio.core.netdesign import design as D
        from pinnstudio.core.netdesign import trust as T
        items = T.untrusted(D.check(config.network_design))
        if items and not args.trust_code:
            print("The designed network runs code or loads files that are not trusted:", file=sys.stderr)
            for it in items:
                print(f"• {it['name']} ({it['kind']}, SHA-256 {it['sha256'][:16]}…) {it['problem']}", file=sys.stderr)
            print("Trust them in the window, or pass --trust-code for this run.", file=sys.stderr)
            return 1
    from pinnstudio.core.runner import run_pinn
    print(f"Run folder: {config.save_dir}", flush=True)
    result = run_pinn(config, on_output=lambda line: print(line, flush=True),
                      scratch=tempfile.mkdtemp(prefix="pinnstudio_run_"))
    return 0 if result == "DONE" else 1


def run_gui(argv):
    _preload_torch_on_windows()
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    # errors are shown and logged instead of closing the window (see errors.py)
    from pinnstudio import errors
    errors.install()
    from pinnstudio.ui.main_window import MainWindow
    # Create the QApplication (and set its style) before importing PyTorch
    # (which happens inside _check_gpu_torch_mismatch()): importing PyTorch
    # first has been observed to leave some widgets built but not painted.
    # "pinnstudio" as the program name, so the desktop matches the window to
    # its shortcut and icon however PINNStudio was started.
    app = QApplication(["pinnstudio"] + argv)
    app.setApplicationName("PINNStudio")
    app.setDesktopFileName("pinnstudio")
    app.setStyle("Fusion")
    _check_gpu_torch_mismatch()
    window = MainWindow()
    window.show()
    files = [a for a in argv if a.endswith(".json") and not a.startswith("-")]
    if files:                      # pinnstudio problem.pinn.json, or a file opened from the desktop
        QTimer.singleShot(0, lambda: window._open_problem(files[0]))
    return app.exec()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    from pinnstudio import shortcut
    code = shortcut.main(argv)
    if code is not None:              # --create-shortcut / --remove-shortcut
        sys.exit(code)
    commands = {"export": cmd_export, "run": cmd_run, "check": cmd_check}
    if argv and (argv[0] in commands or argv[0] in ("-h", "--help", "--version")):
        args = _parser().parse_args(argv)
        sys.exit(commands[args.command](args))
    sys.exit(run_gui(argv))


if __name__ == "__main__":
    main()
