#!/usr/bin/env python3
"""Windows checks for PINNStudio, run in the window as a user would (off screen).

    python win_checks.py paths OUT_DIR
        Heat 1D, a short Solve in the window with its results saved in OUT_DIR,
        then the exported plain script run in OUT_DIR. For folders with spaces,
        non-English letters, or long paths (run it with USERPROFILE, TEMP and
        OUT_DIR set to such folders).
    python win_checks.py stop
        Solve, then Stop while it trains, then Solve and close the window while it
        trains: is the training process gone, and how long did it take?
    python win_checks.py parent-dies
        The window's own process ends while a run trains -- Ctrl+C in the terminal
        it was started from (KeyboardInterrupt), and the process being ended
        outright (closing that terminal, Task Manager, a crash) -- is the
        training process left running?
"""
import copy
import ctypes
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
INSTALLED = os.environ.get("WIN_CHECKS_INSTALLED") == "1"    # test the installed PINNStudio, not this checkout
if not INSTALLED:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))       # benchmarks/, for timing_study's helpers

STILL_ACTIVE = 259


def alive(pid):
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return code.value == STILL_ACTIVE


def gpu_pids():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


def window():
    import timing_study as ts
    if INSTALLED:          # timing_study puts this checkout first on sys.path; take it out again
        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(ROOT)]
    from PyQt6.QtWidgets import QMessageBox
    app, w = ts._window()
    # closing while a run trains asks "Stop it and quit?": answer Yes; "save changes?": No
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    w._maybe_save_changes = lambda: True
    return ts, app, w


def start_solve(ts, app, w, example, iters, out):
    ts.load(app, w, example)
    cfg = ts.config_for(w, iters, out)
    real = w._build_config
    w._build_config = lambda: copy.deepcopy(cfg)
    try:
        w._on_solve()
    finally:
        w._build_config = real


def wait_training(app, w, timeout=180):
    """Until DeepXDE prints its first step line; returns the training process."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.05)
        if any(line.strip()[:1].isdigit() and "[" in line for line in w.log_box.toPlainText().splitlines()):
            return w.thread.process
    raise SystemExit("training did not start:\n" + w.log_box.toPlainText()[-2000:])


def cmd_paths(out):
    os.makedirs(out, exist_ok=True)
    ts, app, w = window()
    r = {"out": out, "out_len": len(out), "USERPROFILE": os.environ.get("USERPROFILE"),
         "TEMP": os.environ.get("TEMP"), "python": sys.executable}
    iters = {"adam": 300, "lbfgs": 20}
    ts.load(app, w, ("Forward", "Heat 1D – Dirichlet"))
    d1 = os.path.join(out, "solve")
    os.makedirs(d1, exist_ok=True)
    s = ts.run_window(app, w, ts.config_for(w, iters, d1), "gpu")
    log = w.log_box.toPlainText()
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(d1) for f in fs]
    r["solve"] = {"ok": s["rc"] == 0, "files": len(files), "longest_path": max((len(f) for f in files), default=0),
                  "errors": [ln for ln in log.splitlines() if "Error" in ln or "Traceback" in ln][-5:]}
    from pinnstudio.core.codegen import generate_clean_script
    d2 = os.path.join(out, "script")
    os.makedirs(d2, exist_ok=True)
    s2 = ts.run_script(generate_clean_script(ts.config_for(w, iters, d2)), "gpu", d2)
    log2 = open(os.path.join(d2, "log.txt"), encoding="utf-8").read()
    files2 = [os.path.join(dp, f) for dp, _, fs in os.walk(d2) for f in fs]
    r["script"] = {"ok": s2["rc"] == 0, "files": len(files2), "longest_path": max((len(f) for f in files2), default=0),
                   "errors": [ln for ln in log2.splitlines() if "Error" in ln or "Traceback" in ln][-5:]}
    print(json.dumps(r, indent=1, ensure_ascii=False))


def cmd_stop():
    import tempfile
    ts, app, w = window()
    res = {}
    for how in ("Stop", "close the window"):
        out = tempfile.mkdtemp(prefix="pinn_stop_")
        start_solve(ts, app, w, ("Forward", "Navier-Stokes 2D – Kovasznay flow"), {"adam": 50000, "lbfgs": 10}, out)
        proc = wait_training(app, w)
        pid = proc.pid
        on_gpu_before = pid in gpu_pids()
        t0 = time.perf_counter()
        if how == "Stop":
            w._on_stop()
        else:
            w.close()
        t_click = time.perf_counter() - t0
        t_gone = None
        for _ in range(200):
            if not alive(pid):
                t_gone = time.perf_counter() - t0
                break
            app.processEvents()
            time.sleep(0.05)
        time.sleep(1.0)
        res[how] = {"training pid": pid, "was on the GPU": on_gpu_before, "handler returned after s": round(t_click, 2),
                    "process gone after s": None if t_gone is None else round(t_gone, 2),
                    "still on the GPU after": pid in gpu_pids(), "Solve enabled again": w.solve_btn.isEnabled()}
        print(how, res[how], flush=True)
        if alive(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        if how == "close the window":
            break
    print(json.dumps(res, indent=1))


def cmd_child(mode):
    """The window process for parent-dies: prints the training pid, then ends as asked."""
    import tempfile
    import threading
    import _thread
    from PyQt6.QtCore import QTimer
    from pinnstudio import errors
    errors.install()                 # as pinnstudio.main does
    ts, app, w = window()
    out = tempfile.mkdtemp(prefix="pinn_parent_")
    start_solve(ts, app, w, ("Forward", "Navier-Stokes 2D – Kovasznay flow"), {"adam": 50000, "lbfgs": 10}, out)
    proc = wait_training(app, w)
    print("TRAINING_PID", proc.pid, flush=True)
    if mode == "sigint":
        # what Ctrl+C in the terminal does to Python: KeyboardInterrupt, raised in the
        # main thread as soon as Python code runs there (a Qt slot, inside app.exec())
        threading.Timer(1.0, _thread.interrupt_main).start()
    QTimer.singleShot(60000, app.quit)
    app.exec()                       # Qt's event loop, as in pinnstudio.main
    print("WINDOW_STILL_RUNNING", flush=True)


def cmd_parent_dies():
    res = {}
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    for mode in ("sigint", "killed"):
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "_child", mode], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                             creationflags=flags)
        pid, lines = None, []
        for line in p.stdout:
            lines.append(line.rstrip())
            if line.startswith("TRAINING_PID"):
                pid = int(line.split()[1])
                break
        if pid is None:
            res[mode] = {"error": "\n".join(lines[-20:])}
            continue
        if mode == "killed":
            time.sleep(1.0)
            p.kill()                 # TerminateProcess, like closing the terminal or Task Manager
        try:
            rest, _ = p.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            p.kill()
            rest, _ = p.communicate()
        time.sleep(3.0)
        res[mode] = {"window process exit code": p.returncode,
                     "window kept running": "WINDOW_STILL_RUNNING" in rest,
                     "training process still running 3 s later": alive(pid),
                     "training process still on the GPU": pid in gpu_pids(),
                     "window output (last lines)": rest.strip().splitlines()[-6:]}
        print(mode, json.dumps(res[mode], ensure_ascii=False), flush=True)
        t_end = time.time() + 240
        while alive(pid) and time.time() < t_end:
            time.sleep(1)
        res[mode]["training process ended after s (max 240)"] = None if alive(pid) else round(243 - (t_end - time.time()))
        if alive(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def main():
    cmd = sys.argv[1]
    if cmd == "paths":
        cmd_paths(sys.argv[2])
    elif cmd == "stop":
        cmd_stop()
    elif cmd == "parent-dies":
        cmd_parent_dies()
    elif cmd == "_child":
        cmd_child(sys.argv[2])
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
