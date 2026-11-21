"""Error reporting for the PINNStudio window.

Started from its desktop shortcut on Windows, PINNStudio runs under
pythonw.exe: there is no console, sys.stdout and sys.stderr are None, and an
error that nobody catches goes nowhere: PySide6 prints it to the missing
stderr and carries on, so it is lost (PyQt6 ended the program instead, and
the window just disappeared). Here:

* output goes to a log file when there is no console (and anything a library
  writes to the missing streams no longer raises);
* an unexpected error is written to the log and shown in a message box, and
  the app keeps running;
* a hard crash (segmentation fault, ...) writes its traceback to the log.

The log is ~/.pinnstudio/pinnstudio.log.
"""
import datetime
import faulthandler
import os
import sys
import threading
import traceback

LOG_DIR = os.path.join(os.path.expanduser("~"), ".pinnstudio")
LOG_PATH = os.path.join(LOG_DIR, "pinnstudio.log")
_MAX_LOG = 2_000_000          # bytes; a longer log starts over
_log_file = None
_showing = False


def _open_log():
    global _log_file
    if _log_file is None:
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            if os.path.isfile(LOG_PATH) and os.path.getsize(LOG_PATH) > _MAX_LOG:
                os.replace(LOG_PATH, LOG_PATH + ".old")
            _log_file = open(LOG_PATH, "a", encoding="utf-8", errors="replace", buffering=1)
        except OSError:
            _log_file = open(os.devnull, "w")
    return _log_file


def log(text):
    f = _open_log()
    try:
        f.write(text if text.endswith("\n") else text + "\n")
        f.flush()
    except Exception:
        pass


def _report(exc_type, exc, tb, where=""):
    global _showing
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n[{stamp}] Unexpected error{where}:\n{text}")
    if sys.stderr is not None and sys.stderr is not _log_file:
        try:
            sys.stderr.write(text)
        except Exception:
            pass
    if _showing:              # an error while the message is open: log only
        return
    try:
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance()
        if app is None or QThread.currentThread() is not app.thread():
            return            # message boxes only from the GUI thread
        _showing = True
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("PINNStudio error")
        box.setText(f"Something went wrong: {exc_type.__name__}: {exc}\n\n"
                    "PINNStudio is still running. If this keeps happening, please open an issue "
                    "on GitHub with the details below.")
        box.setInformativeText(f"The full message is also in {LOG_PATH}")
        box.setDetailedText(text)
        box.exec()
    except Exception:
        pass
    finally:
        _showing = False


def install():
    """Call first thing at startup, before the window is built."""
    if sys.stdout is None:
        sys.stdout = _open_log()
    if sys.stderr is None:
        sys.stderr = _open_log()
    try:
        faulthandler.enable(file=_open_log(), all_threads=True)
    except Exception:
        pass
    sys.excepthook = lambda t, e, tb: _report(t, e, tb)
    threading.excepthook = lambda a: _report(a.exc_type, a.exc_value, a.exc_traceback,
                                            f" in thread {a.thread.name if a.thread else '?'}")


# ── "Not responding": a heartbeat from the UI thread, and a watchdog that logs stalls ──
# Windows marks a window "Not responding" after 5 s without events. When the UI thread misses its beat
# for 3 s, the watchdog writes every thread's stack to the log (read with the GIL held, through
# sys._current_frames: faulthandler's timed dump reads frames without it and can crash the app), again
# every 15 s while the stall lasts, at most 20 times a session, and how long the stall lasted.
_beat = [0.0]
STALL_S = 3.0


def _stack_dump():
    import sys as _sys
    me = threading.get_ident()
    names = {t.ident: t.name for t in threading.enumerate()}
    out = []
    for ident, frame in _sys._current_frames().items():
        if ident == me:
            continue
        out.append(f"--- thread {names.get(ident, ident)} ---\n" + "".join(traceback.format_stack(frame)))
    return "\n".join(out)


def _watch():
    import time
    dumps, since, last_dump = 0, None, 0.0
    while True:
        time.sleep(0.25)
        now = time.monotonic()
        late = now - _beat[0]
        if late > STALL_S:
            if since is None:
                since = _beat[0]
                stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log(f"\n[{stamp}] The window has not responded for {late:.1f} s")
            if dumps < 20 and now - last_dump >= 15.0:
                dumps += 1
                last_dump = now
                log(f"Stacks after {late:.1f} s without a heartbeat:\n{_stack_dump()}")
        elif since is not None:
            log(f"The window responds again, after {now - since:.1f} s")
            since, last_dump = None, 0.0


def start_stall_watch(app):
    """Call once the window is up: a 1 s heartbeat on the UI thread and the watchdog thread."""
    import time
    from PySide6.QtCore import QTimer
    _beat[0] = time.monotonic()
    timer = QTimer(app)
    timer.setInterval(1000)
    timer.timeout.connect(lambda: _beat.__setitem__(0, time.monotonic()))
    timer.start()
    app.setProperty("_pinn_heartbeat", True)
    app._pinn_heartbeat_timer = timer                  # kept alive with the app
    threading.Thread(target=_watch, name="pinnstudio-stall-watch", daemon=True).start()
