import os
os.environ["DDE_BACKEND"] = "pytorch"
from pinnstudio.core import pyexe as _pyexe
import subprocess
import sys
import tempfile
import time
from pinnstudio.core.codegen import generate_script
from pinnstudio.core.config import PINNConfig

def _flush_lines(stream, on_output, interval=0.5):
    """Read `stream` to its end, handing its non-empty lines to on_output in batches every `interval`
    seconds (and the rest at the end)."""
    import queue
    import threading
    q = queue.Queue()

    def _pump():
        try:
            for line in stream:
                q.put(line)
        finally:
            q.put(None)
    threading.Thread(target=_pump, name="pinnstudio-output", daemon=True).start()

    buf = []

    def _flush():
        if on_output and buf:
            for _l in buf:
                on_output(_l)
        buf.clear()
    deadline = time.monotonic() + interval
    while True:
        try:
            item = q.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            item = ""
        if item is None:
            break
        item = item.rstrip()
        if item:
            buf.append(item)
        if time.monotonic() >= deadline:
            _flush()
            deadline = time.monotonic() + interval
    _flush()


def run_pinn(config: PINNConfig, on_output=None, set_process=None, scratch=None):
    """
    Generates the DeepXDE script from config,
    writes it to a temp file, and runs it.
    Calls on_output(line) for each output line.
    Returns 'DONE' or 'ERROR'.

    A config-level problem that codegen itself catches (e.g. an
    incompatible Boundary Condition combination, or an unrecognized BC
    "type") raises directly out of generate_script() below, before any
    subprocess is ever spawned -- unlike every other kind of training
    failure, which only ever surfaces later, inside the generated script's
    own subprocess (already reported gracefully via the on_output stream
    further down). Both of this function's callers (SolverThread.run() and
    sweep_runner.py's per-combination loop) call this with no exception
    handling of their own, since generate_script() never used to raise at
    all -- so without catching it here, that exception propagates straight
    out of a background QThread uncaught, which can abort the whole app
    instead of reporting a clean, readable error the same way every other
    training failure already does.
    """

    # Generate the script
    try:
        script = generate_script(config)
    except Exception as e:
        if on_output:
            on_output(f"ERROR: {e}")
        return "ERROR"

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False,
        encoding='utf-8'
    ) as f:
        f.write(script)
        tmp_path = f.name

    try:
        print(f"Generated script at: {tmp_path}")
        # Run the script as a subprocess
        # Force the child process itself to use UTF-8 for its own stdout/stderr -
        # otherwise Windows defaults a piped (non-console) stream to the legacy
        # system codepage, which can't encode emoji/symbols the script prints.
        _env = _pyexe.script_env()
        if scratch:          # the run's own scratch folder, where it puts plots when there is no save folder
            _env["PINNSTUDIO_SCRATCH"] = scratch
        try:
            process = subprocess.Popen(
                    [_pyexe.script_python(), tmp_path], creationflags=_pyexe.NO_WINDOW,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=_env
                )
        except OSError as e:
            # no Python interpreter at the configured path, or the script file locked: report it like
            # any other failed run (an exception here would end the QThread without its done signal)
            if on_output:
                on_output(f"ERROR: could not start the training script: {e}")
            return "ERROR"
        _pyexe.end_with_pinnstudio(process)   # Windows: it ends when PINNStudio ends
        if set_process:
            set_process(process)

        # Buffer output and pass it on every 0.5 s instead of line by line (this took the GUI's share of
        # the CPU from 23 % to almost nothing). A reader thread fills the buffer and the flush runs on
        # the clock, so the last lines before a long quiet stretch (an L-BFGS phase, the plotting) show
        # at once instead of waiting for the next line, which can be minutes later.
        _flush_lines(process.stdout, on_output)

        process.wait()

        if process.returncode == 0:
            return "DONE"
        else:
            return "ERROR"

    finally:
        try:
            os.unlink(tmp_path)  # clean up temp file
        except OSError:
            pass                 # (Windows: still locked by a scanner) -- never fail the run over it
