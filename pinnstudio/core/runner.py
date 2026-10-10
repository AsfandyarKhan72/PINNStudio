import os
os.environ["DDE_BACKEND"] = "pytorch"
import subprocess
import sys
import tempfile
import time
from pinnstudio.core.codegen import generate_script
from pinnstudio.core.config import PINNConfig

def run_pinn(config: PINNConfig, on_output=None, set_process=None):
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
        _env = os.environ.copy()
        _env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
                    [sys.executable, tmp_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=_env
                )
        if set_process:
            set_process(process)

        # Buffer output — flush every 0.5s instead of every line
        # This reduces GUI overhead from 23% to near zero
        _buffer = []
        _last_flush = time.time()
        for line in process.stdout:
            line = line.rstrip()
            if line:
                _buffer.append(line)
            _now = time.time()
            if _now - _last_flush >= 0.5:
                if on_output and _buffer:
                    for _l in _buffer:
                        on_output(_l)
                _buffer.clear()
                _last_flush = _now
        # Flush any remaining lines
        if on_output and _buffer:
            for _l in _buffer:
                on_output(_l)

        process.wait()

        if process.returncode == 0:
            return "DONE"
        else:
            return "ERROR"

    finally:
        os.unlink(tmp_path)  # clean up temp file
