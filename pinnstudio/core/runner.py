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
    """

    # Generate the script
    script = generate_script(config)

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False
    ) as f:
        f.write(script)
        tmp_path = f.name
        # DEBUG: also save a permanent copy
        with open("/tmp/last_generated_script.py", "w") as dbg:
            dbg.write(script)

    try:
        print(f"Generated script at: {tmp_path}")
        # Run the script as a subprocess
        process = subprocess.Popen(
                    [sys.executable, tmp_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
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