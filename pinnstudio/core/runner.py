import os
os.environ["DDE_BACKEND"] = "pytorch"
import subprocess
import sys
import tempfile
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

        # Stream output line by line
        for line in process.stdout:
            line = line.rstrip()
            if line and on_output:
                on_output(line)

        process.wait()

        if process.returncode == 0:
            return "DONE"
        else:
            return "ERROR"

    finally:
        os.unlink(tmp_path)  # clean up temp file