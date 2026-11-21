"""Check a built Windows program from inside it: PINNStudio.exe packaging/windows/check_build.py
(build_windows.bat runs it after PyInstaller).

What the program must hold, used as a user would: the package's resources and reference data, a training
run started the way Solve starts it (PINNStudio.exe script.py, a second process of the program), its results
and environment.json, the exported script, a designed network, and the window built off screen. A library or
data file the build missed fails here, not in use. Settings and results go to a temporary folder."""
import ast
import dataclasses
import json
import os
import sys
import tempfile

HOME = tempfile.mkdtemp(prefix="pinnstudio_check_")
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DDE_BACKEND"] = "pytorch"
os.environ["PINNSTUDIO_HOME"] = HOME
os.environ["HOME"] = os.environ["USERPROFILE"] = HOME       # the window's settings, not the user's

import pinnstudio  # noqa: E402
from pinnstudio.core import deepxde_examples as DX  # noqa: E402
from pinnstudio.core import pyexe  # noqa: E402
from pinnstudio.core.codegen import generate_clean_script  # noqa: E402
from pinnstudio.core.problem_file import REFERENCE_DATA_DIR  # noqa: E402
from pinnstudio.core.runner import run_pinn  # noqa: E402

assert getattr(sys, "frozen", False), "run this inside the built program: PINNStudio.exe check_build.py"
assert pyexe.script_python() == sys.executable, "training would not run inside the program"
res = os.path.join(os.path.dirname(pinnstudio.__file__), "resources")
assert os.path.isfile(os.path.join(res, "pinnstudio.ico")), "the resources are missing"
n_ref = sum(len(f) for _d, _s, f in os.walk(REFERENCE_DATA_DIR))
assert n_ref > 50, f"the reference data is missing ({REFERENCE_DATA_DIR}: {n_ref} files)"
print(f"reference data: {n_ref} files", flush=True)

# a training run, as Solve starts it: the script in a second process of this program
ex = next(e for e in DX.EXAMPLES if e.name == "Diffusion equation")
cfg = DX.build_config(ex, REFERENCE_DATA_DIR)
run_dir = os.path.join(HOME, "run")
cfg = dataclasses.replace(cfg, save_dir=run_dir, iterations=60, scheduler_phases=json.dumps(
    [{"optimizer": "adam", "iterations": 60, "lr": 0.001, "loss": "MSE"}]))
lines = []
result = run_pinn(cfg, on_output=lambda ln: (lines.append(ln), print("   |", ln, flush=True)),
                  scratch=tempfile.mkdtemp(prefix="pinnstudio_run_"))
assert result == "DONE", "the training run failed:\n" + "\n".join(lines[-40:])
env_files = [os.path.join(d, "environment.json") for d, _s, f in os.walk(run_dir) if "environment.json" in f]
assert env_files, "the run wrote no environment.json"
env = json.load(open(env_files[0], encoding="utf-8"))
assert env.get("torch") and env.get("deepxde"), env
print(f"training run: PyTorch {env['torch']}, DeepXDE {env['deepxde']}", flush=True)

# the exported script, and a designed network's script
ast.parse(generate_clean_script(cfg))
from pinnstudio.core.netdesign import presets as P  # noqa: E402
from pinnstudio.core.netdesign import runtime as R  # noqa: E402

import deepxde as dde  # noqa: E402
import torch  # noqa: E402

ns = {"dde": dde, "torch": torch}
exec(R.net_source(P.msffn([16, 16], [1.0, 10.0]), ["x", "t"], ["u"]), ns)

net = ns["_DesignedNet"](ns["_DESIGN"], "tanh", "Glorot uniform", 0.0)
assert tuple(net(torch.rand(5, 2)).shape) == (5, 1)

# the window, off screen
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication(["check"])
from pinnstudio.ui.main_window import MainWindow  # noqa: E402

w = MainWindow()
w.show()
app.processEvents()
assert w.isVisible()
w._maybe_save_changes = lambda: True
w.close()

print(f"PINNStudio {pinnstudio.__version__}, Python {sys.version.split()[0]}, PyTorch {torch.__version__}: "
      "build check passed", flush=True)
