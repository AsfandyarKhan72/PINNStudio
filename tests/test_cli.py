"""The command line: --version, check, export and run work on a problem file without a display, and
never import Qt."""
import ast
import dataclasses
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio.core import deepxde_examples as DX  # noqa: E402
from pinnstudio.core.problem_file import REFERENCE_DATA_DIR, read, write  # noqa: E402


def _cli(*args, cwd=None):
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    env["PYTHONPATH"] = ROOT
    code = ("import sys, runpy; sys.argv = ['pinnstudio'] + sys.argv[1:]\n"
            "try:\n    runpy.run_module('pinnstudio', run_name='__main__')\n"
            "except SystemExit as e:\n"
            "    print('QT', any(k.startswith('PySide6') for k in sys.modules), file=sys.stderr)\n    raise\n")
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, env=env, cwd=cwd,
                          timeout=600)


def _problem(tmp_path, iterations=50, **kw):
    ex = next(e for e in DX.EXAMPLES if e.name == "Diffusion equation")
    c = DX.build_config(ex, REFERENCE_DATA_DIR)
    c = dataclasses.replace(c, iterations=iterations, scheduler_phases=json.dumps(
        [{"optimizer": "adam", "iterations": iterations, "lr": 0.001, "loss": "MSE"}]), **kw)
    path = str(tmp_path / "diff.pinn.json")
    write(c, path, "test")
    return path


def test_version():
    r = _cli("--version")
    assert r.returncode == 0 and r.stdout.startswith("PINNStudio ")


def test_problem_files_round_trip(tmp_path):
    path = _problem(tmp_path)
    config, name, data = read(path)
    assert name == "diff" and data["pinnstudio_problem_format"] == 1
    assert "@REF" in json.dumps(data["config"]) or not config.ea_files.strip("[]")


def test_check_and_export_without_qt(tmp_path):
    path = _problem(tmp_path)
    r = _cli("check", path)
    assert r.returncode == 0 and r.stdout.strip() == "OK", r.stderr
    assert "QT False" in r.stderr
    r = _cli("export", path, "-o", str(tmp_path / "s.py"))
    assert r.returncode == 0, r.stderr
    ast.parse(open(tmp_path / "s.py", encoding="utf-8").read())
    assert "QT False" in r.stderr


def test_check_lists_what_stops_solve(tmp_path):
    path = _problem(tmp_path, x_min=1.0, x_max=0.0)
    r = _cli("check", path)
    assert r.returncode == 1 and "x_min" in r.stdout


def test_run_trains_into_a_folder_of_its_own(tmp_path):
    path = _problem(tmp_path, iterations=30)
    r = _cli("run", path, "--save-dir", str(tmp_path / "runs"))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    runs = os.listdir(tmp_path / "runs")
    assert len(runs) == 1 and runs[0].startswith("diff__")
    sol = tmp_path / "runs" / runs[0] / "solution_results"
    env = json.load(open(sol / "environment.json", encoding="utf-8"))
    assert env["torch"] and env["deepxde"] and env["python"]
