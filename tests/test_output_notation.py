"""Output names follow the notation of the paper or DeepXDE example each
example reproduces (Wight & Zhao write phi for the 2D/3D Allen-Cahn and the
2D Cahn-Hilliard, Mattey & Ghosh write h), every output name is used in its
own equations, Error Analysis still finds a reference file for every
output, and a problem saved with the old name "u" still opens with its
files found.

The short CPU runs at the end train three renamed templates for a few
steps and check that the script's Error Analysis reads every reference
file: run with PINNSTUDIO_SLOW_TESTS=1 (a few minutes); skipped otherwise.

    QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_output_notation.py
"""
import os
import re
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.core import portable_paths as pp
from pinnstudio.core.codegen import generate_script
from pinnstudio.core.deepxde_examples import EXAMPLES
from pinnstudio.ui.main_window import MainWindow, REFERENCE_DATA_DIR
from tests.test_templates import TEMPLATES_BY_DIM

# the field names each paper uses
PAPER_NAMES = {
    "1D Allen-Cahn": "u",
    "1D Cahn-Hilliard": "u,mu",
    "1D Cahn-Hilliard (Mattey & Ghosh)": "h,mu",
    "2D Allen-Cahn (Mattey & Ghosh)": "h",
    "2D Allen-Cahn (Wight & Zhao)": "phi",
    "2D Cahn-Hilliard (Mattey & Ghosh)": "h,mu",
    "2D Cahn-Hilliard (Wight & Zhao)": "phi,mu",
    "3D Allen-Cahn (Wight & Zhao)": "phi",
}
_DIM_OF = {n: d for d, names in TEMPLATES_BY_DIM.items() for n in names}


def _used(name, pde):
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", pde) or \
        re.search(rf"(?<![A-Za-z0-9_])d{re.escape(name)}_[xyzt]", pde)


@pytest.fixture(scope="module")
def win():
    return MainWindow()


def _load(win, name, inverse=False):
    {"1d": win.radio_1d, "2d": win.radio_2d, "3d": win.radio_3d}[_DIM_OF[name]].setChecked(True)
    win.quick_examples_combo.setCurrentText(name)
    win.radio_forward.setChecked(not inverse)
    win.radio_inverse.setChecked(inverse)
    return win._build_config()


def test_templates_use_the_papers_names(win):
    for name, want in PAPER_NAMES.items():
        c = _load(win, name)
        assert c.output_names.replace(" ", "") == want, (name, c.output_names)


def test_every_output_is_in_its_equations(win):
    for names in TEMPLATES_BY_DIM.values():
        for name in names:
            for inv in (False, True):
                c = _load(win, name, inv)
                for out in c.output_names.split(","):
                    assert _used(out.strip(), c.pde_expressions), (name, inv, out, c.pde_expressions)
    for e in EXAMPLES:
        for s in (e.settings, {**e.settings, **e.inverse} if e.inverse else None):
            if not s:
                continue
            for out in s.get("output_names", "u").split(","):
                assert _used(out.strip(), s["pde_expressions"]), (e.name, out)


def test_error_analysis_finds_every_output(win):
    for name, want in PAPER_NAMES.items():
        _load(win, name)
        files = win._ea_settings["files"]
        sels = {sel for _, _, sel in files}
        n_out = len(want.split(","))
        assert sels == ({None} if n_out == 1 else set(range(n_out))), (name, sels)
        # Inverse measures the first output at the end time by default
        _load(win, name, inverse=True)
        if n_out > 1:
            first = want.split(",")[0]
            assert os.path.basename(win.inv_data_path.text()).endswith(f"_{first}.txt"), \
                (name, win.inv_data_path.text())


def test_old_saved_paths_find_the_renamed_files():
    old = {"ea_files": repr([(0.0, "@REF/2D/cahn_hilliard_wight/t_0_u.txt", 0),
                             (0.0, "@REF/2D/cahn_hilliard_wight/t_0_mu.txt", 1)]),
           "inverse_obs_file": "@REF/1D/cahn_hilliard_mattey/t_1_u.txt",
           "ic_file": "/home/someone/site-packages/reference_data/2D/cahn_hilliard_mattey/t_0_u.txt"}
    new = pp.from_portable(old, REFERENCE_DATA_DIR)
    assert os.path.join("cahn_hilliard_wight", "t_0_phi.txt") in new["ea_files"]
    assert os.path.join("cahn_hilliard_wight", "t_0_mu.txt") in new["ea_files"]
    assert new["inverse_obs_file"].endswith(os.path.join("cahn_hilliard_mattey", "t_1_h.txt"))
    assert new["ic_file"].endswith(os.path.join("cahn_hilliard_mattey", "t_0_h.txt"))
    assert os.path.isfile(new["inverse_obs_file"]) and os.path.isfile(new["ic_file"])
    # a file that was never renamed is left alone
    keep = pp.from_portable({"p": "@REF/2D/burgers_mathias/t_1.0_u.txt"}, REFERENCE_DATA_DIR)
    assert keep["p"].endswith(os.path.join("burgers_mathias", "t_1.0_u.txt"))


def test_old_problem_named_u_still_gets_its_files(win):
    """A problem saved with the outputs still called "u, mu": Error Analysis,
    set up again from the template's folder, gives the renamed files to u."""
    _load(win, "2D Cahn-Hilliard (Wight & Zhao)")
    win.output_name_inputs[0].setText("u")
    win._auto_configure_ea(os.path.join(REFERENCE_DATA_DIR, "2D", "cahn_hilliard_wight"))
    sels = {sel for _, _, sel in win._ea_settings["files"]}
    assert sels == {0, 1}


_SHORT = {"num_domain": 400, "num_boundary": 40, "num_initial": 40, "num_test": 200,
          "iterations": 20, "iterations2": 0, "optimizer2": "none"}


@pytest.mark.skipif(not os.environ.get("PINNSTUDIO_SLOW_TESTS"), reason="set PINNSTUDIO_SLOW_TESTS=1")
@pytest.mark.parametrize("name", ["2D Allen-Cahn (Wight & Zhao)", "2D Cahn-Hilliard (Wight & Zhao)",
                                  "1D Cahn-Hilliard (Mattey & Ghosh)"])
def test_short_cpu_run_reads_every_reference_file(win, name):
    c = _load(win, name)
    for k, v in _SHORT.items():
        setattr(c, k, v)
    c.layers = [c.layers[0], 16, 16, c.layers[-1]]
    c.time_adaptive = False
    d = tempfile.mkdtemp()
    c.save_dir = d
    with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as fh:
        fh.write(generate_script(c))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg")
    p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=1500, env=env)
    assert p.returncode == 0, p.stderr[-2000:]
    n_files = len(win._ea_settings["files"])
    l2 = re.findall(r"L2[^\n]*", p.stdout)
    assert len(l2) >= n_files, (name, n_files, l2[-5:], p.stdout[-1500:])
