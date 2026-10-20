"""Error Analysis fixes: one reader for reference files everywhere (commas or
spaces, header lines skipped, NaN rows dropped); History reads the metric
columns by name; the exported script computes the metrics without figures;
steady references are not filtered by time; the dialog keeps its boxes."""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinnstudio.core import codegen as cg
from pinnstudio.core.config import PINNConfig


def test_reference_reader():
    ns = {"np": np}
    exec(cg._PLOT_TEXT_HELPERS, ns)
    d = tempfile.mkdtemp()
    f = os.path.join(d, "r.csv")
    with open(f, "w") as fh:
        fh.write("x,y,u\n0,0,1\n0.5,0.5,nan\n1,1,2\n")
    assert ns["_load_ref"](f).tolist() == [[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]]
    for gen in (cg.generate_script, cg.generate_clean_script):
        c = PINNConfig()
        c.ea_files = repr([(0.0, f, None)])
        t = gen(c)
        assert "def _load_ref" in t and "np.loadtxt(_ea_fp)" not in t and "np.loadtxt(fp)" not in t


def test_history_reads_metrics_by_name():
    from pinnstudio.ui.results import read_run
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "error_analysis"))
    os.makedirs(os.path.join(d, "solution_results"))
    with open(os.path.join(d, "error_analysis", "error_metrics.txt"), "w") as fh:
        fh.write("t,L2_relative,MSE,Mean_abs_error\n0.5,0.01,4e-06,0.002\n")
    run = read_run(d)
    assert run["metrics"] == [(0.5, 0.01, 0.002)]


def test_export_metrics_without_figures_and_steady_times():
    c = PINNConfig()
    c.ea_files = repr([(0.0, "/tmp/x.txt", None)])
    c.ea_do_line = c.ea_do_surface = False
    assert "error_metrics" in cg.generate_clean_script(c)
    c.steady_state, c.t_min = True, 0.5
    assert "if True or 0.5 - 1e-10 <= _ea_tv0" in cg.generate_script(c)
