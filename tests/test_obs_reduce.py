"""Inverse -> Measured data -> Use: all points, a coarser grid (points per
space axis, evenly spaced, edges kept, each time in a file the same) or a
random subset. The files stay as they are: both scripts keep the chosen
points. The window says what the files hold and what is kept, the setting
is saved with the problem, templates set their own (Allen-Cahn 3D: grid 26)
and the sweep can vary it, with each run's inferred parameters and their
error in the summary."""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from pinnstudio.core import obs_reduce, sweep_registry, sweep_runner
from pinnstudio.core.codegen import OBS_REDUCE_HELPER, generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig

app = QApplication.instance() or QApplication([])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _helper():
    ns = {"np": np}
    exec(OBS_REDUCE_HELPER, ns)
    return ns["_obs_reduce"]


def _grid2d(n, times=(1.0,)):
    xs = np.linspace(0, 1, n)
    rows = [[x, y, t, x + y + t] for t in times for y in xs for x in xs]
    return np.array(rows)


def test_grid_keeps_evenly_spaced_values_and_the_edges():
    reduce_ = _helper()
    d = _grid2d(101)
    r = reduce_(d, 3, 2, "grid", 51)
    assert len(r) == 51 * 51
    assert np.allclose(np.unique(r[:, 0]), np.linspace(0, 1, 51))
    r = reduce_(d, 3, 2, "grid", 30)               # not a divisor: as even as it gets
    xs = np.unique(r[:, 0])
    assert len(r) == 900 and xs[0] == 0 and xs[-1] == 1
    assert len(reduce_(d, 3, 2, "grid", 500)) == len(d), "more than there are: all of them"


def test_each_time_is_reduced_the_same_and_scattered_data_falls_back_to_random():
    reduce_ = _helper()
    d = _grid2d(21, times=(0.5, 1.0))
    r = reduce_(d, 3, 2, "grid", 11)
    assert len(r) == 2 * 11 * 11 and set(np.unique(r[:, 2])) == {0.5, 1.0}
    rng = np.random.default_rng(0)
    s = np.column_stack([rng.random((300, 3)), rng.random(300)])
    assert len(reduce_(s, 3, 3, "grid", 5)) == 125
    a, b = reduce_(s, 3, 3, "random", 100, 7), reduce_(s, 3, 3, "random", 100, 7)
    assert len(a) == 100 and np.array_equal(a, b), "the run's seed: the same points every time"


def test_the_window_note_counts_what_is_kept(tmp_path):
    f = tmp_path / "u.txt"
    np.savetxt(f, _grid2d(101))
    assert obs_reduce.summary([str(f)], "2D", False, "grid", 51) == \
        "101 × 101 grid, 10,201 points → 51 × 51 = 2,601 points"
    assert obs_reduce.summary([str(f)], "2D", False, "all", 51) == "101 × 101 grid, 10,201 points"
    assert obs_reduce.summary([str(f)], "2D", False, "random", 500) == \
        "101 × 101 grid, 10,201 points → 500 points"
    assert obs_reduce.summary([str(tmp_path / "missing.txt")], "2D", False, "grid", 51) == ""


def test_the_scripts_read_every_line_of_a_file_without_a_header(tmp_path):
    """The measured-data loader used to drop the first line always."""
    from PyQt6.QtWidgets import QApplication  # noqa: F401
    f = tmp_path / "u.txt"
    np.savetxt(f, _grid2d(11))
    c = PINNConfig()
    c.problem_type, c.problem_dim = "Inverse", "2D"
    c.inverse_obs_files_json = json.dumps([{"path": str(f), "output_idx": 0, "weight": 100.0}])
    clean = generate_clean_script(c)
    start = clean.index("def _load_obs_data")
    ns = {"np": np}
    exec(clean[start:clean.index("\n\n", clean.index("return np.array(rows)", start))], ns)
    assert len(ns["_load_obs_data"](str(f))) == 121
    with open(tmp_path / "h.txt", "w") as fh:
        fh.write("x,y,t,u\n" + "\n".join(",".join(map(str, r)) for r in _grid2d(3)))
    assert len(ns["_load_obs_data"](str(tmp_path / "h.txt"))) == 9, "a header line is still skipped"
    assert "skiprows=1" not in generate_script(c) and "skiprows=1" not in clean


def _load(w, name):
    w.quick_examples_category_combo.setCurrentText("Inverse")
    app.processEvents()
    w.quick_examples_combo.setCurrentText(name)
    for _ in range(8):
        app.processEvents()


def test_window_setting_scripts_saving_and_templates():
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    _load(w, "Heat 2D")
    c = w._build_config()
    assert c.inverse_obs_reduce == "all" and "_obs_reduce(" not in generate_script(c)
    assert "10,201 points" in w.inv_obs_reduce_note.text()
    w.inv_obs_reduce_combo.setCurrentIndex(w.inv_obs_reduce_combo.findData("grid"))
    w.inv_obs_grid_n.setValue(21)
    app.processEvents()
    assert w.inv_obs_reduce_note.text().endswith("→ 21 × 21 = 441 points")
    assert not w.inv_obs_grid_n.isHidden() and w.inv_obs_random_n.isHidden()
    c = w._build_config()
    assert (c.inverse_obs_reduce, c.inverse_obs_grid_n) == ("grid", 21)
    for gen in (generate_script, generate_clean_script):
        text = gen(c)
        assert "def _obs_reduce(" in text and "'grid', 21" in text
        compile(text, gen.__name__, "exec")
    w2 = MainWindow()
    w2._apply_config(c)
    assert w2._obs_reduce_mode() == "grid" and w2.inv_obs_grid_n.value() == 21
    _load(w, "Allen-Cahn 3D – Wight & Zhao")
    c = w._build_config()
    assert (c.inverse_obs_reduce, c.inverse_obs_grid_n) == ("grid", 26)
    assert json.loads(c.inverse_obs_files_json)[0]["path"].endswith("t_1.txt")
    assert "= 17,576 points" in w.inv_obs_reduce_note.text()
    _load(w, "Heat 3D")
    assert w._build_config().inverse_obs_reduce == "all", "not carried over to the next example"
    w.close()
    w2.close()


def test_sweep_over_the_data_size(tmp_path):
    c = PINNConfig()
    c.problem_type = "Inverse"
    ids = [p.id for p in sweep_registry.available_params(c)]
    assert "inv_obs_grid_n" in ids and "inv_obs_random_n" in ids
    assert "inv_obs_grid_n" not in [p.id for p in sweep_registry.available_params(PINNConfig())]
    c.sweep_mode = "oat"
    c.sweep_parameters = json.dumps([{"id": "inv_obs_random_n", "mode": "list", "values": [100, 1000]}])
    runs = sweep_runner.build_runs_detailed(c)
    assert [(r["config"].inverse_obs_reduce, r["config"].inverse_obs_random_n) for r in runs] == \
        [("random", 100), ("random", 1000)]


def test_sweep_summary_has_the_inferred_parameters(tmp_path):
    c = PINNConfig()
    c.problem_type = "Inverse"
    c.inverse_variables_json = json.dumps([{"name": "alpha", "init": 1.0, "true": 0.4}])
    run = tmp_path / "run"
    run.mkdir()
    (run / "alpha_convergence.txt").write_text("iteration,alpha\n100,0.9\n2000,0.42\n")
    c.save_dir = str(run)
    params = sweep_runner._read_inverse_params(c, ["  [alpha] Iter 1000: 0.5"], 0)
    assert params == {"alpha": {"value": 0.42, "true": 0.4}}
    (run / "alpha_convergence.txt").unlink()
    assert sweep_runner._read_inverse_params(c, ["  [alpha] Iter 1000: 0.5"], 0)["alpha"]["value"] == 0.5
    c.sweep_mode = "oat"
    c.sweep_parameters = json.dumps([{"id": "inv_obs_grid_n", "mode": "list", "values": [11, 51]}])
    label = sweep_registry.get_param("inv_obs_grid_n", c).label
    results = [(f"n={n}", {"status": "done", "final_loss": 1e-4, "metrics": {"L2_relative": 1e-2},
                           "params": {"alpha": {"value": v, "true": 0.4}}, "values": {label: n},
                           "group": 0, "baseline": False, "save_dir": str(tmp_path / f"r{n}")})
               for n, v in ((11, 0.5), (51, 0.404))]
    sweep_runner._write_sweep_summary_csv(str(tmp_path), results)
    rows = open(tmp_path / "sweep_summary.csv").read().splitlines()
    assert rows[0].endswith("alpha,alpha error (%),Folder")
    assert ",0.5,25," in rows[1] and ",0.404,1," in rows[2]
    assert sweep_runner.write_sweep_summary_plot(str(tmp_path), c, results)


def test_window_and_script_agree_on_any_layout():
    """Scattered (x, t) data (one point per time) is not a grid, and times
    with different grids are not one: both fall back to random points,
    n per axis with time counted as an axis, in the window and the script."""
    import numpy as np
    from pinnstudio.core import obs_reduce as o
    from pinnstudio.core.codegen import OBS_REDUCE_HELPER
    ns = {"np": np}
    exec(OBS_REDUCE_HELPER, ns)
    rng = np.random.default_rng(1)
    scattered = np.column_stack([rng.random(2000), rng.random(2000), rng.random(2000)])
    mixed = np.vstack([np.column_stack([np.linspace(0, 1, 21), np.full(21, .5), np.zeros(21)]),
                       np.column_stack([np.linspace(0, 1, 11), np.full(11, 1.), np.zeros(11)])])
    grid = np.array([[x, t, 0] for t in (0, .5, 1) for x in np.linspace(0, 1, 51)])
    assert o.describe(scattered, 2, 1)["grid"] is None
    for d, n, want in ((scattered, 10, 100), (mixed, 5, 25), (grid, 5, 15)):
        assert o.kept(o.describe(d, 2, 1), "grid", n, 1, 2) == len(ns["_obs_reduce"](d, 2, 1, "grid", n)) == want


def test_inverse_names_and_columns():
    """Trainable variables must be Python names; a measured-data file with
    too few columns stops the run instead of training on NaN."""
    import json
    from pinnstudio.core.config import PINNConfig
    c = PINNConfig()
    c.problem_type = "Inverse"
    for name, bad in (("my var", True), ("lambda", True), ("2x", True), ("D_1", False)):
        c.inverse_variables_json = json.dumps([{"name": name, "init": 1}])
        assert any("Trainable variable" in e for e in c.validate()) == bad, name
    from pinnstudio.core import codegen as cg
    c.inverse_variables_json = json.dumps([{"name": "k", "init": 1}])
    for gen in (cg.generate_script, cg.generate_clean_script):
        assert "columns are needed" in gen(c)


def test_whole_token_replacement():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pinnstudio.ui.main_window import _replace_token as r
    assert r("du_t - d*(du_xx + du_yy)", "d", "0.4") == "du_t - 0.4*(du_xx + du_yy)"
    assert r("u*0.45 + 0.4*u", "0.4", "k") == "u*0.45 + k*u"
    assert r("alpha*u - alphabet", "alpha", "a") == "a*u - alphabet"
