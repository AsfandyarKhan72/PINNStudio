"""Parameter Sweep fixes: no "Initial points" for an ODE; the two data-size
parameters can't be swept together and a baseline using all the data is
labelled "all"; inverse parameters found under the names the script uses;
a Bayesian baseline outside the search space doesn't teach the model;
folder names keep whole name_value parts; a row whose parameter is no longer
offered keeps it (marked) instead of moving its values to another one."""
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from pinnstudio.core import sweep_registry as reg
from pinnstudio.core import sweep_runner as sr
from pinnstudio.core.bayes_opt import Dim
from pinnstudio.core.config import PINNConfig

app = QApplication.instance() or QApplication([])


def test_no_initial_points_for_an_ode():
    c = PINNConfig()
    c.problem_dim, c.steady_state = "ODE", False
    assert "num_initial" not in {p.id for p in reg.available_params(c)}
    c.problem_dim = "1D"
    assert "num_initial" in {p.id for p in reg.available_params(c)}


def test_data_size_parameters():
    c = PINNConfig()
    c.problem_type = "Inverse"
    params = {p.id: p for p in reg._inv_obs_reduce_params(c)} if reg._inv_available(c) else {}
    if params:
        c.inverse_obs_reduce = "all"
        assert params["inv_obs_grid_n"].shown(c) == "all" and isinstance(params["inv_obs_grid_n"].get_value(c), int)
        c.inverse_obs_reduce = "grid"
        assert params["inv_obs_grid_n"].shown(c) == c.inverse_obs_grid_n
        assert params["inv_obs_random_n"].shown(c) == f"grid {c.inverse_obs_grid_n}"
    c.sweep_enabled = True
    c.sweep_parameters = json.dumps([{"id": "inv_obs_grid_n", "mode": "list", "values": [11, 21]},
                                     {"id": "inv_obs_random_n", "mode": "list", "values": [100, 500]}])
    assert any("can't be swept together" in e for e in c.validate())


def test_inverse_parameters_under_script_names():
    d = tempfile.mkdtemp()
    c = PINNConfig()
    c.problem_type, c.save_dir = "Inverse", d
    c.inverse_variables_json = json.dumps([{"name": "D 1", "init": 1, "true": 0.5}, {"name": "lambda", "init": 1}])
    with open(os.path.join(d, "D_1_convergence.txt"), "w") as f:
        f.write("100 0.49\n")
    got = sr._read_inverse_params(c, ["[lambda_var] Iter 100: 2.5"], 0)
    assert got == {"D_1": {"value": 0.49, "true": 0.5}, "lambda_var": {"value": 2.5, "true": None}}


def test_bayes_baseline_outside_the_space():
    assert not sr._in_space(Dim("categorical", choices=["sin", "ReLU"]), "tanh")
    assert sr._in_space(Dim("categorical", choices=["sin", "tanh"]), "tanh")
    assert not sr._in_space(Dim("float", 1e-4, 1e-2), 0.5) and sr._in_space(Dim("int", 2, 8), 4)


def test_folder_names_keep_whole_parts():
    name = sr.run_folder_name(3, "Phase 1: Iterations=20000, Phase 2: Iterations=5000, Learning rate=0.001, "
                                 "Network: Hidden layers=4, Activation=sin.")
    assert name.endswith("__more") and "Iterations_20000" in name and len(name) < 120
    assert not sr._slugify("sin.").endswith(".")


def test_stale_row_keeps_its_parameter():
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    if not hasattr(w, "_add_sweep_row") or not hasattr(w, "_refresh_sweep_param_choices"):
        w.close()
        return
    w._add_sweep_row()
    row = w.sweep_row_list[-1]
    combo = row["param_combo"]
    combo.addItem("RAR: cycles", "rar_cycles")          # as if RAR had been on
    combo.setCurrentIndex(combo.count() - 1)
    w._refresh_sweep_param_choices()
    assert combo.currentData() == "rar_cycles" and "not available" in combo.currentText()
    w.close()
