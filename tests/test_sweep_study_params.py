"""Parameter Sweep entries for the training methods and tuning studies: the
random seed, an L-BFGS phase at the end, per-point weights, causal
training, the network type, exact periodicity, an exact initial condition,
time-adaptive training and (Inverse) measured data at every reference time.
Each is offered only where the window allows it, changes the generated
script, and a combination that can't be run is skipped with its reason."""
import copy
import json
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from pinnstudio.core import deepxde_examples as dx  # noqa: E402
from pinnstudio.core import sweep_registry as reg  # noqa: E402
from pinnstudio.core import sweep_runner as sr  # noqa: E402
from pinnstudio.core.codegen import generate_script  # noqa: E402

REF = os.path.join(ROOT, "reference_data")
EX = {e.name: e for e in dx.EXAMPLES}
NEW = {"random_seed", "final_lbfgs", "point_weights", "causal_training", "network_type", "periodic_inputs",
       "exact_ic", "time_adaptive", "inv_obs_times"}
_WIN = None


def _pump(s=0.3):
    t = time.time()
    while time.time() - t < s:
        app.processEvents()


def _template(key, inverse=False):
    """A template's settings as the window loads it."""
    global _WIN
    if _WIN is None:
        from pinnstudio.ui.main_window import MainWindow
        _WIN = MainWindow()
        _WIN._maybe_save_changes = lambda: True
        _pump(1.0)
    w = _WIN
    w.quick_examples_category_combo.setCurrentText("Inverse" if inverse else "Forward")
    _pump(0.2)
    i = w.quick_examples_combo.findData(key)
    assert i >= 0, key
    w.quick_examples_combo.setCurrentIndex(i)
    w.quick_examples_combo.activated.emit(i)
    _pump(1.0)
    return w._build_config()


def _ids(cfg):
    return {p.id for p in reg.available_params(cfg)} & NEW


def _apply(cfg, pid, value):
    cfg = copy.deepcopy(cfg)
    reg.get_param(pid, cfg).set_value(cfg, value)
    return cfg


def _ok(cfg):
    probe = copy.deepcopy(cfg)
    probe.sweep_enabled = False
    errs = probe.validate()
    assert not errs, errs
    compile(generate_script(probe), "run.py", "exec")
    return generate_script(probe)


def test_availability_deepxde_heat():
    cfg = dx.build_config(EX["Heat equation"], REF)
    ids = _ids(cfg)
    assert {"random_seed", "final_lbfgs", "point_weights", "causal_training", "network_type", "time_adaptive"} <= ids
    assert "periodic_inputs" not in ids          # no periodic boundary condition
    assert "inv_obs_times" not in ids            # forward problem


def test_availability_steady_and_ode():
    poisson = dx.build_config(EX["Poisson 1D – Dirichlet"], REF)
    ids = _ids(poisson)
    assert "causal_training" not in ids and "time_adaptive" not in ids and "exact_ic" not in ids
    ode = dx.build_config(EX["Lotka-Volterra equation"], REF)
    assert "causal_training" not in _ids(ode) and "periodic_inputs" not in _ids(ode)


def test_availability_phase_field_and_time_adaptive_templates():
    ac = _template("1D Allen-Cahn (whole interval)")
    assert "periodic_inputs" in _ids(ac)
    ta = _template("2D Allen-Cahn (Wight & Zhao)")
    assert ta.time_adaptive
    ids = _ids(ta)
    assert "time_adaptive" in ids
    # offered with time windows on, so a Specified-combinations sweep can
    # switch the windows off and the exact IC on in the same run
    assert "exact_ic" in ids
    best = _apply(_apply(ta, "time_adaptive", "off"), "exact_ic", "on")
    _ok(best)


def test_availability_inverse():
    inv = _template("2D Allen-Cahn (Wight & Zhao)", inverse=True)
    ids = _ids(inv)
    assert "inv_obs_times" in ids
    assert "time_adaptive" not in ids             # Inverse mode trains on the whole interval


def test_seed_lbfgs_weights_causal_network_change_the_script():
    cfg = dx.build_config(EX["Heat equation"], REF)
    assert "dde.config.set_random_seed(7)" in _ok(_apply(cfg, "random_seed", 7))
    lb = _apply(cfg, "final_lbfgs", 12345)
    phases = json.loads(lb.scheduler_phases)
    assert phases[-1]["optimizer"] == "lbfgs" and phases[-1]["iterations"] == 12345
    assert reg.get_param("final_lbfgs", lb).get_value(lb) == 12345
    _ok(lb)
    nolb = _apply(lb, "final_lbfgs", 0)
    assert json.loads(nolb.scheduler_phases)[-1]["optimizer"] != "lbfgs" or len(json.loads(nolb.scheduler_phases)) == 1
    s = _ok(_apply(cfg, "point_weights", "RBA"))
    assert "_PointWeights(" in s
    assert "_CausalMonitor(" in _ok(_apply(cfg, "causal_training", "on"))
    ms = _apply(cfg, "network_type", "MsFFN")
    assert ms.network_type == "MsFFN" and "MsFFN" in _ok(ms)


def test_time_adaptive_switch():
    cfg = dx.build_config(EX["Heat equation"], REF)
    on = _apply(cfg, "time_adaptive", "on")
    assert on.time_adaptive and on.adapt_method == "Time Adaptive"
    assert "Time-Adaptive Training" in _ok(on)
    off = _apply(on, "time_adaptive", "off")
    assert not off.time_adaptive and off.adapt_method == "None"


def test_exact_ic_and_periodicity_on_a_phase_field_template():
    ac = _template("1D Allen-Cahn (whole interval)")
    for pid in ("exact_ic", "periodic_inputs"):
        p = reg.get_param(pid, ac)
        if p is None:
            continue
        other = "off" if p.get_value(ac) == "on" else "on"
        changed = _apply(ac, pid, other)
        assert reg.get_param(pid, changed).get_value(changed) == other
        n_slots = len(reg._weight_slot_descriptors(changed))
        assert len(changed.loss_weights_multi.split(",")) == n_slots
        for ph in json.loads(changed.scheduler_phases or "[]"):
            assert len(str(ph.get("weights", "")).split(",")) >= n_slots
        _ok(changed)


def test_measured_data_at_every_time():
    inv = _template("2D Allen-Cahn (Wight & Zhao)", inverse=True)
    n0 = len(json.loads(inv.inverse_obs_files_json))
    more = _apply(inv, "inv_obs_times", "every reference time")
    rows = json.loads(more.inverse_obs_files_json)
    assert len(rows) > n0 and all(os.path.isfile(r["path"]) for r in rows)
    _ok(more)


def test_zip_sweep_plan():
    cfg = copy.deepcopy(dx.build_config(EX["Heat equation"], REF))
    cfg.sweep_enabled = True
    cfg.sweep_mode = "zip"
    cfg.sweep_parameters = json.dumps([
        {"id": "random_seed", "mode": "list", "values": [1, 2]},
        {"id": "point_weights", "mode": "list", "values": ["None", "RBA"]},
    ])
    assert not cfg.validate()
    runs = sr.build_runs_detailed(cfg)
    swept = [r for r in runs if not r["baseline"]]
    assert len(swept) == 2
    assert (swept[0]["config"].random_seed, swept[0]["config"].point_weights) == (1, "None")
    assert (swept[1]["config"].random_seed, swept[1]["config"].point_weights) == (2, "RBA")


def test_impossible_combination_is_skipped(tmp_path, monkeypatch):
    """Grid mode turning on time windows and causal training for a steady
    problem can't happen (not offered); a real conflict -- time windows with
    an exact IC -- is skipped with its reason instead of being trained."""
    bu = _template("1D Burgers")
    assert reg.get_param("exact_ic", bu) is not None and reg.get_param("time_adaptive", bu) is not None
    cfg = copy.deepcopy(bu)
    cfg.sweep_enabled = True
    cfg.sweep_mode = "grid"
    cfg.save_dir = str(tmp_path)
    cfg.sweep_save_dir = str(tmp_path)
    cfg.sweep_parameters = json.dumps([
        {"id": "exact_ic", "mode": "list", "values": ["on"]},
        {"id": "time_adaptive", "mode": "list", "values": ["on"]},
    ])
    calls = []
    monkeypatch.setattr(sr, "run_pinn", lambda c, **k: calls.append(c) or "DONE")
    res = sr.run_sweep(cfg)
    statuses = [r["status"] for _l, r in res]
    assert statuses == ["invalid"]
    assert calls == []                            # nothing was trained
