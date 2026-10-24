"""Examples are never changed by saving. A loaded example has no file, so
saving it always asks for a name (suggested: "<example> (copy)") and says
the example itself is unchanged; choosing the example again gives its own
settings. The saved copy remembers which example it came from, so switching
it between Forward and Inverse does what the example does. A problem saved
half done (Inverse with no measured-data file yet) opens again, with what is
missing listed."""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_dialogs(monkeypatch):
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))


def _pump(n=8):
    for _ in range(n):
        app.processEvents()


def _load(w, part):
    w.quick_examples_category_combo.setCurrentText("Forward")
    _pump(2)
    names = [w.quick_examples_combo.itemText(i) for i in range(w.quick_examples_combo.count())]
    name = next(n for n in names if part in n)
    w.quick_examples_combo.setCurrentText(name)
    _pump()
    return name


def _state(w):
    c = w._build_config()
    return dict(pde=w.pde_inputs[0].text(), t_max=w.t_max.value(), ta=c.time_adaptive,
                mode=c.problem_type, data=[e.get("path") for e in json.loads(c.inverse_obs_files_json or "[]")])


@pytest.fixture
def save_to(monkeypatch, tmp_path):
    asked = []

    def _dialog(*a, **k):
        asked.append(a[2])
        return str(tmp_path / asked_name[0]), ""
    asked_name = ["copy.pinn.json"]
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(_dialog))
    return tmp_path, asked, asked_name


def test_saving_an_example_makes_a_copy(save_to):
    tmp, asked, _ = save_to
    w = MainWindow()
    name = _load(w, "Heat 1D – Dirichlet")
    iters = w._build_config().iterations
    w.num_domain.setValue(1234)
    assert w._save_problem()
    assert asked == [f"{name} (copy).pinn.json"], "an example has no file: always asks, names it a copy"
    assert "example itself is unchanged" in w.log_box.toPlainText()
    data = json.load(open(tmp / "copy.pinn.json"))
    assert data["template"]["key"] and data["config"]["num_domain"] == 1234
    assert w._save_problem() and len(asked) == 1, "the copy is saved over without asking"
    w.quick_examples_combo.setCurrentText("None")
    _pump(2)
    _load(w, "Heat 1D – Dirichlet")
    assert w.num_domain.value() != 1234 and w._build_config().iterations == iters, "the example is as it was"
    assert w._current_path is None
    w2 = MainWindow()
    w2._open_problem(str(tmp / "copy.pinn.json"))
    _pump()
    assert w2.num_domain.value() == 1234 and w2._current_template == data["template"]["key"]
    w.close()
    w2.close()


@pytest.mark.parametrize("part", ["Heat 1D – Dirichlet", "Allen-Cahn 1D – Wight", "Allen-Cahn 2D – Wight"])
def test_a_copy_switches_forward_and_inverse_like_its_example(save_to, part):
    tmp, _, asked_name = save_to
    w = MainWindow()
    _load(w, part)
    ref_fwd = _state(w)
    w.radio_inverse.setChecked(True)
    _pump()
    ref_inv = _state(w)
    asked_name[0] = "inv.pinn.json"
    assert w._save_problem()
    w2 = MainWindow()
    w2._open_problem(str(tmp / "inv.pinn.json"))
    _pump()
    assert _state(w2) == ref_inv
    w2.radio_forward.setChecked(True)
    _pump()
    assert _state(w2) == ref_fwd, "constants back in the equation, time-adaptive training back on"
    w2.radio_inverse.setChecked(True)
    _pump()
    assert _state(w2) == ref_inv
    w.close()
    w2.close()


def test_a_problem_saved_half_done_opens_again(save_to):
    tmp, _, asked_name = save_to
    w = MainWindow()
    _load(w, "Heat 1D – Dirichlet")
    w.radio_inverse.setChecked(True)          # no measured-data file picked yet
    _pump()
    asked_name[0] = "half.pinn.json"
    assert w._save_problem()
    w2 = MainWindow()
    w2._open_problem(str(tmp / "half.pinn.json"))
    _pump()
    log = w2.log_box.toPlainText()
    assert "Problem loaded: half" in log and "To fix before Solve" in log and "observed-data file" in log
    assert w2.radio_inverse.isChecked() and w2._current_path.endswith("half.pinn.json")
    w.close()
    w2.close()


def test_allen_cahn_1d_switches_between_its_paper_setup_and_inverse():
    """Allen-Cahn 1D – Wight & Zhao loads with the paper's method (Settings:
    Default, Forward only); Inverse gives the same setup as loading the
    example in Inverse, and Forward gives the paper's method back."""
    import dataclasses
    keys = ("ic_hard", "ic_active", "periodic_inputs", "causal_training", "network_type", "msffn_sigmas",
            "time_adaptive", "scheduler_phases", "inverse_obs_files_json", "loss_weights")

    def cfg(w):
        c = dataclasses.asdict(w._build_config())
        return {k: c[k] for k in keys}

    def load(w, cat, part):
        w.quick_examples_category_combo.setCurrentText(cat)
        _pump(2)
        names = [w.quick_examples_combo.itemText(i) for i in range(w.quick_examples_combo.count())]
        w.quick_examples_combo.setCurrentText(next(n for n in names if part in n))
        _pump()

    w = MainWindow()
    load(w, "Inverse", "Allen-Cahn 1D – Wight & Zhao")
    inv = cfg(w)
    w3 = MainWindow()
    load(w3, "Forward", "Allen-Cahn 1D – Wight & Zhao")
    fwd = cfg(w3)
    assert not fwd["time_adaptive"]                # the paper: the whole interval
    w3.radio_inverse.setChecked(True)
    _pump()
    assert cfg(w3) == inv
    w3.radio_forward.setChecked(True)
    _pump()
    assert cfg(w3) == fwd
    for x in (w, w3):
        x.close()
