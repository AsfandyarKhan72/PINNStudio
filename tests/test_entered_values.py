"""Values the user enters are used as entered: number boxes keep full
precision (a loss weight of 1.2345e-5 used to become 1.23e-5), Plot
settings lists keep a value that is not in the list (a grid of 500 used to
become 80), small learning rates, decay rates and tolerances are not
rounded to 0, a network with different widths per layer reopens as saved,
and typed loss weights survive the rebuilds (a BC's type, a new phase,
Steady-state, the geometry)."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from pinnstudio.ui.main_window import InitGuessLineEdit, MainWindow, SciLineEdit
from pinnstudio.ui.sci_spinbox import SciSpinBox

app = QApplication.instance() or QApplication([])


def _pump(n=5):
    for _ in range(n):
        app.processEvents()


def test_number_boxes_keep_full_precision():
    for v in (1.2345e-5, 1234567.0, 0.00012345, 3.14159265, 1e-6, 2.5e10):
        e = SciLineEdit()
        e.setValue(v)
        assert e.value() == v, (v, e.text())
    g = InitGuessLineEdit()
    g.setValue(1.0)
    assert g.text() == "1.0"
    b = SciSpinBox()
    b.setRange(1e-12, 10)
    for text, v in (("5e-7", 5e-7), ("0.00025", 2.5e-4), ("1e-6", 1e-6)):
        b.lineEdit().setText(text)
        b.interpretText()
        assert b.value() == v and b.text() == f"{v:.12g}"
    b.setDecimals(4)                     # a caller setting decimals doesn't round it
    b.setValue(5e-7)
    assert b.value() == 5e-7


def test_plot_settings_lists_keep_any_value(monkeypatch):
    from PyQt6.QtWidgets import QDialog

    from pinnstudio.ui.figure_style import FigureStyleDialog, _combo
    assert _combo(["80", "100", "200"], "500").currentText() == "500"
    w = MainWindow()
    w._plot_viz_settings.update({"resolution": 500, "dpi": 250, "fps": 12, "colormap": "cividis",
                                 "linewidth": 1.75})
    d = FigureStyleDialog(w, "solve", "General")
    s, _loss = d.values()
    assert (s["resolution"], s["dpi"], s["fps"], s["colormap"], s["linewidth"]) == (500, 250, 12, "cividis", 1.75)
    d.res.setCurrentText("650")
    d.cmap.setCurrentText("not_a_colormap")
    s, _loss = d.values()
    assert s["resolution"] == 650 and s["colormap"] == "cividis", "a name matplotlib lacks keeps the old one"
    d.vmin.setValue(-2.5e-4)
    assert d.values()[0]["vmin"] == -2.5e-4
    w.close()


def test_small_rates_and_wide_limits():
    w = MainWindow()
    if not w.sched_phase_list:
        w._add_scheduler_phase('adam', 1000, 0.001)
    ph = w.sched_phase_list[0]
    ph['lr'].setValue(5e-7)
    assert ph['lr'].value() == 5e-7
    ph['iters'].setValue(2_000_000)
    assert ph['iters'].value() == 2_000_000
    w.rar_tolerance.setValue(5e-7)
    assert w.rar_tolerance.value() == 5e-7
    w.t_min.setValue(-1.0)
    assert w.t_min.value() == -1.0
    w.num_domain.setValue(1_000_000)
    assert w._build_config().num_domain == 1_000_000
    w.timesteps_spin_anim.setValue(120)                 # animations were capped at 20 frames
    w.timesteps_spin_line.setValue(30)
    c = w._build_config()
    assert (c.num_timesteps_anim, c.num_timesteps_line) == (120, 30)
    w.close()


def test_layer_widths_reopen_as_saved():
    w = MainWindow()
    c = w._build_config()
    c.layers = [2, 100, 50, 1]
    w._apply_config(c)
    _pump()
    assert w._build_config().layers == [2, 100, 50, 1] and w.layer_widths_edit.text() == "100, 50"
    c.layers = [2, 64, 64, 64, 1]
    w._apply_config(c)
    _pump()
    assert w._build_config().layers == [2, 64, 64, 64, 1] and not w.layer_widths_edit.text()
    w.layer_widths_edit.setText("32, oops")
    assert w._build_config().layers == [2, 64, 64, 64, 1], "not a list of widths: the boxes above"
    w.close()


def test_typed_loss_weights_survive_rebuilds():
    w = MainWindow()
    w.quick_examples_category_combo.setCurrentText("Forward")
    _pump(2)
    w.quick_examples_combo.setCurrentText("Heat 2D")
    _pump()

    def setw():
        for b in w.weight_widgets.values():
            b.setValue(7.5)

    def kept():
        return all(b.value() == 7.5 for b in w.weight_widgets.values())

    setw()
    t = w.custom_bc_list[0]['type']
    t.setCurrentIndex((t.currentIndex() + 1) % t.count())
    _pump()
    assert kept(), "a BC's type"
    setw()
    w.sched_same_weights_cb.setChecked(not w.sched_same_weights_cb.isChecked())
    _pump()
    assert kept(), "per-phase weights start from the shared ones"
    w.sched_same_weights_cb.setChecked(not w.sched_same_weights_cb.isChecked())
    _pump()
    assert kept(), "and back"
    setw()
    w._add_scheduler_phase('adam', 1000, 0.001)
    _pump()
    assert kept(), "a new phase"
    setw()
    w.steady_state_check.setChecked(True)
    _pump()
    assert kept(), "Steady-state"
    w.close()


def test_cpu_threads_setting():
    from pinnstudio.core import codegen as cg
    w = MainWindow()
    c = w._build_config()
    assert c.cpu_threads == 0
    assert "min(8, (os.cpu_count() or 2) // 2)" in cg.generate_script(c), "Automatic: the old rule"
    c.cpu_threads = 6
    for gen in (cg.generate_script, cg.generate_clean_script):
        assert "torch.set_num_threads(6)" in gen(c)
    w2 = MainWindow()
    w2._apply_config(c)
    assert w2._build_config().cpu_threads == 6
    w2.quick_examples_category_combo.setCurrentText("Forward")
    _pump(2)
    w2.quick_examples_combo.setCurrentText("Heat 1D – Dirichlet")
    _pump()
    assert w2._build_config().cpu_threads == 6, "a machine setting, kept when an example is loaded"
    c.cpu_threads = -1
    assert any("CPU threads" in e for e in c.validate())
    w.close()
    w2.close()
