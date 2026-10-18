"""Figure style: one dialog for every saved figure, opened by the Style...
buttons of Results and Restore & visualize (on the tab of the plot type) and
from the Settings menu. Solve and Restore draw with the same style; the time
steps / frames and the y/z slice stay each one's own. The parameter
convergence plots have their own tab (inverse problems only), and the
Results page's "Outputs to plot" list replaces the old output list plus
"All outputs" tick box."""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from pinnstudio.core.codegen import generate_clean_script, generate_script
from pinnstudio.ui import figure_style as fs
from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_dialogs(monkeypatch):
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))


def _load(w, category, name):
    w.quick_examples_category_combo.setCurrentText(category)
    app.processEvents()
    w.quick_examples_combo.setCurrentText(name)
    for _ in range(10):
        app.processEvents()


def _open(monkeypatch, w, act, **kw):
    """Open the dialog through `act`, let `kw["edit"](dlg)` change it, press OK."""
    seen = {}

    def _exec(dlg):
        seen["dlg"] = dlg
        seen["tab"] = dlg.tabs.tabText(dlg.tabs.currentIndex())
        seen["tabs"] = [dlg.tabs.tabText(i) for i in range(dlg.tabs.count()) if dlg.tabs.isTabVisible(i)]
        if kw.get("edit"):
            kw["edit"](dlg)
        dlg._ok()
        return 1
    monkeypatch.setattr(QDialog, "exec", _exec)
    act()
    return seen


def test_tab_for_plot_types():
    assert fs.tab_for("Surface") == "Surface"
    assert fs.tab_for("Line (time steps)") == "Line"
    assert fs.tab_for("Line Animation (GIF)") == "Animation"
    assert fs.tab_for("Animation Surface (GIF)") == "Animation"
    assert fs.tab_for("Parameter Convergence") == "Parameters"
    assert fs.tab_for("Parameter convergence (GIF)") == "Parameters"
    assert fs.tab_for(None) == "General"


def test_style_buttons_open_the_plot_types_tab_and_no_popup_on_change(monkeypatch):
    w = MainWindow()
    calls = []
    monkeypatch.setattr(QDialog, "exec", lambda dlg: calls.append(dlg) or 0)
    w.plot_type_combo.setCurrentText("Line (time steps)")
    assert not calls, "changing the plot type no longer opens a dialog"
    seen = _open(monkeypatch, w, w.plot_style_btn.click)
    assert seen["tab"] == "Line"
    assert "Parameters" not in seen["tabs"], "forward problem: no parameter tab"
    w.close()


def test_one_style_for_solve_and_restore_steps_kept_apart(monkeypatch):
    w = MainWindow()

    def edit(d):
        d.cmap.setCurrentText("viridis")
        d.dpi.setCurrentText("200")
        d.steps_line.setValue(7)
        d.frames.setValue(13)
        d.lw_line.setCurrentText("1.5")
        d.lw_anim.setCurrentText("4.0")
        d.loss_lw.setCurrentText("3.0")
    _open(monkeypatch, w, lambda: w._on_figure_style("Surface", "solve"), edit=edit)
    cfg = w._build_config()
    assert (cfg.plot_colormap, cfg.plot_dpi) == ("viridis", 200)
    assert (cfg.num_timesteps_line, cfg.num_timesteps_anim) == (7, 13)
    assert (cfg.plot_linewidth, cfg.plot_linewidth_anim, cfg.loss_plot_linewidth) == (1.5, 4.0, 3.0)
    # Restore draws with the same style, but keeps its own steps
    assert w._restore_style()["colormap"] == "viridis"

    def edit_r(d):
        d.steps_line.setValue(9)
        d.cmap.setCurrentText("plasma")
    _open(monkeypatch, w, lambda: w._on_restore_viz_settings("Line (time steps)"), edit=edit_r)
    assert w._restore_viz_settings["n_steps_line"] == 9
    assert w.timesteps_spin_line.value() == 7, "Restore's steps are its own"
    assert w._build_config().plot_colormap == "plasma", "one style"
    # saved with the problem and read back
    w2 = MainWindow()
    w2._apply_config(w._build_config())
    assert w2._plot_viz_settings["colormap"] == "plasma" and w2._loss_plot_settings["linewidth"] == 3.0
    w.close()
    w2.close()


def test_parameter_style_reaches_the_scripts(monkeypatch):
    w = MainWindow()
    _load(w, "Inverse", "Diffusion 1D – unknown coefficient")
    w.plot_type_combo.setCurrentText("Parameter Convergence")

    def edit(d):
        d.p_color.setCurrentIndex(d.p_color.findData("#1c7ed6"))
        d.p_markers.setChecked(True)
        d.p_final.setChecked(False)
        d.p_log.setChecked(True)
    seen = _open(monkeypatch, w, w.plot_style_btn.click, edit=edit)
    assert seen["tab"] == "Parameters"
    cfg = w._build_config()
    assert cfg.param_plot_color == "#1c7ed6" and cfg.param_plot_markers and not cfg.param_plot_show_final
    assert cfg.inv_param_log_scale
    for gen in (generate_script, generate_clean_script):
        text = gen(cfg)
        assert "'color': '#1c7ed6'" in text and "'markers': True" in text and "'show_final': False" in text
        compile(text, gen.__name__, "exec")
    w2 = MainWindow()
    w2._apply_config(cfg)
    assert w2._build_config().param_plot_color == "#1c7ed6"
    w.close()
    w2.close()


def test_outputs_to_plot(monkeypatch):
    w = MainWindow()
    _load(w, "Forward", "Schrödinger 1D")
    c = w.plot_outputs_choice
    items = [c.itemText(i) for i in range(c.count())]
    assert items[-1] == "Custom..." and items[0] in ("All outputs", "Custom...")
    _load(w, "Forward", "Navier-Stokes 2D – Kovasznay flow")
    items = [c.itemText(i) for i in range(c.count())]
    assert items[0] == "All outputs" and items[-1] == "Custom..." and len(items) == 5
    assert c.currentText() == "All outputs"
    cfg = w._build_config()
    assert cfg.plot_all_outputs and cfg.plot_output_idx == 0
    c.setCurrentIndex(2)            # Output 2
    cfg = w._build_config()
    assert not cfg.plot_all_outputs and cfg.plot_output_idx == 1
    w2 = MainWindow()
    w2._apply_config(cfg)
    assert w2.plot_outputs_choice.currentText() == items[2], "read back from a saved problem"
    cfg.plot_all_outputs = True
    w2._apply_config(cfg)
    assert w2.plot_outputs_choice.currentText() == "All outputs"
    _load(w, "Forward", "Heat 1D – Dirichlet")
    assert [c.itemText(i) for i in range(c.count())][0].startswith("Output 1"), "one output: no 'All outputs'"
    w.close()
    w2.close()


def test_restore_reads_forward_or_inverse_from_the_config(tmp_path):
    w = MainWindow()
    run = tmp_path / "run"
    (run / "solution_results").mkdir(parents=True)
    model = run / "solution_results" / "model_lbfgs-phase2-100.pt"
    model.write_bytes(b"")
    cfg = {"problem_dim": "1D", "problem_type": "Inverse", "num_outputs": 1, "output_names": "u",
           "optimizer": "adam", "inverse_variables": [{"name": "C", "init": 2.0, "true": 1.0}]}
    (run / "solution_results" / "model_config.json").write_text(json.dumps(cfg))
    (run / "C_convergence.txt").write_text("iteration,C\n0,2.0\n100,1.5\n")
    w._on_restore_model_chosen(str(model))
    assert w.restore_mode_combo.currentText() == "Inverse Model"
    assert w.restore_viz_combo.currentText() == "Parameter convergence", "selected first for an inverse model"
    assert [r['path'].text() for r in w.restore_param_rows] == [str(run / "C_convergence.txt")]
    assert [r['true'].text() for r in w.restore_param_rows] == ["1.0"]
    assert "inverse" in w.restore_info_label.text() and "C" in w.restore_info_label.text()
    assert not w.restore_inv_vars_widget.isVisibleTo(w), "the config lists the parameters"
    assert not w.restore_optimizer_widget.isVisibleTo(w)
    assert w.restore_optimizer_combo.currentData() == "lbfgs", "from the file name"
    # a forward model takes the parameter plots away again
    cfg.update(problem_type="Forward", inverse_variables=[])
    (run / "solution_results" / "model_config.json").write_text(json.dumps(cfg))
    w._on_restore_config_chosen()
    assert w.restore_mode_combo.currentText() == "Forward Model"
    assert all("arameter" not in w.restore_viz_combo.itemText(i) for i in range(w.restore_viz_combo.count()))
    w.close()
