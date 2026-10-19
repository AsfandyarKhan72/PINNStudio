"""Plot settings: one dialog for every saved figure, opened by the Results
page's Plot settings button (on the tab of the plot type), Settings -> Plot
Settings and Restore's link. Solve, the exported script and Restore draw with
the same settings. An inverse problem's parameter convergence plot is set on
the Line tab and its animation on Animation (red, width 3 by default). The
plot types a problem can have follow one rule on both pages, and the Results
page's "Outputs to plot" list replaces the old output list plus "All
outputs" tick box."""
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
    assert fs.tab_for("Parameter Convergence") == "Line"
    assert fs.tab_for("Parameter Convergence (GIF)") == "Animation"
    assert fs.tab_for(None) == "General"


def test_style_buttons_open_the_plot_types_tab_and_no_popup_on_change(monkeypatch):
    w = MainWindow()
    calls = []
    monkeypatch.setattr(QDialog, "exec", lambda dlg: calls.append(dlg) or 0)
    w.plot_type_combo.setCurrentText("Line (time steps)")
    assert not calls, "changing the plot type no longer opens a dialog"
    seen = _open(monkeypatch, w, w.plot_style_btn.click)
    assert seen["tab"] == "Line"
    assert seen["dlg"].windowTitle() == "Plot Settings"
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
    # Restore draws with the same settings, its steps and frames included
    st = w._restore_style()
    assert (st["colormap"], st["n_steps_line"], st["n_steps_anim"]) == ("viridis", 7, 13)

    def edit_r(d):
        d.cmap.setCurrentText("plasma")
    seen = _open(monkeypatch, w, lambda: w.restore_style_note.linkActivated.emit("#"), edit=edit_r)
    assert seen["dlg"].windowTitle() == "Plot Settings"
    assert w._build_config().plot_colormap == "plasma", "one set of settings"
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
        d.p_true_style.setCurrentIndex(d.p_true_style.findData(":"))
        d.pa_lw.setCurrentText("1.5")
        d.p_log.setChecked(True)
    assert w._build_config().param_plot_color == "#e03131" and w._build_config().param_plot_linewidth == 3.0
    seen = _open(monkeypatch, w, w.plot_style_btn.click, edit=edit)
    assert seen["tab"] == "Line"
    cfg = w._build_config()
    assert cfg.param_plot_color == "#1c7ed6" and cfg.param_plot_markers and not cfg.param_plot_show_final
    assert cfg.param_true_style == ":" and cfg.param_anim_linewidth == 1.5 and cfg.param_anim_color == "#e03131"
    assert cfg.inv_param_log_scale
    for plot_type in ("Parameter Convergence", "Parameter Convergence (GIF)"):
        cfg.plot_type = plot_type
        for gen in (generate_script, generate_clean_script):
            text = gen(cfg)
            assert "'color': '#1c7ed6'" in text and "'markers': True" in text and "'show_final': False" in text
            if gen is generate_clean_script:      # (Solve's script decides when it runs)
                assert ("_param_animation(" in text.split("def _param_animation")[-1]) == plot_type.endswith("(GIF)")
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
    assert w.restore_viz_combo.currentText() == "Parameter Convergence", "selected first for an inverse model"
    assert [w.restore_viz_combo.itemText(i) for i in range(w.restore_viz_combo.count())] == \
        ["Surface", "Line (time steps)", "Animation Line (GIF)", "Parameter Convergence", "Parameter Convergence (GIF)"]
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


def test_plot_types_follow_the_problem():
    f = MainWindow.plot_types_for
    assert f("ODE", False, False) == ["Line"] and f("1D", True, False) == ["Line"]
    assert f("1D", False, False) == ["Surface", "Line (time steps)", "Line Animation (GIF)"]
    assert f("2D", True, False) == ["Surface", "Line (time steps)"]
    assert f("3D", False, False)[-1] == "Surface Animation (GIF)"
    assert f("ODE", False, True) == ["Line", "Parameter Convergence", "Parameter Convergence (GIF)"]
    assert f("1D", False, False, restore=True)[-1] == "Animation Line (GIF)"
    assert f("2D", True, False, restore=True) == ["Surface"]


def test_results_plot_types_per_example():
    w = MainWindow()

    def items():
        return [w.plot_type_combo.itemText(i) for i in range(w.plot_type_combo.count())]
    _load(w, "Forward", "Linear ODE system")
    assert items() == ["Line"]
    _load(w, "Forward", "Poisson 1D – Dirichlet")
    assert items() == ["Line"]
    _load(w, "Forward", "Heat 1D – Dirichlet")
    assert "Surface Animation (GIF)" not in items()
    _load(w, "Inverse", "Lorenz ODE – unknown C₁, C₂, C₃")
    assert items() == ["Line", "Parameter Convergence", "Parameter Convergence (GIF)"]
    assert w.plot_type_combo.currentText() == "Parameter Convergence", "an inverse example opens on its parameters"
    _load(w, "Inverse", "Diffusion 1D – unknown coefficient")
    assert w.plot_type_combo.currentText() == "Parameter Convergence"
    w.close()


def test_convergence_file_rows_are_not_squeezed(tmp_path):
    """With an example loaded first, the convergence-file rows got less height
    than their fields (a wrapped label above them kept a stale height)."""
    import time
    w = MainWindow()
    w.resize(1300, 650)
    w.show()
    _load(w, "Inverse", "Allen-Cahn 2D – Wight & Zhao")
    w.workspace.select("restore")
    run = tmp_path / "run"
    (run / "solution_results").mkdir(parents=True)
    model = run / "solution_results" / "model_lbfgs-phase2-100.pt"
    model.write_bytes(b"")
    (run / "solution_results" / "model_config.json").write_text(json.dumps(
        {"problem_dim": "2D", "problem_type": "Inverse", "num_outputs": 1, "output_names": "u",
         "inverse_variables": [{"name": n, "init": 1.0, "true": 2.0} for n in ("a", "b", "c")]}))
    for n in ("a", "b", "c"):
        (run / f"{n}_convergence.txt").write_text(f"iteration,{n}\n0,1.0\n100,1.5\n")
    w._on_restore_model_chosen(str(model))
    w.restore_viz_combo.setCurrentText("Parameter Convergence (GIF)")
    w._add_restore_param_row()
    t = time.time()
    while time.time() - t < 0.5:
        app.processEvents()
    assert len(w.restore_param_rows) == 4
    for r in w.restore_param_rows:
        assert r["widget"].height() >= r["path"].minimumHeight(), (r["widget"].height(), r["path"].minimumHeight())
    w.close()


def test_restore_shows_its_figure_alone(tmp_path):
    """A restore shows one panel across the Overview (with "Show loss plot"
    when the model's loss plot is there); a training run's results bring
    the loss panel back."""
    from types import SimpleNamespace
    w = MainWindow()
    w.show()
    run = tmp_path / "Heat_2D__2026-10-18_12-00-00"
    (run / "solution_results").mkdir(parents=True)
    w._last_restore_model_dirs = [str(run / "solution_results"), str(run)]
    w._restore_loss_path = str(run / "solution_results" / "loss_plot.png")
    w._show_restore_panel(w._restore_title("Surface"))
    assert w._loss_container.isHidden()
    assert w.solution_header_label.text() == "Restored solution · Heat 2D · Surface"
    assert not w.restore_loss_link.isHidden()
    w._display_run_result_plots(SimpleNamespace(plot_type="Surface"), str(tmp_path / "none"))
    assert not w._loss_container.isHidden() and w.solution_header_label.text() == "Solution"
    assert w.restore_loss_link.isHidden()
    w.close()


def test_parameter_animation_log_axis_holds_the_curve(tmp_path):
    """On a log axis the animation's value range was set by the two
    reference lines only, so the curve was drawn off the axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from pinnstudio.core.codegen import _param_axis_helper_code
    ns = {"np": np, "plt": plt}
    exec(_param_axis_helper_code(10, 10), ns)
    it = np.arange(0, 4001, 100)
    vals = 0.9 * np.exp(-it / 300) + 2.6e-3
    seen = {}
    real_save = matplotlib.animation.Animation.save

    def save(self, *a, **k):
        ax = self._fig.axes[0]
        seen["ylim"], seen["scale"], seen["label"] = ax.get_ylim(), ax.get_yscale(), ax.get_ylabel()
    matplotlib.animation.Animation.save = save
    try:
        ns["_param_animation"]([("eps", it, vals, 2.5e-3)], str(tmp_path / "a.gif"), log_scale=True)
    finally:
        matplotlib.animation.Animation.save = real_save
    lo, hi = seen["ylim"]
    assert seen["scale"] == "log" and seen["label"] == "eps"
    assert lo <= 2.5e-3 and hi >= vals.max(), seen
