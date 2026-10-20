"""The newer Plot settings reach every script: PINN/Reference line colours,
the |Error| colormap, panels per row in the error-analysis figure, the
animation resolution and the font sizes. Also the Inverse save period, the
unweighted final loss (used by the sweep), and the live loss log axis."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re

from PyQt6.QtWidgets import QApplication

from pinnstudio.core import codegen as cg
from pinnstudio.core.config import PINNConfig

app = QApplication.instance() or QApplication([])


def test_helpers_read_config_and_dict():
    c = PINNConfig()
    assert cg.err_cmap(c) == c.plot_colormap == "jet"
    c.plot_err_colormap = "magma"
    assert cg.err_cmap(c) == "magma"
    assert cg.err_cmap({"colormap": "viridis", "err_colormap": ""}) == "viridis"
    assert cg.line_colors(c) == ("#ff6b6b", "#4dabf7")
    assert cg.line_colors({"pinn_color": "#000000", "ref_color": "#111111"}) == ("#000000", "#111111")
    assert cg.font_sizes({"title_fontsize": 16, "subplot_title_fontsize": "bad"}) == (16.0, 10.0, 10.0)
    assert cg.anim_dpi({"dpi_anim": 5}) == 20 and cg.anim_dpi({}) == 150


def test_inverse_save_period():
    c = PINNConfig()
    for text, n in (("No", 0), ("Every 100 iters", 100), ("Every 250 iters", 250), ("1,000", 1000), ("junk", 100)):
        c.inv_param_save = text
        assert cg.inv_param_period(c) == n, text


def _example(name):
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w.quick_examples_category_combo.setCurrentText("Forward")
    app.processEvents()
    w.quick_examples_combo.setCurrentText(name)
    for _ in range(3):
        app.processEvents()
    c = w._build_config()
    w.close()
    return c


def test_scripts_carry_the_settings():
    c = _example("Heat 2D")
    c.plot_type = "Line"
    c.plot_pinn_color, c.plot_ref_color = "#212529", "#0b7285"
    c.plot_err_colormap, c.plot_ea_cols, c.plot_dpi_anim = "magma", 3, 90
    c.plot_title_fontsize = 15.0
    for gen in (cg.generate_script, cg.generate_clean_script):
        t = gen(c)
        compile(t, gen.__name__, "exec")
        assert "_PLOT_DPI_ANIM = 90" in t
        assert "#212529" in t and "#0b7285" in t
        assert "_PLOT_FONTS = (15.0, 10.0, 10.0)" in t
        if "|Error|" in t:
            assert "magma" in t
        if "Error analysis" in t or "error_analysis" in t:
            assert "min(3," in t, "EA panels per row"
        assert not re.search(r"ani\.save\([^)]*dpi=\d", t), "an animation with a fixed dpi"
    assert "Final loss (unweighted)" in cg.generate_script(c), "the sweep reads it from the Solve log"


def test_helper_rcparams_and_unweighted_loss():
    import matplotlib
    ns = {}
    exec(cg._figsize_helper_code("Default", 7, 5, fonts=(17.0, 12.0, 11.0), dpi_anim=80), ns)
    assert matplotlib.rcParams["figure.titlesize"] == 17.0
    assert matplotlib.rcParams["axes.titlesize"] == 12.0 and matplotlib.rcParams["axes.labelsize"] == 11.0

    class LH:
        loss_train = [[9, 9], [2.0, 0.5]]

    class M:
        loss_weights = [1.0, 10.0]
    assert ns["_final_losses"](M(), LH()) == (2.5, 2.05)
    M.loss_weights = None
    assert ns["_final_losses"](M(), LH()) == (2.5, 2.5)


def test_sweep_reads_the_unweighted_loss():
    from pinnstudio.core import sweep_runner as sr
    log = "Final loss: 1.5e-3\nFinal loss (unweighted): 4.0e-4\n"
    assert [float(m.group(1)) for m in sr._FINAL_LOSS_RE.finditer(log)] == [1.5e-3]
    assert [float(m.group(1)) for m in sr._FINAL_LOSS_UNW_RE.finditer(log)] == [4.0e-4]


def test_dialog_and_window_round_trip():
    from pinnstudio.ui.figure_style import FigureStyleDialog
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w._plot_viz_settings.update({"pinn_color": "#123456", "err_colormap": "magma", "ea_cols": 2, "dpi_anim": 75})
    d = FigureStyleDialog(w, "solve", "General")
    s, _loss = d.values()
    assert (s["pinn_color"], s["err_colormap"], s["ea_cols"], s["dpi_anim"]) == ("#123456", "magma", 2, 75)
    c = w._build_config()
    assert (c.plot_pinn_color, c.plot_err_colormap, c.plot_ea_cols, c.plot_dpi_anim) == ("#123456", "magma", 2, 75)
    w2 = MainWindow()
    w2._apply_config(c)
    c2 = w2._build_config()
    assert (c2.plot_pinn_color, c2.plot_err_colormap, c2.plot_ea_cols, c2.plot_dpi_anim) == ("#123456", "magma", 2, 75)
    assert w2._remembered_save_dir() == "", "tests never read a remembered folder"
    i = w2.sweep_bayes_obj_combo.findData("loss_unweighted")
    assert i >= 0
    w.close()
    w2.close()
