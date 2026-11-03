"""GUI regression checks for the v1.6.0 audit (P2 group): settings that the window must offer, keep through
Save/Open, and pass to both generated scripts."""
import dataclasses
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from pinnstudio.core import codegen  # noqa: E402
from pinnstudio.ui.main_window import MainWindow  # noqa: E402


def _settle(s=0.2):
    t = time.time()
    while time.time() - t < s:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()


def _window():
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    _settle(0.3)
    return w


def test_ic_file_anchors_has_a_checkbox_and_survives_save_and_open():
    """P2: 'ic_file_anchors' was used by both generators but had no widget, so the GUI reset it to False and a
    problem saved with it on lost it."""
    w = _window()
    w.radio_1d.setChecked(True)
    _settle()
    assert w._build_config().ic_file_anchors is False
    w.ic_from_file[0].setChecked(True)
    w.ic_file_paths[0].setText("/tmp/ic.txt")
    w.ic_file_anchor_cb.setChecked(True)
    c = w._build_config()
    assert c.forward_ic_from_file and c.ic_file_anchors
    for s in (codegen.generate_script(c), codegen.generate_clean_script(c)):
        assert "anchors=_ic_xt" in s or "anchors=_ic_xyt" in s
    w2 = _window()
    w2._apply_config(c)
    _settle(0.3)
    assert w2._build_config().ic_file_anchors is True
    w2.ic_from_file[0].setChecked(False)              # no IC file: the setting has no meaning
    assert w2._build_config().ic_file_anchors is False
    w.close(); w2.close()


def test_one_precision_setting_and_restore_uses_it():
    """P2: Settings > Float Precision and the Training page's Precision were two controls; float64 in either ran the
    whole training in float64, but the status bar and the log showed only the first, and model_config.json did not
    record it, so Restore always rebuilt the model in float32."""
    w = _window()
    w.lbfgs_float_combo.setCurrentText("float64")
    _settle(0.1)
    c = w._build_config()
    assert (c.float_type, c.lbfgs_float_type) == ("float64", "float64") and w.status_env.text().startswith("float64")
    w2 = _window()
    w2._apply_config(dataclasses.replace(c, float_type="float32", lbfgs_float_type="float64"))
    _settle(0.2)
    c2 = w2._build_config()
    assert (c2.float_type, c2.lbfgs_float_type) == ("float64", "float64")
    s = codegen.generate_script(c2)
    assert 'Float type:   {_effective_float}' in s and '"float_type": _effective_float' in s
    mc = dict(dataclasses.asdict(c2), float_type="float64")       # what model_config.json holds
    rs = w2._build_restore_script("m.pt", mc, "adam", "Line", 0, 3, "/tmp")
    assert "dde.config.set_default_float('float64')" in rs
    w.close(); w2.close()


def test_the_window_offers_only_working_loss_functions():
    w = _window()
    for combo in [w.loss_combo, w.ic_pretrain_loss_combo] + [p["loss"] for p in w.sched_phase_list if "loss" in p]:
        assert [combo.itemText(i) for i in range(combo.count())] == ["MSE", "MAE"]
    w.close()


def test_small_gui_fixes_found_while_writing_the_documentation():
    """P2/P3: cosine decay kept Step's gamma 0.9 as Eta min (the rate rose); a new 3D problem kept the 201-point
    time-window grid (201^3 points); the BC 'Output' box stopped at output 8 while Outputs go to 10."""
    import json
    w = _window()
    dt = w.sched_phase_list[0]["decay_type"]
    dt.setCurrentIndex(dt.findData("cosine"))
    _settle(0.1)
    assert json.loads(w._build_config().scheduler_phases)[0]["decay_p2"] == 0.0
    w.ta_grid.setCurrentText("201")
    w.radio_3d.setChecked(True)
    _settle(0.2)
    assert w.ta_grid.currentText() == "21"
    w._add_custom_bc_entry("dirichlet", 9, "True", "0")
    assert w.custom_bc_list[-1]["component"].value() == 9
    w.close()


def test_time_window_resampling_candidates_can_be_set():
    """P2: the candidates per resampling round were fixed at 10,000 with no control, so more than 10,000 points
    per round gave an error the window could not resolve."""
    from pinnstudio.core import codegen as cg
    w = _window()
    w.ta_rs_rounds.setValue(1)
    w.ta_rs_points.setValue(20000)
    w.ta_rs_cand.setValue(50000)
    c = w._build_config()
    assert c.ta_resample_candidates == 50000
    assert not any("candidates" in e for e in __import__("dataclasses").replace(c, time_adaptive=True).validate())
    assert "_ta_rs_cand = 50000" in cg.generate_script(__import__("dataclasses").replace(c, time_adaptive=True))
    w2 = _window()
    w2._apply_config(c)
    _settle(0.2)
    assert w2.ta_rs_cand.value() == 50000
    w.close(); w2.close()


def test_renamed_outputs_show_their_new_names_on_other_pages():
    """Renaming an output left the Loss weights rows and the Results/Restore output lists with the old name
    until the problem was reopened."""
    from PySide6.QtWidgets import QLabel
    w = _window()
    w._new_problem("ODE")
    _settle()
    w.num_outputs_spin.setValue(2)
    _settle(0.3)
    w.weight_widgets[sorted(w.weight_widgets)[0]].setText("7")
    w.output_name_inputs[0].setText("y1")
    w.output_name_inputs[1].setText("y2")
    _settle(0.6)
    labels = [lb.text() for lb in w.findChildren(QLabel) if lb.text().startswith(("PDE", "IC"))]
    assert any("(y1)" in t for t in labels) and not any("(u)" in t for t in labels), labels
    assert w.plot_output_combo.itemText(0) == "Output 1 (y1)" and w.restore_output_combo.itemText(1) == "Output 2 (y2)"
    assert "7" in [x.text() for x in w.weight_widgets.values()], "a typed weight is kept"


def test_restore_finds_the_formula_reference_files_of_the_run(tmp_path):
    """Restore with a formula reference looked for reference_from_formula/ under the base Save to folder
    (model_config.json has no run folder) and never found the files the run wrote."""
    import dataclasses as dc
    from pinnstudio.core.config import PINNConfig
    from pinnstudio.core.codegen import _formula_reference_entries
    run = tmp_path / "1D_Heat__2026-11-01_00-00-00"
    (run / "solution_results").mkdir(parents=True)
    model = run / "solution_results" / "model_lbfgs-phase2-400.pt"
    model.write_bytes(b"")
    c = dc.replace(PINNConfig(), save_dir=str(run), ea_ref_mode="formula", ea_ref_formula="sin(pi*x)*exp(-t)")
    entries = _formula_reference_entries(c)
    assert entries
    for _t, pth, _i, _e in entries:
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        open(pth, "w").write("0 0 0\n")
    cfg = dc.asdict(dc.replace(c, save_dir=str(tmp_path / "elsewhere")))   # what model_config.json would give
    w = _window()
    w.restore_model_path.setText(str(model))
    ea = w._ea_with_formula_files({"ref_mode": "formula", "ref_formula": c.ea_ref_formula}, cfg)
    assert len(ea["files"]) == len(entries)


def test_restore_skips_error_analysis_set_up_for_another_problem():
    """Restore used the Error Analysis reference files of whatever problem was in the window: a 1D Burgers
    model restored with the steady 2D L-shape example loaded failed inside matplotlib ("Input z must be at
    least a (2, 2) shaped array, but has shape (1, 200)")."""
    w = _window()
    w.quick_examples_category_combo.setCurrentText("Forward")
    _settle(0.1)
    w.quick_examples_combo.setCurrentIndex(w.quick_examples_combo.findText("Poisson 2D – L-shape, Lu et al."))
    _settle(0.8)
    burgers_1d = {"problem_dim": "1D", "steady_state": False, "ode": False, "output_names": "u"}
    assert "the model is 1D" in w._restore_ea_mismatch(burgers_1d)
    w.quick_examples_combo.setCurrentIndex(w.quick_examples_combo.findText("Burgers 1D – Raissi et al."))
    _settle(0.8)
    assert w._restore_ea_mismatch(burgers_1d) == ""
    assert "time-dependent" in w._restore_ea_mismatch(dict(burgers_1d, steady_state=True))
    assert "outputs u" in w._restore_ea_mismatch(dict(burgers_1d, output_names="v"))
    # an ODE model (prepared as 1D with ode=True) against an ODE problem in the window
    w._new_problem("ODE")
    _settle(0.3)
    assert w._restore_ea_mismatch({"problem_dim": "1D", "ode": True, "steady_state": True,
                                   "output_names": w._build_config().output_names}) == ""
    w.close()
