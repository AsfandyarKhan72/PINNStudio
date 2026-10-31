"""GUI regression checks for the v1.6.0 audit (P2 group): settings that the window must offer, keep through
Save/Open, and pass to both generated scripts."""
import dataclasses
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

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
