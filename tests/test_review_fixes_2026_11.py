"""Fixes from the November 2026 code review, and the GUI additions made with them.

Window: BC loss weights follow their rows when one is deleted; training phases are renumbered (with their
weights) when one is removed; sweep rows follow the rows they sweep; typed initial conditions, equations
and output transforms survive a change of shape or of the number of outputs; the command palette; the
System and High Contrast themes. Core: precise and cached sketch edges, SVG import (rotated ellipse,
compact arc flags), sweep folder names of negative values, the ODE monitor-point size, anchors inside the
domain, quoted plot labels in generated scripts, and the time-adaptive loss history.

Run directly:
    QT_QPA_PLATFORM=offscreen python tests/test_review_fixes_2026_11.py
"""
import ast
import dataclasses
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

app = QApplication.instance() or QApplication([])

from pinnstudio.ui import theme  # noqa: E402
from pinnstudio.ui.main_window import MainWindow  # noqa: E402


def _window(dim="1D"):
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w._new_problem(dim)
    app.processEvents()
    return w


def _close(w):
    w.close()
    w.deleteLater()
    app.processEvents()


def _remove_phase(w, k):
    btn = [b for b in w.sched_phase_list[k]['widget'].findChildren(QPushButton) if b.text() == "✕"][0]
    btn.click()
    app.processEvents()


def test_bc_weights_follow_their_rows():
    w = _window()
    try:
        while len(w.custom_bc_list) < 3:
            w.add_custom_bc_btn.click()
            app.processEvents()
        for j, v in enumerate([100.0, 2.0, 3.0]):
            w.weight_widgets[f"bc_{j}"].setValue(v)
        w.custom_bc_list[0]['remove_btn'].click()
        app.processEvents()
        assert [w.weight_widgets[f"bc_{j}"].value() for j in range(2)] == [2.0, 3.0]
    finally:
        _close(w)


def test_phases_renumbered_with_their_weights_and_sweeps():
    w = _window()
    try:
        w._add_scheduler_phase()                     # three phases
        app.processEvents()
        w.sched_same_weights_cb.setChecked(False)
        app.processEvents()
        w.weight_widgets["pde_0_p3"].setValue(7.0)
        w.sweep_enable_cb.setChecked(True)
        app.processEvents()
        while len(w.sweep_row_list) < 2:
            w._add_sweep_row()
            app.processEvents()
        r0, r1 = w.sweep_row_list[:2]
        r0['param_combo'].setCurrentIndex(r0['param_combo'].findData("phase2_lr"))
        r1['param_combo'].setCurrentIndex(r1['param_combo'].findData("phase0_lr"))
        _remove_phase(w, 0)
        assert [p['phase_num'] for p in w.sched_phase_list] == [1, 2]
        assert [p['header'].text() for p in w.sched_phase_list] == ["── Phase 1 ──", "── Phase 2 ──"]
        assert w.weight_widgets["pde_0_p2"].value() == 7.0, "old phase 3 keeps its weight as phase 2"
        w._add_scheduler_phase()
        app.processEvents()
        assert [p['phase_num'] for p in w.sched_phase_list] == [1, 2, 3], "no duplicate number"
        assert r0['param_combo'].currentData() == "phase1_lr", "the sweep follows its phase"
        assert "not available" in r1['param_combo'].currentText(), "a sweep on the removed phase is marked"
    finally:
        _close(w)


def test_typed_values_survive_rebuilds():
    w = _window("2D")
    try:
        w.ic_inputs[0].setText("exp(-10*(x**2+y**2))")
        w.pde_inputs[0].setText("du_t - du_xx")
        w.output_name_inputs[0].setText("c")
        w.output_transform_rows[0]['scale'].setValue(5.0)
        w.geometry_type_combo.setCurrentText("Disk")
        app.processEvents()
        assert w.ic_inputs[0].text() == "exp(-10*(x**2+y**2))"
        w.num_outputs_spin.setValue(2)
        app.processEvents()
        assert w.ic_inputs[0].text() == "exp(-10*(x**2+y**2))"
        assert w.pde_inputs[0].text() == "du_t - du_xx" and w.output_name_inputs[0].text() == "c"
        assert w.output_transform_rows[0]['scale'].value() == 5.0
        assert len(w.ic_inputs) == 2
    finally:
        _close(w)


def test_command_palette():
    from pinnstudio.ui.command_palette import CommandPalette, score
    assert score("lbfgs", "L-BFGS Options") and score("loss w", "Loss weights") and not score("xyz", "Loss")
    w = _window()
    try:
        d = CommandPalette(w)
        d.edit.setText("loss w")
        assert d.list.count() and d.list.item(0).text().startswith("Loss weights")
        d._run_current()
        assert w.workspace.current_key() == "weights"
        d.deleteLater()
        d = CommandPalette(w)
        d.edit.setText("save as")
        assert "Save Problem As" in d.list.item(0).text()
        d.deleteLater()
    finally:
        _close(w)


def test_system_and_high_contrast_themes():
    assert theme.THEME_CHOICES[0] == theme.SYSTEM_THEME and "High Contrast" in theme.THEME_CHOICES
    assert theme.resolve(theme.SYSTEM_THEME) in ("Studio Light", "Studio Dark")
    assert theme.resolve("Monokai") == theme.DEFAULT_THEME
    hc = theme.tokens("High Contrast")
    assert hc["text"] == "#ffffff" and hc["bg"] == "#000000"
    qss = theme.build_qss(hc, "Sans", 14, 18, "", "", "Mono", 13)
    assert "QProgressBar::chunk" in qss and "QPushButton:focus" in qss and "outline: none; }" not in qss


def test_sketch_edges_far_from_origin_and_cached():
    from pinnstudio.ui import sketch_edges as se
    pts = [(100, 0), (101, 0), (100.731, 0.913), (100.117, 0.677)]
    shape = {"type": "Polygon", "params": {"vertices_text": ";".join(f"{x},{y}" for x, y in pts)}}
    for e in se.boundary_edges([shape]):
        x, y = e["mid"]
        assert eval(e["expr"], {"np": np, "abs": abs, "x": x, "y": y}), e["expr"]


def test_svg_import():
    from pinnstudio.ui import sketch_edges as se
    from pinnstudio.ui import sketch_import as si
    shapes, _ = si.to_shapes(si.read_svg('<svg xmlns="http://www.w3.org/2000/svg">'
                                         '<ellipse rx="2" ry="1" transform="rotate(30)"/></svg>'))
    assert se.shape_inside(shapes[0], 1.6, -0.92) and not se.shape_inside(shapes[0], 1.6, 0.92)
    (pts, closed), = si._path_subpaths("M0 0A1 1 0 012 0Z")
    assert closed and abs(pts[-1][0] - 2.0) < 1e-9
    # a regular 12-gon stays a polygon
    import math
    gon = [(math.cos(2 * math.pi * i / 12), math.sin(2 * math.pi * i / 12)) for i in range(12)]
    res = si.ImportResult()
    res.items.append(("poly", gon))
    shapes, _ = si.to_shapes(res)
    assert shapes[0]["type"] == "Polygon"


def test_sweep_folder_names_keep_the_sign():
    from pinnstudio.core.sweep_runner import _slugify
    assert _slugify("-0.5") != _slugify("0.5")
    assert _slugify("1e-05") == "1e-05"


def test_ode_monitor_point_size():
    from pinnstudio.core.config import PINNConfig
    c = PINNConfig()
    c.problem_dim, c.steady_state = "ODE", False
    c.training_monitors_enabled = True
    c.training_monitors = '[{"name": "m", "expr": "u", "period": 100, "points": [[0.5]]}]'
    errs = [e for e in (c.validate() or []) if "Training Monitor" in e]
    assert not errs, errs


def test_generated_scripts_quote_labels():
    from pinnstudio.core import deepxde_examples as dx
    from pinnstudio.core.codegen import generate_clean_script, generate_script
    ex = next(e for e in dx.EXAMPLES if e.name == "Burgers equation with RAR")
    cfg = dx.build_config(ex, os.path.join(ROOT, "reference_data"))
    cfg = dataclasses.replace(cfg, plot_custom_expr="u**2", plot_custom_label='u "sq" $\\xi$')
    for gen in (generate_script, generate_clean_script):
        ast.parse(gen(cfg))


def test_time_adaptive_loss_history_counts_each_step_once():
    from pinnstudio.core import codegen
    src = codegen.generate_script(dataclasses.replace(_ta_config(), optimizer_scheduler=False))
    i = src.index("def _ta_accumulate_loss")
    body = src[i:src.index("\n\n", i)]
    ns = {}
    exec("_ta_all_train_loss, _ta_all_test_loss, _ta_all_steps, _ta_loss_offset, _ta_lh_taken = [], [], [], 0, {}\n"
         + body.replace("global _ta_loss_offset", "global _ta_loss_offset"), ns)

    class LH:
        def __init__(self):
            self.steps, self.loss_train, self.loss_test = [], [], []
    lh = LH()
    lh.steps, lh.loss_train, lh.loss_test = [0, 100], [[1], [0.5]], [[1], [0.5]]
    ns["_ta_accumulate_loss"](lh)               # window 1, phase 1 (Adam)
    lh.steps += [150]; lh.loss_train += [[0.1]]; lh.loss_test += [[0.1]]
    ns["_ta_accumulate_loss"](lh)               # window 1, phase 2 (L-BFGS): same history, steps go on
    lh2 = LH()
    lh2.steps, lh2.loss_train, lh2.loss_test = [0, 100], [[1], [0.5]], [[1], [0.5]]
    ns["_ta_accumulate_loss"](lh2)              # window 2
    assert ns["_ta_all_steps"] == [0, 100, 150, 150, 250], ns["_ta_all_steps"]


def test_constants_cannot_reach_python():
    """A constant's value is evaluated as a problem file opens: only numbers, arithmetic, math functions
    and earlier constants are accepted."""
    import json
    from pinnstudio.core import constants as C
    vals = C.evaluate(json.dumps([{"name": "Re", "value": "20"}, {"name": "nu", "value": "1/Re"},
                                  {"name": "l", "value": "1/(2*nu) - sqrt(1/(4*nu**2) + 4*pi**2)"}]))
    assert abs(vals[2][2] + 1.8100981200139667) < 1e-12
    for bad in ("().__class__.__mro__", "(lambda: 1)()", "[1][0]", "'a'", "True", '__import__("os")'):
        try:
            C.evaluate(json.dumps([{"name": "a", "value": bad}]))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_atomic_write_keeps_permissions(tmp_path=None):
    import stat
    import tempfile
    from pinnstudio.ui.main_window import _write_atomic
    d = str(tmp_path) if tmp_path else tempfile.mkdtemp()
    p = os.path.join(d, "a.pinn.json")
    _write_atomic(p, "{}")
    os.chmod(p, 0o640)
    _write_atomic(p, '{"x": 1}')
    assert open(p, encoding="utf-8").read() == '{"x": 1}'
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o640
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")]


def test_error_panel_names_the_outputs():
    """Several outputs at one time used to read as rows "time 0, time 0, time 0" over "3 reference times"."""
    from pinnstudio.ui.results import ErrorAnalysisPanel, RunProgress
    p = RunProgress(None)
    for ln in ("  ── Output group: u (1 files) ──", "  t=0.0000 — L2=1.0836e-03, Max=1e-2, MeanAbs=9.971e-04",
               "  ── Output group: v (1 files) ──", "  t=0.0000 — L2=4.6096e-03, Max=1e-2, MeanAbs=6.563e-04"):
        p.feed(ln)
    assert p.metric_groups == ["u", "v"]
    e = ErrorAnalysisPanel()
    e.show_run(p.metrics, "", groups=p.metric_groups)
    assert e.table.horizontalHeaderItem(0).text() == "output" and e.table.item(1, 0).text() == "v"
    assert "over 2 outputs" in e.summary.text()
    e.show_run(p.metrics[:1], "")
    assert e.table.columnCount() == 3 and "reference time" in e.summary.text()
    import shiboken6
    shiboken6.delete(e)          # not left to the end of the interpreter (a parentless widget crashes there)


def _ta_config():
    from pinnstudio.core import deepxde_examples as dx
    ex = next(e for e in dx.EXAMPLES if e.name == "Diffusion equation")
    cfg = dx.build_config(ex, os.path.join(ROOT, "reference_data"))
    return dataclasses.replace(cfg, time_adaptive=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("REVIEW FIXES TEST PASSED")
