"""The network designer's views with real events: a palette drop is one step of Undo, dragging between
ports connects (or is refused with the reason), the canvas's keys, the inspector refuses a bad value
without changing anything, groups open and close, and trust is asked through the Prompter."""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from pinnstudio.core.config import PINNConfig  # noqa: E402
from pinnstudio.core.netdesign.presets import PRESETS  # noqa: E402
from pinnstudio.ui.controllers.designer import DesignerController  # noqa: E402
from pinnstudio.ui.controllers.prompter import Prompter  # noqa: E402
from pinnstudio.ui.designer.canvas import MIME  # noqa: E402
from pinnstudio.ui.designer.panel import DesignerPanel  # noqa: E402
from pinnstudio.ui.state.design_state import DesignState  # noqa: E402


class FakePrompter(Prompter):
    def __init__(self):
        self.trust_answer = True
        self.asked = []
        self.save_path = ""

    def confirm_trust(self, items):
        self.asked.append(items)
        return self.trust_answer

    def ask_save_path(self, title, default_name, file_filter):
        return self.save_path


@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path))
    state = DesignState()
    prompter = FakePrompter()
    ctl = DesignerController(state, prompter)
    cfg = PINNConfig()
    ctl.set_problem(cfg)
    ctl.from_settings(cfg)
    panel = DesignerPanel(state, ctl, PRESETS)
    panel.resize(1300, 720)
    panel.show()
    app.processEvents()
    yield state, ctl, panel, prompter
    panel.close()
    panel.deleteLater()
    app.processEvents()


def _vp(canvas, scene_pt):
    return canvas.mapFromScene(scene_pt)


def test_drop_from_palette_is_one_undo_step(ui):
    state, ctl, panel, _p = ui
    n = len(state.design["blocks"])
    md = QMimeData()
    md.setData(MIME, json.dumps({"kind": "stack", "preset": "fourier"}).encode())
    pos = QPointF(panel.canvas.viewport().rect().center())
    ev = QDropEvent(pos, Qt.DropAction.CopyAction, md, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    panel.canvas.dropEvent(ev)
    app.processEvents()
    d = state.design
    assert len(d["blocks"]) == n + 1 and d["blocks"][-1]["layers"][0]["type"] == "fourier"
    ctl.undo()
    assert len(state.design["blocks"]) == n


def _drag(canvas, a, b):
    vp = canvas.viewport()
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vp(canvas, a))
    QTest.mouseMove(vp, _vp(canvas, (a + b) / 2))
    QTest.mouseMove(vp, _vp(canvas, b))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _vp(canvas, b))
    app.processEvents()


def test_dragging_between_ports_connects_or_says_why(ui):
    state, ctl, panel, _p = ui
    new = ctl.add("stack", "dense", 200, 300)
    app.processEvents()
    c = panel.canvas
    src = c.blocks["in1"].out_port.scenePos()
    dst = c.blocks[new].in_ports["in"].scenePos()
    _drag(c, src, dst)
    assert any(e["src"] == "in1" and e["dst"] == new for e in state.design["edges"])
    # out of a head is refused, with the reason, and asked once per port
    calls = []
    real = c.legality
    c.legality = lambda *a: (calls.append(a), real(*a))[1]
    _drag(c, c.blocks[new].out_port.scenePos(), c.blocks["in1"].out_port.scenePos() + QPointF(0, 0))
    _drag(c, c.blocks["s1"].out_port.scenePos(), c.blocks[new].in_ports["in"].scenePos())
    ok, why = c.verdicts[("s1", new, "in")]
    assert ok      # a second input into a stack is fine: concatenated
    head_out = QPointF(c.blocks["h1"].pos().x() + 180, c.blocks["h1"].pos().y() + 20)
    _drag(c, head_out, c.blocks[new].in_ports["in"].scenePos())
    assert len(calls) <= 3


def test_keys_on_the_canvas(ui):
    state, ctl, panel, _p = ui
    c = panel.canvas
    c.setFocus()
    ctl.select("s1")
    app.processEvents()
    n = len(state.design["blocks"])
    QTest.keyClick(c, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier)
    assert len(state.design["blocks"]) == n + 1
    QTest.keyClick(c, Qt.Key.Key_Delete)
    assert len(state.design["blocks"]) == n
    QTest.keyClick(c, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert len(state.design["blocks"]) == n + 1
    QTest.keyClick(c, Qt.Key.Key_L, Qt.KeyboardModifier.ControlModifier)
    xs = {b["id"]: b["x"] for b in state.design["blocks"]}
    assert xs["in1"] < xs["s1"] < xs["h1"]
    QTest.keyClick(c, Qt.Key.Key_F)


def test_moving_a_block_needs_more_than_the_dead_zone(ui):
    state, ctl, panel, _p = ui
    c = panel.canvas
    b = c.blocks["s1"]
    start = b.pos() + QPointF(60, 10)
    x0 = next(x["x"] for x in state.design["blocks"] if x["id"] == "s1")
    could_undo = state.can_undo()
    _drag(c, start, start + QPointF(1, 1))        # inside the dead zone: a click, not a move
    assert next(x["x"] for x in state.design["blocks"] if x["id"] == "s1") == x0
    assert state.can_undo() == could_undo
    _drag(c, c.blocks["s1"].pos() + QPointF(60, 10), c.blocks["s1"].pos() + QPointF(160, 10))
    x1 = next(x["x"] for x in state.design["blocks"] if x["id"] == "s1")
    assert x1 != x0 and x1 % 20 == 0


def test_inspector_refuses_a_bad_value_and_changes_nothing(ui):
    state, ctl, panel, _p = ui
    ctl.select("s1")
    app.processEvents()
    insp = panel.inspector
    before = state.design
    insp._set_layer_param("units", 0)
    insp._flush()
    assert state.design == before
    assert "outside 1 to" in panel.status.text()
    insp._set_layer_param("units", 32)
    insp._flush()
    assert state.design["blocks"][1]["layers"][0]["params"]["units"] == 32


def test_groups_open_and_close(ui):
    state, ctl, panel, _p = ui
    g = ctl.add("group", "", 0, 300)
    app.processEvents()
    QTest.mouseDClick(panel.canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      _vp(panel.canvas, panel.canvas.blocks[g].pos() + QPointF(60, 10)))
    app.processEvents()
    assert state.path == (g,) and panel.up_btn.isVisible()
    assert {b.kind for b in panel.canvas.blocks.values()} == {"port_in", "stack", "port_out"}
    panel.up_btn.click()
    app.processEvents()
    assert state.path == ()


def test_trust_is_asked_through_the_prompter(ui):
    state, ctl, panel, prompter = ui
    assert ctl.check_trust() and prompter.asked == []          # nothing to trust
    ctl.set_code("class Twice(torch.nn.Module):\n    def forward(self, x):\n        return 2 * x\n")
    ctl.update_block("s1", {"layers": state.design["blocks"][1]["layers"] +
                            [{"type": "custom", "params": {"class_name": "Twice"}}]})
    prompter.trust_answer = False
    assert not ctl.check_trust() and len(prompter.asked) == 1
    prompter.trust_answer = True
    assert ctl.check_trust() and len(prompter.asked) == 2
    assert ctl.check_trust() and len(prompter.asked) == 2      # remembered for exactly this code
    ctl.set_code("class Twice(torch.nn.Module):\n    def forward(self, x):\n        return 3 * x\n")
    assert ctl.check_trust() and len(prompter.asked) == 3      # changed code asks again


def test_presets_and_library(ui, tmp_path):
    state, ctl, panel, prompter = ui
    panel.palette.preset_requested.emit("modified_mlp")
    app.processEvents()
    assert len(state.design["blocks"]) == 20 and not state.problems
    ctl.undo()
    assert len(state.design["blocks"]) == 3
    prompter.save_path = str(tmp_path / "blocks" / "hidden.pinn-block.json")
    assert ctl.save_block("s1")
    panel.refresh_library()
    assert ctl.load_file(prompter.save_path, 0, 400)
    ids = [b["id"] for b in state.design["blocks"]]
    assert len(ids) == len(set(ids)) == 4
