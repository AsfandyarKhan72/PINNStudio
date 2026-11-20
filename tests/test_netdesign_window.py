"""The network designer in the main window: the Designed type, the Network tab, Edit as blocks, the
design following the problem, unsaved changes, saving and opening, and trust before Solve and Restore."""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from pinnstudio.ui.main_window import MainWindow  # noqa: E402


class Answers:
    """Stands in for the window's dialogs."""

    def __init__(self):
        self.trust = False
        self.asked = 0

    def confirm_trust(self, items):
        self.asked += 1
        return self.trust

    def inform(self, title, text):
        pass

    def ask_open_path(self, *a):
        return ""

    def ask_save_path(self, *a):
        return ""

    def ask_text(self, *a, **k):
        return None


@pytest.fixture
def win(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path / "home"))
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    for _ in range(3):          # the start-up settings are captured two event-loop passes after start
        app.processEvents()
    w._new_problem("1D")
    w.resize(1500, 950)
    w.show()
    app.processEvents()
    yield w
    w.close()
    w.deleteLater()
    app.processEvents()


def test_designed_type_shows_the_tab_and_starts_from_the_form(win):
    w = win
    w.network_type_combo.setCurrentText("MsFFN")
    app.processEvents()
    w.network_type_combo.setCurrentText("Designed")
    app.processEvents()
    assert w.right_tabs.isTabVisible(w._network_tab_index)
    assert w.right_tabs.currentWidget() is w.designer_panel
    assert w.layers_row.isHidden() and not w.designer_row.isHidden()
    d = w.design_state.design
    assert d.get("name") == "MsFFN" and not w.design_state.problems
    cfg = w._build_config()
    assert cfg.network_type == "Designed" and json.loads(cfg.network_design)["blocks"]
    assert not [e for e in cfg.validate() if "Network" in e], cfg.validate()
    w.network_type_combo.setCurrentText("FNN")
    app.processEvents()
    assert not w.right_tabs.isTabVisible(w._network_tab_index) and not w.design_kept_note.isHidden()


def test_the_design_follows_the_problem(win):
    w = win
    w.network_type_combo.setCurrentText("Designed")
    app.processEvents()
    w.num_outputs_spin.setValue(2)
    app.processEvents()
    w._check_problems()
    assert w.design_state.outputs == ["u", "v"]
    assert not w.design_state.problems, "the FNN design's head gives every output"
    w.designer.update_block("h1", {"outputs": ["u"]})
    assert w.designer.missing_outputs() == ["v"]
    assert w.designer.add_missing_heads() and not w.design_state.problems


def test_design_edits_count_as_unsaved_and_round_trip(win, tmp_path):
    w = win
    w.network_type_combo.setCurrentText("Designed")
    app.processEvents()
    path = str(tmp_path / "p.pinn.json")
    w._save_problem(path)
    w._check_problems()
    assert not w._dirty
    w.designer.update_block("s1", {"title": "Trunk"})
    w._check_problems()
    assert w._dirty, "a design edit is an unsaved change"
    w._save_problem(path)
    saved = json.load(open(path, encoding="utf-8"))["config"]
    assert saved["network_type"] == "Designed" and "Trunk" in saved["network_design"]
    w._new_problem("1D")
    app.processEvents()
    assert w.network_type_combo.currentText() == "FNN"
    w._open_problem(path)
    app.processEvents()
    assert w.network_type_combo.currentText() == "Designed"
    assert any(b["title"] == "Trunk" for b in w.design_state.design["blocks"])
    assert not w.design_state.can_undo(), "opening a problem starts a new history"


def test_solve_asks_trust_for_custom_code_and_runs_nothing_without_it(win, monkeypatch):
    w = win
    answers = Answers()
    w.designer.prompter = answers
    w.network_type_combo.setCurrentText("Designed")
    app.processEvents()
    w.designer.set_code("class Twice(torch.nn.Module):\n    def forward(self, x):\n        return 2 * x\n")
    layers = w.design_state.design["blocks"][1]["layers"] + [{"type": "custom", "params": {"class_name": "Twice"}}]
    w.designer.update_block("s1", {"layers": layers})
    started = []
    monkeypatch.setattr(w, "_start_progress", lambda *a: started.append(1))
    w._on_solve()
    assert answers.asked == 1 and not started and "not trusted" in w.log_box.toPlainText()


def test_restore_asks_trust_for_the_runs_design(win, tmp_path):
    w = win
    answers = Answers()
    w.designer.prompter = answers
    design = {"version": 1, "code": "class Twice(torch.nn.Module):\n    pass\n",
              "blocks": [{"id": "in1", "kind": "input"},
                         {"id": "s1", "kind": "stack", "layers": [{"type": "custom", "params": {"class_name": "Twice"}}]},
                         {"id": "h1", "kind": "head"}],
              "edges": [{"src": "in1", "dst": "s1"}, {"src": "s1", "dst": "h1"}]}
    assert not w.designer.check_trust(json.dumps(design))
    assert answers.asked == 1


def test_the_network_node_opens_the_tab(win):
    w = win
    w.network_type_combo.setCurrentText("Designed")
    w.workspace.select("problem")
    app.processEvents()
    w.workspace.select("network")
    app.processEvents()
    assert w.right_tabs.currentWidget() is w.designer_panel
