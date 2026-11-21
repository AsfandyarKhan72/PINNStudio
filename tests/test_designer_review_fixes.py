"""Fixes from the review of the network-designer branch: code in a design is always asked about, titles and
ids cannot become code, a shared copy of a group calls the group's own port, malformed design files are
refused with a message, user paths under /tmp stay as they are in Solve's script, a typed value reaches the
design when the selection moves on, a settings Undo keeps the designer's own history, and a finished run's
scratch folder is removed."""
import ast
import dataclasses
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from pinnstudio.core import codegen  # noqa: E402
from pinnstudio.core.config import PINNConfig  # noqa: E402
from pinnstudio.core.netdesign import design as D  # noqa: E402
from pinnstudio.core.netdesign import presets as P  # noqa: E402
from pinnstudio.core.netdesign import runtime as R  # noqa: E402
from pinnstudio.core.netdesign import shapes as S  # noqa: E402
from pinnstudio.core.netdesign import trust as T  # noqa: E402
from pinnstudio.core.netdesign.params import ParamError  # noqa: E402


def test_code_is_asked_about_even_without_a_custom_layer(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path))
    d = P.fnn([8])
    d["code"] = "import subprocess\nHELPER_SCALE = 1\n"
    d = D.check(d)
    assert "import subprocess" in R.net_source(d, ["x"], ["u"]), "the script embeds it"
    items = T.untrusted(d)
    assert [it["kind"] for it in items] == ["code"]


def test_titles_and_ids_cannot_become_code():
    d = P.fnn([8])
    d["blocks"][1]["title"] = "hidden\nimport os; os.system('echo pwned')"
    d["name"] = "net\r\nimport sys"
    d = D.check(d)
    assert "\n" not in d["blocks"][1]["title"] and "\n" not in d["name"]
    src = R.net_source(d, ["x"], ["u"])
    tree = ast.parse(src)
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(a.name == "os" for a in n.names)
                   for n in tree.body), "the title stayed in a comment"
    assert not any(ln.lstrip().startswith("import os; os.system") for ln in src.splitlines())
    bad = P.fnn([8])
    bad["blocks"][1]["id"] = "s1\n"           # re.match('^…$') let a final newline through
    with pytest.raises(ParamError):
        D.check(bad)


@pytest.mark.parametrize("raw", [
    {"version": "1.0a", "blocks": [], "edges": []},
    {"blocks": [{"id": "in1", "kind": "input", "x": "left"}], "edges": []},
    {"blocks": [{"id": "g1", "kind": "group", "inner": [1, 2]}], "edges": []},
])
def test_malformed_designs_are_refused_with_a_message(raw):
    with pytest.raises(ParamError):
        D.check(json.dumps(raw))
    assert PINNConfig(network_type="Designed", network_design=json.dumps(raw)).validate(), \
        "validate() lists it instead of raising"


def _group_design(port_names):
    d = P.fnn([6])
    inner_blocks = [{"id": f"pi{i}", "kind": "port_in", "port": p} for i, p in enumerate(port_names)]
    inner_blocks += [{"id": "s9", "kind": "stack", "layers": [{"type": "dense", "params": {"units": 6}}]},
                     {"id": "po1", "kind": "port_out"}]
    inner_edges = [{"src": f"pi{i}", "dst": "s9"} for i in range(len(port_names))] + [{"src": "s9", "dst": "po1"}]
    d["blocks"].append({"id": "g1", "kind": "group", "inner": {"blocks": inner_blocks, "edges": inner_edges}})
    d["blocks"].append({"id": "sh1", "kind": "shared", "ref": "g1"})
    d["blocks"].append({"id": "m1", "kind": "merge", "merge": "add"})
    d["edges"] = ([{"src": "in1", "dst": "s1"}]
                  + [{"src": "s1", "dst": "g1", "port": p} for p in port_names]
                  + [{"src": "s1", "dst": "sh1"}, {"src": "g1", "dst": "m1"}, {"src": "sh1", "dst": "m1"},
                     {"src": "m1", "dst": "h1"}])
    return D.check(d)


def test_a_shared_copy_of_a_group_calls_the_groups_own_port():
    torch = pytest.importorskip("torch")
    dde = pytest.importorskip("deepxde")
    d = _group_design(["u"])                  # the group's one Port in is not called "in"
    assert S.problems(d, ["x"], ["u"]) == []
    ns = {"dde": dde, "torch": torch}
    exec(R.net_source(d, ["x"], ["u"]), ns)
    net = ns["_DesignedNet"](ns["_DESIGN"], "tanh", "Glorot uniform", 0.0)
    assert net(torch.rand(5, 1)).shape == (5, 1)


def test_a_shared_copy_of_a_group_with_two_ports_is_a_problem_not_a_crash():
    d = _group_design(["a", "b"])
    probs = " ".join(str(p) for p in S.problems(d, ["x"], ["u"]))
    assert "one Port in" in probs


def test_user_paths_under_tmp_stay_in_solves_script():
    d = P.fnn([8])
    d["blocks"].insert(2, {"id": "sv1", "kind": "saved", "path": "/tmp/pretrained/m.pt", "width_in": 8,
                           "width_out": 8})
    d["edges"] = [{"src": "in1", "dst": "s1"}, {"src": "s1", "dst": "sv1"}, {"src": "sv1", "dst": "h1"}]
    c = dataclasses.replace(PINNConfig(), save_dir="/tmp/my_results", network_type="Designed",
                            network_design=json.dumps(d))
    s = codegen.generate_script(c)
    ast.parse(s)
    assert "/tmp/my_results" in s and "/tmp/pretrained/m.pt" in s
    assert "_PS_TMP" in s and '"/tmp/param_' not in s and '"/tmp/loss_plot' not in s


def test_the_scratch_folder_of_a_finished_run_is_removed(tmp_path):
    from pinnstudio.ui.main_window import _drop_scratch
    mine = tempfile.mkdtemp(prefix="pinnstudio_run_")
    open(os.path.join(mine, "loss_plot.png"), "wb").close()
    _drop_scratch(mine)
    assert not os.path.exists(mine)
    other = tmp_path / "pinnstudio_run_keep"      # not in the temporary folder: never touched
    other.mkdir()
    _drop_scratch(str(other))
    assert other.exists()


# ── the window ──────────────────────────────────────────────────────────────
@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNSTUDIO_HOME", str(tmp_path))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    for _ in range(3):
        app.processEvents()
    w._new_problem("1D")
    w.show()
    app.processEvents()
    w.network_type_combo.setCurrentText("Designed")
    app.processEvents()
    yield w, app
    w.close()
    w.deleteLater()
    app.processEvents()


def test_a_typed_value_reaches_the_design_when_another_block_is_selected(window):
    w, app = window
    stack = next(b for b in w.design_state.design["blocks"] if b["kind"] == "stack")
    w.designer.select(stack["id"])
    app.processEvents()
    insp = w.designer_panel.inspector
    insp._draft["layers"][0]["params"]["units"] = 37      # as the Units box does while typing
    insp._timer.start()
    w.designer.select("")                                  # clicked elsewhere within the pause
    app.processEvents()
    now = next(b for b in w.design_state.design["blocks"] if b["id"] == stack["id"])
    assert now["layers"][0]["params"]["units"] == 37


def test_a_settings_undo_keeps_the_designers_own_history(window):
    w, app = window
    w._check_problems()
    w.designer.add("stack", "dense", 0, 300)
    assert w.design_state.can_undo()
    w._check_problems()
    w.pde_inputs[0].setText("du_t - 0.3*du_xx")
    w._check_problems()
    w.menuBar().setFocus()
    w.undo_action.trigger()                              # undoes the equation, not the design
    assert w.pde_inputs[0].text() != "du_t - 0.3*du_xx"
    assert w.design_state.can_undo(), "the designer's undo history is kept"
