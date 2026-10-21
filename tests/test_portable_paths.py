"""A saved problem file writes reference_data paths as "@REF/...", so it opens
with its reference and measured-data files on another computer."""
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio.core.portable_paths import from_portable, to_portable  # noqa: E402

REF = os.path.join(ROOT, "reference_data")


def test_repr_list_of_tuples_round_trip():
    p = os.path.join(REF, "1D", "heat", "t_1.0.txt")
    cfg = {"ea_files": repr([(1.0, p, None), (0.5, p, 0)])}
    out = to_portable(cfg, REF)
    assert "@REF/1D/heat/t_1.0.txt" in out["ea_files"] and REF not in out["ea_files"]
    back = from_portable(out, REF)
    assert eval(back["ea_files"]) == [(1.0, p, None), (0.5, p, 0)]


def test_json_list_of_dicts_round_trip():
    p = os.path.join(REF, "2D", "poisson_lshape", "u_obs.txt")
    rows = [{"path": p, "output_idx": 0, "weight": 100.0, "custom_expr": ""}]
    out = to_portable({"inverse_obs_files_json": json.dumps(rows)}, REF)
    assert json.loads(out["inverse_obs_files_json"])[0]["path"] == "@REF/2D/poisson_lshape/u_obs.txt"
    back = from_portable(out, REF)
    assert json.loads(back["inverse_obs_files_json"]) == rows


def test_plain_path_and_untouched_values():
    p = os.path.join(REF, "1D", "heat", "t_1.0.txt")
    cfg = {"some_path": p, "layers": [2, 20, 1], "a": "[1, 2]", "b": "du_xx - 0.4*du_t",
           "c": "/tmp/elsewhere.txt", "d": "[ 1,2 ]", "e": "", "n": 5}
    out = to_portable(cfg, REF)
    assert out["some_path"] == "@REF/1D/heat/t_1.0.txt"
    for k in ("layers", "a", "b", "c", "d", "e", "n"):
        assert out[k] == cfg[k], k
    assert from_portable(out, REF) == cfg


def test_folder_names_keep_their_case(monkeypatch):
    # Windows compares paths case-insensitively (normcase lowers them); the
    # saved "@REF/2D/..." must still say "2D", or Linux and macOS can't find it.
    monkeypatch.setattr(os.path, "normcase", lambda s: s.lower())
    p = os.path.join(REF, "2D", "allen_cahn_wight", "t_2.5.txt")
    out = to_portable({"ea_files": repr([(2.5, p, None)])}, REF)
    assert "@REF/2D/allen_cahn_wight/t_2.5.txt" in out["ea_files"]


def test_windows_style_ref_paths_are_resolved():
    out = from_portable({"x": "@REF/1D/heat/t_1.0.txt"}, REF)
    assert out["x"] == os.path.join(REF, "1D", "heat", "t_1.0.txt") and os.path.isfile(out["x"])


def test_path_saved_on_another_computer_is_remapped():
    here = os.path.join(REF, "1D", "heat", "t_1.0.txt")
    for foreign in ("/home/someone/.local/share/uv/tools/pinnstudio/lib/python3.12/site-packages/"
                    "reference_data/1D/heat/t_1.0.txt",
                    r"C:\Users\someone\AppData\Roaming\uv\tools\pinnstudio\Lib\site-packages"
                    r"\reference_data\1D\heat\t_1.0.txt"):
        cfg = {"ea_files": repr([(1.0, foreign, None)])}
        assert eval(from_portable(cfg, REF)["ea_files"])[0][1] == here
    # not under reference_data, or no such file here: left as it is
    for other in ("/home/someone/data/t_1.0.txt", "/home/someone/reference_data/1D/heat/nope.txt"):
        assert from_portable({"p": other}, REF)["p"] == other


# --- through the GUI: save, then open in a fresh window ------------------

def _gui():
    from PyQt6.QtCore import QCoreApplication, QEvent
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from pinnstudio.ui.main_window import MainWindow

    def settle(s=0.3):
        t = time.time()
        while time.time() - t < s:
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
            app.processEvents()

    def window():
        w = MainWindow()
        w._maybe_save_changes = lambda: True
        settle()
        return w
    return MainWindow, window, settle


def _pick(w, settle, category, key):
    w.quick_examples_category_combo.setCurrentText(category)
    settle(0.1)
    combo = w.quick_examples_combo
    i = combo.findData(key)
    assert i >= 0, key
    combo.setCurrentIndex(i)
    combo.activated.emit(i)
    settle(0.3)


def _check_round_trip(category, key, field):
    MainWindow, window, settle = _gui()
    w = window()
    _pick(w, settle, category, key)
    before = w._build_config()
    assert "reference_data" in getattr(before, field), field   # escaped inside the string on Windows
    d = tempfile.mkdtemp()
    p = os.path.join(d, "copy.pinn.json")
    assert w._save_problem(p)
    text = open(p, encoding="utf-8").read()
    assert "@REF/" in text
    saved = json.load(open(p, encoding="utf-8"))["config"]
    assert "reference_data" not in json.dumps(saved[field])
    w.close()
    w2 = window()
    w2._open_problem(p)
    settle()
    after = w2._build_config()
    assert getattr(after, field) == getattr(before, field)
    assert MainWindow._config_key(after) == MainWindow._config_key(before)
    w2.close()
    return after


def test_gui_reference_files_round_trip():
    after = _check_round_trip("Forward", "1D Heat", "ea_files")
    files = [f for _t, f, *_ in eval(after.ea_files)]
    assert files and all(os.path.isfile(f) for f in files)


def test_gui_measured_data_round_trip():
    after = _check_round_trip("Inverse", "1D Burgers", "inverse_obs_files_json")
    rows = json.loads(after.inverse_obs_files_json)
    assert rows and all(os.path.isfile(r["path"]) for r in rows)
