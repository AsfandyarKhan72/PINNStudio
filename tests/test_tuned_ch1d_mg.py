"""1D Cahn-Hilliard (Mattey & Ghosh): Settings Default loads the paper's method (bc-PINN, paper_mg) and Tuned loads
the benchmark's selected setup, windows2 (twice as many time windows, 38), in the window itself, in the script
Solve runs and the exported script, and in a saved problem file opened again.

windows2 is what the documented rule selects from the corrected records (tools/make_benchmarks_readme.py: the
lowest of max(h, mu) at the last reference time, then the fastest within 1.25x): windows2 0.0622 against
base 0.124, iters 0.238, net 0.375, plain 0.772, paper_mg 1.07 (benchmarks/records/reruns_2026-11-01)."""
import dataclasses
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from pinnstudio.core import codegen  # noqa: E402
from pinnstudio.core import example_catalog as cat  # noqa: E402
from pinnstudio.core import example_settings as es  # noqa: E402
from pinnstudio.ui.main_window import MainWindow  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
KEY = "1D Cahn-Hilliard (Mattey & Ghosh)"


def _load(w, mode):
    e = next(x for x in cat.ENTRIES if x.key == KEY)
    w.quick_examples_settings_combo.setCurrentText(mode)
    w.quick_examples_category_combo.setCurrentText(e.category if hasattr(e, "category") else "Forward")
    app.processEvents()
    i = w.quick_examples_combo.findText(e.name)
    if i < 0:                                       # templates may sit in another category
        for c in range(w.quick_examples_category_combo.count()):
            w.quick_examples_category_combo.setCurrentIndex(c)
            app.processEvents()
            i = w.quick_examples_combo.findText(e.name)
            if i >= 0:
                break
    assert i >= 0
    w.quick_examples_combo.setCurrentIndex(-1)
    app.processEvents()
    w.quick_examples_combo.setCurrentIndex(i)
    for _ in range(30):
        app.processEvents()
    return w._build_config()


def _intervals(script):
    """The time windows a generated script trains."""
    import ast
    import re
    for l in script.split("\n"):
        if l.strip().startswith("intervals = ["):                          # the exported script
            return ast.literal_eval(l.split("=", 1)[1].strip())
        m = re.search(r"_ta_groups = _json_ta_groups\.loads\('(.*)'\)", l)    # Solve's script
        if m:
            out = []
            for g in json.loads(m.group(1)):
                d = (g["t_end"] - g["t_start"]) / g["steps"]
                out += [(g["t_start"] + i * d, g["t_start"] + (i + 1) * d) for i in range(g["steps"])]
            return out
    raise AssertionError("no time windows in the script")


RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "records",
                        "reruns_2026-11-01", "run_outputs", "p2_ch_tune_1d", "out")
RUNS = {es.DEFAULT: "Template_1D_Cahn_Hilliard_Mattey_Ghosh/paper_mg_s2026/1D_Cahn-Hilliard_Mattey__Ghosh__2026-10-31_14-31-34/"
                    "solution_results/model_config.json",
        es.TUNED: "Template_1D_Cahn_Hilliard_Mattey_Ghosh_best_setup/windows2_s2026/"
                  "1D_Cahn-Hilliard_Mattey__Ghosh__2026-10-31_16-55-23/solution_results/model_config.json"}


def _windows(c):
    return sum(int(g["steps"]) for g in json.loads(c.ta_step_groups)) if (c.ta_step_groups or "").strip() \
        else int(c.ta_num_steps)


def test_shipped_tuned_entry_is_windows2():
    t = es.tuned_table()[KEY]["forward"]
    assert t["variant"] == "windows2", t
    assert es.has_tuned(KEY, False)


def test_gui_default_is_the_paper_method_and_tuned_is_windows2(tmp_path):
    w = MainWindow()
    try:
        d = _load(w, es.DEFAULT)
        assert d.layers == [2] + [128] * 6 + [2] and d.ta_backward and _windows(d) == 10
        assert [(p["optimizer"], p["iterations"]) for p in json.loads(d.scheduler_phases)] == \
            [("adam", 10000), ("lbfgs", 50000)]
        t = _load(w, es.TUNED)
        assert t.layers == [2] + [128] * 4 + [2] and not t.ta_backward
        assert t.periodic_inputs and t.point_weights == "SA-PINN"          # the best setup's switches
        assert json.loads(t.ta_step_groups) == [{"t_start": 0.0, "t_end": 0.1, "steps": 20},
                                                {"t_start": 0.1, "t_end": 1.0, "steps": 18}]
        assert _windows(t) == 38 and int(t.ta_num_steps) == 38
        assert [(p["optimizer"], p["iterations"]) for p in json.loads(t.scheduler_phases)] == \
            [("adam", 5000), ("lbfgs", 5000)]
        log = w.log_box.toPlainText()
        assert "Tuned settings: IC as a loss term, exact periodicity, time windows, SA-PINN weights; " \
               "twice as many time windows" in log

        # the same settings as the benchmark runs the records come from (their saved model_config.json, which
        # holds the configuration after exact periodicity is turned into input features)
        for cfg, run in ((d, RUNS[es.DEFAULT]), (t, RUNS[es.TUNED])):
            saved = json.load(open(os.path.join(RUNS_DIR, run), encoding="utf-8"))
            saved = saved.get("config", saved)
            got = dataclasses.asdict(codegen._exact_constraints(cfg))
            common = [k for k in got if k in saved]
            assert len(common) >= 45
            assert {k: (got[k], saved[k]) for k in common if got[k] != saved[k] and k != "layers"} == {}, run
            # the input layer's width is set by the script from the input features (cos, sin of x, and t)
            assert got["layers"][1:] == saved["layers"][1:], run
            for gen in (codegen.generate_script, codegen.generate_clean_script):
                assert str(saved["layers"]) in gen(cfg), (run, gen.__name__)

        # the script Solve runs and the exported script carry the 38 windows
        for gen in (codegen.generate_script, codegen.generate_clean_script):
            iv = _intervals(gen(t))
            assert len(iv) == 38 and abs(iv[20][0] - 0.1) < 1e-9 and abs(iv[-1][1] - 1.0) < 1e-9, gen.__name__

        # saved and opened again
        path = str(tmp_path / "ch1d_mg_tuned.pinn.json")
        assert w._save_problem(path)
        saved = json.load(open(path, encoding="utf-8"))["config"]
        assert json.loads(saved["ta_step_groups"]) == json.loads(t.ta_step_groups)
        assert saved["layers"] == t.layers and saved["ta_backward"] is False
        w2 = MainWindow()
        try:
            w2._open_problem(path)
            for _ in range(30):
                app.processEvents()
            r = w2._build_config()
            assert json.loads(r.ta_step_groups) == json.loads(t.ta_step_groups) and r.layers == t.layers
            assert r.scheduler_phases == t.scheduler_phases and not r.ta_backward
        finally:
            w2.close()
    finally:
        w.close()
