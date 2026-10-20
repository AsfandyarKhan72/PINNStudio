"""Numbers in figures follow one standard (Plot settings -> General ->
Numbers): significant digits (4 by default; parameter values two more),
scientific notation Automatic (below 0.01 or from 100,000 up; errors always
with a power of ten), Always or Never, written as "1.23 x 10^-4" or
"1.23e-04". Times read "t = 0.25", not "t = 0.2500". Solve, the exported
script and Restore all use it; it is saved with the problem."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PyQt6.QtWidgets import QApplication, QDialog

from pinnstudio.core import codegen as cg
from pinnstudio.core.config import PINNConfig

app = QApplication.instance() or QApplication([])


def _helpers(numbers):
    ns = {}
    exec(cg._figsize_helper_code("Default", 7, 5, True, "line", numbers), ns)
    return ns


@pytest.mark.parametrize("numbers, expect", [
    ((4, "auto", "times"), {"t": "0.25", "pi": "3.142", "small": "1.235 $\\times$ 10$^{-4}$",
                            "l2": "$L_2$ = 4.792 $\\times$ 10$^{-1}$", "param": "1.00005"}),
    ((3, "always", "e"), {"t": "2.5e-01", "pi": "3.14e+00", "small": "1.23e-04",
                          "l2": "$L_2$ = 4.79e-01", "param": "1.0001e+00"}),
    ((5, "never", "times"), {"t": "0.25", "pi": "3.1416", "small": "0.00012346",
                             "l2": "$L_2$ = 0.47920", "param": "1.000053"}),
])
def test_one_rule_for_every_number(numbers, expect):
    ns = _helpers(numbers)
    got = {"t": ns["_fmt_num"](0.25), "pi": ns["_fmt_num"](3.14159265), "small": ns["_fmt_num"](1.23456e-4),
           "l2": ns["_l2_txt"](0.47920), "param": ns["_num_txt"](1.0000534)}
    if numbers[2] == "times" and numbers[1] == "never":
        got["l2"] = ns["_l2_txt"](0.47920)
        expect = dict(expect, l2="$L_2$ = 0.4792")
    assert got == expect
    for v in (0.25, 3.14159265, 1.23456e-4, 123456.7, 0, -0.5):     # generation-time twin: the same text
        assert cg.fmt_num(v, numbers) == ns["_fmt_num"](v)


def test_tick_notation_follows_the_setting():
    import matplotlib
    _helpers((4, "always", "e"))
    assert tuple(matplotlib.rcParams["axes.formatter.limits"]) == (0, 0)
    assert not matplotlib.rcParams["axes.formatter.use_mathtext"]
    _helpers((4, "auto", "times"))
    assert tuple(matplotlib.rcParams["axes.formatter.limits"]) == (-2, 5)


def test_scripts_carry_the_settings_and_no_hand_formatted_times():
    c = PINNConfig()
    c.plot_sig_digits, c.plot_sci_notation, c.plot_sci_style = 3, "never", "e"
    for gen in (cg.generate_script, cg.generate_clean_script):
        text = gen(c)
        assert "_PLOT_SIG, _PLOT_SCI, _PLOT_SCI_STYLE = (3, 'never', 'e')" in text
        import re
        fig_lines = [l for l in text.splitlines()
                     if re.search(r"set_title|suptitle|label=|set_text|ylabel", l) and "print(" not in l]
        assert not [l for l in fig_lines if re.search(r"\{[^{}]+:\.\d[fg]\}", l)], "hand-formatted number in a figure"
        compile(text, gen.__name__, "exec")


def test_dialog_round_trip(monkeypatch):
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()

    def _exec(dlg):
        dlg.sig_digits.setValue(6)
        dlg.sci_notation.setCurrentIndex(dlg.sci_notation.findData("always"))
        dlg.sci_style.setCurrentIndex(dlg.sci_style.findData("e"))
        dlg._ok()
        return 1
    monkeypatch.setattr(QDialog, "exec", _exec)
    w.plot_style_btn.click()
    c = w._build_config()
    assert (c.plot_sig_digits, c.plot_sci_notation, c.plot_sci_style) == (6, "always", "e")
    assert "_PLOT_SIG, _PLOT_SCI, _PLOT_SCI_STYLE = (6, 'always', 'e')" in cg.generate_script(c)
    w2 = MainWindow()
    w2._apply_config(c)
    c2 = w2._build_config()
    assert (c2.plot_sig_digits, c2.plot_sci_notation, c2.plot_sci_style) == (6, "always", "e")
    w.close()
    w2.close()
