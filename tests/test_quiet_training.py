"""While a run trains the window does less: the settings checks (a full
config rebuild every 1.5 s) wait until the run ends, the live loss plot can
be told not to redraw until then ("Update while training" off), and the log
keeps a bounded number of lines."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


def test_checks_pause_and_live_plot_can_wait():
    w = MainWindow()
    cfg = w._build_config()
    assert w._problems_timer.isActive()
    w._start_progress(cfg)
    assert not w._problems_timer.isActive(), "settings checks wait while a run trains"

    draws = []
    real = w.live_loss._redraw
    w.live_loss._redraw = lambda *a: draws.append(1)
    w.live_loss_cb.setChecked(False)
    w._progress_feed("1000      [1.00e-01]    [1.10e-01]    []")
    w.live_loss._dirty = True
    w.live_loss._tick()
    assert not draws, "no redraw while training with 'Update while training' off"

    w._finish_progress("Finished")
    assert draws, "the curve is drawn once when the run ends"
    assert w._problems_timer.isActive(), "settings checks resume after the run"
    w.live_loss._redraw = real
    w.live_loss_cb.setChecked(True)

    assert w.log_box.document().maximumBlockCount() == 20000
    w.close()
