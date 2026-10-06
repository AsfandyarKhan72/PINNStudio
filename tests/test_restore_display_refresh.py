#!/usr/bin/env python3
"""
After Restore && Visualize finished, the right-side results panel
sometimes showed a stale leftover plot/animation from a COMPLETELY
different, earlier restore run in the same save folder, instead of what
the just-finished run actually produced -- reported by the user: running
a steady-state Surface restore (their Soomro_Paper triangular-cavity
model) correctly saved restored_plot.png, but the right panel displayed
an animated GIF of an unrelated Burgers-equation example instead, because
a restored_animation.gif from an earlier, different restore was still
sitting in the same save directory.

Root cause: _on_restore_done() (main_window.py) decides what to show in
the right panel purely by `os.path.exists(...)`:

    plot_path = os.path.join(save_dir, "restored_plot.png")
    gif_path = os.path.join(save_dir, "restored_animation.gif")
    if os.path.exists(gif_path):
        self._set_solution_gif(gif_path)          # <-- picked first
    elif os.path.exists(plot_path):
        self.solution_label.setPixmap(...)

save_dir is just whatever folder the user pointed "Save results to" at
for THIS restore -- the same folder is commonly reused across many
different restores of different models/problems over time. A Surface
restore (this run) only ever writes restored_plot.png, never a gif -- so
if an animation GIF was left over in that folder from some earlier,
unrelated Animation-type restore, os.path.exists(gif_path) is still True
and the stale GIF wins, every time, regardless of what viz_type was just
requested. The same blind-existence-check problem applies to the Error
Analysis comparison PNGs logged right below it.

Original fix: _on_restore() deleted any pre-existing restored_plot.png,
restored_animation.gif, error_analysis/surface_comparison_restore.png,
error_analysis/line_comparison_restore.png, and
error_analysis/error_metrics_restore.txt in save_dir BEFORE launching the
new restore subprocess -- so by the time _on_restore_done()'s
os.path.exists(...) checks run, the only files that can possibly be there
are ones THIS run actually just wrote. Mirrors the existing
_last_restore_is_param fix for the same class of bug (see that flag's own
comment) -- stale files misreported as fresh.

Superseded (dev-branch round, standardized per-run results folders): every
restore now writes into its own fresh "<label>__<timestamp>" subfolder
under the configured restore save path (see MainWindow._on_restore /
_timestamped_save_dir) instead of directly into that path -- so two
restores can never share a folder in the first place, and the deletion
loop above (left in place, harmless) never actually has anything to
delete any more. This test now points its checks at the REAL resolved
folder (win._last_restore_save_dir) instead of the configured base path,
and simulates "stale file from an unrelated earlier restore" as a file
sitting in that base path directly (e.g. left over from before this
per-run-folder feature existed) -- proving it's simply never looked at,
rather than proving it gets deleted.

Checks:
 - A stale restored_animation.gif (simulating a leftover GIF from an
   unrelated earlier Animation-type restore, or from before per-run
   folders existed) sitting directly in the configured base save path
   is untouched and ignored by a steady-state Surface restore, which
   shows the real restored_plot.png it just produced under its own
   resolved subfolder -- not the stale GIF.
 - A stale restored_plot.png sitting in that base path is likewise
   ignored by an Animation Surface (GIF) restore (a transient model,
   since Animation types are hidden for steady configs), which shows
   the real restored_animation.gif it just produced.
 - Stale error_analysis/*_restore.png files in the base path are
   likewise ignored by a restore that doesn't configure any Error
   Analysis files at all (so nothing new gets written there either).
 - Verified this test fails (stale GIF still shown for the steady Surface
   case) against the original pre-fix code, and passes with both the
   original fix and the current per-run-folder design.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_restore_display_refresh.py
"""
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow


def _train_and_save_model(tmpdir, is_3d, is_2d, steady, layers, tag):
    """Same minimal real-checkpoint helper as test_restore_steady_state.py."""
    import deepxde as dde

    if is_3d:
        geom = dde.geometry.Cuboid([0, 0, 0], [1, 1, 1])
    elif is_2d:
        geom = dde.geometry.Rectangle([0, 0], [1, 1])
    else:
        geom = dde.geometry.Interval(0, 1)

    if steady:
        data = dde.data.PDE(geom, lambda x, y: y[:, 0:1] * 0, [], num_domain=20, num_test=20)
    else:
        timedomain = dde.geometry.TimeDomain(0, 1)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        data = dde.data.TimePDE(geomtime, lambda x, y: y[:, 0:1] * 0, [], num_domain=20,
                                 num_test=20, num_initial=5)

    net = dde.nn.FNN(layers, "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    model.train(iterations=1, display_every=1000)
    return model.save(os.path.join(tmpdir, tag))


def _run_restore_and_wait(win, timeout_s=60):
    """Drives _on_restore() through its real QThread+subprocess path and
    blocks until _on_restore_done() has actually run -- QThread.wait()
    blocks for the background OS thread regardless of the Qt event loop,
    but the cross-thread done_signal connection is a queued connection
    that still needs the event loop pumped at least once to be
    delivered, hence the processEvents() loop after it."""
    win._on_restore()
    thread = win._restore_thread
    thread.wait(timeout_s * 1000)
    deadline = time.time() + 5
    while win.restore_btn.text() != "🔄  Restore && Visualize" and time.time() < deadline:
        _app.processEvents()
        time.sleep(0.02)
    _app.processEvents()


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    try:
        import deepxde as _dde_probe  # noqa: F401
        import torch as _torch_probe  # noqa: F401
    except ImportError:
        print("SKIPPED: deepxde/torch not installed in this environment.")
        return failures

    win = MainWindow()
    _app.processEvents()
    win._on_restore_viz_changed = lambda text: None  # skip the pre-existing
    # settings-dialog side effect (dialog.exec()) -- unrelated to this fix,
    # and blocking under headless/offscreen test runs with no one to close it
    # (same workaround test_restore_steady_state.py already uses).
    win._restore_viz_settings = dict(win._restore_viz_settings or {})
    win._restore_viz_settings["resolution"] = 16  # speed only

    # ── Case 1: steady 2D Surface restore, stale animation GIF present ──
    with tempfile.TemporaryDirectory() as tmpdir:
        layers = [2, 16, 16, 1]
        model_path = _train_and_save_model(tmpdir, False, True, True, layers, "steady_2d")
        cfg = {
            "layers": layers, "activation": "tanh",
            "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0, "t_min": 0.0, "t_max": 1.0,
            "problem_dim": "2D", "steady_state": True, "loss_type": "MSE",
            "output_names": "u", "geometry_type": "Rectangle",
        }
        config_path = os.path.join(tmpdir, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(cfg, f)

        # Simulate a leftover animation GIF + stale EA pngs from some
        # earlier, unrelated restore run in this exact same folder --
        # this is the user's reported "shows a Burgers animation instead"
        # scenario (a GIF genuinely saved by a past run, still on disk).
        stale_gif = os.path.join(tmpdir, "restored_animation.gif")
        with open(stale_gif, "wb") as f:
            f.write(b"stale gif from an unrelated earlier restore")
        os.makedirs(os.path.join(tmpdir, "error_analysis"), exist_ok=True)
        stale_ea = os.path.join(tmpdir, "error_analysis", "surface_comparison_restore.png")
        with open(stale_ea, "wb") as f:
            f.write(b"stale EA png from an unrelated earlier restore")

        win.restore_mode_combo.setCurrentText("Forward Model")
        win.restore_model_path.setText(model_path)
        win.restore_config_path.setText(config_path)
        win._refresh_restore_output_combo()
        win._refresh_restore_viz_options()
        # setCurrentText fires the REAL currentTextChanged signal, connected
        # at __init__ time straight to the original _on_restore_viz_changed
        # (monkeypatching the instance attribute above doesn't retarget an
        # already-made connect() call) -- which opens a modal Settings
        # dialog via _on_restore_viz_settings. blockSignals avoids that.
        win.restore_viz_combo.blockSignals(True)
        win.restore_viz_combo.setCurrentText("Surface")
        win.restore_viz_combo.blockSignals(False)
        win.restore_optimizer_combo.setCurrentIndex(win.restore_optimizer_combo.findData("adam"))
        win.restore_save_path.setText(tmpdir)
        win._ea_settings = {}  # no Error Analysis files configured for this restore

        _run_restore_and_wait(win)

        run_dir = win._last_restore_save_dir
        check(run_dir and run_dir != tmpdir and os.path.dirname(run_dir) == tmpdir,
              f"this restore should have gotten its own fresh subfolder under {tmpdir!r}, got {run_dir!r}")
        check(os.path.exists(stale_gif) and os.path.exists(stale_ea),
              "stale files sitting directly in the configured base save path (left over from "
              "before per-run folders existed) should be left alone, not touched by this restore")
        check(run_dir and os.path.exists(os.path.join(run_dir, "restored_plot.png")),
              "this steady-state Surface restore should have saved its own restored_plot.png "
              "under its own resolved subfolder")
        shown_path = getattr(win.solution_label, "_source_path", None)
        check(shown_path is not None and os.path.basename(shown_path) == "restored_plot.png",
              f"right panel should show THIS run's restored_plot.png, got {shown_path!r} "
              f"(the user's reported bug: a stale leftover animation GIF gets shown instead)")

    # ── Case 2: transient 2D Animation Surface restore, stale plot.png ──
    with tempfile.TemporaryDirectory() as tmpdir:
        layers = [3, 16, 16, 1]
        model_path = _train_and_save_model(tmpdir, False, True, False, layers, "transient_2d")
        cfg = {
            "layers": layers, "activation": "tanh",
            "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0, "t_min": 0.0, "t_max": 1.0,
            "problem_dim": "2D", "steady_state": False, "loss_type": "MSE",
            "output_names": "u", "geometry_type": "Rectangle",
        }
        config_path = os.path.join(tmpdir, "model_config.json")
        with open(config_path, "w") as f:
            json.dump(cfg, f)

        stale_plot = os.path.join(tmpdir, "restored_plot.png")
        with open(stale_plot, "wb") as f:
            f.write(b"stale static plot from an unrelated earlier restore")

        win.restore_mode_combo.setCurrentText("Forward Model")
        win.restore_model_path.setText(model_path)
        win.restore_config_path.setText(config_path)
        win._refresh_restore_output_combo()
        win._refresh_restore_viz_options()
        win.restore_viz_combo.blockSignals(True)
        win.restore_viz_combo.setCurrentText("Animation Surface (GIF)")
        win.restore_viz_combo.blockSignals(False)
        win.restore_optimizer_combo.setCurrentIndex(win.restore_optimizer_combo.findData("adam"))
        win.restore_tsteps_spin.setValue(3)
        win.restore_save_path.setText(tmpdir)
        win._ea_settings = {}

        _run_restore_and_wait(win)

        run_dir = win._last_restore_save_dir
        check(run_dir and run_dir != tmpdir and os.path.dirname(run_dir) == tmpdir,
              f"this restore should have gotten its own fresh subfolder under {tmpdir!r}, got {run_dir!r}")
        check(os.path.exists(stale_plot),
              "a stale file sitting directly in the configured base save path should be left "
              "alone, not touched by this restore")
        check(run_dir and os.path.exists(os.path.join(run_dir, "restored_animation.gif")),
              "this transient Animation Surface restore should have saved its own "
              "restored_animation.gif under its own resolved subfolder")
        shown_path = getattr(win.solution_label, "_source_path", None)
        check(shown_path is not None and os.path.basename(shown_path) == "restored_animation.gif",
              f"right panel should show THIS run's restored_animation.gif, got {shown_path!r}")

    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL RESTORE DISPLAY-REFRESH TESTS PASSED")
    return 0


def test_restore_display_refresh():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
