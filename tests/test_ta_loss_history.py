#!/usr/bin/env python3
"""
Time-Adaptive training's loss plot only ever showed the LAST time
sub-domain's loss history -- titled "Loss — Last Time Sub-domain". The
user asked whether this was intentional/useful and, per the chat
discussion, the fix is to show the full cumulative loss across every
time sub-domain of the run instead.

Root cause: codegen.py's Time-Adaptive loop re-binds `lh_i`/`ts_i` on
EVERY `model_i.train(...)` call -- and there can be several per
sub-domain (one per optimizer-scheduler phase), plus a separate one for
the legacy Adam-then-L-BFGS path. The loss plot below the loop read
`lh_i.loss_train`/`lh_i.loss_test` directly, which by then only ever held
whichever single train() call happened to run LAST, for the LAST
sub-domain -- every earlier phase's and every earlier sub-domain's loss
history was silently discarded.

Fix: a new `_ta_accumulate_loss(lh)` helper (defined once, before the
per-sub-domain loop) appends each phase's `loss_train`/`loss_test`/
`steps` into running `_ta_all_*` lists, with a cumulative iteration
offset added to `steps` so the x-axis keeps increasing across phases and
sub-domains instead of restarting at 0 each time (DeepXDE's LossHistory
numbers every train() call's own steps from scratch). Called after each
of the 5 `model_i.train(...)` call sites in the Time-Adaptive loop. The
loss plot now reads `_ta_all_train_loss`/`_ta_all_test_loss`/
`_ta_all_steps` instead of the bare last `lh_i`, with light vertical
markers at each sub-domain boundary and a "Loss — All N Time
Sub-domain(s)" title.

Checks:
 - The generated Time-Adaptive script contains exactly 5
   `_ta_accumulate_loss(lh_i)` calls (one per train() call site) and no
   longer assigns `train_loss_ta`/`test_loss_ta` directly from a bare
   `lh_i` (the single-phase/single-sub-domain bug).
 - A REAL, tiny, 2-sub-domain Time-Adaptive run (via a side-channel JSON
   dump appended to the generated script, same technique used in
   test_restore_geometry_masking.py) produces a cumulative step list that
   is strictly increasing across the whole run, has exactly one boundary
   mark per sub-domain, and whose final value is larger than the first
   sub-domain's own boundary -- proving the SECOND sub-domain's loss
   actually got appended on top of the first's, not reset or dropped.
 - loss_plot_ta.png (or whatever _ta_loss_path resolves to) is actually
   written and non-empty.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_ta_loss_history.py
"""
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.codegen import generate_script


def _run_script(script, tmpdir, tag):
    sp = os.path.join(tmpdir, f"{tag}_script.py")
    with open(sp, "w") as f:
        f.write(script)
    proc = subprocess.run(
        [sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=180,
    )
    return proc


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
    win.quick_examples_combo.setCurrentText("1D Heat")
    win.adapt_combo.setCurrentText("Time Adaptive Training")
    _app.processEvents()

    # Time-Adaptive's per-step training always goes through the
    # scheduler-phases loop (the GUI-hidden, always-on scheduler), not the
    # legacy flat config.iterations field -- see test_sweep_execution.py's
    # own v73 comment for the same discovery. Shrink every phase so this
    # test actually runs fast.
    for ph in win.sched_phase_list:
        ph["iters"].setValue(5)
    win.iter1_spin.setValue(5)

    # Two separate, tiny sub-domains (1 step each) -- the smallest setup
    # that can actually distinguish "only the last sub-domain" (the bug)
    # from "the full cumulative run" (the fix): if the fix works, the
    # second sub-domain's steps must be appended ON TOP OF the first's,
    # not replace them.
    for row in list(win.ta_group_rows):
        row["widget"].deleteLater()
    win.ta_group_rows.clear()
    win._add_ta_step_group(0.0, 0.5, 1)
    win._add_ta_step_group(0.5, 1.0, 1)

    config = win._build_config()
    config.iterations = 5  # belt-and-braces, same caveat as test_sweep_execution.py's v73 case
    check(config.validate() == [], f"Time-Adaptive config should validate clean: {config.validate()}")
    check(config.time_adaptive, "config should record time_adaptive=True")

    # ── real, tiny, 2-sub-domain training run ────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        config.save_dir = tmpdir  # so _use_save is True and _ta_loss_path resolves under tmpdir, not /tmp
        script = generate_script(config)

        # ── static checks on the generated script itself ────────────────
        check(script.count("_ta_accumulate_loss(lh_i)") == 5,
              f"expected exactly 5 _ta_accumulate_loss(lh_i) calls (one per model_i.train() site), "
              f"got {script.count('_ta_accumulate_loss(lh_i)')}")
        check("train_loss_ta = _ta_all_train_loss" in script,
              "loss plot should read the accumulated _ta_all_train_loss, not a bare lh_i")
        check("train_loss_ta = lh_i.loss_train" not in script,
              "the old single-phase/single-sub-domain bug (train_loss_ta = lh_i.loss_train) should be gone")
        check("Loss — All " in script and "Time Sub-domain(s)" in script,
              "loss plot title should reflect the full run, not just the last sub-domain")

        diag_path = os.path.join(tmpdir, "ta_loss_diag.json")
        diag = (
            "\nimport json as _json_diag\n"
            "with open(" + repr(diag_path) + ", 'w') as _f_diag:\n"
            "    _json_diag.dump({\n"
            "        'all_steps': _ta_all_steps,\n"
            "        'boundaries': _ta_step_boundaries,\n"
            "        'offset': _ta_loss_offset,\n"
            "        'n_train_points': len(_ta_all_train_loss),\n"
            "        'n_test_points': len(_ta_all_test_loss),\n"
            "    }, _f_diag)\n"
        )
        proc = _run_script(script + diag, tmpdir, "ta_loss_history")
        check(proc.returncode == 0,
              f"Time-Adaptive script should run cleanly, got exit {proc.returncode}:\n"
              f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

        check(os.path.exists(diag_path) and os.path.getsize(diag_path) > 0,
              "diagnostic side-channel file should have been written (non-empty)")
        diag_data = None
        if os.path.exists(diag_path) and os.path.getsize(diag_path) > 0:
            try:
                with open(diag_path) as f:
                    diag_data = json.load(f)
            except Exception as e:
                check(False, f"diagnostic side-channel file should be valid JSON, got {type(e).__name__}: {e}")
        if diag_data is not None:
            all_steps = diag_data["all_steps"]
            boundaries = diag_data["boundaries"]

            check(len(all_steps) > 0, "accumulated step list should not be empty")
            check(diag_data["n_train_points"] == len(all_steps),
                  "accumulated train-loss list should be the same length as the accumulated step list")
            check(diag_data["n_test_points"] == len(all_steps),
                  "accumulated test-loss list should be the same length as the accumulated step list")

            check(len(boundaries) == 2,
                  f"expected one sub-domain boundary per sub-domain (2 sub-domains configured), "
                  f"got {len(boundaries)}: {boundaries}")

            check(all(b2 >= b1 for b1, b2 in zip(all_steps, all_steps[1:])),
                  "accumulated steps should be monotonically non-decreasing across the whole run "
                  "(each phase's offset should chain onto the previous one, not reset)")

            if len(boundaries) == 2:
                check(boundaries[1] > boundaries[0],
                      f"the SECOND sub-domain's cumulative offset ({boundaries[1]}) should be strictly "
                      f"greater than the FIRST's ({boundaries[0]}) -- proving its loss was appended on "
                      f"top of the first sub-domain's, not just replacing it (the original bug)")
                check(max(all_steps) > boundaries[0],
                      f"the final accumulated step ({max(all_steps)}) should extend past the first "
                      f"sub-domain's own boundary ({boundaries[0]}) -- i.e. the plot actually covers "
                      f"more than just one sub-domain")

        # _ta_loss_path = os.path.join(_sol_dir, "loss_plot.png"), and
        # _sol_dir = os.path.join(_save_dir, "solution_results") since
        # config.save_dir (== tmpdir) was set above, making _use_save True.
        loss_png = os.path.join(tmpdir, "solution_results", "loss_plot.png")
        check(os.path.exists(loss_png) and os.path.getsize(loss_png) > 0,
              f"Time-Adaptive loss plot PNG should be written and non-empty, checked {loss_png!r} "
              f"(dir contents: {os.listdir(tmpdir)})")

    win.close()
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL TIME-ADAPTIVE LOSS-HISTORY TESTS PASSED")
    return 0


def test_ta_loss_history():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
