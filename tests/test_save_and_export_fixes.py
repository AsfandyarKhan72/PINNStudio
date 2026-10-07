#!/usr/bin/env python3
"""
Checks four fixes, all found-and-flagged in earlier rounds and fixed in
this one:

  1. generate_script()'s plain Adam-only path (scheduler off, optimizer2
     "none") now actually calls model.save() when Save is enabled, for
     both Forward and Inverse problems. Previously the Forward case wrote
     only a model_adam-<iters>.json config file and never called
     model.save() at all -- a silent data-loss bug (flagged Round 15):
     anyone training that exact configuration with Save on would find no
     checkpoint for Model Restore to load. The Inverse case was worse --
     it wrote nothing at all for this configuration, not even the config
     JSON. Both scheduler-on and optimizer2-set configurations are
     confirmed unaffected (they already saved correctly; this fix must not
     double-save or disturb that).

  2. generate_clean_script()'s Time-Adaptive loss plot now stitches the
     loss history across EVERY optimizer phase of EVERY time sub-domain,
     mirroring generate_script()'s own _ta_accumulate_loss mechanism.
     Previously it plotted only loss_history -- whichever phase happened
     to run last for the very last sub-domain -- a known limitation
     flagged since Round 8 and carried forward, unfixed, through every
     round since.

  3. generate_clean_script()'s "Custom..." plot-field expression is now
     derivative-aware (du_x, du_xx, ...), via the same dde.Model.predict(
     x, operator=...) mechanism Round 17 wired into generate_script() and
     both Restore script builders -- reusing Training Monitors' own
     _tm_build_dvars, not a second derivative-building copy. Round 17
     explicitly left this exporter on the old NumPy-only path; this closes
     that gap.

  4. Error Analysis's per-group (expr, label) custom selector -- the rarer
     multi-output-reference-file case -- is now also derivative-aware,
     in both generate_script() paths (Standard and Time-Adaptive). This
     was Round 17's other explicitly-flagged gap: only the common
     _ea_sel is None case got derivative support "for free" via delegation
     to _extract_plot_field; the tuple-selector case stayed NumPy-only.

Each fix is checked both statically (AST-valid generated code, the right
wiring present/absent) and with a real, briefly-trained dde.Model run as a
subprocess, confirming the actual behavior, not just the generated text.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_save_and_export_fixes.py
"""
import os
import sys
import ast
import glob
import subprocess
import dataclasses

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

import numpy as np

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import generate_script, generate_clean_script


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    base_cfg = PINNConfig()
    base_cfg.ic_expressions = "sin(pi*x)"
    base_cfg.iterations = 5

    # =====================================================================
    # 1) model.save() for the plain Adam-only save path
    # =====================================================================
    save_cfg = dataclasses.replace(
        base_cfg, save_dir="/tmp/_fake_save_dir", optimizer2="none",
        optimizer_scheduler=False, scheduler_phases="[]",
    )
    fwd_script = generate_script(save_cfg)
    ast.parse(fwd_script)
    check("model.save(_adam_model_path)" in fwd_script,
          "Forward plain-Adam-only + Save-on should now call model.save(), "
          "not just write the config JSON (Round 15's flagged bug)")

    inv_cfg = dataclasses.replace(save_cfg, problem_type="Inverse")
    inv_script = generate_script(inv_cfg)
    ast.parse(inv_script)
    check("model.save(_adam_model_path)" in inv_script,
          "Inverse plain-Adam-only + Save-on should also now call "
          "model.save() -- previously this configuration saved nothing "
          "at all, not even a config JSON")

    # With optimizer2 set, the early save in the Phase-1 block must be
    # skipped (the Phase 2 block below saves the real final checkpoint) --
    # no double/premature save.
    phase2_cfg = dataclasses.replace(save_cfg, optimizer2="lbfgs", iterations2=5)
    phase2_script = generate_script(phase2_cfg)
    ast.parse(phase2_script)
    check("model saved after scheduler phases / Phase 2 below" in phase2_script,
          "with optimizer2 set, the Phase-1 block should skip its own "
          "early save (Phase 2 saves the real final checkpoint)")

    # With the scheduler active, the early save must also be skipped (each
    # scheduler phase already saves its own checkpoint).
    sched_cfg = dataclasses.replace(
        save_cfg, optimizer_scheduler=True,
        scheduler_phases='[{"optimizer": "adam", "iterations": 5, "weights": "1,1,1,1"}]',
    )
    sched_script = generate_script(sched_cfg)
    ast.parse(sched_script)
    check("model saved after scheduler phases / Phase 2 below" in sched_script,
          "with the scheduler active, the Phase-1 block should also skip "
          "its own early save")

    # Real end-to-end: train a plain Adam-only run with Save on, confirm an
    # actual .pt checkpoint lands on disk (previously none would exist).
    import tempfile
    _tmpdir = tempfile.mkdtemp(prefix="pinnstudio_test_save_")
    e2e_cfg = dataclasses.replace(
        base_cfg, save_dir=_tmpdir, iterations=15,
        optimizer2="none", optimizer_scheduler=False, scheduler_phases="[]",
    )
    e2e_script = generate_script(e2e_cfg)
    _script_path = os.path.join(_tmpdir, "run.py")
    with open(_script_path, "w") as _f:
        _f.write(e2e_script)
    _r = subprocess.run([sys.executable, _script_path], cwd=_tmpdir,
                         capture_output=True, text=True, timeout=180)
    check(_r.returncode == 0,
          f"plain Adam-only training script should run to completion "
          f"(stderr tail: {_r.stderr[-800:]})")
    _pt_files = glob.glob(os.path.join(_tmpdir, "**", "*.pt"), recursive=True)
    check(len(_pt_files) >= 1,
          "a real .pt checkpoint should now exist after a plain Adam-only "
          "+ Save-on run (this is the actual data-loss bug this fix closes)")

    # =====================================================================
    # 2) Time-Adaptive clean-script loss plot: stitched across all steps
    # =====================================================================
    ta_clean_cfg = dataclasses.replace(base_cfg, time_adaptive=True, ta_num_steps=2)
    ta_clean_script = generate_clean_script(ta_clean_cfg)
    ast.parse(ta_clean_script)
    check("_ta_all_steps, _ta_all_train_loss, _ta_all_test_loss = [], [], []" in ta_clean_script,
          "TA clean-script should initialize the stitched-loss accumulators")
    check(ta_clean_script.count("_ta_all_steps.extend") >= 1,
          "TA clean-script should accumulate loss after each phase, not just read loss_history once")
    check("plt.semilogy(_ta_all_steps, train_loss" in ta_clean_script,
          "TA clean-script's final loss plot should use the stitched arrays, not loss_history directly")
    check("last Time-Adaptive step's loss curve" not in ta_clean_script,
          "the old 'last step only' limitation comment should be gone (regression guard)")

    plain_clean_script = generate_clean_script(base_cfg)
    ast.parse(plain_clean_script)
    check("plt.semilogy(loss_history.steps, train_loss" in plain_clean_script,
          "the non-TA clean script should be unaffected -- still plots loss_history directly")

    # Real end-to-end: a 3-step TA clean-script run should accumulate loss
    # points spanning ALL steps, not just the last one.
    import tempfile as _tempfile2
    _ta_tmpdir = _tempfile2.mkdtemp(prefix="pinnstudio_test_ta_clean_")
    ta_e2e_cfg = dataclasses.replace(base_cfg, time_adaptive=True, ta_num_steps=3, iterations=10)
    ta_e2e_script = generate_clean_script(ta_e2e_cfg)
    _marker = "plt.savefig(loss_path"
    _idx = ta_e2e_script.index(_marker)
    _debug_line = ('print("TEST_DEBUG_TA_STEPS", len(_ta_all_steps), '
                    'max(_ta_all_steps) if _ta_all_steps else None)\n')
    ta_e2e_script_debug = ta_e2e_script[:_idx] + _debug_line + ta_e2e_script[_idx:]
    _ta_script_path = os.path.join(_ta_tmpdir, "run.py")
    with open(_ta_script_path, "w") as _f:
        _f.write(ta_e2e_script_debug)
    _r2 = subprocess.run([sys.executable, _ta_script_path], cwd=_ta_tmpdir,
                          capture_output=True, text=True, timeout=180)
    check(_r2.returncode == 0,
          f"3-step TA clean-script should run to completion (stderr tail: {_r2.stderr[-800:]})")
    _debug_line_out = next((l for l in _r2.stdout.splitlines() if "TEST_DEBUG_TA_STEPS" in l), "")
    check(bool(_debug_line_out), "the TA clean-script run should have reached the debug print")
    if _debug_line_out:
        _parts = _debug_line_out.split()
        _n_points, _max_step = int(_parts[1]), float(_parts[2])
        check(_n_points >= 3,
              f"stitched loss array should have at least one point per step (3 steps); got {_n_points}")
        check(_max_step > ta_e2e_cfg.iterations,
              f"stitched loss array's max iteration ({_max_step}) should exceed a single "
              f"step's own iteration count ({ta_e2e_cfg.iterations}) -- confirms offsetting/"
              f"stitching across steps actually happened, not just the last step's own history")

    # =====================================================================
    # 3) generate_clean_script() derivative-aware custom plotting
    # =====================================================================
    deriv_clean_cfg = dataclasses.replace(base_cfg, plot_custom_expr="du_x", plot_custom_label="du_dx")
    deriv_clean_script = generate_clean_script(deriv_clean_cfg)
    ast.parse(deriv_clean_script)
    check("operator=_plot_custom_op" in deriv_clean_script,
          "a configured custom expression in the clean-script exporter should now route "
          "through model.predict(..., operator=_plot_custom_op)")
    check("_tm_build_dvars" in deriv_clean_script,
          "the clean-script exporter's custom-expression operator should reuse Training "
          "Monitors' own _tm_build_dvars, not a second derivative-building copy")
    check("torch.sin" in deriv_clean_script and "torch.sqrt" in deriv_clean_script,
          "the clean-script exporter's custom-expression math namespace should use torch "
          "equivalents (autograd-safe), not np.*")

    plain_clean_script2 = generate_clean_script(base_cfg)
    ast.parse(plain_clean_script2)
    # Checking for "def _tm_build_dvars(" (the actual helper definition),
    # not just the substring "_tm_build_dvars" -- that substring is always
    # present as dead code inside _plot_custom_op's own (always-embedded)
    # body, same "helper always present, only reached conditionally"
    # convention Training Monitors itself established; a bare substring
    # check here would be exactly the kind of trap Round 17 already hit
    # once with "OperatorPredictor".
    check("def _tm_build_dvars(" not in plain_clean_script2,
          "with no custom expression and Training Monitors off, the clean-script exporter "
          "should NOT embed the Training Monitors runtime code's helper definitions (keeps "
          "the minimal/tutorial-style exporter's own 'only what's configured' philosophy)")

    # Real end-to-end: a trained clean-script with plot_custom_expr="du_x"
    # should run to completion and produce a solution plot.
    import tempfile as _tempfile3
    _clean_tmpdir = _tempfile3.mkdtemp(prefix="pinnstudio_test_clean_deriv_")
    clean_e2e_cfg = dataclasses.replace(base_cfg, iterations=30,
                                         plot_custom_expr="du_x", plot_custom_label="du_dx")
    clean_e2e_script = generate_clean_script(clean_e2e_cfg)
    _clean_script_path = os.path.join(_clean_tmpdir, "run.py")
    with open(_clean_script_path, "w") as _f:
        _f.write(clean_e2e_script)
    _r3 = subprocess.run([sys.executable, _clean_script_path], cwd=_clean_tmpdir,
                          capture_output=True, text=True, timeout=180)
    check(_r3.returncode == 0,
          f"derivative-custom-expr clean-script should run to completion "
          f"(stderr tail: {_r3.stderr[-800:]})")
    check(os.path.exists("/tmp/solution_plot.png"),
          "the derivative-custom-expr clean-script should produce a solution plot")

    # =====================================================================
    # 4) Error Analysis per-group (expr, label) selector: derivative-aware
    # =====================================================================
    ea_cfg = dataclasses.replace(
        base_cfg,
        ea_files=repr([(0.0, "/tmp/_fake_ref.txt", ["du_x", "dudx"])]),
        ea_do_line=True,
    )
    ea_script = generate_script(ea_cfg)
    ast.parse(ea_script)
    check("def _ea_custom_op(" in ea_script,
          "a per-group (expr, label) EA selector should now get its own derivative-aware operator helper")
    check("_ea_m.predict(_ea_grid, operator=_ea_custom_op)" in ea_script,
          "the per-group tuple-selector branch should route through model.predict(..., operator=...), "
          "not the old pre-computed-prediction NumPy eval")

    ea_ta_cfg = dataclasses.replace(ea_cfg, time_adaptive=True, ta_num_steps=2)
    ea_ta_script = generate_script(ea_ta_cfg)
    ast.parse(ea_ta_script)
    check("_ea_model.predict(_ea_grid, operator=_ea_custom_op)" in ea_ta_script,
          "the Time-Adaptive path's per-group tuple-selector branch should also be derivative-aware")

    # int selector and None selector must be unaffected (no regression):
    # these still go through the original, simpler branches.
    int_sel_cfg = dataclasses.replace(
        base_cfg, ea_files=repr([(0.0, "/tmp/_fake_ref.txt", 0)]), ea_do_line=True)
    int_sel_script = generate_script(int_sel_cfg)
    ast.parse(int_sel_script)
    check("_ea_m.predict(_ea_grid)[:, _ea_sel]" in int_sel_script,
          "an int EA selector should still take the plain column-indexing branch, unchanged")

    # Real end-to-end: a per-group derivative selector should run to
    # completion against a real reference file (numeric correctness of the
    # underlying operator mechanism itself was already verified in Round
    # 17's own test; this confirms the NEW per-group wiring reaches it
    # without crashing, end to end, as a real subprocess).
    import tempfile as _tempfile4
    _ea_tmpdir = _tempfile4.mkdtemp(prefix="pinnstudio_test_ea_deriv_")
    _ref_path = os.path.join(_ea_tmpdir, "ref_t0.txt")
    _x = np.linspace(0.0, 1.0, 11)
    _t = np.zeros_like(_x)
    _u = np.sin(np.pi * _x)
    np.savetxt(_ref_path, np.column_stack([_x, _t, _u]))

    ea_e2e_cfg = dataclasses.replace(
        base_cfg, iterations=20, save_dir="",
        ea_files=repr([(0.0, _ref_path, ["du_x", "dudx"])]),
        ea_do_line=True,
    )
    ea_e2e_script = generate_script(ea_e2e_cfg)
    _ea_script_path = os.path.join(_ea_tmpdir, "run.py")
    with open(_ea_script_path, "w") as _f:
        _f.write(ea_e2e_script)
    _r4 = subprocess.run([sys.executable, _ea_script_path], cwd=_ea_tmpdir,
                          capture_output=True, text=True, timeout=180)
    check(_r4.returncode == 0,
          f"derivative per-group EA selector script should run to completion "
          f"(stderr tail: {_r4.stderr[-800:]})")
    check("Group 'dudx' analysis complete" in _r4.stdout,
          "the derivative-aware per-group EA selector should complete its analysis "
          "group, confirming the operator-based path executed without error")

    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nALL SAVE/EXPORT-FIX TESTS PASSED")


if __name__ == "__main__":
    main()


def test_save_and_export_fixes():
    failures = run()
    assert not failures, "\n".join(failures)
