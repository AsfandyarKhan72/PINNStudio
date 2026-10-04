#!/usr/bin/env python3
"""
Real, end-to-end Parameter Sweep execution check: actually runs
pinnstudio.core.sweep_runner.run_sweep() against a live config -- real
subprocess training, not mocked -- on 1D Heat with iterations shrunk
down purely for test speed (not training quality). This is the one
sweep test that needs torch/deepxde actually installed, which is why
it's kept out of test_sweep_tab.py and NOT wired into the CI
smoke-test workflow (that workflow only installs PyQt6/numpy/
matplotlib/pandas). Run this locally instead, wherever PINNStudio's own
install.sh/install.bat environment (or an equivalent one with torch +
deepxde) is available.

Checks:
 - A tiny real sweep (baseline + one hidden_layers variant) on 1D Heat
   runs to completion, both runs reporting status "done" with a real,
   distinct parsed final_loss, and l2_relative=None (1D Heat has no
   Error Analysis reference data configured by default, so nothing
   should ever be reported there).
 - The same sweep on 3D Poisson (Sphere) -- which DOES have Error
   Analysis reference data configured by default -- reports a real,
   distinct l2_relative for each run, confirming sweep_runner reads
   back each run's own freshly-written error_metrics.txt rather than a
   stale one left over from an earlier run (the exact bug caught by
   hand while building this feature).
 - A real sweep over a loss weight (sweep_registry.py's "SCOPE (v2)"
   entries) on 1D Heat: pushing the Initial Condition weight to an
   extreme value (100000, vs. the default 100) for a handful of
   iterations measurably changes the real, parsed final_loss relative
   to the baseline. test_sweep_registry.py already checks -- without
   training -- that the swept value reaches every scheduler phase's
   embedded weights string in the generated script; this is the one
   check that it also reaches an ACTUAL training run's real result, not
   just the script text.
 - Inverse-problem entries ("SCOPE (v3)"): a real sweep over a
   trainable variable's own initial guess, and a real sweep over a
   measured-data file's own loss weight, both on 2D Poisson (Disk)
   (which has a real bundled u_obs.txt and a known true value
   auto-populated the moment Inverse mode is selected) -- each
   measurably changes the real, parsed final_loss relative to baseline.
 - v73: a real RAR sweep (sweeping rar_cycles) and a real Time-Adaptive
   sweep (sweeping ta_grid_size, one group/one step for speed) each run
   to completion; a combined sweep over an output-transform scale and
   Activation (one value each, to bound the number of new real runs this
   round adds) also runs to completion -- test_sweep_param_expansion.py
   already confirms every new registry category reaches the generated
   script's text without running it; this confirms each one also
   actually trains for real.
 - Stop hard-kills the CURRENTLY RUNNING training subprocess instead of
   only preventing the NEXT run from starting -- a huge-iteration-count
   run is started for real, stopped shortly after it begins, and the
   subprocess/thread are confirmed to actually exit within seconds
   rather than continuing to train in the background.
 - Once a real sweep finishes, the right panel shows the last
   successfully-completed run's own real loss/solution figure (the
   signal-handler wiring itself, with synthetic files, is covered in
   test_sweep_tab.py; this confirms it also works end-to-end against a
   real run_sweep() call).

Run directly (this trains for real, so budget a few minutes depending
on the machine):
    QT_QPA_PLATFORM=offscreen python3 tests/test_sweep_execution.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.sweep_runner import run_sweep


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── 1D Heat: no Error Analysis data configured ───────────────────
    win = MainWindow()
    _app.processEvents()
    for ph in win.sched_phase_list:
        ph["iters"].setValue(15)  # shrunk purely for test speed
    row = win.sweep_row_list[0]
    row["param_combo"].setCurrentIndex(row["param_combo"].findData("hidden_layers"))
    row["mode_combo"].setCurrentIndex(row["mode_combo"].findData("list"))
    row["list_edit"].setText("2")
    win.sweep_enable_cb.setChecked(True)
    config = win._build_config()
    check(config.validate() == [], f"1D Heat sweep config should validate clean: {config.validate()}")
    check(config.ea_files in ("[]", []), "1D Heat should have no Error Analysis data configured by default")

    results = run_sweep(config)
    check(len(results) == 2, f"expected 2 runs (baseline + hidden_layers=2), got {len(results)}")
    for label, result in results:
        check(result["status"] == "done", f"[1D Heat] run {label!r} should finish 'done', got {result}")
        check(result["final_loss"] is not None, f"[1D Heat] run {label!r} should report a parsed final_loss, got {result}")
        check(result["l2_relative"] is None,
              f"[1D Heat] run {label!r} should report l2_relative=None (no EA data configured), got {result}")

    # ── 3D Poisson (Sphere): DOES have Error Analysis data configured ─
    win2 = MainWindow()
    _app.processEvents()
    win2.radio_3d.setChecked(True)
    win2.quick_examples_combo.setCurrentText("3D Poisson (Sphere)")
    _app.processEvents()
    for ph in win2.sched_phase_list:
        ph["iters"].setValue(15)
    row2 = win2.sweep_row_list[0]
    row2["param_combo"].setCurrentIndex(row2["param_combo"].findData("hidden_layers"))
    row2["mode_combo"].setCurrentIndex(row2["mode_combo"].findData("list"))
    row2["list_edit"].setText("2")
    win2.sweep_enable_cb.setChecked(True)
    config2 = win2._build_config()
    check(config2.validate() == [], f"3D Sphere sweep config should validate clean: {config2.validate()}")
    check(config2.ea_files not in ("[]", []), "3D Poisson (Sphere) should have Error Analysis data configured by default")

    results2 = run_sweep(config2)
    check(len(results2) == 2, f"expected 2 runs (baseline + hidden_layers=2), got {len(results2)}")
    l2s = []
    for label, result in results2:
        check(result["status"] == "done", f"[3D Sphere] run {label!r} should finish 'done', got {result}")
        check(result["l2_relative"] is not None,
              f"[3D Sphere] run {label!r} should report a real l2_relative (EA data IS configured), got {result}")
        l2s.append(result["l2_relative"])
    check(len(set(l2s)) == len(l2s),
          f"each 3D Sphere run should read back its OWN freshly-written error_metrics.txt, not a shared/stale one -- got identical values: {l2s}")

    # ── Loss weight sweep: a real training run, not just generated-
    # script text (test_sweep_registry.py already checks the text) ────
    win3 = MainWindow()
    _app.processEvents()
    win3.quick_examples_combo.setCurrentText("1D Heat")
    _app.processEvents()
    for ph in win3.sched_phase_list:
        ph["iters"].setValue(20)
    row3 = win3.sweep_row_list[0]
    row3["param_combo"].setCurrentIndex(row3["param_combo"].findData("weight_ic_0"))
    row3["mode_combo"].setCurrentIndex(row3["mode_combo"].findData("list"))
    row3["list_edit"].setText("100000")  # the default IC weight is 100 -- a 1000x jump
    win3.sweep_enable_cb.setChecked(True)
    config3 = win3._build_config()
    check(config3.validate() == [], f"1D Heat weight-sweep config should validate clean: {config3.validate()}")

    results3 = run_sweep(config3)
    check(len(results3) == 2, f"expected 2 runs (baseline + weight_ic_0=100000), got {len(results3)}")
    losses3 = [r.get("final_loss") for _, r in results3]
    check(all(v is not None for v in losses3),
          f"[1D Heat weight sweep] both runs should report a parsed final_loss, got {losses3}")
    if all(v is not None for v in losses3):
        baseline_loss3, swept_loss3 = losses3
        check(abs(swept_loss3 - baseline_loss3) > 1e-6 * max(abs(baseline_loss3), 1e-12),
              f"[1D Heat weight sweep] pushing the IC weight to 100000 should measurably change "
              f"the real final_loss vs. baseline -- got baseline={baseline_loss3}, "
              f"swept={swept_loss3} (if these are too close, the swept weight may not actually "
              "be reaching training)")

    # ── v68: real per-run save folders + manifest/summary CSV ────────
    # (sweep_save_dir set -> every run's own save_dir gets pointed at a
    # distinct, descriptively-named subfolder, and the EXISTING codegen
    # save pipeline -- already exercised above for plain save_dir runs
    # -- writes real model/plot/log files into each one automatically.)
    import glob
    import json as _json
    import shutil
    import tempfile

    tmp_root = tempfile.mkdtemp(prefix="v68_sweep_exec_")
    try:
        win4 = MainWindow()
        _app.processEvents()
        win4.quick_examples_combo.setCurrentText("1D Heat")
        _app.processEvents()
        for ph in win4.sched_phase_list:
            ph["iters"].setValue(15)
        row4 = win4.sweep_row_list[0]
        row4["param_combo"].setCurrentIndex(row4["param_combo"].findData("hidden_layers"))
        row4["mode_combo"].setCurrentIndex(row4["mode_combo"].findData("list"))
        row4["list_edit"].setText("2")
        win4.sweep_enable_cb.setChecked(True)
        win4.save_dir_input.setText(tmp_root)
        config4 = win4._build_config()
        check(config4.sweep_save_dir == tmp_root, "sweep_save_dir should round-trip into the config "
              "(from the shared 'Save to:' field, not a dedicated sweep one)")
        check(config4.validate() == [], f"per-run-folder sweep config should validate clean: {config4.validate()}")

        # Also drive the real GUI signal handlers (not just run_sweep()
        # directly) with this sweep's REAL results, piggybacking on the
        # 2 runs this section already executes rather than paying for
        # more real training elsewhere -- confirms the right panel ends
        # up showing the LAST completed run's own real loss/solution
        # figure once a real sweep finishes (the handler wiring itself,
        # with synthetic files, is covered in test_sweep_tab.py; this is
        # the one check that it also works end-to-end against a real
        # run_sweep() call).
        win4._sweep_base_config = config4
        win4._sweep_run_count = 0
        win4._sweep_last_run_dir = None
        win4._sweep_last_run_label = None

        captured_root = []
        results4 = run_sweep(
            config4, on_sweep_root=lambda r: captured_root.append(r),
            on_run_done=lambda i, total, label, result: win4._on_sweep_run_done(i, total, label, result))
        check(len(captured_root) == 1, "on_sweep_root should fire exactly once with the real sweep root")
        sweep_root = captured_root[0] if captured_root else None
        check(bool(sweep_root) and sweep_root.startswith(tmp_root),
              f"the real sweep root should live under the configured sweep_save_dir, got {sweep_root!r}")
        check(len(results4) == 2, f"expected 2 runs (baseline + hidden_layers=2), got {len(results4)}")

        expected_folders = ["run_000_baseline", "run_001_Hidden_layers=2"]
        if sweep_root:
            for folder in expected_folders:
                run_dir = os.path.join(sweep_root, folder)
                check(os.path.isdir(run_dir), f"expected a per-run folder at {run_dir}")
                check(os.path.isdir(os.path.join(run_dir, "solution_results")),
                      f"expected the existing codegen save pipeline to populate solution_results/ under {run_dir}")
                check(len(glob.glob(os.path.join(run_dir, "solution_results", "model*.pt"))) > 0,
                      f"expected a real saved model checkpoint under {run_dir}/solution_results")
                check(os.path.isfile(os.path.join(run_dir, "solution_results", "loss_plot.png")),
                      f"expected a real loss_plot.png under {run_dir}/solution_results")
                check(os.path.isfile(os.path.join(run_dir, "solution_results", "solution_plot.png")),
                      f"expected a real solution_plot.png under {run_dir}/solution_results")

            win4._on_sweep_finished()
            last_run_dir = os.path.join(sweep_root, "run_001_Hidden_layers=2")
            check(win4._sweep_last_run_dir == last_run_dir,
                  f"the last completed run's folder should be remembered, got {win4._sweep_last_run_dir}")
            check(win4.loss_label._source_path == os.path.join(last_run_dir, "solution_results", "loss_plot.png"),
                  f"after a real sweep finishes, the right panel should show the last run's real loss figure, got {win4.loss_label._source_path!r}")
            check(win4.solution_label._source_path == os.path.join(last_run_dir, "solution_results", "solution_plot.png"),
                  f"after a real sweep finishes, the right panel should show the last run's real solution figure, got {win4.solution_label._source_path!r}")

            for label, result in results4:
                check(result["save_dir"] in (os.path.join(sweep_root, "run_000_baseline"),
                                              os.path.join(sweep_root, "run_001_Hidden_layers=2")),
                      f"[{label}] result save_dir should point at that run's own per-run folder, got {result['save_dir']}")

            manifest_path = os.path.join(sweep_root, "sweep_manifest.json")
            csv_path = os.path.join(sweep_root, "sweep_summary.csv")
            check(os.path.isfile(manifest_path), f"expected a sweep_manifest.json at {manifest_path}")
            check(os.path.isfile(csv_path), f"expected a sweep_summary.csv at {csv_path}")
            if os.path.isfile(manifest_path):
                with open(manifest_path) as f:
                    manifest = _json.load(f)
                check(manifest.get("sweep_mode") == "oat",
                      f"manifest should record the real sweep_mode, got {manifest.get('sweep_mode')}")
                check(len(manifest.get("runs", [])) == 2, f"manifest should have one entry per run, got {manifest.get('runs')}")
                for run_entry in manifest.get("runs", []):
                    check(run_entry.get("status") == "done" and run_entry.get("final_loss") is not None,
                          f"manifest run entry should reflect the real completed result: {run_entry}")
            if os.path.isfile(csv_path):
                with open(csv_path) as f:
                    csv_text = f.read()
                check(csv_text.startswith("Run,Status,Final loss,L2 relative error,Folder"),
                      f"sweep_summary.csv should have the expected header, got {csv_text.splitlines()[:1]}")
                check("run_000_baseline" in csv_text and "run_001_Hidden_layers=2" in csv_text,
                      f"sweep_summary.csv should list both runs' own folder names, got:\n{csv_text}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # ── v68: a real "zip" (Specified Combinations) sweep ──────────────
    # 2 parameters, 2 values each -> zip() pairs them up (3 runs: 1
    # baseline + 2 paired), NOT a 5-run cross product like grid mode.
    win5 = MainWindow()
    _app.processEvents()
    win5.quick_examples_combo.setCurrentText("1D Heat")
    _app.processEvents()
    for ph in win5.sched_phase_list:
        ph["iters"].setValue(15)
    row5a = win5.sweep_row_list[0]
    row5a["param_combo"].setCurrentIndex(row5a["param_combo"].findData("hidden_layers"))
    row5a["mode_combo"].setCurrentIndex(row5a["mode_combo"].findData("list"))
    row5a["list_edit"].setText("2, 3")
    win5._add_sweep_row()
    row5b = win5.sweep_row_list[1]
    row5b["param_combo"].setCurrentIndex(row5b["param_combo"].findData("neurons_per_layer"))
    row5b["mode_combo"].setCurrentIndex(row5b["mode_combo"].findData("list"))
    row5b["list_edit"].setText("16, 24")
    win5.sweep_enable_cb.setChecked(True)
    win5.sweep_mode_combo.setCurrentIndex(win5.sweep_mode_combo.findData("zip"))
    config5 = win5._build_config()
    check(config5.sweep_mode == "zip", "config should record sweep_mode='zip'")
    check(config5.validate() == [], f"real zip-mode sweep config should validate clean: {config5.validate()}")

    results5 = run_sweep(config5)
    check(len(results5) == 3, f"zip mode with 2+2 equal-length values should give 3 real runs "
                               f"(1 baseline + 2 paired), got {len(results5)}: {[l for l, _ in results5]}")
    labels5 = [l for l, _ in results5]
    if len(labels5) == 3:
        check(labels5[0] == "baseline"
              and labels5[1].endswith("=2, Neurons per layer=16")
              and labels5[2].endswith("=3, Neurons per layer=24"),
              f"zip mode should pair each row's Nth value together, got {labels5}")
    for label, result in results5:
        check(result["status"] == "done", f"[zip sweep] run {label!r} should finish 'done', got {result}")
        check(result["final_loss"] is not None, f"[zip sweep] run {label!r} should report a real final_loss, got {result}")

    # ── Inverse: initial guess + observation loss weight sweep together
    # ("SCOPE v3") -- one "one-at-a-time" sweep over BOTH new Inverse
    # entries at once (baseline + 1 variant per entry = 3 runs total,
    # instead of running two separate 2-run sweeps) -- real training on
    # 2D Poisson (Disk), which has a real, bundled u_obs.txt and a known
    # true source_term value (see main_window.py's INVERSE_AUTO_VARS/
    # INVERSE_AUTO_OBS) auto-populated the moment Inverse mode is
    # selected for this template. Switching a row's parameter to an
    # Inverse-only id requires a fresh _refresh_sweep_param_choices()
    # call first: the combo is only populated against whatever template/
    # mode was active the last time it refreshed, and nothing auto-
    # refreshes it just from flipping Forward/Inverse.
    win6 = MainWindow()
    _app.processEvents()
    win6.radio_2d.setChecked(True)
    win6.quick_examples_combo.setCurrentText("2D Poisson (Disk)")
    win6.radio_inverse.setChecked(True)
    _app.processEvents()
    for ph in win6.sched_phase_list:
        ph["iters"].setValue(20)
    win6._refresh_sweep_param_choices()
    row6a = win6.sweep_row_list[0]
    _idx6a = row6a["param_combo"].findData("inv_var_init_0")
    check(_idx6a >= 0, "inv_var_init_0 should be selectable once Inverse mode is on for a template with a known trainable variable")
    win6._add_sweep_row()
    win6._refresh_sweep_param_choices()
    row6b = win6.sweep_row_list[1]
    _idx6b = row6b["param_combo"].findData("inv_obs_weight_0")
    check(_idx6b >= 0, "inv_obs_weight_0 should be selectable once Inverse mode is on for a template with a measured-data file")
    if _idx6a >= 0 and _idx6b >= 0:
        row6a["param_combo"].setCurrentIndex(_idx6a)
        row6a["mode_combo"].setCurrentIndex(row6a["mode_combo"].findData("list"))
        row6a["list_edit"].setText("5.0")
        row6b["param_combo"].setCurrentIndex(_idx6b)
        row6b["mode_combo"].setCurrentIndex(row6b["mode_combo"].findData("list"))
        row6b["list_edit"].setText("100000")  # the template's own default is 100.0 -- a 1000x jump
        win6.sweep_enable_cb.setChecked(True)
        win6.sweep_mode_combo.setCurrentIndex(win6.sweep_mode_combo.findData("oat"))
        config6 = win6._build_config()
        check(config6.problem_type == "Inverse", "sanity: this config should be Inverse")
        check(config6.validate() == [], f"Inverse one-at-a-time sweep config should validate clean: {config6.validate()}")

        results6 = run_sweep(config6)
        check(len(results6) == 3, f"expected 3 runs (baseline + 1 initial-guess variant + 1 obs-weight variant), got {len(results6)}")
        labels6 = [l for l, _ in results6]
        losses6 = {l: r.get("final_loss") for l, r in results6}
        check(all(v is not None for v in losses6.values()),
              f"[Inverse oat sweep] every run should report a parsed final_loss, got {losses6}")
        baseline6 = losses6.get("baseline")
        init_run6 = next((l for l in labels6 if "initial guess" in l), None)
        obs_run6 = next((l for l in labels6 if "Data loss weight" in l), None)
        check(init_run6 is not None and obs_run6 is not None,
              f"expected one run per swept Inverse entry, got labels {labels6}")
        if baseline6 is not None and init_run6 and losses6.get(init_run6) is not None:
            check(abs(losses6[init_run6] - baseline6) > 1e-6 * max(abs(baseline6), 1e-12),
                  f"[Inverse initial-guess sweep] starting the trainable variable at 5.0 "
                  f"instead of the template's own default should measurably change the "
                  f"real final_loss after a few iterations -- got baseline={baseline6}, "
                  f"swept={losses6[init_run6]}")
        if baseline6 is not None and obs_run6 and losses6.get(obs_run6) is not None:
            check(abs(losses6[obs_run6] - baseline6) > 1e-6 * max(abs(baseline6), 1e-12),
                  f"[Inverse obs-weight sweep] pushing the observation loss weight to "
                  f"100000 should measurably change the real final_loss vs. baseline -- "
                  f"got baseline={baseline6}, swept={losses6[obs_run6]}")

    # ── v73: RAR sweep, real training ─────────────────────────────────
    # test_sweep_param_expansion.py already confirms rar_cycles reaches
    # the generated script's text without running it; this confirms a
    # swept RAR config actually trains to completion for real (RAR's own
    # knobs -- cycles/candidates/add_points/adam_iters -- are all forced
    # tiny here purely for test speed).
    win9 = MainWindow()
    _app.processEvents()
    win9.quick_examples_combo.setCurrentText("1D Heat")
    win9.adapt_combo.setCurrentText("Residual-based Adaptive Refinement (RAR)")
    _app.processEvents()
    for ph in win9.sched_phase_list:
        ph["iters"].setValue(10)
    win9.rar_candidates.setValue(500)
    win9.rar_add_points.setValue(50)
    win9.rar_adam_iters.setValue(10)
    win9.rar_lbfgs_iters.setValue(0)
    # The one sweep row already present on a fresh MainWindow was built
    # at construction time, before RAR was selected above -- its
    # param_combo's choices are stale until refreshed (same gotcha the
    # Inverse sweep section above already works around). Without this,
    # findData("rar_cycles") returns -1, setCurrentIndex(-1) silently
    # leaves the combo on whatever it already had selected, and the
    # sweep runs against the WRONG parameter instead of raising anything
    # -- exactly what happened here before this fix was added (real
    # training errors against a stale "Weight decay" selection).
    win9._refresh_sweep_param_choices()
    row9 = win9.sweep_row_list[0]
    _idx9 = row9["param_combo"].findData("rar_cycles")
    check(_idx9 >= 0, "rar_cycles should be selectable once RAR is the adaptive method")
    row9["param_combo"].setCurrentIndex(_idx9)
    row9["mode_combo"].setCurrentIndex(row9["mode_combo"].findData("list"))
    row9["list_edit"].setText("1, 2")
    win9.sweep_enable_cb.setChecked(True)
    config9 = win9._build_config()
    check(config9.validate() == [], f"RAR sweep config should validate clean: {config9.validate()}")
    check(config9.adapt_method == "RAR", "config should record adapt_method='RAR'")

    results9 = run_sweep(config9)
    check(len(results9) == 3, f"expected 1 baseline + 2 swept RAR runs, got {len(results9)}")
    for label, result in results9:
        check(result["status"] == "done", f"[RAR sweep] run {label!r} should finish 'done', got {result}")
        check(result["final_loss"] is not None, f"[RAR sweep] run {label!r} should report a parsed final_loss, got {result}")

    # ── v73: Time-Adaptive sweep, real training ───────────────────────
    # ta_grid_size swept between two small values, with a single tiny
    # time step. IMPORTANT (found while verifying this real-training run
    # -- it was taking 30+ minutes per run before this fix): Time-
    # Adaptive's own per-step training does NOT read the legacy flat
    # config.iterations field at all once the (GUI-hidden, always-on)
    # scheduler is active -- it runs through the SAME scheduler-phases
    # loop the Standard (non-Time-Adaptive) path uses, i.e.
    # win.sched_phase_list's own iteration counts, once PER STEP. With
    # the template's default phases (10000 Adam + 5000 L-BFGS) and even
    # just 1 step, that's 15000+ real iterations per run -- so, exactly
    # like win9 (RAR) and win11 (Output-transform/Activation) already do,
    # the scheduler phases themselves must be shrunk for this to actually
    # be a fast test.
    win10 = MainWindow()
    _app.processEvents()
    win10.quick_examples_combo.setCurrentText("1D Heat")
    win10.adapt_combo.setCurrentText("Time Adaptive Training")
    _app.processEvents()
    for ph in win10.sched_phase_list:
        ph["iters"].setValue(10)
    win10.iter1_spin.setValue(10)
    for row in list(win10.ta_group_rows):
        row["widget"].deleteLater()
    win10.ta_group_rows.clear()
    win10._add_ta_step_group(0.0, 1.0, 1)  # one group, one step -- as fast as Time-Adaptive gets
    win10._refresh_sweep_param_choices()  # see win9's comment above -- same stale-combo gotcha
    row10 = win10.sweep_row_list[0]
    _idx10 = row10["param_combo"].findData("ta_grid_size")
    check(_idx10 >= 0, "ta_grid_size should be selectable once Time Adaptive is the adaptive method")
    row10["param_combo"].setCurrentIndex(_idx10)
    row10["list_edit"].setText("11, 21")
    win10.sweep_enable_cb.setChecked(True)
    config10 = win10._build_config()
    config10.iterations = 10  # belt-and-braces only (see comment above -- this field is NOT
                               # actually what Time-Adaptive's per-step training reads whenever
                               # the scheduler is active, which it always is; shrinking
                               # win10.sched_phase_list above is what actually matters). Still
                               # set directly in case a future GUI change ever makes the legacy
                               # field reachable again.
    check(config10.validate() == [], f"Time-Adaptive sweep config should validate clean: {config10.validate()}")
    check(config10.time_adaptive, "config should record time_adaptive=True")

    results10 = run_sweep(config10)
    check(len(results10) == 3, f"expected 1 baseline + 2 swept Time-Adaptive runs, got {len(results10)}")
    for label, result in results10:
        check(result["status"] == "done", f"[Time-Adaptive sweep] run {label!r} should finish 'done', got {result}")

    # ── v73: Output transform + Activation sweep, real training ───────
    # Two independent, cheap one-value sweep rows combined into a single
    # OAT sweep (1 baseline + 1 + 1 = 3 runs) rather than two separate
    # tests, purely to bound the number of new real training runs this
    # round adds -- each still exercises a genuinely different new
    # registry category end-to-end.
    win11 = MainWindow()
    _app.processEvents()
    win11.quick_examples_combo.setCurrentText("1D Heat")
    win11.output_transform_cb.setChecked(True)
    _app.processEvents()
    for ph in win11.sched_phase_list:
        ph["iters"].setValue(10)
    win11._refresh_sweep_param_choices()  # see win9's comment above -- same stale-combo gotcha
    row11a = win11.sweep_row_list[0]
    _idx11a = row11a["param_combo"].findData("ot_scale_0")
    check(_idx11a >= 0, "ot_scale_0 should be selectable once Output transform is enabled")
    row11a["param_combo"].setCurrentIndex(_idx11a)
    row11a["mode_combo"].setCurrentIndex(row11a["mode_combo"].findData("list"))
    row11a["list_edit"].setText("2.0")
    win11._add_sweep_row()  # _add_sweep_row() itself refreshes choices for the new row
    row11b = win11.sweep_row_list[1]
    _idx11b = row11b["param_combo"].findData("activation")
    check(_idx11b >= 0, "activation should always be selectable")
    row11b["param_combo"].setCurrentIndex(_idx11b)
    row11b["list_edit"].setText("relu")
    win11.sweep_enable_cb.setChecked(True)
    config11 = win11._build_config()
    check(config11.validate() == [], f"Output-transform/Activation sweep config should validate clean: {config11.validate()}")

    results11 = run_sweep(config11)
    check(len(results11) == 3, f"expected 1 baseline + 1 output-transform run + 1 activation run, got {len(results11)}")
    for label, result in results11:
        check(result["status"] == "done", f"[OT/Activation sweep] run {label!r} should finish 'done', got {result}")
        check(result["final_loss"] is not None, f"[OT/Activation sweep] run {label!r} should report a parsed final_loss, got {result}")

    # ── Stop hard-kills the CURRENTLY RUNNING subprocess ──────────────
    # Earlier behavior (v71 and before) only set a flag SweepThread.run()
    # checked BETWEEN runs -- the in-flight run kept training to
    # completion first, which looked like clicking Stop wasn't doing
    # anything (the user's own report: iterations kept printing for a
    # while after clicking it). Give this run a huge iteration count so
    # there is plenty of real remaining work to interrupt -- if Stop
    # genuinely hard-kills it, the thread finishes in a few seconds
    # regardless; if it silently went back to the old graceful-finish
    # behavior, it would still be training 200000 iterations and this
    # check would time out.
    from pinnstudio.ui.main_window import SweepThread
    import time

    win8 = MainWindow()
    _app.processEvents()
    win8.quick_examples_combo.setCurrentText("1D Heat")
    _app.processEvents()
    for ph in win8.sched_phase_list:
        ph["iters"].setValue(200000)
    row8 = win8.sweep_row_list[0]
    row8["param_combo"].setCurrentIndex(row8["param_combo"].findData("hidden_layers"))
    row8["mode_combo"].setCurrentIndex(row8["mode_combo"].findData("list"))
    row8["list_edit"].setText("2")
    win8.sweep_enable_cb.setChecked(True)
    config8 = win8._build_config()
    check(config8.validate() == [], f"hard-stop sweep config should validate clean: {config8.validate()}")

    sweep_thread = SweepThread(config8)
    sweep_thread.start()
    _deadline = time.time() + 15
    while time.time() < _deadline and sweep_thread.process is None:
        time.sleep(0.1)
    check(sweep_thread.process is not None, "the sweep's training subprocess should have started within 15s")
    if sweep_thread.process is not None:
        time.sleep(1.5)  # let it actually make some real training progress first
        _proc = sweep_thread.process
        _t0 = time.time()
        sweep_thread.stop()
        check(time.time() - _t0 < 10, ".stop() should return within its own bounded terminate/kill window")
        check(_proc.poll() is not None, "the training subprocess should have actually exited after .stop()")
        finished_in_time = sweep_thread.wait(10000)
        check(finished_in_time,
              "SweepThread should finish within 10s of being stopped, not keep running a "
              "200000-iteration training job to completion in the background")
    sweep_thread.wait(5000)

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL SWEEP EXECUTION TESTS PASSED")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
