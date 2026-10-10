#!/usr/bin/env python3
"""
RAR (Residual-based Adaptive Refinement) per-round diagnostics
(user-requested): before this round, RAR only ever showed the final
model's own solution/loss plot after ALL rounds finished -- nothing
showed how the model (or the points RAR chose to add) evolved round to
round, so a sharp residual region or a growing cluster of added points
near an edge/front was invisible until training was already over.

Each RAR round saves, under this run's own
solution_results/rar_rounds/round_NN/ subfolder:

 - solution_plot.png: a snapshot of whatever field is configured (raw
   output index or a derivative-aware Custom expression -- the same
   _extract_plot_field every other plot in the generated script uses).
   For 1D this is the x-t plane (axis orientation follows the same
   Plot Settings "Swap axes" choice -- config.plot_swap_xt -- the main
   Surface plot type and every Error Analysis comparison plot already
   use: t on the x-axis, x on the y-axis, by default). For 2D/3D this
   file is now the SOLUTION *AND* the error comparison merged into one:
   when at least one Error Analysis reference file is configured, it is
   a 3-panel PINN | Reference | |Error| figure (same color scheme the
   full Inline Error Analysis uses -- PINN and Reference share one
   color scale, |Error| gets its own, "inferno"), evaluated against the
   single reference whose time is closest to t_max (not the full
   multi-file/multi-time Inline Error Analysis sweep that still runs
   once, at the very end, unchanged); the accompanying
   "L2 rel error = ..., MSE = ..." log line is still printed either
   way. With no reference configured, it's just the PINN prediction
   panel, same as before. 1D keeps its own separate two-file structure
   (solution_plot.png + error_compare.png) -- no merge there.
 - collocation_points.png: a 3-color scatter -- original domain points
   (gray), points added in EARLIER rounds (orange), points added THIS
   round (red). 1D: x-t plane (same axis-orientation fix as above). 3D:
   a 3D (x,y,z) scatter with time folded into per-category alpha (4 real
   coordinate dims can't all be plot axes at once) -- unchanged this
   round. 2D: now TWO panels side by side -- the existing 3D (x,y,t)
   view, plus a new plain 2D (x,y) view collapsing time, so the spatial
   concentration of added points is readable without rotating a 3D plot.

Also, from the previous round: "Points from:" (rar_output_combo in the
RAR settings panel) -- for a multi-equation PDE, model.predict(x,
operator=pde) returns a list of residuals, one per equation (in
PDE-editor entry order, which lines up with output order for every
current template); this lets a sweep's point-selection be restricted to
just one equation's residual instead of the default combined (summed)
one. config.rar_output_selector: -1 is "All outputs (combined)" (the
original, unchanged behavior).

Checks:
 - rar_output_combo exists, seeded with "All outputs (combined)" +
   "Output 1 (u)" for the single-output default, and refreshes correctly
   (clears/repopulates, keeps the "All outputs" entry first) when
   num_outputs changes.
 - _build_config()/_apply_config() round-trip rar_output_selector
   correctly (index 0 <-> -1, index k<->k-1), including restoring a
   selection after num_outputs changes.
 - A REAL, tiny 2-round RAR run (1D, no reference file) actually writes
   round_01/ and round_02/, each with solution_plot.png and
   collocation_points.png (non-empty).
 - A REAL, tiny 1-round RAR run WITH a reference file configured (1D)
   also writes error_compare.png (its own separate file, 1D only) and
   its L2 relative error matches (within floating point tolerance) what
   the full, separate Inline Error Analysis section computes for the
   exact same model/reference at the end of the very same run --
   proving the lightweight per-round compare and the full one agree,
   not just that a file got written.
 - A REAL, tiny 2-round RAR run (2D, WITHOUT a reference file): each
   round's solution_plot.png is the single-panel PINN prediction, and
   round_02's collocation_points.png shows more accumulated points than
   round_01's would (2 colors -- original + this round, no "earlier
   rounds" yet after round 1, 3 colors by round 2).
 - A REAL, tiny 1-round RAR run (2D, WITH a reference file): the merged
   solution_plot.png exists (3-panel comparison) and error_compare.png
   does NOT exist as a separate file (merged away) -- and the round's
   L2 log line is still printed and parses to a finite number.
 - A REAL, tiny RAR run on a 2-output, 2D problem with
   rar_output_selector=1 (equation index 1, "Output 2") runs cleanly
   end to end (model.predict(..., operator=pde) returns a list there,
   exercising the per-equation residual-selection branch, not the
   single-array branch the 1D single-output case above exercises).
 - A REAL, tiny 1-round RAR run (3D, WITH a reference file): the merged
   solution_plot.png exists (3-panel scatter comparison) and
   error_compare.png does NOT exist as a separate file, same as the 2D
   case -- exercising the 3D scatter-comparison branch.

RAR is also now available for STEADY-STATE problems (e.g. a Poisson
equation) -- previously the GUI's whole "Adaptive Training" panel was
hidden whenever Steady-state was on, even though only Time Adaptive
Training (not RAR) actually needs a time axis. Steady 1D/2D/3D each get
their own no-time-axis round diagnostics (a plain u(x) line + 3-row
scatter for 1D; the usual 2D/3D panels evaluated once, with no time
slicing, for 2D/3D). Fixing this also surfaced two related pre-existing
bugs in Error Analysis for steady problems, fixed alongside it: a steady
1D reference file (x, u -- 2 columns) crashed with an IndexError because
the loader unconditionally assumed the time-dependent 3-column (x, t, u)
layout, and a steady 3D reference file on the default Cuboid geometry
crashed the Surface Comparison with a matmul shape mismatch because its
box-face prediction grid always appended a placeholder time column.
 - A REAL, tiny 2-round RAR run (1D STEADY, no reference): round_01 and
   round_02 each write a non-empty solution_plot.png and
   collocation_points.png.
 - A REAL, tiny 1-round RAR run (1D STEADY) WITH a reference file (x, u):
   error_compare.png is written, Error Analysis loads the file without
   an IndexError, and the (inapplicable) x-t surface comparison is
   skipped gracefully instead of crashing.
 - A REAL, tiny 1-round RAR run (2D STEADY) WITH a reference file (x, y,
   u): the merged solution_plot.png is written, and the Inline Error
   Analysis surface_comparison.png is also written.
 - A REAL, tiny 1-round RAR run (3D STEADY, default Cuboid geometry)
   WITH a reference file (x, y, z, u) and surface comparison enabled:
   both round_01/solution_plot.png and the Inline Error Analysis
   surface_comparison.png are written without the matmul crash.

RAR's own candidate-point sampling had a separate, independent bug
(found after shipping steady-state RAR above, on a real 2D Poisson
L-Shape run): candidates were built by uniformly sampling each axis
over the domain's bounding box (np.random.uniform(x_min, x_max, ...),
same for y/z/t) rather than drawing from the actual geometry. For a
Rectangle/Cuboid/Interval domain the bounding box IS the domain, so
this was invisible, but for any non-box geometry (L-Shape/Polygon,
Disk, Triangle, an Annulus or other CSG shape with a hole, ...) the
bounding box is strictly larger than the real domain -- so candidate
points, and then (since RAR just keeps whichever candidates have the
largest predicted PDE residual) sometimes the actual added points too,
could land outside the geometry entirely, e.g. inside the L-Shape's
own missing quadrant. Fixed by sampling candidates from
geomtime.random_points(...) instead (geomtime is geom itself when
steady, or geom wrapped in GeometryXTime otherwise) -- exactly what
dde.data.PDE/TimePDE already use for ordinary training collocation
points elsewhere in the very same generated script, which performs
real rejection sampling against the geometry for every non-box shape
(see DeepXDE's own CSGDifference/Polygon/Disk random_points()).
 - A REAL, tiny 1-round RAR run on the 2D Poisson L-Shape template's own
   Polygon geometry (steady-state) exits cleanly, and the generated
   script's RAR candidate-sampling line is geomtime.random_points(...)
   rather than independent per-axis np.random.uniform(...) calls.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_rar_round_diagnostics.py
"""
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
from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import generate_script


def _run_script(script, tmpdir, tag, timeout_s=150):
    sp = os.path.join(tmpdir, f"{tag}_script.py")
    with open(sp, "w") as f:
        f.write(script)
    return subprocess.run(
        [sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=timeout_s,
    )


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── UI: rar_output_combo exists, seeded, and refreshes correctly ──
    win = MainWindow()
    _app.processEvents()
    check(hasattr(win, "rar_output_combo"), "MainWindow should have a rar_output_combo")
    items0 = [win.rar_output_combo.itemText(i) for i in range(win.rar_output_combo.count())]
    check(items0 == ["All outputs (combined)", "Output 1 (u)"],
          f"single-output default should seed ['All outputs (combined)', 'Output 1 (u)'], got {items0}")

    win.num_outputs_spin.setValue(3)
    _app.processEvents()
    items3 = [win.rar_output_combo.itemText(i) for i in range(win.rar_output_combo.count())]
    check(items3[0] == "All outputs (combined)" and len(items3) == 4,
          f"with 3 outputs, expected 4 items (combined + 3 outputs), got {items3}")

    win.rar_output_combo.setCurrentIndex(2)  # "Output 2"
    cfg_built = win._build_config()
    check(cfg_built.rar_output_selector == 1,
          f"combo index 2 ('Output 2') should map to rar_output_selector=1, got {cfg_built.rar_output_selector}")

    win.num_outputs_spin.setValue(1)
    _app.processEvents()
    win.rar_output_combo.setCurrentIndex(0)
    cfg_default = win._build_config()
    check(cfg_default.rar_output_selector == -1,
          f"'All outputs (combined)' should map to rar_output_selector=-1, got {cfg_default.rar_output_selector}")

    # Round-trip through _apply_config: num_outputs back to 2, selector=1 ("Output 2").
    win.num_outputs_spin.setValue(2)
    _app.processEvents()
    restore_cfg = win._build_config()
    restore_cfg.rar_output_selector = 1
    win.rar_output_combo.setCurrentIndex(0)  # perturb away first
    win._apply_config(restore_cfg)
    _app.processEvents()
    check(win.rar_output_combo.currentIndex() == 2,
          f"_apply_config with rar_output_selector=1 should select combo index 2, got {win.rar_output_combo.currentIndex()!r} "
          f"({win.rar_output_combo.currentText()!r})")

    try:
        import deepxde as _dde_probe  # noqa: F401
        import torch as _torch_probe  # noqa: F401
    except ImportError:
        print("Partial run (UI-only): deepxde/torch not installed in this environment, "
              "skipping the real-training checks below.")
        if failures:
            print("FAILURES:")
            for f in failures:
                print(f"  - {f}")
        return failures

    def _extract_l2(stdout, marker):
        for line in stdout.splitlines():
            if marker in line and "L2 rel error" in line:
                try:
                    return float(line.split("L2 rel error = ")[1].split(",")[0])
                except Exception:
                    return None
        return None

    # ── Real, tiny 2-round RAR run (1D, no reference) ─────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = PINNConfig()
        cfg.problem_dim = "1D"; cfg.steady_state = False
        cfg.adapt_method = "RAR"
        cfg.rar_cycles = 2; cfg.rar_candidates = 200; cfg.rar_add_points = 10
        cfg.rar_adam_iters = 5; cfg.rar_lbfgs_iters = 0
        cfg.iterations = 5; cfg.optimizer_scheduler = False; cfg.iterations2 = 0
        cfg.save_dir = tmpdir
        cfg.pde_expression = "du_t - 0.01*du_xx"; cfg.pde_expressions = "du_t - 0.01*du_xx"
        cfg.ic_expression = "sin(pi*x)"; cfg.ic_expressions = "sin(pi*x)"

        proc = _run_script(generate_script(cfg), tmpdir, "rar_2round")
        check(proc.returncode == 0, f"2-round RAR script should exit cleanly, got {proc.returncode}: {proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

        rounds_dir = os.path.join(tmpdir, "solution_results", "rar_rounds")
        r1 = os.path.join(rounds_dir, "round_01")
        r2 = os.path.join(rounds_dir, "round_02")
        for rd, label in [(r1, "round_01"), (r2, "round_02")]:
            sol_p = os.path.join(rd, "solution_plot.png")
            coll_p = os.path.join(rd, "collocation_points.png")
            check(os.path.exists(sol_p) and os.path.getsize(sol_p) > 0,
                  f"{label}/solution_plot.png should exist and be non-empty")
            check(os.path.exists(coll_p) and os.path.getsize(coll_p) > 0,
                  f"{label}/collocation_points.png should exist and be non-empty")

        final_sol = os.path.join(tmpdir, "solution_results", "solution_plot.png")
        check(os.path.exists(final_sol), "the final run-level solution_plot.png should still exist (unchanged)")

    # ── Real, tiny 1-round RAR run (1D) WITH a reference file ─────────
    with tempfile.TemporaryDirectory() as tmpdir2:
        import numpy as np
        ref_path = os.path.join(tmpdir2, "ref_1d.txt")
        xr = np.linspace(0, 1, 20)
        np.savetxt(ref_path, np.column_stack([xr, np.full(20, 1.0), np.sin(np.pi * xr)]))

        cfg2 = PINNConfig()
        cfg2.problem_dim = "1D"; cfg2.steady_state = False
        cfg2.adapt_method = "RAR"
        cfg2.rar_cycles = 1; cfg2.rar_candidates = 200; cfg2.rar_add_points = 10
        cfg2.rar_adam_iters = 5; cfg2.rar_lbfgs_iters = 0
        cfg2.iterations = 5; cfg2.optimizer_scheduler = False; cfg2.iterations2 = 0
        cfg2.save_dir = tmpdir2
        cfg2.pde_expression = "du_t - 0.01*du_xx"; cfg2.pde_expressions = "du_t - 0.01*du_xx"
        cfg2.ic_expression = "sin(pi*x)"; cfg2.ic_expressions = "sin(pi*x)"
        cfg2.ea_files = [(1.0, ref_path, None)]
        cfg2.ea_do_line = True

        proc2 = _run_script(generate_script(cfg2), tmpdir2, "rar_ea")
        check(proc2.returncode == 0, f"RAR+EA script should exit cleanly, got {proc2.returncode}: {proc2.stdout[-2000:]}\n{proc2.stderr[-2000:]}")

        err_png = os.path.join(tmpdir2, "solution_results", "rar_rounds", "round_01", "error_compare.png")
        check(os.path.exists(err_png) and os.path.getsize(err_png) > 0,
              "1D round_01/error_compare.png should exist and be non-empty when a reference file is configured "
              "(1D keeps its own separate solution+error files, no merge)")

        l2_round = _extract_l2(proc2.stdout, "[RAR round 1]")
        l2_full = None
        for line in proc2.stdout.splitlines():
            if line.strip().startswith("t=1.0000") and "L2=" in line:
                try:
                    l2_full = float(line.split("L2=")[1].split(",")[0])
                except Exception:
                    pass
        check(l2_round is not None, f"round-level L2 should be printed in the log, got stdout tail: {proc2.stdout[-1500:]}")
        check(l2_full is not None, f"the full Inline Error Analysis section's own L2 should also be printed, got stdout tail: {proc2.stdout[-1500:]}")
        if l2_round is not None and l2_full is not None:
            check(abs(l2_round - l2_full) < 1e-3,
                  f"the lightweight per-round L2 ({l2_round}) should match the full Inline Error Analysis L2 "
                  f"({l2_full}) for the same model/reference at the end of the same run")

    # ── Real, tiny 2-round RAR run (2D, no reference) ─────────────────
    with tempfile.TemporaryDirectory() as tmpdir4:
        cfg4 = PINNConfig()
        cfg4.problem_dim = "2D"; cfg4.steady_state = False
        cfg4.layers = [3, 16, 16, 1]
        cfg4.adapt_method = "RAR"
        cfg4.rar_cycles = 2; cfg4.rar_candidates = 150; cfg4.rar_add_points = 10
        cfg4.rar_adam_iters = 5; cfg4.rar_lbfgs_iters = 0
        cfg4.iterations = 5; cfg4.optimizer_scheduler = False; cfg4.iterations2 = 0
        cfg4.save_dir = tmpdir4
        cfg4.pde_expression = "du_t - 0.01*(du_xx+du_yy)"; cfg4.pde_expressions = cfg4.pde_expression
        cfg4.ic_expression = "sin(pi*x)*sin(pi*y)"; cfg4.ic_expressions = cfg4.ic_expression

        proc4 = _run_script(generate_script(cfg4), tmpdir4, "rar_2d_noref")
        check(proc4.returncode == 0, f"2D no-reference RAR script should exit cleanly, got {proc4.returncode}: {proc4.stdout[-2000:]}\n{proc4.stderr[-2000:]}")

        for rn, label in [("round_01", "round_01"), ("round_02", "round_02")]:
            rd4 = os.path.join(tmpdir4, "solution_results", "rar_rounds", rn)
            sol4 = os.path.join(rd4, "solution_plot.png")
            coll4 = os.path.join(rd4, "collocation_points.png")
            err4 = os.path.join(rd4, "error_compare.png")
            check(os.path.exists(sol4) and os.path.getsize(sol4) > 0,
                  f"2D no-ref {label}/solution_plot.png should exist and be non-empty")
            check(os.path.exists(coll4) and os.path.getsize(coll4) > 0,
                  f"2D no-ref {label}/collocation_points.png (2-panel: 3D view + spatial x-y view) should exist and be non-empty")
            check(not os.path.exists(err4),
                  f"2D no-ref {label} should have no separate error_compare.png (nothing to compare against, and 2D merges into solution_plot.png anyway)")

    # ── Real, tiny 1-round RAR run (2D) WITH a reference file ─────────
    with tempfile.TemporaryDirectory() as tmpdir5:
        import numpy as np
        ref2d = os.path.join(tmpdir5, "ref_2d.txt")
        n = 15
        xg, yg = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        xg = xg.ravel(); yg = yg.ravel()
        uu = np.sin(np.pi * xg) * np.sin(np.pi * yg)
        np.savetxt(ref2d, np.column_stack([xg, yg, np.full(len(xg), 1.0), uu]))

        cfg5 = PINNConfig()
        cfg5.problem_dim = "2D"; cfg5.steady_state = False
        cfg5.layers = [3, 16, 16, 1]
        cfg5.adapt_method = "RAR"
        cfg5.rar_cycles = 1; cfg5.rar_candidates = 150; cfg5.rar_add_points = 10
        cfg5.rar_adam_iters = 5; cfg5.rar_lbfgs_iters = 0
        cfg5.iterations = 5; cfg5.optimizer_scheduler = False; cfg5.iterations2 = 0
        cfg5.save_dir = tmpdir5
        cfg5.pde_expression = "du_t - 0.01*(du_xx+du_yy)"; cfg5.pde_expressions = cfg5.pde_expression
        cfg5.ic_expression = "sin(pi*x)*sin(pi*y)"; cfg5.ic_expressions = cfg5.ic_expression
        cfg5.ea_files = [(1.0, ref2d, None)]

        proc5 = _run_script(generate_script(cfg5), tmpdir5, "rar_2d_ref")
        check(proc5.returncode == 0, f"2D+reference RAR script should exit cleanly, got {proc5.returncode}: {proc5.stdout[-2000:]}\n{proc5.stderr[-2000:]}")

        rd5 = os.path.join(tmpdir5, "solution_results", "rar_rounds", "round_01")
        sol5 = os.path.join(rd5, "solution_plot.png")
        err5 = os.path.join(rd5, "error_compare.png")
        check(os.path.exists(sol5) and os.path.getsize(sol5) > 0,
              "2D+ref round_01/solution_plot.png (merged PINN|Reference|Error) should exist and be non-empty")
        check(not os.path.exists(err5),
              "2D+ref round_01 should have NO separate error_compare.png -- it's merged into solution_plot.png")
        l2_2d = _extract_l2(proc5.stdout, "[RAR round 1]")
        check(l2_2d is not None, f"2D+ref round-level L2 should be printed in the log, got stdout tail: {proc5.stdout[-1500:]}")

    # ── Real, tiny RAR run on a 2-output 2D problem, rar_output_selector=1 ──
    with tempfile.TemporaryDirectory() as tmpdir3:
        cfg3 = PINNConfig()
        cfg3.problem_dim = "2D"; cfg3.steady_state = False
        cfg3.num_outputs = 2; cfg3.output_names = "u,v"
        cfg3.layers = [3, 16, 16, 2]
        cfg3.adapt_method = "RAR"
        cfg3.rar_cycles = 1; cfg3.rar_candidates = 150; cfg3.rar_add_points = 8
        cfg3.rar_adam_iters = 5; cfg3.rar_lbfgs_iters = 0
        cfg3.rar_output_selector = 1
        cfg3.iterations = 5; cfg3.optimizer_scheduler = False; cfg3.iterations2 = 0
        cfg3.save_dir = tmpdir3
        cfg3.pde_expression = "du_t - 0.01*du_xx"
        cfg3.pde_expressions = "du_t - 0.01*(du_xx+du_yy)|dv_t - 0.01*(dv_xx+dv_yy)"
        cfg3.ic_expression = "sin(pi*x)"
        cfg3.ic_expressions = "sin(pi*x)*sin(pi*y)|sin(pi*x)*sin(pi*y)"

        proc3 = _run_script(generate_script(cfg3), tmpdir3, "rar_multiout", timeout_s=150)
        check(proc3.returncode == 0,
              f"2-output RAR run with rar_output_selector=1 should exit cleanly, got {proc3.returncode}: "
              f"{proc3.stdout[-2000:]}\n{proc3.stderr[-2000:]}")
        mo_coll = os.path.join(tmpdir3, "solution_results", "rar_rounds", "round_01", "collocation_points.png")
        check(os.path.exists(mo_coll) and os.path.getsize(mo_coll) > 0,
              "2-output run's round_01/collocation_points.png (2-panel: 3D x-y-t scatter + spatial x-y view) should exist and be non-empty")

    # ── Real, tiny 1-round RAR run (3D) WITH a reference file ─────────
    with tempfile.TemporaryDirectory() as tmpdir6:
        import numpy as np
        ref3d = os.path.join(tmpdir6, "ref_3d.txt")
        n3 = 8
        xg3, yg3, zg3 = np.meshgrid(np.linspace(0, 1, n3), np.linspace(0, 1, n3), np.linspace(0, 1, n3))
        xg3 = xg3.ravel(); yg3 = yg3.ravel(); zg3 = zg3.ravel()
        uu3 = np.sin(np.pi * xg3) * np.sin(np.pi * yg3) * np.sin(np.pi * zg3)
        np.savetxt(ref3d, np.column_stack([xg3, yg3, zg3, np.full(len(xg3), 1.0), uu3]))

        cfg6 = PINNConfig()
        cfg6.problem_dim = "3D"; cfg6.steady_state = False
        cfg6.layers = [4, 16, 16, 1]
        cfg6.adapt_method = "RAR"
        cfg6.rar_cycles = 1; cfg6.rar_candidates = 120; cfg6.rar_add_points = 8
        cfg6.rar_adam_iters = 5; cfg6.rar_lbfgs_iters = 0
        cfg6.iterations = 5; cfg6.optimizer_scheduler = False; cfg6.iterations2 = 0
        cfg6.save_dir = tmpdir6
        cfg6.pde_expression = "du_t - 0.01*(du_xx+du_yy+du_zz)"; cfg6.pde_expressions = cfg6.pde_expression
        cfg6.ic_expression = "sin(pi*x)*sin(pi*y)*sin(pi*z)"; cfg6.ic_expressions = cfg6.ic_expression
        cfg6.ea_files = [(1.0, ref3d, None)]

        proc6 = _run_script(generate_script(cfg6), tmpdir6, "rar_3d_ref", timeout_s=150)
        check(proc6.returncode == 0, f"3D+reference RAR script should exit cleanly, got {proc6.returncode}: {proc6.stdout[-2000:]}\n{proc6.stderr[-2000:]}")

        rd6 = os.path.join(tmpdir6, "solution_results", "rar_rounds", "round_01")
        sol6 = os.path.join(rd6, "solution_plot.png")
        err6 = os.path.join(rd6, "error_compare.png")
        check(os.path.exists(sol6) and os.path.getsize(sol6) > 0,
              "3D+ref round_01/solution_plot.png (merged PINN|Reference|Error 3D scatter) should exist and be non-empty")
        check(not os.path.exists(err6),
              "3D+ref round_01 should have NO separate error_compare.png -- it's merged into solution_plot.png")
        l2_3d = _extract_l2(proc6.stdout, "[RAR round 1]")
        check(l2_3d is not None, f"3D+ref round-level L2 should be printed in the log, got stdout tail: {proc6.stdout[-1500:]}")

    # ── Real, tiny 2-round RAR run (1D STEADY, e.g. a Poisson equation,
    # no reference) -- RAR previously only ever ran for time-dependent
    # problems; the GUI didn't even show the Adaptive Training panel for
    # a steady one. Steady-1D's own round diagnostics are a plain u(x)
    # line (solution_plot.png) and a 3-row (original/earlier/this-round)
    # scatter (collocation_points.png) -- no x-t axes, since there's no
    # time axis at all. ──
    with tempfile.TemporaryDirectory() as tmpdir7:
        cfg7 = PINNConfig()
        cfg7.problem_dim = "1D"; cfg7.steady_state = True
        cfg7.layers = [1, 20, 20, 1]  # steady 1D: x only, no t input column
        cfg7.adapt_method = "RAR"
        cfg7.rar_cycles = 2; cfg7.rar_candidates = 200; cfg7.rar_add_points = 10
        cfg7.rar_adam_iters = 5; cfg7.rar_lbfgs_iters = 0
        cfg7.iterations = 5; cfg7.optimizer_scheduler = False; cfg7.iterations2 = 0
        cfg7.save_dir = tmpdir7
        cfg7.pde_expression = "du_xx + 1"; cfg7.pde_expressions = "du_xx + 1"

        proc7 = _run_script(generate_script(cfg7), tmpdir7, "rar_1d_steady")
        check(proc7.returncode == 0,
              f"1D steady 2-round RAR script should exit cleanly, got {proc7.returncode}: {proc7.stdout[-2000:]}\n{proc7.stderr[-2000:]}")

        for rn, label in [("round_01", "round_01"), ("round_02", "round_02")]:
            rd7 = os.path.join(tmpdir7, "solution_results", "rar_rounds", rn)
            sol7 = os.path.join(rd7, "solution_plot.png")
            coll7 = os.path.join(rd7, "collocation_points.png")
            check(os.path.exists(sol7) and os.path.getsize(sol7) > 0,
                  f"1D steady {label}/solution_plot.png should exist and be non-empty")
            check(os.path.exists(coll7) and os.path.getsize(coll7) > 0,
                  f"1D steady {label}/collocation_points.png should exist and be non-empty")

    # ── Real, tiny 1-round RAR run (1D STEADY) WITH a reference file --
    # steady reference files have no time column at all (x, u -- 2
    # columns, one fewer than the time-dependent x, t, u layout), and
    # this is the format the live-Solve Error Analysis loader needs to
    # read correctly to avoid an IndexError. ──
    with tempfile.TemporaryDirectory() as tmpdir8:
        import numpy as np
        ref1ds = os.path.join(tmpdir8, "ref_1d_steady.txt")
        xr8 = np.linspace(0, 1, 20)
        np.savetxt(ref1ds, np.column_stack([xr8, np.sin(np.pi * xr8)]))

        cfg8 = PINNConfig()
        cfg8.problem_dim = "1D"; cfg8.steady_state = True
        cfg8.layers = [1, 20, 20, 1]  # steady 1D: x only, no t input column
        cfg8.adapt_method = "RAR"
        cfg8.rar_cycles = 1; cfg8.rar_candidates = 200; cfg8.rar_add_points = 10
        cfg8.rar_adam_iters = 5; cfg8.rar_lbfgs_iters = 0
        cfg8.iterations = 5; cfg8.optimizer_scheduler = False; cfg8.iterations2 = 0
        cfg8.save_dir = tmpdir8
        cfg8.pde_expression = "du_xx + 1"; cfg8.pde_expressions = "du_xx + 1"
        cfg8.ea_files = [(0.0, ref1ds, None)]
        cfg8.ea_do_line = True; cfg8.ea_do_surface = True

        proc8 = _run_script(generate_script(cfg8), tmpdir8, "rar_1d_steady_ref")
        check(proc8.returncode == 0,
              f"1D steady+reference RAR script should exit cleanly, got {proc8.returncode}: {proc8.stdout[-2000:]}\n{proc8.stderr[-2000:]}")

        err8 = os.path.join(tmpdir8, "solution_results", "rar_rounds", "round_01", "error_compare.png")
        check(os.path.exists(err8) and os.path.getsize(err8) > 0,
              "1D steady round_01/error_compare.png should exist and be non-empty when a reference file is configured")
        check("Loaded ground truth (steady-state)" in proc8.stdout,
              f"Error Analysis should load the steady 2-column (x, u) reference file without an IndexError, "
              f"got stdout tail: {proc8.stdout[-1500:]}\nstderr tail: {proc8.stderr[-1500:]}")
        check("Skipping surface comparison" in proc8.stdout and "IndexError" not in proc8.stderr,
              f"1D steady Error Analysis should skip the (inapplicable, no time axis) x-t surface comparison "
              f"gracefully rather than crashing, got stdout tail: {proc8.stdout[-1500:]}\nstderr tail: {proc8.stderr[-1500:]}")
        l2_round8 = _extract_l2(proc8.stdout, "[RAR round 1]")
        check(l2_round8 is not None, f"1D steady round-level L2 should be printed, got stdout tail: {proc8.stdout[-1500:]}")

    # ── Real, tiny 1-round RAR run (2D STEADY, e.g. 2D Poisson) WITH a
    # reference file (x, y, u -- 3 columns, no time column) ──
    with tempfile.TemporaryDirectory() as tmpdir9:
        import numpy as np
        ref2ds = os.path.join(tmpdir9, "ref_2d_steady.txt")
        n9 = 12
        xg9, yg9 = np.meshgrid(np.linspace(0, 1, n9), np.linspace(0, 1, n9))
        xg9 = xg9.ravel(); yg9 = yg9.ravel()
        uu9 = np.sin(np.pi * xg9) * np.sin(np.pi * yg9)
        np.savetxt(ref2ds, np.column_stack([xg9, yg9, uu9]))

        cfg9 = PINNConfig()
        cfg9.problem_dim = "2D"; cfg9.steady_state = True
        cfg9.layers = [2, 16, 16, 1]
        cfg9.adapt_method = "RAR"
        cfg9.rar_cycles = 1; cfg9.rar_candidates = 150; cfg9.rar_add_points = 10
        cfg9.rar_adam_iters = 5; cfg9.rar_lbfgs_iters = 0
        cfg9.iterations = 5; cfg9.optimizer_scheduler = False; cfg9.iterations2 = 0
        cfg9.save_dir = tmpdir9
        cfg9.pde_expression = "du_xx + du_yy + 1"; cfg9.pde_expressions = cfg9.pde_expression
        cfg9.ea_files = [(0.0, ref2ds, None)]
        cfg9.ea_do_surface = True

        proc9 = _run_script(generate_script(cfg9), tmpdir9, "rar_2d_steady_ref")
        check(proc9.returncode == 0,
              f"2D steady+reference RAR script should exit cleanly, got {proc9.returncode}: {proc9.stdout[-2000:]}\n{proc9.stderr[-2000:]}")

        rd9 = os.path.join(tmpdir9, "solution_results", "rar_rounds", "round_01")
        sol9 = os.path.join(rd9, "solution_plot.png")
        check(os.path.exists(sol9) and os.path.getsize(sol9) > 0,
              "2D steady+ref round_01/solution_plot.png (merged PINN|Reference|Error) should exist and be non-empty")
        l2_2ds = _extract_l2(proc9.stdout, "[RAR round 1]")
        check(l2_2ds is not None, f"2D steady round-level L2 should be printed, got stdout tail: {proc9.stdout[-1500:]}")
        surf9 = os.path.join(tmpdir9, "error_analysis", "surface_comparison.png")
        check(os.path.exists(surf9) and os.path.getsize(surf9) > 0,
              "2D steady Inline Error Analysis surface_comparison.png should exist and be non-empty")

    # ── Real, tiny 1-round RAR run (3D STEADY, default Cuboid geometry)
    # WITH a reference file AND surface comparison enabled -- this is the
    # exact combination that used to crash the live-Solve Error Analysis
    # ("mat1 and mat2 shapes cannot be multiplied"): the box-face surface
    # comparison for a Cuboid always appended a placeholder time column
    # to its prediction grid, even though a steady-state network has no
    # time input at all. ──
    with tempfile.TemporaryDirectory() as tmpdir10:
        import numpy as np
        ref3ds = os.path.join(tmpdir10, "ref_3d_steady.txt")
        n10 = 6
        xg10, yg10, zg10 = np.meshgrid(np.linspace(0, 1, n10), np.linspace(0, 1, n10), np.linspace(0, 1, n10))
        xg10 = xg10.ravel(); yg10 = yg10.ravel(); zg10 = zg10.ravel()
        uu10 = np.sin(np.pi * xg10) * np.sin(np.pi * yg10) * np.sin(np.pi * zg10)
        np.savetxt(ref3ds, np.column_stack([xg10, yg10, zg10, uu10]))

        cfg10 = PINNConfig()
        cfg10.problem_dim = "3D"; cfg10.steady_state = True
        cfg10.layers = [3, 16, 16, 1]
        cfg10.adapt_method = "RAR"
        cfg10.rar_cycles = 1; cfg10.rar_candidates = 120; cfg10.rar_add_points = 8
        cfg10.rar_adam_iters = 5; cfg10.rar_lbfgs_iters = 0
        cfg10.iterations = 5; cfg10.optimizer_scheduler = False; cfg10.iterations2 = 0
        cfg10.save_dir = tmpdir10
        cfg10.pde_expression = "du_xx + du_yy + du_zz + 1"; cfg10.pde_expressions = cfg10.pde_expression
        cfg10.ea_files = [(0.0, ref3ds, None)]
        cfg10.ea_do_surface = True

        proc10 = _run_script(generate_script(cfg10), tmpdir10, "rar_3d_steady_ref", timeout_s=150)
        check(proc10.returncode == 0,
              f"3D steady+reference (Cuboid, surface comparison on) RAR script should exit cleanly, "
              f"got {proc10.returncode}: {proc10.stdout[-2000:]}\n{proc10.stderr[-2000:]}")

        rd10 = os.path.join(tmpdir10, "solution_results", "rar_rounds", "round_01")
        sol10 = os.path.join(rd10, "solution_plot.png")
        check(os.path.exists(sol10) and os.path.getsize(sol10) > 0,
              "3D steady+ref round_01/solution_plot.png (merged PINN|Reference|Error 3D scatter) should exist and be non-empty")
        surf10 = os.path.join(tmpdir10, "error_analysis", "surface_comparison.png")
        check(os.path.exists(surf10) and os.path.getsize(surf10) > 0,
              "3D steady Cuboid Inline Error Analysis surface_comparison.png should exist and be non-empty "
              "(previously crashed with a matmul shape mismatch)")

    # ── Real, tiny 1-round RAR run on a non-rectangular geometry
    # (2D Poisson L-Shape's own Polygon, steady-state) -- confirms RAR's
    # candidate sampling now respects the real geometry instead of its
    # bounding box, so no candidate (and so no added point) can land in
    # the L-Shape's own missing quadrant. ──
    with tempfile.TemporaryDirectory() as tmpdir11:
        cfg11 = PINNConfig()
        cfg11.problem_dim = "2D"; cfg11.steady_state = True
        cfg11.num_outputs = 1; cfg11.output_names = "u"
        cfg11.geometry_type = "Polygon"
        cfg11.geom_polygon_vertices = "0,0;1,0;1,-1;-1,-1;-1,1;0,1"
        cfg11.x_min, cfg11.x_max = -1.0, 1.0
        cfg11.y_min, cfg11.y_max = -1.0, 1.0
        cfg11.num_domain = 300; cfg11.num_boundary = 60; cfg11.num_test = 300
        cfg11.layers = [2, 20, 20, 1]
        import json as _json11
        cfg11.custom_bc_json = _json11.dumps([
            {"type": "dirichlet", "location": "True", "value": "0", "component": 0}
        ])
        cfg11.adapt_method = "RAR"
        cfg11.rar_cycles = 1; cfg11.rar_candidates = 2000; cfg11.rar_add_points = 100
        cfg11.rar_adam_iters = 10; cfg11.rar_lbfgs_iters = 0
        cfg11.iterations = 10; cfg11.optimizer_scheduler = False; cfg11.iterations2 = 0
        cfg11.save_dir = tmpdir11
        cfg11.pde_expression = "-du_xx - du_yy - 1"; cfg11.pde_expressions = cfg11.pde_expression

        script11 = generate_script(cfg11)
        rar_block11 = script11.split("Starting RAR Adaptive Refinement")[1] if "Starting RAR Adaptive Refinement" in script11 else ""
        check("geomtime.random_points(" in rar_block11,
              "RAR candidate sampling should call geomtime.random_points(...) so it respects the real geometry")
        check("np.random.uniform" not in rar_block11,
              "RAR candidate sampling should no longer build candidates with independent per-axis "
              "np.random.uniform(...) over the domain's bounding box -- that let candidates (and "
              "potentially added points) land outside a non-rectangular geometry like the L-Shape's "
              "own missing quadrant")

        proc11 = _run_script(script11, tmpdir11, "rar_2d_lshape_polygon", timeout_s=150)
        check(proc11.returncode == 0,
              f"2D Poisson L-Shape (Polygon, steady) RAR script should exit cleanly, got {proc11.returncode}: "
              f"{proc11.stdout[-2000:]}\n{proc11.stderr[-2000:]}")

        rd11 = os.path.join(tmpdir11, "solution_results", "rar_rounds", "round_01")
        coll11 = os.path.join(rd11, "collocation_points.png")
        check(os.path.exists(coll11) and os.path.getsize(coll11) > 0,
              "L-Shape RAR round_01/collocation_points.png should exist and be non-empty")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("OK: RAR per-round diagnostics (solution snapshot, collocation-points scatter, "
              "lightweight error compare -- including the merged 2D/3D solution+error figure and "
              "the 2D dual-panel collocation plot), the 'Points from:' output selector, RAR for "
              "steady-state problems (1D/2D/3D, with and without a reference file, including the "
              "3D Cuboid surface-comparison fix), and RAR candidate sampling on a non-rectangular "
              "geometry (2D Poisson L-Shape's own Polygon) staying inside the real domain -- "
              "not just its bounding box -- all work correctly.")
    return failures


def test_rar_round_diagnostics():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
