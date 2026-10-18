#!/usr/bin/env python3
"""
Round 41: codebase cleanup, phase 4 (audit §2d).

generate_clean_script()'s Error Analysis section used to predict and
error-analyze exactly ONE field (whichever the main plot shows), silently
DROPPING any ea_files entry whose own output_selector didn't match that one
field -- confirmed by reading the old filtering code (`_ea_sel_matches`),
which kept a file only if its selector was None (implicit default), or an
int matching plot_output_idx, or a [expr, label] pair matching the main
plot's own custom expression. generate_script() (the live Solve path) has
never had this limitation: it groups ea_files by output_selector and runs a
full metrics/line/surface report PER GROUP. This closes that gap for
"Export as DeepXDE Script" too, grouping AT GENERATION TIME (consistent
with this generator's own "decide the shape once" design) rather than at
runtime like generate_script() does.

This test exercises, against REAL exported scripts run as real DeepXDE
subprocesses:

 1. Backward compatibility -- a single-group (old-style 2-tuple, no
    selector) ea_files config still produces the exact same UNSUFFIXED
    output filenames as before this round (error_metrics.txt,
    line_comparison.png, surface_comparison.png).

 2. Multi-output grouping -- three ea_files entries with three different
    selectors (implicit default, a custom derived expression, and a raw
    output column on a genuine 2-output problem) each get their OWN
    suffixed metrics/line files, instead of two of the three being
    silently dropped the way they were before this round. The custom-
    expression group's selector is deliberately a CONSTANT field
    ("0*u+7.5") compared against a reference file that is itself uniformly
    7.5 -- this is a training-accuracy-independent check: regardless of
    how undertrained the tiny test network is, that group's L2 error must
    come out ~0 if (and only if) the custom selector is actually being
    evaluated and compared against its own matching reference file,
    rather than silently compared against the wrong field or dropped.

 3. Time-Adaptive + multi-output -- the same constant-field trick, but
    routed through _predict_at_time's new per-group `sel` parameter (it
    used to hard-code _extract_plot_field with no selector at all), to
    confirm per-time-step model routing still works once grouping is
    selector-aware.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round41_ea_multi_output.py
"""
import os
import sys
import glob
import tempfile
import subprocess

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import generate_clean_script


def _run_script(script, tmpdir, tag, timeout_s=150):
    sp = os.path.join(tmpdir, f"{tag}_script.py")
    with open(sp, "w", encoding="utf-8") as f:
        f.write(script)
    return subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=timeout_s)


def _base_1d_cfg(save_dir, num_outputs=1):
    cfg = PINNConfig()
    cfg.problem_dim = "1D"
    cfg.steady_state = False
    cfg.x_min, cfg.x_max = 0.0, 1.0
    cfg.t_min, cfg.t_max = 0.0, 1.0
    cfg.num_domain = 20; cfg.num_boundary = 10; cfg.num_initial = 10; cfg.num_test = 15
    cfg.iterations = 3; cfg.optimizer_scheduler = False; cfg.iterations2 = 0
    cfg.plot_type = "Line (time steps)"
    cfg.plot_resolution = 12
    cfg.num_timesteps_line = 1
    cfg.save_dir = save_dir
    cfg.ea_do_line = True
    cfg.ea_do_surface = True
    if num_outputs == 1:
        cfg.num_outputs = 1
        cfg.output_names = "u"
        cfg.layers = [2, 8, 1]
        cfg.pde_expressions = "du_t - 0.1*du_xx"
        cfg.pde_expression = cfg.pde_expressions
        cfg.ic_expressions = "0.0"
        cfg.ic_expression = cfg.ic_expressions
    else:
        cfg.num_outputs = 2
        cfg.output_names = "u,v"
        cfg.layers = [2, 8, 2]
        cfg.pde_expressions = "du_t - 0.1*du_xx|dv_t - 0.1*dv_xx"
        cfg.pde_expression = "du_t - 0.1*du_xx"
        cfg.ic_expressions = "0.0|0.0"
        cfg.ic_expression = "0.0"
    cfg.custom_bc_json = "[]"
    cfg.adapt_method = "None"
    return cfg


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
            print("FAIL:", msg)
        else:
            print("ok:", msg)

    try:
        import deepxde as _dde_probe  # noqa: F401
        import torch as _torch_probe  # noqa: F401
    except ImportError:
        print("SKIPPED: deepxde/torch not installed in this environment.")
        return failures

    # ── 1. Backward compatibility: single-group, old-style 2-tuple ───────
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "run1")
        ref1 = os.path.join(tmpdir, "ref_single.txt")
        xg = np.linspace(0, 1, 15)
        np.savetxt(ref1, np.column_stack([xg, np.full(15, 0.5), np.full(15, 0.5)]))
        ref2 = os.path.join(tmpdir, "ref_single_2.txt")
        np.savetxt(ref2, np.column_stack([xg, np.full(15, 0.8), np.full(15, 0.5)]))

        cfg1 = _base_1d_cfg(save_dir)
        cfg1.ea_files = repr([(0.5, ref1), (0.8, ref2)])  # old-style 2-tuples, no selector
        clean1 = generate_clean_script(cfg1)
        check('error_metrics.txt' in clean1 and 'error_metrics{_ea_suffix}' not in clean1,
              "single-group config should still reference an unsuffixed error_metrics.txt literal "
              "(the f-string {_ea_suffix} placeholder itself must not leak into the OUTPUT -- it "
              "should already have been resolved to an empty string at generation time)")

        proc1 = _run_script(clean1, tmpdir, "backcompat")
        check(proc1.returncode == 0,
              f"backward-compat single-group script should run cleanly, got {proc1.returncode}:\n"
              f"{proc1.stdout[-2000:]}\n{proc1.stderr[-2000:]}")
        ea_dir1 = os.path.join(save_dir, "error_analysis")
        check(os.path.exists(os.path.join(ea_dir1, "error_metrics.txt")),
              "single-group run should produce the unsuffixed error_metrics.txt")
        check(os.path.exists(os.path.join(ea_dir1, "line_comparison.png")),
              "single-group run should produce the unsuffixed line_comparison.png")
        check(os.path.exists(os.path.join(ea_dir1, "surface_comparison.png")),
              "single-group run (2 time snapshots) should produce the unsuffixed surface_comparison.png")
        # No suffixed variants should exist at all for a single-group config.
        check(len(glob.glob(os.path.join(ea_dir1, "error_metrics_*.txt"))) == 0,
              "single-group run should NOT produce any suffixed error_metrics_*.txt")

    # ── 2. Multi-output grouping: default + constant-custom-expr + raw int ──
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "run2")
        xg = np.linspace(0, 1, 15)

        ref_default = os.path.join(tmpdir, "ref_default.txt")
        np.savetxt(ref_default, np.column_stack([xg, np.full(15, 0.5), np.full(15, 0.3)]))

        ref_const = os.path.join(tmpdir, "ref_const.txt")
        np.savetxt(ref_const, np.column_stack([xg, np.full(15, 0.5), np.full(15, 7.5)]))

        ref_v = os.path.join(tmpdir, "ref_v.txt")
        np.savetxt(ref_v, np.column_stack([xg, np.full(15, 0.5), np.full(15, 0.2)]))

        cfg2 = _base_1d_cfg(save_dir, num_outputs=2)
        cfg2.plot_output_idx = 0  # main plot shows "u" -- the implicit-default group
        cfg2.ea_files = repr([
            (0.5, ref_default),                       # sel=None -> implicit default ("u")
            (0.5, ref_const, ["0*u+7.5", "const"]),    # custom derived field, training-independent
            (0.5, ref_v, 1),                           # raw output column 1 ("v")
        ])
        clean2 = generate_clean_script(cfg2)
        check("_ea_extract_group" in clean2,
              "multi-output config should embed the shared _ea_extract_group helper")
        check("error_metrics_const.txt" in clean2,
              "the custom-expr group should get a 'const'-suffixed metrics filename")
        check("error_metrics_v.txt" in clean2,
              "the raw-int-selector group (output index 1, named 'v') should get a 'v'-suffixed "
              "metrics filename")

        proc2 = _run_script(clean2, tmpdir, "multiout")
        check(proc2.returncode == 0,
              f"multi-output script should run cleanly, got {proc2.returncode}:\n"
              f"{proc2.stdout[-3000:]}\n{proc2.stderr[-3000:]}")

        ea_dir2 = os.path.join(save_dir, "error_analysis")
        # The implicit-default group's own label is "" (see _ea_group_label);
        # once there's more than one group it's suffixed "_default" (Round 50
        # fixed both generators together -- it used to be a bare "_", e.g.
        # "error_metrics_.txt"), not the unsuffixed "error_metrics.txt" test 1
        # checks for.
        default_metrics = os.path.join(ea_dir2, "error_metrics_default.txt")
        const_metrics = os.path.join(ea_dir2, "error_metrics_const.txt")
        v_metrics = os.path.join(ea_dir2, "error_metrics_v.txt")
        check(os.path.exists(default_metrics), "default-group error_metrics_default.txt should exist")
        check(os.path.exists(const_metrics), "custom-expr-group error_metrics_const.txt should exist")
        check(os.path.exists(v_metrics), "raw-int-group error_metrics_v.txt should exist")

        if os.path.exists(const_metrics):
            with open(const_metrics, encoding="utf-8", errors="replace") as f:
                rows = [ln for ln in f.read().splitlines() if ln.strip() and not ln.lower().startswith("t")]
            # header row's exact column names vary by config (RMSE/L2/etc select-
            # able metrics) -- just confirm at least one data row parses and its
            # L2 relative error (2nd numeric column) is ~0, since "0*u+7.5" is a
            # constant the network's own (untrained) weights cannot perturb, and
            # the reference file is uniformly 7.5 too.
            check(len(rows) >= 1, "const-group error_metrics_const.txt should have at least one data row")
            if rows:
                vals = [float(v) for v in rows[0].split(",")]
                l2_const = vals[1]
                check(l2_const < 1e-6,
                      f"const-group L2 error should be ~0 (constant field vs constant reference), got {l2_const}")

        check(os.path.exists(os.path.join(ea_dir2, "line_comparison_default.png")),
              "default-group line_comparison_default.png should exist (see the error_metrics_default.txt comment above)")
        check(os.path.exists(os.path.join(ea_dir2, "line_comparison_const.png")),
              "const-group line_comparison_const.png should exist")
        check(os.path.exists(os.path.join(ea_dir2, "line_comparison_v.png")),
              "raw-int-group line_comparison_v.png should exist")

    # ── 3. Time-Adaptive + multi-output (constant-field trick again) ─────
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "ta_run")
        xg = np.linspace(0, 1, 12)

        ref_default_ta = os.path.join(tmpdir, "ref_default_ta.txt")
        np.savetxt(ref_default_ta, np.column_stack([xg, np.full(12, 0.5), np.full(12, 0.3)]))

        ref_const_ta = os.path.join(tmpdir, "ref_const_ta.txt")
        np.savetxt(ref_const_ta, np.column_stack([xg, np.full(12, 0.5), np.full(12, 4.0)]))

        cfg3 = _base_1d_cfg(save_dir, num_outputs=1)
        cfg3.time_adaptive = True
        cfg3.ta_num_steps = 2
        cfg3.ea_files = repr([
            (0.5, ref_default_ta),
            (0.5, ref_const_ta, ["0*u+4.0", "const"]),
        ])
        clean3 = generate_clean_script(cfg3)
        check("_predict_at_time(xt_no_time, tv, sel=None)" in clean3,
              "Time-Adaptive + Error Analysis should emit the sel-aware _predict_at_time")

        proc3 = _run_script(clean3, tmpdir, "ta_multiout", timeout_s=180)
        check(proc3.returncode == 0,
              f"Time-Adaptive multi-output script should run cleanly, got {proc3.returncode}:\n"
              f"{proc3.stdout[-3000:]}\n{proc3.stderr[-3000:]}")

        ea_dir3 = os.path.join(save_dir, "error_analysis")
        const_metrics_ta = os.path.join(ea_dir3, "error_metrics_const.txt")
        check(os.path.exists(os.path.join(ea_dir3, "error_metrics_default.txt")),
              "TA default-group error_metrics_default.txt should exist (see the suffix comment above)")
        check(os.path.exists(const_metrics_ta),
              "TA const-group error_metrics_const.txt should exist")
        if os.path.exists(const_metrics_ta):
            with open(const_metrics_ta, encoding="utf-8", errors="replace") as f:
                rows = [ln for ln in f.read().splitlines() if ln.strip() and not ln.lower().startswith("t")]
            check(len(rows) >= 1, "TA const-group metrics should have at least one data row")
            if rows:
                vals = [float(v) for v in rows[0].split(",")]
                l2_const_ta = vals[1]
                check(l2_const_ta < 1e-6,
                      f"TA const-group L2 error should be ~0 (per-step-model routing through the "
                      f"correct constant field), got {l2_const_ta}")

    return failures


def main():
    failures = run()
    print()
    if failures:
        print(f"FAILURES: {len(failures)}")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("ALL ROUND 41 EA-MULTI-OUTPUT TESTS PASSED")
        sys.exit(0)


def test_round41_ea_multi_output():
    """pytest entry point."""
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    main()
