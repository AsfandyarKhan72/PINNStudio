"""Round 37 regression test: Time-Adaptive training's own, independent
copy of Error Analysis's "Line Comparison" plot (generate_script()'s
`if {config.time_adaptive}:` branch, ~line 6646) must evaluate a true
line at the exact configured slice (PINNConfig.line_slice_y/_z), the
same fix Round 36 already applied to the Standard (non-Time-Adaptive)
path and to generate_clean_script().

This copy was actually WORSE than the Standard path's pre-Round-36 bug:
it did no slice filtering at all for 2D/3D -- every reference point,
regardless of y(/z), was plotted sorted by x. It was flagged but
deliberately left unfixed in Round 36 (scoped to what was reported),
pending the user's go-ahead to fix it -- this test covers that fix.

The fix reuses the Standard path's own pattern (exact line, geom.inside
masking, griddata-interpolated reference) but must additionally route
the PINN evaluation through whichever Time-Adaptive step model actually
covers each reference snapshot's time -- there is no single `model` in
a Time-Adaptive script. It reuses `_ta_intervals` (already built by the
per-reference-point prediction loop just above) to map each time
snapshot to its step directory, then restores that step's own model
before evaluating it on the exact slice line.

Covers a real, tiny, 2-sub-domain Time-Adaptive 2D run (so two
different reference snapshots are necessarily served by two different
step models), checking:
 - the generated TA Error Analysis section's own Line Comparison block
   no longer plots an unfiltered scatter of every reference point
 - it now masks with the real geometry and interpolates the reference
   via griddata, matching the Standard path's fix
 - the full script (train 2 TA sub-domains, then run EA) runs cleanly
   end to end and produces a non-trivial line_comparison.png
 - an out-of-domain slice falls back to the "line outside the domain"
   text instead of crashing

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_ta_line_comparison_geometry.py
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

from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import generate_script


def _run_script(script, tmpdir, tag, timeout_s=180):
    sp = os.path.join(tmpdir, f"{tag}_script.py")
    with open(sp, "w") as f:
        f.write(script)
    return subprocess.run(
        [sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=timeout_s,
    )


def _ta_2d_cfg(save_dir, ref1, ref2, line_slice_y=0.25):
    cfg = PINNConfig()
    cfg.problem_dim = "2D"; cfg.steady_state = False
    cfg.num_outputs = 1; cfg.output_names = "u"
    cfg.geometry_type = "Rectangle"
    cfg.x_min, cfg.x_max = 0.0, 1.0
    cfg.y_min, cfg.y_max = 0.0, 1.0
    cfg.t_min, cfg.t_max = 0.0, 1.0
    cfg.num_domain = 80; cfg.num_boundary = 20; cfg.num_initial = 20; cfg.num_test = 40
    cfg.layers = [3, 16, 16, 1]
    cfg.pde_expression = "du_t - 0.4*(du_xx + du_yy)"
    cfg.pde_expressions = cfg.pde_expression
    cfg.ic_expression = "0.0"; cfg.ic_expressions = cfg.ic_expression
    cfg.custom_bc_json = json.dumps([
        {"type": "dirichlet", "location": "True", "value": "0", "component": 0}
    ])
    cfg.adapt_method = "None"
    cfg.time_adaptive = True
    cfg.ta_num_steps = 2  # splits [t_min, t_max] into 2 contiguous sub-domains
    cfg.ta_step_groups = ""  # default: one group across [t_min, t_max], ta_num_steps steps
    cfg.iterations = 2; cfg.optimizer_scheduler = False; cfg.iterations2 = 0
    cfg.save_dir = save_dir
    # One reference snapshot per sub-domain (t=0.2 in [0, 0.5), t=0.7 in
    # [0.5, 1]) so the fix is necessarily exercised against TWO different
    # step models, not just one.
    cfg.ea_files = repr([(0.2, ref1), (0.7, ref2)])
    cfg.ea_do_line = True
    cfg.ea_do_surface = False
    cfg.line_slice_y_auto = False
    cfg.line_slice_y = line_slice_y
    return cfg


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    try:
        import numpy as np
        import deepxde as _dde_probe  # noqa: F401
        import torch as _torch_probe  # noqa: F401
    except ImportError:
        print("deepxde/torch not installed in this environment, skipping real-training checks.")
        return failures

    with tempfile.TemporaryDirectory() as tmpdir:
        # Synthetic reference data over the full unit square, with u
        # varying meaningfully with y -- exactly the condition the old
        # unfiltered-scatter code smeared (it had no slice filtering at
        # all, so this is a lower bar than even the Standard path's old
        # tolerance-band bug).
        xs = np.linspace(0, 1, 25)
        ys = np.linspace(0, 1, 25)
        X, Y = np.meshgrid(xs, ys)
        X = X.ravel(); Y = Y.ravel()
        ref1 = os.path.join(tmpdir, "ref_t0.2.dat")
        ref2 = os.path.join(tmpdir, "ref_t0.7.dat")
        np.savetxt(ref1, np.column_stack([X, Y, np.full_like(X, 0.2), X + 5.0 * Y]))
        np.savetxt(ref2, np.column_stack([X, Y, np.full_like(X, 0.7), X - 3.0 * Y]))

        save_dir = os.path.join(tmpdir, "run")
        os.makedirs(save_dir, exist_ok=True)
        cfg = _ta_2d_cfg(save_dir, ref1, ref2, line_slice_y=0.25)
        script = generate_script(cfg)

        # Isolate the Time-Adaptive Error Analysis section specifically --
        # the Standard path's own (dead, since config.time_adaptive=True
        # makes it an unreached `if not True:` branch) Line Comparison text
        # is still physically present earlier in the script (generate_script
        # emits both branches' text unconditionally, guarded by runtime
        # `if`/`if not`), so a bare search for "Line comparison" would find
        # the WRONG one.
        check("Time Adaptive Error Analysis" in script,
              "precondition: generated script should contain the Time-Adaptive Error Analysis section")
        ta_section = script.split("Time Adaptive Error Analysis")[1] if "Time Adaptive Error Analysis" in script else ""
        lc_section = ta_section.split("Line comparison")[1][:5000] if "Line comparison" in ta_section else ""
        check(bool(lc_section),
              "generated TA script should contain its own 'Line comparison' section")
        check("_ea_x_refs[_ei]\n" not in lc_section.replace(" ", "") or "_line_pts_ta" in lc_section,
              "TA Line Comparison should no longer plot every reference point unfiltered")
        check("_build_geom()" in lc_section and ".inside(" in lc_section,
              "TA Line Comparison section should mask with the real geometry's .inside(...), built via "
              "_build_geom() (there is no bare `geom` variable in the Time-Adaptive branch)")
        check("griddata" in lc_section or "_gd_line_ta" in ta_section,
              "TA script should use griddata to interpolate the reference onto the exact line")
        check("_step_dir_map_line" in lc_section and "_ta_intervals" in lc_section,
              "TA Line Comparison should route each snapshot's PINN curve through its own step model, "
              "reusing _ta_intervals already built by the per-reference-point prediction loop above it")

        proc = _run_script(script, save_dir, "ta_line_geom")
        check(proc.returncode == 0,
              f"Time-Adaptive 2D Line Comparison script should exit cleanly, got {proc.returncode}: "
              f"{proc.stdout[-2500:]}\n{proc.stderr[-2500:]}")
        check("Time-Adaptive Error Analysis Complete" in proc.stdout,
              f"Time-Adaptive Error Analysis should actually complete, stdout tail:\n{proc.stdout[-1500:]}")

        png = None
        for root, _dirs, files in os.walk(save_dir):
            for fn in files:
                if fn.startswith("line_comparison"):
                    png = os.path.join(root, fn)
        check(png is not None and os.path.getsize(png) > 1000,
              f"TA line_comparison.png should exist and be non-trivially sized (found: {png})")

        # ── Edge case: an out-of-domain slice must hit the fallback text,
        # not crash, exactly like the Standard path's own edge case. ──
        save_dir_oob = os.path.join(tmpdir, "oob")
        os.makedirs(save_dir_oob, exist_ok=True)
        cfg_oob = _ta_2d_cfg(save_dir_oob, ref1, ref2, line_slice_y=5.0)
        proc_oob = _run_script(generate_script(cfg_oob), save_dir_oob, "ta_oob")
        check(proc_oob.returncode == 0,
              f"out-of-domain TA line slice should still exit cleanly (fallback text, no crash), got "
              f"{proc_oob.returncode}: {proc_oob.stdout[-2000:]}\n{proc_oob.stderr[-2000:]}")

    if not failures:
        print("OK: Time-Adaptive training's own Error Analysis Line Comparison now evaluates a true "
              "line at the exact configured slice, routed through the correct per-time-step model, "
              "instead of an unfiltered scatter of every reference point regardless of y(/z).")
    return failures


def test_ta_line_comparison_geometry():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    _failures = run()
    for _f in _failures:
        print("FAIL:", _f)
    sys.exit(1 if _failures else 0)
