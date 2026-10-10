"""Round 36 regression test: Error Analysis's "Line Comparison" plot for
2D/3D problems must evaluate a true line at the exact configured slice
(PINNConfig.line_slice_y/_z), not a tolerance-band scatter of whichever
reference-file points happened to fall near it.

The tolerance-band approach smeared multiple different-y reference/PINN
values together at the same x whenever u varies meaningfully with y
within the band -- reported by the user on the official DeepXDE 2D
Poisson L-Shape example, where the plot showed a dense smeared cloud
instead of a clean comparison line. See the matching fix/comment in
codegen.py's generate_script() and generate_clean_script() Line
Comparison blocks.

Covers both codegen paths (generate_script -- live Solve, and
generate_clean_script -- Export as DeepXDE Script), on the exact
reported scenario (2D steady Poisson, L-Shape Polygon geometry), plus
an edge case confirming the new "line outside the domain at this slice"
fallback doesn't crash when the configured slice has no valid line
segment in range.
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
from pinnstudio.core.codegen import generate_script, generate_clean_script


def _run_script(script, tmpdir, tag, timeout_s=150):
    sp = os.path.join(tmpdir, f"{tag}_script.py")
    with open(sp, "w") as f:
        f.write(script)
    return subprocess.run(
        [sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=timeout_s,
    )


def _lshape_cfg(save_dir, ref_path, line_slice_y=0.0):
    cfg = PINNConfig()
    cfg.problem_dim = "2D"; cfg.steady_state = True
    cfg.num_outputs = 1; cfg.output_names = "u"
    cfg.geometry_type = "Polygon"
    cfg.geom_polygon_vertices = "0,0;1,0;1,-1;-1,-1;-1,1;0,1"
    cfg.x_min, cfg.x_max = -1.0, 1.0
    cfg.y_min, cfg.y_max = -1.0, 1.0
    cfg.num_domain = 200; cfg.num_boundary = 40; cfg.num_test = 200
    cfg.layers = [2, 16, 16, 1]
    cfg.custom_bc_json = json.dumps([
        {"type": "dirichlet", "location": "True", "value": "0", "component": 0}
    ])
    cfg.adapt_method = "None"
    cfg.iterations = 10; cfg.optimizer_scheduler = False; cfg.iterations2 = 0
    cfg.save_dir = save_dir
    cfg.pde_expression = "-du_xx - du_yy - 1"; cfg.pde_expressions = cfg.pde_expression
    cfg.ea_files = repr([(0.0, ref_path)])
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
        # A synthetic "reference" dataset over the L-Shape domain only,
        # with u varying meaningfully with y near y=0 -- exactly the
        # condition that made the old tolerance-band code smear.
        xs = np.linspace(-1, 1, 50)
        ys = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(xs, ys)
        X = X.ravel(); Y = Y.ravel()
        inside = ~((X > 0) & (Y > 0))
        X, Y = X[inside], Y[inside]
        U = 0.3 * (1 - X**2) + 0.5 * Y
        ref_path = os.path.join(tmpdir, "reference.dat")
        np.savetxt(ref_path, np.column_stack([X, Y, U]))

        for name, gen in [("generate_script", generate_script), ("generate_clean_script", generate_clean_script)]:
            save_dir = os.path.join(tmpdir, name)
            os.makedirs(save_dir, exist_ok=True)
            cfg = _lshape_cfg(save_dir, ref_path, line_slice_y=0.0)
            script = gen(cfg)

            lc_section = script.split("Line comparison")[1][:4000] if "Line comparison" in script else ""
            check(bool(lc_section), f"{name}: generated script should contain a 'Line comparison' section")
            check("y_tol" not in lc_section and "y_mid - " not in lc_section.replace(" ", "") and "np.abs(ea_y_refs" not in lc_section.replace(" ", "") and "np.abs(_ea_y_refs" not in lc_section.replace(" ", ""),
                  f"{name}: Line Comparison section should no longer use tolerance-band masking")
            check(("geom.inside" in lc_section or "_build_geom().inside" in lc_section),
                  f"{name}: Line Comparison section should mask with geom.inside(...)")
            check("griddata" in script, f"{name}: script should import/use scipy's griddata for the reference line")

            proc = _run_script(script, save_dir, name)
            check(proc.returncode == 0,
                  f"{name}: L-Shape Line Comparison script should exit cleanly, got {proc.returncode}: "
                  f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

            png = None
            for root, _dirs, files in os.walk(save_dir):
                for fn in files:
                    if fn.startswith("line_comparison"):
                        png = os.path.join(root, fn)
            check(png is not None and os.path.getsize(png) > 1000,
                  f"{name}: line_comparison.png should exist and be non-trivially sized")

        # ── Edge case: a slice with no valid in-domain line segment must
        # hit the new fallback text, not crash. ──
        save_dir_oob = os.path.join(tmpdir, "out_of_bounds")
        os.makedirs(save_dir_oob, exist_ok=True)
        cfg_oob = _lshape_cfg(save_dir_oob, ref_path, line_slice_y=5.0)
        proc_oob = _run_script(generate_script(cfg_oob), save_dir_oob, "oob")
        check(proc_oob.returncode == 0,
              f"out-of-bounds line slice should still exit cleanly (fallback text, no crash), got "
              f"{proc_oob.returncode}: {proc_oob.stdout[-2000:]}\n{proc_oob.stderr[-2000:]}")

    if not failures:
        print("OK: Error Analysis Line Comparison now evaluates a true line at the exact configured "
              "slice (geom.inside-masked, griddata-interpolated reference) for 2D/3D problems, in both "
              "generate_script() and generate_clean_script(), instead of smearing a tolerance-band "
              "scatter of reference points -- verified on the reported 2D Poisson L-Shape scenario, "
              "plus an out-of-domain slice correctly falling back without crashing.")
    return failures


def test_line_comparison_geometry():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
