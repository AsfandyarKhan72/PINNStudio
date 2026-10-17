#!/usr/bin/env python3
"""
Round 50 -- codebase cleanup, phase 13 (open item from Round 22): matplotlib's
contourf ignores vmin/vmax when choosing level boundaries if `levels` is a
plain integer -- fill colors still use the vmin/vmax norm, but the bands and
the COLORBAR auto-range to each panel's own data. So panels meant to share
one scale (PINN vs Reference) showed different colorbars, and a manual color
range from Plot Settings never showed on the colorbar. A handful of
comparison plots worked around it with a hardcoded np.linspace(vmin, vmax,
41) (ignoring the configured level count). Every site -- ~30 across
generate_script(), generate_clean_script() and the restore builders -- now
goes through one emitted runtime helper, _cf_levels(n, vmin, vmax, *data).

Also fixed in the same area: generate_clean_script()'s exported Error
Analysis surface comparison gave PINN and Reference independent color
scales (1D, 2D and 3D) and, in 2D, never masked to the real geometry
(fabricated fill_value=0.0 reference values and extrapolated PINN values
outside the domain) -- both unlike the live Solve path.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round50_shared_color_scale.py
"""
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("MPLBACKEND", "Agg")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import _contour_levels_helper_code, generate_clean_script, generate_script


def _ea_2d_disk_cfg(save_dir, ref_path):
    cfg = PINNConfig()
    cfg.problem_dim = "2D"; cfg.steady_state = False
    cfg.geometry_type = "Disk"
    cfg.geom_center_x, cfg.geom_center_y, cfg.geom_radius = 0.5, 0.5, 0.4
    cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max = 0.0, 1.0, 0.0, 1.0
    cfg.t_min, cfg.t_max = 0.0, 1.0
    cfg.num_outputs = 1; cfg.output_names = "u"; cfg.layers = [3, 8, 1]
    cfg.pde_expressions = "du_t - 0.1*(du_xx + du_yy)"; cfg.pde_expression = cfg.pde_expressions
    cfg.ic_expressions = "0.0"; cfg.ic_expression = cfg.ic_expressions
    cfg.custom_bc_json = json.dumps([{"type": "dirichlet", "location": "True", "value": "0", "component": 0}])
    cfg.adapt_method = "None"
    cfg.num_domain = 40; cfg.num_boundary = 10; cfg.num_initial = 10; cfg.num_test = 20
    cfg.iterations = 2; cfg.optimizer_scheduler = False; cfg.iterations2 = 0
    cfg.plot_type = "Surface"; cfg.plot_resolution = 16; cfg.plot_n_2d_snapshots = 1
    cfg.save_dir = save_dir
    cfg.ea_do_line = False; cfg.ea_do_surface = True
    cfg.ea_files = repr([(0.5, ref_path)])
    return cfg


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = {"np": np}
    exec(_contour_levels_helper_code(), ns)
    cf = ns["_cf_levels"]

    # ── 1. helper semantics ───────────────────────────────────────────────
    check(cf(30) == {"levels": 30} and cf(30, None, 1.0) == {"levels": 30},
          "no range -> plain integer levels, unchanged behavior")
    kw = cf(10, 0.0, 1.0, np.array([0.1, 0.9]))
    check(kw["levels"][0] <= 0.0 and kw["levels"][-1] >= 1.0 and 8 <= len(kw["levels"]) <= 12
          and kw["extend"] == "neither", "in-range data -> ~10 nice bands covering [0, 1], no extension")
    kw2 = cf(20, 0.0023, 0.9168)
    check(all(abs(v * 100 - round(v * 100)) < 1e-9 for v in kw2["levels"]),
          "boundaries for an arbitrary range should still be round numbers (readable colorbar ticks)")
    check(cf(10, 0.0, 1.0, np.array([-1.0, 0.5]))["extend"] == "min", "data below vmin -> extend='min'")
    check(cf(10, 0.0, 1.0, np.array([-1.0, 2.0]))["extend"] == "both", "data on both sides -> extend='both'")
    check(cf(10, 0.0, 1.0, np.array([np.nan, 0.5]))["extend"] == "neither", "NaN (masked) cells are ignored")
    deg = cf(10, 2.0, 2.0)
    check("extend" not in deg and deg["levels"] == 10, "degenerate range (constant field) falls back safely")
    x, y = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    fig, axes = plt.subplots(1, 3)
    try:
        cbs = [fig.colorbar(axes[i].contourf(x, y, z, cmap="viridis", **cf(20, 0.0, 1.0, z)), ax=axes[i])
               for i, z in enumerate((x, 0.5 * x, 2.0 * x))]
        check(all((cb.vmin, cb.vmax) == (cbs[0].vmin, cbs[0].vmax) for cb in cbs)
              and cbs[0].vmin <= 0.0 and cbs[0].vmax >= 1.0,
              "every panel sharing [0, 1] should show the same colorbar covering [0, 1], whatever its own data")
        fig2, ax2 = plt.subplots()
        ax2.contourf(x, y, np.full_like(x, 3.0), cmap="viridis", **cf(20, 3.0, 3.0))
        plt.close(fig2)
    except Exception as e:  # noqa: BLE001
        check(False, f"contourf with _cf_levels kwargs should not raise: {e!r}")
    plt.close(fig)

    # ── 2. every generator emits the helper and no int-levels+vmin call ──
    cfg_txt = _ea_2d_disk_cfg("/tmp/x", "/tmp/ref.txt")
    for name, script in (("generate_script", generate_script(cfg_txt)),
                         ("generate_clean_script", generate_clean_script(cfg_txt))):
        check("def _cf_levels(" in script, f"{name}: should define _cf_levels")
        bad = [ln for ln in script.splitlines() if "contourf(" in ln and "vmin=" in ln and "_cf_levels" not in ln]
        check(not bad, f"{name}: contourf calls still pass vmin with integer levels: {bad[:2]}")

    try:
        import deepxde as _dde_probe  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures

    # ── 3. exported 2D Error Analysis: masked to the Disk, runs cleanly ──
    with tempfile.TemporaryDirectory() as tmpdir:
        g = np.linspace(0, 1, 21)
        X, Y = np.meshgrid(g, g)
        inside = (X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.4 ** 2
        ref = np.column_stack([X[inside], Y[inside], np.full(inside.sum(), 0.5), np.sin(X[inside])])
        ref_path = os.path.join(tmpdir, "ref.txt")
        np.savetxt(ref_path, ref)
        save_dir = os.path.join(tmpdir, "out")
        os.makedirs(save_dir, exist_ok=True)
        script = generate_clean_script(_ea_2d_disk_cfg(save_dir, ref_path))
        mask_path = os.path.join(tmpdir, "mask.npy")
        sp = os.path.join(tmpdir, "export.py")
        with open(sp, "w", encoding="utf-8") as f:
            f.write(script + f"\nnp.save({mask_path!r}, inside_ea)\n")
        proc = subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=300)
        check(proc.returncode == 0, f"export with 2D Error Analysis should run cleanly, got {proc.returncode}:\n"
                                    f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        found = [os.path.join(r, f) for r, _, fs in os.walk(save_dir) for f in fs if f.startswith("surface_comparison")]
        check(bool(found), "export should save surface_comparison.png")
        check(os.path.exists(mask_path), "inside_ea mask should exist in the exported script")
        if os.path.exists(mask_path):
            m = np.load(mask_path)
            check(bool(m.any()) and bool((~m).any()), "export EA mask should be genuinely mixed on a Disk")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 50 SHARED-COLOR-SCALE TESTS PASSED")
    return 0


def test_round50_shared_color_scale():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
