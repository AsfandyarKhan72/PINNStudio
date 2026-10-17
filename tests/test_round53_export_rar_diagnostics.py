#!/usr/bin/env python3
"""
Round 53 -- codebase cleanup, phase 16 (long-standing open item):
generate_clean_script()'s RAR loop had none of the live Solve path's
per-round diagnostics and ignored the "Points from:" setting
(rar_output_selector), always summing every equation's residual.

Now the export (a) honors rar_output_selector, and (b) writes, for every
RAR round, rar_rounds/round_NN/{model.pt, collocation_points.png,
solution_plot.png} -- the same files the live path writes (minus its
per-round reference-comparison panels; the export keeps comparison to its
end-of-run Error Analysis). The loop records each round's weights/points;
a block after _extract_plot_field replays them and restores the final
weights.

Checks every generation branch end to end (1D transient, 1D steady, 2D
steady L-Shape, 2D transient Disk, 3D transient), that model.pt restores
through the Restore tab's own loader convention, that the final weights are
restored afterwards, and that "Points from:" reaches the export.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round53_export_rar_diagnostics.py
"""
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import generate_clean_script

_DIRICHLET = json.dumps([{"type": "dirichlet", "location": "True", "value": "0", "component": 0}])


def _cfg(save_dir, dim, steady, geometry="Rectangle", n_out=1, **extra):
    cfg = PINNConfig()
    cfg.problem_dim = dim; cfg.steady_state = steady; cfg.geometry_type = geometry
    cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max, cfg.z_min, cfg.z_max = 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
    cfg.t_min, cfg.t_max = 0.0, 1.0
    n_sp = {"1D": 1, "2D": 2, "3D": 3}[dim]
    n_in = n_sp + (0 if steady else 1)
    lap = {"1D": "du_xx", "2D": "du_xx + du_yy", "3D": "du_xx + du_yy + du_zz"}[dim]
    if n_out == 1:
        cfg.num_outputs = 1; cfg.output_names = "u"; cfg.layers = [n_in, 8, 1]
        cfg.pde_expressions = (f"-({lap}) - 1" if steady else f"du_t - 0.1*({lap})")
        cfg.ic_expressions = "0.0"
    else:
        lapv = lap.replace("du_", "dv_")
        cfg.num_outputs = 2; cfg.output_names = "u,v"; cfg.layers = [n_in, 8, 2]
        cfg.pde_expressions = f"du_t - 0.1*({lap})|dv_t - 0.1*({lapv})"
        cfg.ic_expressions = "0.0|0.0"
    cfg.pde_expression = cfg.pde_expressions.split("|")[0]; cfg.ic_expression = cfg.ic_expressions.split("|")[0]
    cfg.custom_bc_json = _DIRICHLET
    cfg.num_domain = 40; cfg.num_boundary = 16; cfg.num_initial = 10; cfg.num_test = 20
    cfg.iterations = 2; cfg.optimizer_scheduler = False; cfg.iterations2 = 0
    cfg.adapt_method = "RAR"; cfg.rar_cycles = 2; cfg.rar_candidates = 200; cfg.rar_add_points = 5
    cfg.rar_adam_iters = 2; cfg.rar_lbfgs_iters = 0
    cfg.plot_type = "Surface"; cfg.plot_resolution = 16; cfg.plot_n_2d_snapshots = 1
    cfg.save_dir = save_dir
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    try:
        import deepxde as _dde_probe  # noqa: F401
        import torch
    except ImportError:
        print("SKIPPED: deepxde/torch not installed in this environment.")
        return failures

    cases = [
        ("1d_transient", dict(dim="1D", steady=False)),
        ("1d_steady", dict(dim="1D", steady=True)),
        ("2d_steady_lshape", dict(dim="2D", steady=True, geometry="Polygon",
                                  geom_polygon_vertices="0,0;1,0;1,0.5;0.5,0.5;0.5,1;0,1")),
        ("2d_transient_disk_sel", dict(dim="2D", steady=False, geometry="Disk", n_out=2, rar_output_selector=1,
                                       geom_center_x=0.5, geom_center_y=0.5, geom_radius=0.4)),
        ("3d_transient", dict(dim="3D", steady=False, geometry="Cuboid")),
    ]
    for tag, kw in cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "out")
            os.makedirs(save_dir, exist_ok=True)
            script = generate_clean_script(_cfg(save_dir, **kw))
            if kw.get("rar_output_selector") == 1:
                check("residuals = np.abs(res[1]).flatten()" in script,
                      f"{tag}: 'Points from: v' should select equation 1's residual in the export")
            check("RAR per-round diagnostics" in script, f"{tag}: export should include the diagnostics block")
            sp = os.path.join(tmpdir, "export.py")
            with open(sp, "w", encoding="utf-8") as f:
                f.write(script + "\nnp.save('final_w.npy', next(iter(model.net.state_dict().values())).detach().cpu().numpy())\n")
            proc = subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=300)
            check(proc.returncode == 0, f"{tag}: export should run cleanly, got {proc.returncode}:\n"
                                        f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
            for r in (1, 2):
                rd = os.path.join(save_dir, "rar_rounds", f"round_{r:02d}")
                for fn in ("model.pt", "collocation_points.png", "solution_plot.png"):
                    check(os.path.exists(os.path.join(rd, fn)), f"{tag}: {rd}/{fn} should exist")
            ck = os.path.join(save_dir, "rar_rounds", "round_02", "model.pt")
            if os.path.exists(ck) and os.path.exists(os.path.join(tmpdir, "final_w.npy")):
                import numpy as np
                state = torch.load(ck, map_location="cpu").get("model_state_dict")
                check(state is not None, f"{tag}: model.pt should hold a model_state_dict (Restore-tab format)")
                last = next(iter(state.values())).cpu().numpy()
                check(np.allclose(last, np.load(os.path.join(tmpdir, "final_w.npy"))),
                      f"{tag}: the last round's weights should equal the final weights left in the model")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 53 EXPORT-RAR-DIAGNOSTICS TESTS PASSED")
    return 0


def test_round53_export_rar_diagnostics():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
