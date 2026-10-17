#!/usr/bin/env python3
"""
Round 75 -- wave propagation, Helmholtz sound-hard scattering, and
Bayesian search in Parameter Sweep.

* "Scale by initial losses" (loss weight_i = c / L_i(0), DeepXDE's wave
  example): config, GUI, both generators (same weights).
* Numbers inside math calls are folded (sqrt(3) -> 1.732...), so
  torch.sqrt(3) no longer breaks a formula.
* A Robin BC value can use every output by name (-k0*u_im on u_re).
* Parameter Sweep mode "bayes": the optimizer, the run source (baseline
  first, then model-chosen runs, outputs written), validation, and the
  GUI round-trip.
* The three DeepXDE examples build and validate.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round75_wave_helmholtz_bayes.py
"""
import json
import math
import os
import re
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

from pinnstudio.core import deepxde_examples as dx
from pinnstudio.core import sweep_runner
from pinnstudio.core.bayes_opt import BayesOptimizer, Dim
from pinnstudio.core.codegen import _clean_bc_row_code, _simplify_pde_expr, generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow

NEW = ["Wave propagation – space-time multi-scale Fourier features",
       "Helmholtz sound-hard scattering (absorbing BC)", "Helmholtz 2D – hyperparameter optimization"]


def _cfg(d=".", **kw):
    c = PINNConfig()
    base = dict(problem_dim="1D", steady_state=True, geometry_type="Interval", x_min=-1.0, x_max=1.0, num_initial=0,
                pde_expressions="du_xx - 2", layers=[1, 20, 20, 1], iterations=300, optimizer2="none", iterations2=0,
                optimizer_scheduler=False, num_domain=16, num_boundary=2, num_test=40,
                custom_bc_json=json.dumps([{"type": "dirichlet", "location": "left", "value": "0", "component": 0},
                                           {"type": "neumann", "location": "right", "value": "4", "component": 0}]),
                ea_ref_mode="formula", ea_ref_formula="(x + 1)**2", save_dir=d, plot_resolution=40)
    base.update(kw)
    for k, v in base.items():
        setattr(c, k, v)
    c.pde_expression = c.pde_expressions
    return c


def run():
    failures = []

    def check(cond, msg):
        print(("ok: " if cond else "FAIL: ") + msg)
        if not cond:
            failures.append(msg)

    # -- numbers inside math calls
    check(_simplify_pde_expr("(x - 0.5)*2*sqrt(3)") == "(x - 0.5)*2*1.7320508075688772", "sqrt(3) folded")
    check(_simplify_pde_expr("x*log(1e-3)**2") == "x*(-6.907755278982137)**2", "negative values keep parentheses")
    check(_simplify_pde_expr("sin(pi*x) + sqrt(x)") == "torch.sin(np.pi*x) + torch.sqrt(x)", "calls on variables unchanged")

    # -- weights from the initial losses
    c = _cfg(loss_weights_from_initial=5.0)
    check(c.validate() == [], "initial-loss weights validate")
    check(any("c > 0" in e for e in _cfg(loss_weights_from_initial=-1.0).validate()), "negative c rejected")
    live, exp = generate_script(c), generate_clean_script(c)
    check("_multi_weights = [5.0 / max(float(_l), 1e-30)" in live, "live script computes the weights")
    check("loss_weights = [5.0 / max(float(_l), 1e-30)" in exp and "loss_weights=loss_weights" in exp,
          "export computes and uses the weights")
    check("_lh0" not in generate_script(_cfg()), "off by default")

    # -- Robin with other outputs
    d, _a = _clean_bc_row_code(1, {"type": "robin", "location": "left or right", "value": "-k0*u_im", "component": 0},
                               True, False, 2, True, n_out=2, out_names="u_re,u_im")
    check("    u_im = Y[:, 1:2]" in d, "export Robin binds the other output")
    two = _cfg(problem_dim="2D", geometry_type="Rectangle", y_min=-1.0, y_max=1.0, num_outputs=2, output_names="a,b",
               pde_expressions="-da_xx - da_yy - a|-db_xx - db_yy - b", layers=[2, 8, 2], ea_ref_mode="files",
               ea_ref_formula="", custom_bc_json=json.dumps([
                   {"type": "robin", "location": "True", "value": "-b", "component": 0},
                   {"type": "robin", "location": "True", "value": "a", "component": 1}]))
    two.pde_expression = "-da_xx - da_yy - a"
    check("for _oi, _on in enumerate(_out_names_list)" in generate_script(two), "live Robin binds every output")

    # -- Bayesian optimizer
    dims = [Dim("float", 1e-4, 5e-2, log=True), Dim("int", 1, 10), Dim("categorical", choices=["sin", "tanh"])]
    f = lambda p: (math.log10(p[0]) + 2.5) ** 2 + 0.1 * (p[1] - 4) ** 2 + (0 if p[2] == "sin" else 1) + 0.01
    opt = BayesOptimizer(dims, n_initial=4, seed=1)
    for _ in range(25):
        p = opt.ask()
        check_types = isinstance(p[0], float) and isinstance(p[1], int) and p[2] in ("sin", "tanh")
        if not check_types:
            check(False, f"point types {p}")
        opt.tell(p, f(p))
    check(opt.best()[1] < 0.15, f"finds a good point: {opt.best()}")
    opt.tell([1e-3, 2, "sin"], None)
    check(math.isfinite(opt.y[-1]) and opt.y[-1] >= max(opt.y[:-1]), "a failed run counts as the worst")

    # -- sweep mode "bayes" with a stand-in training run
    ex = dx.build_config(dx.BY_NAME["Helmholtz 2D – hyperparameter optimization"], "/r")
    check(ex.validate() == [], f"HPO example validates {ex.validate()}")
    bad = dx.build_config(dx.BY_NAME["Helmholtz 2D – hyperparameter optimization"], "/r")
    bad.sweep_bayes_objective, bad.ea_ref_mode, bad.ea_files = "l2", "files", "[]"
    check(any("reference data" in e for e in bad.validate()), "L2 objective needs reference data")
    seen = []

    def fake_run(cfg, on_output=None, set_process=None):
        lr = json.loads(cfg.scheduler_phases)[0]["lr"]
        seen.append((lr, len(cfg.layers) - 2, cfg.layers[1], cfg.activation))
        on_output(f"Final loss: {abs(math.log10(lr) + 2.5) + 0.01 * len(cfg.layers):.6e}")
        return "DONE"

    real = sweep_runner.run_pinn
    sweep_runner.run_pinn = fake_run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ex.sweep_save_dir, ex.sweep_bayes_calls, ex.sweep_bayes_initial = tmp, 8, 3
            res = sweep_runner.run_sweep(ex)
            check(len(res) == 8 and res[0][0] == "baseline", f"baseline + 7 runs: {[r[0] for r in res]}")
            check(seen[0] == (1e-3, 4, 50, "sin"), f"baseline is the example's own settings {seen[0]}")
            check(all(1e-4 <= s[0] <= 5e-2 and 1 <= s[1] <= 10 and 5 <= s[2] <= 500 and s[3] in ("sin", "sigmoid", "tanh")
                      for s in seen[1:]), "every run inside the search ranges")
            root = os.path.dirname(res[0][1]["save_dir"])
            check(os.path.exists(os.path.join(root, "bayes_search.csv")) and
                  os.path.exists(os.path.join(root, "bayes_convergence.png")), "search table and convergence plot written")
            man = json.load(open(os.path.join(root, "sweep_manifest.json"), encoding="utf-8", errors="replace"))
            check("bayes_best" in man, f"manifest records the best run {man.get('bayes_best')}")
    finally:
        sweep_runner.run_pinn = real

    # -- GUI
    w = MainWindow()
    w._apply_deepxde_example("Helmholtz 2D – hyperparameter optimization")
    g = w._build_config()
    check(g.sweep_enabled and g.sweep_mode == "bayes" and g.sweep_bayes_calls == 50 and g.sweep_bayes_initial == 10,
          "GUI keeps the Bayesian sweep settings")
    gp = json.loads(g.sweep_parameters)
    check([e["id"] for e in gp] == ["phase0_lr", "hidden_layers", "neurons_per_layer", "activation"]
          and abs(gp[0]["min"] - 1e-4) < 1e-12, f"GUI keeps the search space {gp}")
    w._apply_deepxde_example("Wave propagation – space-time multi-scale Fourier features")
    g = w._build_config()
    check(g.loss_weights_from_initial == 5.0 and w.loss_w_init_cb.isChecked() and not g.sweep_enabled,
          "GUI: initial-loss weights on, no sweep")

    for n in NEW:
        e = dx.build_config(dx.BY_NAME[n], "/r")
        check(e.validate() == [], f"example validates: {n}")
        check(bool(generate_script(e)) and bool(generate_clean_script(e)), f"both generators write {n}")

    try:
        import deepxde  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures
    with tempfile.TemporaryDirectory() as tmp:
        out = {}
        for gen in (generate_script, generate_clean_script):
            dd = os.path.join(tmp, gen.__name__)
            os.makedirs(dd)
            with open(os.path.join(dd, "run.py"), "w", encoding="utf-8") as fh:
                fh.write(gen(_cfg(dd, loss_weights_from_initial=5.0)))
            p = subprocess.run([sys.executable, "run.py"], cwd=dd, capture_output=True, text=True, timeout=600)
            check(p.returncode == 0, f"{gen.__name__} ran:\n{p.stderr[-1500:]}")
            out[gen.__name__] = (re.findall(r"Loss weights from the initial losses: (.*)", p.stdout),
                                 re.findall(r"L2=(\S+),", p.stdout))
        check(out["generate_script"][0] and out["generate_script"] == out["generate_clean_script"],
              f"same weights and result live and exported: {out}")
    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 75 TESTS PASSED")
    return 0


def test_round75_wave_helmholtz_bayes():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
