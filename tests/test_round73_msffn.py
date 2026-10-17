#!/usr/bin/env python3
"""
Round 73 -- multi-scale Fourier feature networks (MsFFN, STMsFFN).

Network type MsFFN / STMsFFN with "Fourier scales (σ)" (and "Time scales"
for STMsFFN): PINNStudio's own implementation of DeepXDE's dde.nn.MsFFN /
STMsFFN (Wang, Wang & Perdikaris 2021), so it works with the released
DeepXDE and restores from a checkpoint with the same random features
(they're buffers in the state dict). Checks: GUI fields and config
round-trip, validation, the network's shape and features, checkpoint
reload, both generators train it, and the DeepXDE multi-scale Poisson
example.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round73_msffn.py
"""
import json
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
from pinnstudio.core.codegen import (_net_construction_helper_code, _parse_sigmas, generate_clean_script,
                                     generate_script)
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow


def _cfg(d=".", **kw):
    c = PINNConfig()
    base = dict(problem_dim="1D", steady_state=True, geometry_type="Interval", x_min=0.0, x_max=1.0, num_initial=0,
                pde_expressions="du_xx + pi**2*sin(pi*x)", network_type="MsFFN", msffn_sigmas="1, 5",
                custom_bc_json=json.dumps([{"type": "dirichlet", "location": "True", "value": "0", "component": 0}]),
                layers=[1, 32, 32, 1], iterations=1500, optimizer2="none", iterations2=0, optimizer_scheduler=False,
                num_domain=64, num_boundary=2, num_test=50, ea_ref_mode="formula", ea_ref_formula="sin(pi*x)",
                save_dir=d, plot_resolution=50)
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

    check(_parse_sigmas("1, 10") == [1.0, 10.0] and _parse_sigmas("") == [], "scales parse")
    check(_cfg().validate() == [], "a MsFFN config validates")
    check(any("positive" in e for e in _cfg(msffn_sigmas="a").validate()), "bad scales are reported")
    check(any("time-dependent" in e for e in _cfg(network_type="STMsFFN").validate()), "STMsFFN needs time")
    check("class _MsFFN" in generate_script(_cfg()) and "class _MsFFN" in generate_clean_script(_cfg()),
          "both generators embed the network")
    check("class _MsFFN" not in generate_script(_cfg(network_type="FNN")), "FNN scripts unchanged")

    w = MainWindow()
    w.network_type_combo.setCurrentText("MsFFN")
    w.msffn_sigmas_edit.setText("1, 20")
    c = w._build_config()
    check((c.network_type, c.msffn_sigmas) == ("MsFFN", "1, 20"), "GUI -> config")
    c.msffn_sigmas = "2, 4"
    w._apply_config(c)
    check(w.msffn_sigmas_edit.text() == "2, 4", "config -> GUI")

    ex = dx.build_config(dx.BY_NAME["Poisson 1D – multi-scale Fourier features"], "/r")
    check(ex.network_type == "MsFFN" and ex.layers == [1, 100, 100, 100, 1], "DeepXDE multi-scale example")

    try:
        import deepxde as dde
        import torch
    except ImportError:
        print("SKIPPED (network/end-to-end part): deepxde not installed.")
        return failures
    import numpy as np
    for kind, n_in, st in (("MsFFN", 1, ""), ("STMsFFN", 2, "1, 3")):
        ns = {"dde": dde, "torch": torch, "np": np}
        exec(_net_construction_helper_code(kind, "1, 10", st or "1, 10"), ns)
        net = ns["_make_net"]([n_in, 20, 20, 1], "tanh", "Glorot uniform")
        x = torch.rand(7, n_in)
        y = net(x)
        check(tuple(y.shape) == (7, 1), f"{kind}: output shape")
        bufs = [k for k in net.state_dict() if k.startswith("B_")]
        check(len(bufs) == (2 if kind == "MsFFN" else 4), f"{kind}: Fourier matrices are saved buffers {bufs}")
        check(not any(k.startswith("B_") for k, _p in net.named_parameters()), f"{kind}: features are not trained")
        torch.manual_seed(123)
        net2 = ns["_make_net"]([n_in, 20, 20, 1], "tanh", "Glorot uniform")
        net2.load_state_dict(net.state_dict())
        check(torch.allclose(net2(x), y), f"{kind}: a reloaded checkpoint gives the same network")

    with tempfile.TemporaryDirectory() as tmp:
        l2 = {}
        for gen in (generate_script, generate_clean_script):
            d = os.path.join(tmp, gen.__name__)
            os.makedirs(d)
            with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as fh:
                fh.write(gen(_cfg(d)))
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
            check(p.returncode == 0, f"{gen.__name__} ran:\n{p.stderr[-1500:]}")
            l2[gen.__name__] = [float(v) for v in re.findall(r"L2=(\S+),", p.stdout)]
        check(l2["generate_script"] and l2["generate_script"][0] < 0.05 and l2["generate_script"] == l2["generate_clean_script"],
              f"MsFFN trains, live and export agree: {l2}")
        d = os.path.join(tmp, "st")
        os.makedirs(d)
        c = _cfg(d, network_type="STMsFFN", msffn_sigmas="1", msffn_sigmas_t="1", steady_state=False, t_min=0.0, t_max=1.0,
                 pde_expressions="du_t - 0.1*du_xx", ic_expressions="sin(pi*x)", num_initial=20, iterations=200,
                 layers=[2, 16, 16, 1], ea_ref_formula="exp(-0.1*pi**2*t)*sin(pi*x)")
        c.ic_expression = c.ic_expressions
        with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as fh:
            fh.write(generate_script(c))
        p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
        check(p.returncode == 0, f"STMsFFN runs:\n{p.stderr[-1500:]}")
    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 73 MsFFN TESTS PASSED")
    return 0


def test_round73_msffn():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
