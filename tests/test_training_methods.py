#!/usr/bin/env python3
"""Causal training and per-point weights, RBA and SA-PINN (Adaptive training page):
the GUI options round-trip through a saved config, validation, both
generated scripts, and the generated code itself on a tiny DeepXDE model.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_training_methods.py
"""
import ast
import dataclasses
import json
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication

from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import (_training_methods_defs, generate_clean_script,
                                     generate_script)
from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication(sys.argv)


def _settle(secs=0.2):
    t = time.time()
    while time.time() - t < secs:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()


def _window(example="1D Burgers"):
    w = MainWindow()
    w.show()
    _settle()
    w.quick_examples_combo.setCurrentText(example)
    _settle()
    return w


def test_gui_round_trip():
    w = _window()
    assert w.causal_widget.isHidden() and w.rba_widget.isHidden() and w.sa_widget.isHidden()
    w.causal_cb.setChecked(True)
    assert not w.causal_widget.isHidden()
    w.causal_chunks.setValue(16)
    assert w.causal_anneal_cb.isChecked() and not w.causal_eps.isEnabled()   # the paper's schedule by default
    w.causal_anneal_cb.setChecked(False)
    assert w.causal_eps.isEnabled()
    w.causal_eps.setValue(25.0)
    assert w.causal_pw_warn.isHidden()
    w._set_combo_data(w.point_weights_combo, "RBA")
    assert not w.causal_pw_warn.isHidden()           # causal + per-point weights: warned
    assert not w.rba_widget.isHidden() and w.sa_widget.isHidden()
    w.rba_gamma.setValue(0.99)
    w.rba_eta.setValue(0.02)
    w._set_combo_data(w.point_weights_combo, "SA-PINN")
    assert w.rba_widget.isHidden() and not w.sa_widget.isHidden()
    w.sa_lr.setValue(0.01)
    assert not w.sa_normalize_cb.isChecked()         # the paper's SA-PINN by default
    w.sa_normalize_cb.setChecked(True)
    assert not w.pw_lbfgs_widget.isHidden()
    w._set_combo_data(w.pw_lbfgs_combo, "keep")
    c = w._build_config()
    assert (c.causal_training, c.causal_chunks, c.causal_eps, c.causal_anneal) == (True, 16, 25.0, False)
    assert (c.point_weights, c.rba_gamma, c.rba_eta, c.sa_lr) == ("SA-PINN", 0.99, 0.02, 0.01)
    assert c.sa_normalize is True and c.point_weights_lbfgs == "keep"
    assert c.validate() == [], c.validate()
    w2 = _window()
    w2._apply_config(PINNConfig(**json.loads(json.dumps(dataclasses.asdict(c)))))   # as saved to .pinn.json
    _settle()
    c2 = w2._build_config()
    assert (c2.causal_training, c2.causal_chunks, c2.causal_eps, c2.causal_anneal) == (True, 16, 25.0, False)
    assert w2.causal_eps.isEnabled() and not w2.causal_pw_warn.isHidden()
    assert (c2.point_weights, c2.rba_gamma, c2.rba_eta, c2.sa_lr) == ("SA-PINN", 0.99, 0.02, 0.01)
    assert c2.sa_normalize is True and c2.point_weights_lbfgs == "keep"
    assert not w2.causal_widget.isHidden() and not w2.sa_widget.isHidden()
    for x in (w, w2):
        x.close()


def test_validation():
    c = PINNConfig()
    c.causal_training = True
    c.steady_state = True
    assert any("Causal training" in e for e in c.validate())
    c.steady_state = False
    c.causal_chunks = 1
    c.causal_eps = 0.0
    errs = c.validate()
    assert any("time chunks" in e for e in errs) and any("epsilon" in e for e in errs)
    c = PINNConfig()
    c.point_weights = "RBA"
    c.rba_gamma = 1.0
    c.rba_eta = 0.0
    errs = c.validate()
    assert any("gamma" in e for e in errs) and any("eta" in e for e in errs)
    c = PINNConfig()
    c.point_weights = "SA-PINN"
    c.sa_lr = 0.0
    assert any("SA-PINN" in e for e in c.validate())
    c.point_weights = "bogus"
    assert any("unknown method" in e for e in c.validate())


def test_scripts():
    w = _window("1D Allen-Cahn")
    for adapt in (0, None):          # plain training, then the template's own Time-Adaptive
        if adapt is not None:
            w.adapt_combo.setCurrentIndex(adapt)
        w.causal_cb.setChecked(True)
        w._set_combo_data(w.point_weights_combo, "RBA")
        c = w._build_config()
        for gen in (generate_script, generate_clean_script):
            src = gen(c)
            ast.parse(src)
            assert "pde = _causal_wrap(pde)" in src and "pde = _pw_wrap(pde)" in src
            assert "_CausalMonitor(1000)" in src and "_PointWeights(1000)" in src
            # the callbacks list is defined before it is used
            assert src.index("_PointWeights(1000)") < src.rindex("callbacks=")
        w = _window("1D Allen-Cahn")
    c = PINNConfig()
    assert _training_methods_defs(c) == ""
    assert "_CausalMonitor" not in generate_script(c) and "_PointWeights" not in generate_script(c)
    w.close()


def _tiny(c):
    import numpy as np
    import torch
    import deepxde as dde
    ns = {"dde": dde, "torch": torch, "np": np}

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - 0.1 * dy_xx
    ns["pde"] = pde
    exec(_training_methods_defs(c), ns)
    geom = dde.geometry.GeometryXTime(dde.geometry.Interval(0, 1), dde.geometry.TimeDomain(0, 1))
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on: on)
    ic = dde.icbc.IC(geom, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on: on)
    data = dde.data.TimePDE(geom, ns["pde"], [bc, ic], num_domain=200, num_boundary=20,
                            num_initial=20, num_test=150)
    net = dde.nn.FNN([2, 16, 1], "tanh", "Glorot normal")
    return ns, dde, torch, np, dde.Model(data, net)


def test_generated_code_runs():
    """The generated causal wrapper and per-point weights on a tiny heat problem."""
    c = PINNConfig()
    c.causal_training = True
    c.causal_chunks = 4
    c.causal_eps = 10.0
    c.causal_anneal = False
    c.point_weights = "RBA"
    ns, dde, torch, np, model = _tiny(c)
    # causal weights: one per chunk, 1 for the first chunk, never increasing
    x = torch.rand(200, 2, requires_grad=True)
    r = ns["_causal_wrap"](lambda x, y: x[:, 1:2] * 0 + y)(x, model.net(x))
    wts = ns["_CAUSAL"]["w"]
    assert r.shape == (200, 1) and wts.shape == (4,) and abs(float(wts[0]) - 1.0) < 1e-6
    assert bool((wts[1:] <= wts[:-1] + 1e-7).all())

    model.compile("adam", lr=1e-3)
    cbs = [ns["_CausalMonitor"](10), ns["_PointWeights"](10)]
    model.train(iterations=20, display_every=1000, verbose=0, callbacks=cbs)
    lam = ns["_PW"]["lam"]
    n_train = len(model.train_state.X_train)
    assert lam is not None and tuple(lam.shape) == (n_train, 1)
    # 20 updates from 1: lambda = 0.999^20 + 0.01 * sum(...) -- some above, all below 1.2
    assert float(lam.min()) > 0.97 and float(lam.max()) <= 0.999 ** 20 + 0.01 * 20 + 1e-6
    assert float(lam.max()) > 0.999 ** 20 + 0.001
    # the test points (a different set) are not weighted: their loss is the plain one
    assert not ns["_pw_is_train"](torch.as_tensor(model.data.test_x))
    held_c, held_l = ns["_CAUSAL"]["w"].clone(), lam.clone()
    dde.optimizers.set_LBFGS_options(maxiter=5)
    model.compile("L-BFGS")
    model.train(display_every=1000, verbose=0, callbacks=cbs)
    assert ns["_CAUSAL"]["frozen"] and ns["_PW"]["frozen"]
    assert torch.equal(ns["_CAUSAL"]["w"], held_c) and torch.equal(ns["_PW"]["lam"], held_l)
    # default "plain": during L-BFGS the training residual is not weighted
    assert not ns["_PW"]["keep"]
    X = torch.as_tensor(model.train_state.X_train, dtype=torch.float32).requires_grad_(True)
    plain = lambda x, y: x[:, 0:1] * 0 + 1.0
    assert torch.equal(ns["_pw_wrap"](plain)(X, model.net(X)), torch.ones(len(X), 1))
    ns["_PW"]["keep"] = True        # "keep": the held weights apply
    assert torch.allclose(ns["_pw_wrap"](plain)(X, model.net(X)), held_l[:, 0:1])


def test_causal_epsilon_schedule():
    """The paper's schedule: epsilon starts at 0.01 and moves up once every
    chunk weight is above 0.99; a new Adam run starts it over; L-BFGS holds it."""
    c = PINNConfig()
    c.causal_training = True
    c.causal_chunks = 4
    assert c.causal_anneal                    # on by default
    ns, dde, torch, np, model = _tiny(c)
    C = ns["_CAUSAL"]
    assert C["sched"] == [0.01, 0.1, 1.0, 10.0, 100.0] and C["eps"] == 0.01
    model.compile("adam", lr=1e-3)
    cb = ns["_CausalMonitor"](1000)
    model.train(iterations=30, display_every=1000, verbose=0, callbacks=[cb])
    assert C["k"] >= 1 and C["eps"] == C["sched"][C["k"]]
    k = C["k"]
    dde.optimizers.set_LBFGS_options(maxiter=5)
    model.compile("L-BFGS")
    model.train(display_every=1000, verbose=0, callbacks=[cb])
    assert C["frozen"] and C["k"] == k        # not advanced during L-BFGS
    model.compile("adam", lr=1e-3)
    model.train(iterations=0, display_every=1000, verbose=0, callbacks=[cb])
    assert C["k"] == 0 and C["eps"] == 0.01   # a new Adam run (next time window) starts over
    c.causal_anneal = False
    assert '"sched": None' in _training_methods_defs(c)


def test_sa_pinn_ascends():
    for norm in (False, True):
        c = PINNConfig()
        c.point_weights = "SA-PINN"
        c.sa_lr = 0.01
        c.sa_normalize = norm
        ns, dde, torch, np, model = _tiny(c)
        model.compile("adam", lr=1e-3)
        model.train(iterations=30, display_every=1000, verbose=0, callbacks=[ns["_PointWeights"](10)])
        lam = ns["_PW"]["lam"]
        assert lam.requires_grad and float(lam.min()) >= 0.0
        if norm:      # mean 1, but no longer uniform
            assert abs(float(lam.mean()) - 1.0) < 1e-4 and float(lam.max() - lam.min()) > 1e-3
        else:         # gradient ascent: the weights only grow from 1 (d loss / d lambda >= 0)
            assert float(lam.min()) >= 1.0 - 1e-6 and float(lam.max()) > 1.05


def test_cahn_hilliard_template_scaling():
    """The template's best setup (training methods study): mu's equation
    divided by gamma_2 and the network's mu scaled by 0.01, the exact initial
    condition u = IC + t*N, periodic inputs, SA-PINN (the paper's: no
    normalization, held during L-BFGS); none of it
    carries over to the next template; the inverse substitution reaches both
    0.01s of the equation."""
    w = _window("1D Cahn-Hilliard")
    c = w._build_config()
    assert c.output_names == "u,mu" and c.ic_active == "False,False" and c.ic_hard == "True,False"
    assert c.periodic_inputs and not c.input_transform_enabled
    assert c.pde_expressions.split("|")[1] == "(mu - 0.01*(u**3 - u) + 1e-6*du_xx)/0.01"
    assert c.output_transform_enabled and c.output_transform_mode == "scale"
    assert list(c.output_transform_scale) == [1.0, 0.01]
    assert (c.point_weights, c.point_weights_lbfgs, c.sa_normalize) == ("SA-PINN", "keep", False)
    assert not w.ic_active[0].isEnabled() and not w.ic_active[0].isChecked()
    assert not c.time_adaptive and c.validate() == [], c.validate()
    from pinnstudio.core.codegen import _effective_config
    e = _effective_config(c)
    assert e.output_transform_mode == "formula" and e.output_transform_formula == "(-cos(2*pi*x)) + t*u|0.01*mu"
    assert e.input_transform_enabled and e.input_feature_formula == "cos(pi*x); sin(pi*x); t"
    for gen in (generate_script, generate_clean_script):
        src = gen(c)
        ast.parse(src)
        assert "pde = _pw_wrap(pde)" in src
        assert "torch.cat([(-torch.cos(2*np.pi*x)) + t*u, 0.01*mu], dim=1)" in src
        assert "torch.cat([torch.cos(np.pi*x), torch.sin(np.pi*x), t], dim=1)" in src
    assert not w.exact_inputs_note.isHidden() and not w.exact_outputs_note.isHidden()
    w.radio_inverse.setChecked(True)
    _settle()
    assert w.pde_inputs[1].text() == "(mu - gamma_2*(u**3 - u) + gamma_1*du_xx)/gamma_2"
    w.radio_inverse.setChecked(False)
    w.radio_forward.setChecked(True)
    _settle()
    for nxt in ("1D Allen-Cahn", "2D Heat"):
        w.quick_examples_combo.setCurrentText("1D Cahn-Hilliard")
        _settle()
        w.quick_examples_combo.setCurrentText(nxt)
        _settle()
        c = w._build_config()
        assert not c.output_transform_enabled and not c.input_transform_enabled, nxt
        assert c.point_weights == "None" and not (c.output_transform_formula or "").strip("|"), nxt
        assert not c.periodic_inputs and "True" not in (c.ic_hard or ""), nxt
        assert all(cb.isEnabled() for cb in w.ic_active), nxt
    w.close()


def test_exact_constraints():
    """Exact IC / periodicity: the formulas they generate, validation."""
    from pinnstudio.core.codegen import _exact_constraints
    c = PINNConfig()
    c.problem_dim, c.num_outputs, c.output_names = "2D", 1, "u"
    c.x_min, c.x_max, c.y_min, c.y_max, c.t_min, c.t_max = 0.0, 1.0, 0.0, 2.0, 0.0, 10.0
    c.ic_expressions = "sin(4*pi*x)*cos(4*pi*y)"
    c.ic_hard, c.ic_active = "True", "False"
    c.periodic_inputs = True
    c.custom_bc_json = json.dumps([{"type": "periodic", "component": 0, "axis": "x", "location": "left"}])
    assert c.validate() == [], c.validate()
    e = _exact_constraints(c)
    assert e.output_transform_formula == "(sin(4*pi*x)*cos(4*pi*y)) + (t/10)*u"
    assert e.input_feature_formula == "cos(2*pi*x); sin(2*pi*x); y; t"      # only x is periodic
    c.t_min = 1.0
    assert _exact_constraints(c).output_transform_formula.endswith("((t - 1)/9)*u")
    # what can't be combined
    bad = dataclasses.replace(c, time_adaptive=True)
    assert any("time-adaptive" in m for m in bad.validate())
    bad = dataclasses.replace(c, input_transform_enabled=True)
    assert any("Exact periodicity" in m for m in bad.validate())
    bad = dataclasses.replace(c, output_transform_enabled=True, output_transform_mode="formula",
                              output_transform_formula="x*u")
    assert any("output transform formula" in m for m in bad.validate())
    bad = dataclasses.replace(c, ic_expressions="np.sin(np.pi * x[:, 0])")
    assert any("as a formula" in m for m in bad.validate())
    bad = dataclasses.replace(c, custom_bc_json="[]")
    assert any("no boundary condition is periodic" in m for m in bad.validate())
    off = dataclasses.replace(c, ic_hard="False", periodic_inputs=False)
    assert _exact_constraints(off) is off


def test_exact_ic_toggle_keeps_weights_in_step():
    """Ticking Exact removes the IC loss: its weight row leaves every training
    phase's weights and the other weights keep their values; unticking it
    brings the IC row back with weight 100. (The 3D Allen-Cahn template
    starts with the IC as a loss term, weight 100.)"""
    import json as _j
    w = _window("3D Allen-Cahn (Wight & Zhao)")
    c = w._build_config()
    assert c.ic_active == "True"
    assert [p["weights"] for p in _j.loads(c.scheduler_phases)] == ["1.0,1.0,1.0,1.0,100.0"] * 2
    w.weight_widgets["pde_0"].setValue(3.0)
    w.ic_hard[0].setChecked(True)
    _settle()
    c = w._build_config()
    assert [p["weights"] for p in _j.loads(c.scheduler_phases)] == ["3.0,1.0,1.0,1.0"] * 2
    w.ic_hard[0].setChecked(False)
    _settle()
    c = w._build_config()
    assert c.ic_active == "True"
    assert [p["weights"] for p in _j.loads(c.scheduler_phases)] == ["3.0,1.0,1.0,1.0,100.0"] * 2
    w.close()


def test_point_weights_reset_on_new_points():
    """Per-point weights belong to points: new training points start over."""
    c = PINNConfig()
    c.point_weights = "RBA"
    ns, dde, torch, np, model = _tiny(c)
    model.compile("adam", lr=1e-3)
    cb = ns["_PointWeights"](1000)
    model.train(iterations=5, display_every=1000, verbose=0, callbacks=[cb])
    assert ns["_PW"]["lam"] is not None and float(ns["_PW"]["lam"].max()) > 1.0 - 1e-6
    model.data.resample_train_points()
    model.train_state.set_data_train(*model.data.train_next_batch())
    cb._mark()
    assert ns["_PW"]["lam"] is None          # reset; re-created at 1 on the next step
    model.train(iterations=2, display_every=1000, verbose=0, callbacks=[cb])
    assert ns["_PW"]["lam"] is not None


if __name__ == "__main__":
    test_exact_constraints()
    test_point_weights_reset_on_new_points()
    test_cahn_hilliard_template_scaling()
    test_gui_round_trip()
    test_validation()
    test_scripts()
    test_generated_code_runs()
    test_sa_pinn_ascends()
    print("PASSED")
