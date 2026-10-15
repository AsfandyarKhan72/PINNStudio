#!/usr/bin/env python3
"""Causal training and gradient-norm loss balancing (Adaptive training page):
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
    assert w.causal_widget.isHidden() and w.balance_widget.isHidden()
    w.causal_cb.setChecked(True)
    w.balance_cb.setChecked(True)
    assert not w.causal_widget.isHidden() and not w.balance_widget.isHidden()
    w.causal_chunks.setValue(16)
    w.causal_eps.setValue(25.0)
    w.balance_every.setValue(500)
    w.balance_alpha.setValue(0.8)
    c = w._build_config()
    assert (c.causal_training, c.causal_chunks, c.causal_eps) == (True, 16, 25.0)
    assert (c.loss_balancing, c.loss_balancing_every, c.loss_balancing_alpha) == (True, 500, 0.8)
    assert c.validate() == [], c.validate()
    w2 = _window()
    w2._apply_config(PINNConfig(**json.loads(json.dumps(dataclasses.asdict(c)))))   # as saved to .pinn.json
    _settle()
    c2 = w2._build_config()
    assert (c2.causal_training, c2.causal_chunks, c2.causal_eps) == (True, 16, 25.0)
    assert (c2.loss_balancing, c2.loss_balancing_every, c2.loss_balancing_alpha) == (True, 500, 0.8)
    assert not w2.causal_widget.isHidden()
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
    c.loss_balancing = True
    c.loss_balancing_alpha = 1.0
    assert any("Loss balancing" in e for e in c.validate())


def test_scripts():
    w = _window("1D Allen-Cahn")
    for adapt in (0, None):          # plain training, then the template's own Time-Adaptive
        if adapt is not None:
            w.adapt_combo.setCurrentIndex(adapt)
        w.causal_cb.setChecked(True)
        w.balance_cb.setChecked(True)
        c = w._build_config()
        for gen in (generate_script, generate_clean_script):
            src = gen(c)
            ast.parse(src)
            assert "pde = _causal_wrap(pde)" in src
            assert "_CausalMonitor(1000)" in src and "_GradNormBalancer()" in src
            # the callbacks list is defined before it is used
            assert src.index("_GradNormBalancer()") < src.rindex("callbacks=")
        w = _window("1D Allen-Cahn")
    c = PINNConfig()
    assert _training_methods_defs(c) == ""
    assert "_CausalMonitor" not in generate_script(c) and "_GradNormBalancer" not in generate_script(c)
    w.close()


def test_generated_code_runs():
    """The generated causal wrapper and balancer on a tiny heat problem."""
    import numpy as np
    import torch
    import deepxde as dde

    c = PINNConfig()
    c.causal_training = True
    c.causal_chunks = 4
    c.causal_eps = 10.0
    c.loss_balancing = True
    c.loss_balancing_every = 5
    ns = {"dde": dde, "torch": torch, "np": np}

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - 0.1 * dy_xx
    ns["pde"] = pde
    exec(_training_methods_defs(c), ns)
    # weights: one per chunk, 1 for the first chunk, never increasing
    x = torch.rand(200, 2, requires_grad=True)
    net = dde.nn.FNN([2, 16, 1], "tanh", "Glorot normal")
    r = ns["pde"](x, net(x))
    assert r.shape == (200, 1)
    wts = ns["_CAUSAL"]["w"]
    assert wts.shape == (4,) and abs(float(wts[0]) - 1.0) < 1e-6
    assert bool((wts[1:] <= wts[:-1] + 1e-7).all())

    geom = dde.geometry.GeometryXTime(dde.geometry.Interval(0, 1), dde.geometry.TimeDomain(0, 1))
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on: on)
    ic = dde.icbc.IC(geom, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on: on)
    data = dde.data.TimePDE(geom, ns["pde"], [bc, ic], num_domain=200, num_boundary=20, num_initial=20)
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss_weights=[1.0, 1.0, 10.0])
    model.train(iterations=20, display_every=1000, verbose=0,
                callbacks=[ns["_CausalMonitor"](10), ns["_GradNormBalancer"]()])
    bw = ns["_BALANCE"]["w"]
    assert bw is not None and len(bw) == 3 and all(np.isfinite(bw)) and bw != [1.0, 1.0, 10.0]
    assert bw[2] > 5.0, bw          # started from the Loss Weights (10), not from 1
    assert model.loss_weights == bw
    assert not ns["_CAUSAL"]["frozen"]
    # L-BFGS: causal weights and loss weights held
    held = ns["_CAUSAL"]["w"].clone()
    dde.optimizers.set_LBFGS_options(maxiter=5)
    model.compile("L-BFGS", loss_weights=list(bw))
    model.train(display_every=1000, verbose=0,
                callbacks=[ns["_CausalMonitor"](10), ns["_GradNormBalancer"]()])
    assert ns["_CAUSAL"]["frozen"]
    assert torch.equal(ns["_CAUSAL"]["w"], held)
    assert ns["_BALANCE"]["w"] == bw


def test_cahn_hilliard_template_scaling():
    """mu's equation divided by gamma_2 and the network's mu scaled by 0.01;
    the scaling is switched off again for the next template; the inverse
    substitution reaches both 0.01s."""
    w = _window("1D Cahn-Hilliard")
    c = w._build_config()
    assert c.output_names == "u,mu" and c.ic_active == "True,False"
    assert c.pde_expressions.split("|")[1] == "(mu - 0.01*(u**3 - u) + 1e-6*du_xx)/0.01"
    assert c.output_transform_enabled and c.output_transform_mode == "scale"
    assert list(c.output_transform_scale) == [1.0, 0.01]
    w.radio_inverse.setChecked(True)
    _settle()
    assert w.pde_inputs[1].text() == "(mu - gamma_2*(u**3 - u) + gamma_1*du_xx)/gamma_2"
    w.radio_inverse.setChecked(False)
    w.radio_forward.setChecked(True)
    _settle()
    w.quick_examples_combo.setCurrentText("1D Allen-Cahn")
    _settle()
    c = w._build_config()
    assert not c.output_transform_enabled and list(c.output_transform_scale) == [1.0]
    w.close()


if __name__ == "__main__":
    test_cahn_hilliard_template_scaling()
    test_gui_round_trip()
    test_validation()
    test_scripts()
    test_generated_code_runs()
    print("PASSED")
