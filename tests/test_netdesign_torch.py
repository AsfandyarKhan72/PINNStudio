"""A designed network in PyTorch: every preset runs, the FNN and MsFFN presets compute exactly what
DeepXDE's FNN and PINNStudio's _MsFFN compute, groups and shared blocks reuse weights, and the shape the
canvas states is the shape PyTorch gives."""
import os
import sys

import pytest

os.environ.setdefault("DDE_BACKEND", "pytorch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
dde = pytest.importorskip("deepxde")

from pinnstudio.core.codegen import _MSFFN_CODE  # noqa: E402
from pinnstudio.core.config import PINNConfig  # noqa: E402
from pinnstudio.core.netdesign import design as D  # noqa: E402
from pinnstudio.core.netdesign import presets as P  # noqa: E402
from pinnstudio.core.netdesign import runtime as R  # noqa: E402
from pinnstudio.core.netdesign import shapes as S  # noqa: E402
from pinnstudio.core.netdesign.layers import LAYERS  # noqa: E402
from pinnstudio.core.netdesign.params import ParamError  # noqa: E402


def build(design, ins, outs, act="tanh", wd=0.0):
    ns = {"dde": dde, "torch": torch}
    exec(R.net_source(design, ins, outs), ns)
    return ns["_DesignedNet"](ns["_DESIGN"], act, "Glorot uniform", wd)


def test_fnn_preset_is_deepxde_fnn():
    net = build(P.fnn([12, 12]), ["x", "t"], ["u"])
    ref = dde.nn.FNN([2, 12, 12, 1], "tanh", "Glorot uniform")
    mine = [m.lin for m in net.top.mods["s1"].layers] + [net.top.mods["h1"].lin]
    for a, b in zip(mine, ref.linears):
        a.weight.data.copy_(b.weight.data)
        a.bias.data.copy_(b.bias.data)
    x = torch.rand(40, 2)
    assert torch.allclose(net(x), ref(x), atol=1e-6)


def test_msffn_preset_is_pinnstudios_msffn():
    ns = {"dde": dde, "torch": torch}
    exec(_MSFFN_CODE, ns)
    ms = ns["_MsFFN"]([2, 16, 16, 1], "tanh", "Glorot uniform", [1.0, 5.0])
    d = P.msffn([16, 16], [1.0, 5.0])
    net = build(d, ["x", "t"], ["u"])
    four = [b for b in d["blocks"] if b["kind"] == "stack" and b["layers"][0]["type"] == "fourier"]
    for k, b in enumerate(four):
        net.top.mods[b["id"]].layers[0].B.copy_(getattr(ms, f"B_x{k}"))
    hid = next(b for b in d["blocks"] if b["title"] == "Hidden layers")
    for a, b in zip(net.top.mods[hid["id"]].layers, ms.linears):
        a.lin.weight.data.copy_(b.weight.data)
        a.lin.bias.data.copy_(b.bias.data)
    h = next(b for b in d["blocks"] if b["kind"] == "head")
    net.top.mods[h["id"]].lin.weight.data.copy_(ms.dense.weight.data)
    net.top.mods[h["id"]].lin.bias.data.copy_(ms.dense.bias.data)
    x = torch.rand(30, 2)
    assert torch.allclose(net(x), ms(x), atol=1e-6)
    assert any(k.endswith(".B") for k in net.state_dict()), "fixed Fourier matrices are saved with the weights"


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_runs_with_transforms_and_weight_decay(name):
    c = PINNConfig()
    c.layers = [2, 8, 8, 2]
    d = P.make(name, c, ["x", "t"], ["u", "v"])
    net = build(d, ["x", "t"], ["u", "v"], wd=1e-4)
    net.apply_feature_transform(lambda x: x * 2.0)
    net.apply_output_transform(lambda x, y: y * x[:, :1])
    x = torch.rand(5, 2, requires_grad=True)
    y = net(x)
    assert y.shape == (5, 2) and net.regularizer == ["l2", 1e-4]
    du = torch.autograd.grad(y[:, :1].sum(), x, create_graph=True)[0]
    assert du.shape == (5, 2), "derivatives with respect to the inputs, as a PDE residual needs"


def test_heads_put_outputs_in_order():
    d = P.pfnn([4], ["u", "v", "w"])
    net = build(d, ["x"], ["u", "v", "w"])
    for b in d["blocks"]:
        if b["kind"] == "head":
            lin = net.top.mods[b["id"]].lin
            lin.weight.data.zero_()
            lin.bias.data.fill_(float(["u", "v", "w"].index(b["outputs"][0])))
    assert net(torch.rand(3, 1))[0].tolist() == [0.0, 1.0, 2.0]


def test_shared_block_uses_the_same_weights():
    d = P.msffn([8, 8], [1.0, 2.0, 3.0])
    net = build(d, ["x"], ["u"])
    n_hidden = sum(1 for b in d["blocks"] if b["kind"] == "stack" and b["title"] == "Hidden layers")
    assert n_hidden == 1
    shared_params = [k for k in net.state_dict() if "Hidden" in k]
    assert shared_params == []      # modules are keyed by id; one hidden stack, called three times
    assert sum(1 for b in d["blocks"] if b["kind"] == "shared") == 2


def test_group_runs_its_inside():
    d = P.fnn([6])
    d["blocks"].append({"id": "g1", "kind": "group", "title": "Twice", "inner": {
        "blocks": [{"id": "pi1", "kind": "port_in", "port": "a"},
                   {"id": "s9", "kind": "stack", "layers": [{"type": "dense", "params": {"units": 6}}]},
                   {"id": "po1", "kind": "port_out"}],
        "edges": [{"src": "pi1", "dst": "s9"}, {"src": "s9", "dst": "po1"}]}})
    d["edges"] = [{"src": "in1", "dst": "s1"}, {"src": "s1", "dst": "g1", "port": "a"}, {"src": "g1", "dst": "h1"}]
    d = D.check(d)
    assert S.problems(d, ["x"], ["u"]) == []
    net = build(d, ["x"], ["u"])
    assert net(torch.rand(4, 1)).shape == (4, 1)
    assert any(k.startswith("top.mods.g1.mods.s9") for k in net.state_dict())


@pytest.mark.parametrize("layer", sorted(set(LAYERS) - {"custom"}))
def test_widths_the_canvas_states_are_what_pytorch_gives(layer):
    params = {"dense": {"units": 5}, "fourier": {"features": 3}, "periodic": {"harmonics": 2}}.get(layer, {})
    d = P.fnn([4])
    d["blocks"][1]["layers"].append({"type": layer, "params": params})
    d = D.check(d)
    widths, problems = S.infer(d, ["x", "t"], ["u"])
    assert problems == []
    net = build(d, ["x", "t"], ["u"])
    out = net.top.mods["s1"](torch.rand(3, 2))
    assert out.shape[1] == widths["s1"], layer


def test_custom_layer_runs_the_designs_code():
    d = P.fnn([4])
    d["blocks"][1]["layers"].append({"type": "custom", "params": {"class_name": "Twice", "kwargs": '{"k": 2}',
                                                                  "width": "same"}})
    d["code"] = ("class Twice(torch.nn.Module):\n    def __init__(self, k):\n        super().__init__()\n"
                 "        self.k = k\n\n    def forward(self, x):\n        return self.k * x\n")
    d = D.check(d)
    assert D.uses_code(d)
    net = build(d, ["x"], ["u"])
    assert net(torch.rand(2, 1)).shape == (2, 1)


def test_a_broken_design_is_refused_with_the_reason():
    d = P.fnn([4])
    d["blocks"][2]["outputs"] = ["u"]
    with pytest.raises(ParamError, match="no head gives output 'v'"):
        R.net_source(d, ["x"], ["u", "v"])


def test_pfnn_preset_is_deepxdes_pfnn():
    ref = dde.nn.PFNN([2, [8, 8], [8, 8], 2], "tanh", "Glorot uniform")
    d = P.pfnn([8, 8], ["u", "v"])
    net = build(d, ["x", "t"], ["u", "v"])
    for o, name in enumerate(["u", "v"]):
        branch = next(b for b in d["blocks"] if b["title"] == f"Branch {name}")
        head = next(b for b in d["blocks"] if b["title"] == f"Output {name}")
        for i, layer in enumerate(net.top.mods[branch["id"]].layers):
            layer.lin.weight.data.copy_(ref.layers[i][o].weight.data)
            layer.lin.bias.data.copy_(ref.layers[i][o].bias.data)
        net.top.mods[head["id"]].lin.weight.data.copy_(ref.layers[-1][o].weight.data)
        net.top.mods[head["id"]].lin.bias.data.copy_(ref.layers[-1][o].bias.data)
    x = torch.rand(20, 2)
    assert torch.allclose(net(x), ref(x), atol=1e-6)


def test_stmsffn_preset_is_pinnstudios_space_time_msffn():
    ns = {"dde": dde, "torch": torch}
    exec(_MSFFN_CODE, ns)
    ms = ns["_MsFFN"]([3, 16, 16, 1], "tanh", "Glorot uniform", [1.0, 4.0], [1.0, 2.0])
    d = P.stmsffn([16, 16], [1.0, 4.0], [1.0, 2.0], ["x", "y"])
    net = build(d, ["x", "y", "t"], ["u"])
    four = [b for b in d["blocks"] if b["kind"] == "stack" and b["layers"][0]["type"] == "fourier"]
    xs = [b for b in four if " x " in b["title"]]
    ts = [b for b in four if " t " in b["title"]]
    for k, b in enumerate(xs):
        net.top.mods[b["id"]].layers[0].B.copy_(getattr(ms, f"B_x{k}"))
    for k, b in enumerate(ts):
        net.top.mods[b["id"]].layers[0].B.copy_(getattr(ms, f"B_t{k}"))
    hid = next(b for b in d["blocks"] if b["title"] == "Hidden layers")
    for a, b in zip(net.top.mods[hid["id"]].layers, ms.linears):
        a.lin.weight.data.copy_(b.weight.data)
        a.lin.bias.data.copy_(b.bias.data)
    h = next(b for b in d["blocks"] if b["kind"] == "head")
    net.top.mods[h["id"]].lin.weight.data.copy_(ms.dense.weight.data)
    net.top.mods[h["id"]].lin.bias.data.copy_(ms.dense.bias.data)
    x = torch.rand(25, 3)
    assert torch.allclose(net(x), ms(x), atol=1e-6)


def test_float64_runs_in_float64():
    dde.config.set_default_float("float64")
    try:
        d = P.msffn([8, 8], [1.0])
        d["blocks"][1]["layers"].append({"type": "layer_norm", "params": {}})
        d["blocks"][1]["layers"].append({"type": "adaptive", "params": {}})
        net = build(D.check(d), ["x"], ["u"])
        assert all(p.dtype == torch.float64 for p in net.parameters())
        assert all(b.dtype == torch.float64 for b in net.buffers())
        assert net(torch.rand(3, 1, dtype=torch.float64)).dtype == torch.float64
    finally:
        dde.config.set_default_float("float32")


def test_a_custom_layer_that_gives_another_width_is_named():
    d = P.fnn([4])
    d["blocks"][1]["layers"].append({"type": "custom", "params": {"class_name": "Wide", "width": "same"}})
    d["code"] = ("class Wide(torch.nn.Module):\n    def forward(self, x):\n"
                 "        return torch.cat([x, x], dim=1)\n")
    net = build(D.check(d), ["x"], ["u"])
    with pytest.raises(RuntimeError, match="Hidden layers.*Wide gives"):
        net(torch.rand(2, 1))


def test_code_may_not_take_the_scripts_names():
    d = P.fnn([4])
    d["code"] = "model = None\n"
    with pytest.raises(ParamError, match="'model'"):
        D.check(d)
    d["code"] = "def _helper():\n    pass\n"
    with pytest.raises(ParamError, match="'_helper'"):
        D.check(d)
