"""A designed network in the scripts PINNStudio writes: Solve's script and the exported one parse for
every preset (forward, inverse and time-adaptive), the training and restore scripts build identical
parameters, the design is recorded for Restore, and a broken design is refused before anything runs.
The last test trains a designed network for real (marked slow in tests/conftest.py)."""
import ast
import dataclasses
import json
import os
import sys

import pytest

os.environ.setdefault("DDE_BACKEND", "pytorch")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio.core import codegen as CG  # noqa: E402
from pinnstudio.core import deepxde_examples as DX  # noqa: E402
from pinnstudio.core.netdesign import design as D  # noqa: E402
from pinnstudio.core.netdesign import presets as P  # noqa: E402
from pinnstudio.core.netdesign import problem as PR  # noqa: E402

REF = os.path.join(ROOT, "reference_data")


def _example(name, inverse=False):
    ex = next(e for e in DX.EXAMPLES if e.name == name)
    return DX.build_config(ex, REF, inverse=inverse)


def _designed(cfg, preset="msffn"):
    ins, outs = PR.network_io(cfg)
    d = P.make(preset, cfg, ins, outs)
    return dataclasses.replace(cfg, network_type="Designed", network_design=D.to_json(d))


@pytest.mark.parametrize("preset", list(P.PRESETS))
@pytest.mark.parametrize("example, inverse, ta", [
    ("Diffusion equation", False, False),
    ("Diffusion equation", False, True),
    ("Inverse: diffusion equation", False, False),
    ("Kovasznay flow", False, False),
])
def test_scripts_parse(preset, example, inverse, ta):
    cfg = _example(example, inverse)
    if ta:
        cfg = dataclasses.replace(cfg, time_adaptive=True)
    if preset == "stmsffn" and cfg.steady_state:
        return
    cfg = _designed(cfg, preset)
    assert not [e for e in (cfg.validate() or []) if "etwork" in e], cfg.validate()
    for gen in (CG.generate_script, CG.generate_clean_script):
        src = gen(cfg)
        ast.parse(src)
        assert "class _DesignedNet(dde.nn.NN)" in src and "_network_type = 'Designed'" in src


def _net_from_helper(helper_src):
    import deepxde as dde
    import torch
    ns = {"dde": dde, "torch": torch, "np": __import__("numpy")}
    exec(helper_src, ns)
    return ns["_make_net"]


def test_training_and_restore_build_the_same_parameters():
    cfg = _designed(_example("Diffusion equation"), "modified_mlp")
    eff = CG._effective_config(cfg)
    ins, outs = PR.names_effective(eff)
    train = _net_from_helper(CG._net_construction_helper_code("Designed", design=cfg.network_design, io=(ins, outs)))
    model_config = {"network_type": "Designed", "network_design": cfg.network_design,
                    "network_inputs": ins, "network_outputs": outs}
    restore = _net_from_helper(CG._net_construction_helper_code(
        model_config["network_type"], design=model_config["network_design"],
        io=(model_config["network_inputs"], model_config["network_outputs"])))
    a = train(eff.layers, "tanh", "Glorot uniform")
    b = restore([2, 64, 1], "tanh", "Glorot uniform")      # the layers a restore passes do not matter
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(sa[k].shape == sb[k].shape for k in sa)
    b.load_state_dict(sa)


def test_model_and_step_configs_record_the_design():
    cfg = dataclasses.replace(_designed(_example("Diffusion equation"), "resnet"), time_adaptive=True)
    src = CG.generate_script(cfg)
    assert '"network_design":' in src and '"network_inputs":' in src
    i = src.index('"step": step_i + 1')
    block = src[i:i + 2500]
    assert '"network_design"' in block and '"network_type"' in block and '"network_inputs"' in block


def test_a_broken_design_is_refused_by_validate():
    cfg = _designed(_example("Diffusion equation"), "fnn")
    d = json.loads(cfg.network_design)
    d["blocks"] = [b for b in d["blocks"] if b["kind"] != "head"]
    d["edges"] = [e for e in d["edges"] if e["dst"] != "h1"]
    bad = dataclasses.replace(cfg, network_design=json.dumps(d))
    errs = [e for e in bad.validate() if "Network design" in e]
    assert errs and ("goes nowhere" in errs[0] or "Head" in errs[0]), bad.validate()
    empty = dataclasses.replace(cfg, network_design="")
    assert any("Network design" in e for e in empty.validate())


def test_odes_and_input_formulas_name_the_inputs():
    ode = _example("ODE system")
    assert PR.network_io(ode)[0] == ["t"]
    per = dataclasses.replace(_example("Diffusion equation"), input_transform_enabled=True,
                              input_transform_mode="formula", input_feature_formula="cos(pi*x); sin(pi*x); t")
    names = PR.network_io(per)[0]
    forms = PR.feature_formulas(per)
    assert names[-1] == "t" and names[0] == "f1" and "cos" in forms["f1"], (names, forms)


@pytest.mark.slow
def test_a_designed_network_trains_saves_and_restores(tmp_path):
    """Poisson 1D with a designed Fourier network: a short real run through Solve's own runner."""
    from pinnstudio.core.runner import run_pinn
    cfg = _example("Poisson 1D – Dirichlet")
    cfg = _designed(cfg, "msffn")
    cfg = dataclasses.replace(cfg, save_dir=str(tmp_path), optimizer_scheduler=True,
                              scheduler_phases=json.dumps([{"optimizer": "adam", "iterations": 300, "lr": 0.001,
                                                            "loss": "MSE"}]),
                              iterations=300, optimizer2="none", ea_files="[]", plot_type="Line")
    lines = []
    assert run_pinn(cfg, on_output=lines.append) == "DONE", "\n".join(lines[-40:])
    run = next(os.path.join(r, d) for r, ds, _f in os.walk(tmp_path) for d in ds if d == "solution_results")
    mc = json.load(open(os.path.join(run, "model_config.json"), encoding="utf-8"))
    assert mc["network_type"] == "Designed" and mc["network_inputs"] == ["x"]
    import glob
    ckpt = sorted(glob.glob(os.path.join(os.path.dirname(run), "**", "*.pt"), recursive=True))
    assert ckpt, os.listdir(os.path.dirname(run))
    import torch
    net = _net_from_helper(CG._net_construction_helper_code(
        mc["network_type"], design=mc["network_design"], io=(mc["network_inputs"], mc["network_outputs"])))(
        mc["layers"], mc["activation"], "Glorot uniform")
    state = torch.load(ckpt[-1], map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("model_state_dict", state))
