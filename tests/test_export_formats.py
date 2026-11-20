"""Solution export as VTK for ParaView (in the order VTK expects, with a .pvd of the times) and the
trained network as TorchScript and torch.export: a short real run of a 3D problem."""
import dataclasses
import glob
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DDE_BACKEND", "pytorch")

from pinnstudio.core import deepxde_examples as DX  # noqa: E402
from pinnstudio.core.problem_file import REFERENCE_DATA_DIR  # noqa: E402


def _read_vtk(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    dims = [int(v) for v in lines[4].split()[1:]]
    org = [float(v) for v in lines[5].split()[1:]]
    spc = [float(v) for v in lines[6].split()[1:]]
    i0 = lines.index("LOOKUP_TABLE default") + 1
    n = dims[0] * dims[1] * dims[2]
    return dims, org, spc, np.array([float(v) for v in lines[i0:i0 + n]])


@pytest.mark.slow
def test_vtk_and_network_exports(tmp_path):
    from pinnstudio.core.runner import run_pinn
    ex = next(e for e in DX.EXAMPLES if e.name == "Beltrami flow")
    c = DX.build_config(ex, REFERENCE_DATA_DIR)
    c = dataclasses.replace(c, iterations=20, save_dir=str(tmp_path), ea_files="[]",
                            scheduler_phases=json.dumps([{"optimizer": "adam", "iterations": 20, "lr": 0.001,
                                                          "loss": "MSE"}]),
                            export_enabled=True, export_format="both", export_grid_size=7, export_t_steps=3,
                            export_torchscript=True, num_domain=100, num_boundary=40, num_initial=40)
    log = []
    assert run_pinn(c, on_output=log.append) == "DONE", "\n".join(log[-30:])
    data = glob.glob(os.path.join(str(tmp_path), "**", "solution_data"), recursive=True)[0]
    csv = np.loadtxt(os.path.join(data, "solution_t0.5000.txt"), delimiter=",", skiprows=1)
    dims, org, spc, u = _read_vtk(os.path.join(data, "solution_t0.5000.vtk"))
    assert dims == [7, 7, 7]
    for row in csv[::11]:
        i, j, k = (round((row[a] - org[a]) / spc[a]) for a in range(3))
        assert abs(u[i + dims[0] * (j + dims[1] * k)] - row[4]) < 1e-6
    pvd = open(os.path.join(data, "solution.pvd"), encoding="utf-8").read()
    assert pvd.count("<DataSet ") == 3
    sol = os.path.dirname(data)
    import torch
    ts = torch.jit.load(os.path.join(sol, "model_torchscript.pt"))
    assert ts(torch.rand(5, 4)).shape == (5, 4)
    if os.path.exists(os.path.join(sol, "model_exported.pt2")):
        ep = torch.export.load(os.path.join(sol, "model_exported.pt2"))
        assert ep.module()(torch.rand(9, 4)).shape == (9, 4)
