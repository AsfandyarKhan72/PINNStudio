"""DeepXDE (PyTorch) works out its L-BFGS chunk size once, when imported, and
set_LBFGS_options does not update it: a 30-iteration L-BFGS phase used to
run 1000. Both scripts keep it in step, so a phase runs what it says.
Also: an expression typed with np. (the default initial condition) is not
given a second np. or a second [:, 0]."""
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnstudio.core import codegen as cg
from pinnstudio.core.config import PINNConfig


def test_expressions_with_np_are_left_alone():
    f = cg._simplify_expr
    assert f("np.sin(np.pi * x[:, 0])") == "np.sin(np.pi * x[:, 0])"
    assert f("np.exp(-x)") == "np.exp(-x[:, 0])"
    assert f("sin(pi*x)") == "np.sin(np.pi*x[:, 0])"
    assert f("x*y", is_2d=True) == "x[:, 0]*x[:, 1]"


def test_short_lbfgs_phase_runs_its_iterations():
    c = PINNConfig()
    c.num_domain, c.num_boundary, c.num_initial, c.num_test = 150, 20, 20, 100
    c.optimizer_scheduler = True
    c.scheduler_phases = json.dumps([{"optimizer": "adam", "iterations": 20, "lr": 1e-3},
                                     {"optimizer": "lbfgs", "iterations": 15}])
    d = tempfile.mkdtemp()
    c.save_dir = d
    for gen in (cg.generate_script, cg.generate_clean_script):
        with open(os.path.join(d, "run.py"), "w") as fh:
            fh.write(gen(c))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
        p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=900, env=env)
        assert p.returncode == 0, p.stderr[-1500:]
        steps = [int(s) for s in re.findall(r"^(\d+)\s+\[", p.stdout, re.M)]
        assert steps and max(steps) <= 35, (gen.__name__, steps[-3:])


def test_training_method_checks():
    """MsFFN needs an even first width; weight decay can't go with PFNN or
    MsFFN; causal training isn't available for an ODE; RAR rounds train the
    unknowns of an inverse problem with the phases' optimizer and callbacks."""
    c = PINNConfig()
    c.network_type, c.layers = "MsFFN", [2, 21, 20, 1]
    assert any("even number" in e for e in c.validate())
    c.layers, c.weight_decay = [2, 20, 20, 1], 1e-4
    assert any("Weight decay can't" in e for e in c.validate())
    c = PINNConfig()
    c.problem_dim, c.causal_training = "ODE", True
    assert any("Causal training isn't available for an ODE" in e for e in c.validate())
    c = PINNConfig()
    c.problem_type, c.adapt_method = "Inverse", "RAR"
    c.optimizer_scheduler = True
    c.scheduler_phases = json.dumps([{"optimizer": "adam", "iterations": 100, "lr": 1e-4}])
    t = cg.generate_script(c)
    assert "external_trainable_variables=_rar_ext" in t and "callbacks=list(_train_cbs)" in t
    assert 'model.compile("adam", lr=0.0001' in t
    assert "exists('/tmp')" not in t
