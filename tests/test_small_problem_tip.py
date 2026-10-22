"""A small 1D problem training on the GPU gets a tip in the Training Log
that the CPU is often faster; larger problems and an explicit device choice
get none, and nothing about the run is changed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnstudio.core import deepxde_examples as dx  # noqa: E402
from pinnstudio.core.codegen import generate_script  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_data")
EX = {e.name: e for e in dx.EXAMPLES}


def _script(name, **over):
    cfg = dx.build_config(EX[name], REF)
    for k, v in over.items():
        setattr(cfg, k, v)
    s = generate_script(cfg)
    compile(s, name, "exec")
    return s


def test_small_1d_problem_gets_the_tip():
    assert "[Tip] A small 1D problem" in _script("Heat equation")


def test_2d_problem_and_large_1d_problem_get_none():
    assert "[Tip]" not in _script("Kovasznay flow")
    assert "[Tip]" not in _script("Heat equation", num_domain=20000)


def test_explicit_device_choice_gets_none():
    assert "[Tip]" not in _script("Heat equation", train_device="cpu")
    assert "[Tip]" not in _script("Heat equation", train_device="gpu")
