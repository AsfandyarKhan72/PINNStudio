"""Error Analysis also reports the relative L2 over every reference point at
every reference time together (error_spacetime.txt), the measure papers
usually give, on the standard and the time-adaptive path."""
import dataclasses

import numpy as np

from pinnstudio.core.codegen import generate_script
from pinnstudio.core.config import PINNConfig


def test_both_paths_write_the_space_time_error():
    for ta in (False, True):
        s = generate_script(dataclasses.replace(PINNConfig(), time_adaptive=ta))
        assert s.count("error_spacetime") >= 1 and "reference times together" in s
        compile(s, "<script>", "exec")


def test_space_time_error_is_the_pooled_norm():
    # the formula the scripts use, on two snapshots of different size
    refs = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
    pinn = [np.array([1.1, 2.0]), np.array([3.0, 3.9, 5.2])]
    num = sum(float(np.sum((p - r) ** 2)) for p, r in zip(pinn, refs))
    den = sum(float(np.sum(r ** 2)) for r in refs)
    pooled = np.linalg.norm(np.concatenate(pinn) - np.concatenate(refs)) / np.linalg.norm(np.concatenate(refs))
    assert abs((num / den) ** 0.5 - pooled) < 1e-12
