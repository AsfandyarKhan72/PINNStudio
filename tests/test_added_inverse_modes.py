#!/usr/bin/env python3
"""
Inverse modes PINNStudio added to DeepXDE forward examples
(deepxde_examples.ADDED_INVERSE): each one's settings follow the same
recipe (measured data weighted 100, an L-BFGS phase at the end, the
unknowns' values saved), and a few of them train on the CPU for a few
iterations and print the found values of their unknowns.

Run directly:
    python3 tests/test_added_inverse_modes.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinnstudio.core import deepxde_examples as dx  # noqa: E402
from pinnstudio.core.codegen import generate_script  # noqa: E402

REF = os.path.join(ROOT, "reference_data")
SMOKE = ["Euler beam", "ODE system", "Second-order ODE"]


def _ends_with_lbfgs(cfg):
    if cfg.optimizer_scheduler:
        phases = json.loads(cfg.scheduler_phases or "[]")
        return bool(phases) and phases[-1].get("optimizer", "").lower() == "lbfgs"
    return (cfg.optimizer2 or "").lower() == "lbfgs" and cfg.iterations2 > 0


def run(train=True):
    failures = []

    def check(cond, msg):
        print(("ok: " if cond else "FAIL: ") + msg)
        if not cond:
            failures.append(msg)

    check(len(dx.ADDED_INVERSE) >= 10, f"{len(dx.ADDED_INVERSE)} added inverse modes")
    for name in dx.ADDED_INVERSE:
        ex = dx.BY_NAME[name]
        check(ex.inverse and "PINNStudio's addition" in ex.inverse_note, f"{name}: has its Inverse mode and note")
        cfg = dx.build_config(ex, REF, inverse=True)
        obs = json.loads(cfg.inverse_obs_files_json)
        check(all(float(o["weight"]) == 100.0 for o in obs), f"{name}: measured data weighted 100")
        check(_ends_with_lbfgs(cfg), f"{name}: ends with an L-BFGS phase")
        check(bool(cfg.inv_param_save), f"{name}: saves the unknowns' values")

    if not train:
        return failures
    for name in SMOKE:
        ex = dx.BY_NAME[name]
        with tempfile.TemporaryDirectory() as d:
            cfg = dx.build_config(ex, REF, keep={"save_dir": d}, inverse=True)
            cfg.iterations, cfg.iterations2 = 30, 10
            if cfg.optimizer_scheduler:
                phases = json.loads(cfg.scheduler_phases)
                for ph in phases:
                    ph["iterations"] = 30 if ph.get("optimizer", "").lower() != "lbfgs" else 10
                cfg.scheduler_phases = json.dumps(phases)
            with open(os.path.join(d, "run.py"), "w") as fh:
                fh.write(generate_script(cfg))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True,
                               timeout=900, env=env)
            found = dict(re.findall(r"=== (\w+) final value: (\S+) ===", p.stdout))
            want = [v["name"] for v in json.loads(cfg.inverse_variables_json)]
            check(p.returncode == 0 and sorted(found) == sorted(want),
                  f"{name}: trains on the CPU and prints the final values {found}"
                  + ("" if p.returncode == 0 else "\n" + p.stderr[-1500:]))
    return failures


def test_added_inverse_modes():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    f = run()
    print(f"\n{len(f)} FAILURE(S)" if f else "\nALL ADDED INVERSE MODE TESTS PASSED")
    sys.exit(1 if f else 0)
