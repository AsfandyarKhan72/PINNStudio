#!/usr/bin/env python3
"""
Round 78 -- examples train through the Training Phases the GUI shows.

45 DeepXDE examples stored their training in the older single-optimizer
fields (hidden in the GUI) with the scheduler off, so Solve trained from
those while the visible Training Phases panel showed unrelated phases and
edits to it were ignored (the plate kept stopping at 5000 iterations).
Now every example -- and any config loaded with the scheduler off -- gets
the equivalent phases. Checks: every example has phases matching its
DeepXDE iterations; editing a phase in the GUI reaches both scripts; an
old-style config loads as phases; the two paths train identically; and
the "Changed from the example" log lines.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round78_example_phases.py
"""
import copy
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
from pinnstudio.core.codegen import generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig, phases_from_legacy, use_training_phases
from pinnstudio.ui.main_window import MainWindow


def _small(d, **kw):
    c = PINNConfig()
    for k, v in dict(problem_dim="1D", steady_state=True, geometry_type="Interval", x_min=-1.0, x_max=1.0,
                     num_initial=0, pde_expressions="du_xx - 2", layers=[1, 16, 16, 1], optimizer="adam",
                     learning_rate=1e-3, iterations=400, optimizer2="lbfgs", iterations2=60, lbfgs_use_default=False,
                     optimizer_scheduler=False, scheduler_phases="", num_domain=16, num_boundary=2, num_test=40,
                     custom_bc_json=json.dumps([{"type": "dirichlet", "location": "left", "value": "0", "component": 0},
                                                {"type": "neumann", "location": "right", "value": "4", "component": 0}]),
                     loss_weights_multi="1,1,1", ea_ref_mode="formula", ea_ref_formula="(x + 1)**2", save_dir=d,
                     plot_resolution=40).items():
        setattr(c, k, v)
    for k, v in kw.items():
        setattr(c, k, v)
    c.pde_expression = c.pde_expressions
    return c


def run():
    failures = []

    def check(cond, msg):
        print(("ok: " if cond else "FAIL: ") + msg)
        if not cond:
            failures.append(msg)

    # -- every example trains through phases that match DeepXDE's iterations
    for ex in dx.EXAMPLES:
        c = dx.build_config(ex, "/r")
        ph = json.loads(c.scheduler_phases or "[]")
        if not c.optimizer_scheduler or not ph:
            check(False, f"{ex.name}: no training phases")
            continue
        if not ex.settings.get("optimizer_scheduler"):
            want = int(c.iterations) + (int(c.iterations2) if c.optimizer2 != "none" else 0)
            check(sum(int(p["iterations"]) for p in ph) == want, f"{ex.name}: phases = DeepXDE's iterations")
    plate = json.loads(dx.build_config(dx.BY_NAME["Linear elastostatic plate"], "/r").scheduler_phases)
    check([(p["optimizer"], p["iterations"]) for p in plate] == [("adam", 20000), ("lbfgs", 20000)],
          f"plate: Adam 20000 then L-BFGS 20000 {plate}")
    lor = json.loads(dx.build_config(dx.BY_NAME["Inverse: Lorenz system"], "/r").scheduler_phases)
    check([p["optimizer"] for p in lor] == ["adam", "lbfgs"], "Adam + L-BFGS -> two phases")

    old = _small(".")
    check(phases_from_legacy(old) == [{"optimizer": "adam", "iterations": 400, "lr": 0.001, "decay_type": "none"},
                                      {"optimizer": "lbfgs", "iterations": 60, "lr": 0.001, "decay_type": "none"}],
          "legacy fields -> phases")

    # -- GUI: what the panel says is what runs
    w = MainWindow()
    for name in ("Linear elastostatic plate", "Inverse: Lorenz system", "Burgers equation with RAR", "Euler beam"):
        w._apply_deepxde_example(name)
        n0 = w.sched_phase_list[0]["iters"].value()
        w.sched_phase_list[0]["iters"].setValue(n0 + 12345)
        c = w._build_config()
        ph = json.loads(c.scheduler_phases)
        check(c.optimizer_scheduler and int(ph[0]["iterations"]) == n0 + 12345, f"{name}: edited phase in the config")
        live, exp = generate_script(c), generate_clean_script(c)
        check(f"'iterations': {n0 + 12345}" in live or f'"iterations": {n0 + 12345}' in live, f"{name}: live script uses it")
        check(f"iterations={n0 + 12345}" in exp, f"{name}: exported script uses it")
        msgs = dx.describe_changes(w._example_snapshot[1], c)
        check(msgs == [f"Phase 1 iterations: {n0} -> {n0 + 12345}"], f"{name}: change reported {msgs}")
        w.log_box.clear()
        w._log_example_changes(c)
        check("Changed from the" in w.log_box.toPlainText(), f"{name}: logged")
    w._apply_deepxde_example("Euler beam")
    check(dx.describe_changes(w._example_snapshot[1], w._build_config()) == [], "unchanged example: nothing reported")
    w.quick_examples_combo.setCurrentText("1D Heat")
    check(w._example_snapshot is None, "a PINNStudio template clears the example snapshot")
    w._apply_config(_small("."))
    ph = [(p["optimizer"].currentText().lower() if hasattr(p["optimizer"], "currentText") else None, p["iters"].value())
          for p in w.sched_phase_list] if w.sched_phase_list and "optimizer" in w.sched_phase_list[0] else \
        [(None, p["iters"].value()) for p in w.sched_phase_list]
    check([n for _o, n in ph] == [400, 60], f"an old-style config loads as phases {ph}")
    check(w._build_config().optimizer_scheduler, "and keeps the scheduler on")

    try:
        import deepxde  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures
    with tempfile.TemporaryDirectory() as tmp:
        res = {}
        for tag in ("legacy", "phases"):
            d = os.path.join(tmp, tag)
            os.makedirs(d)
            c = _small(d)
            if tag == "phases":
                c = use_training_phases(copy.deepcopy(c))
            with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as fh:
                fh.write(generate_script(c))
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
            check(p.returncode == 0, f"{tag} ran:\n{p.stderr[-1500:]}")
            res[tag] = re.findall(r"L2=(\S+),", p.stdout)
        check(res["legacy"] and res["legacy"] == res["phases"], f"old fields and phases train identically {res}")
    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 78 EXAMPLE-PHASES TESTS PASSED")
    return 0


def test_round78_example_phases():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
