"""A growing time range in 2D: every window takes the true initial condition at t_start, so there is no
previous-window grid to add as anchor points. The script used to stop at the second window with
NameError: _xyt_ic_anchor (found in the Figure 7 runs, 2D Allen-Cahn of Mattey & Ghosh)."""
import dataclasses
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))


def test_growing_range_2d_runs_past_the_first_window():
    import tuning_study as ts
    from pinnstudio.core.codegen import generate_script
    cfg = ts.template_config("2D Allen-Cahn (Mattey & Ghosh)", False)
    phases = [dict(ts._phases(cfg)[0], optimizer="adam", iterations=20)]
    cfg = dataclasses.replace(
        cfg, time_adaptive=True, adapt_method="Time Adaptive", ta_growing=True, ta_transfer_learning=True,
        ta_backward=False, ta_icgl_fraction=0.0, ta_resample_rounds=0, ta_num_steps=2, ta_step_groups="",
        ta_grid_size=11, num_domain=200, num_boundary=40, num_initial=40, num_test=100,
        optimizer_scheduler=True, scheduler_phases=json.dumps(phases), iterations=20,
        train_device="cpu")
    with tempfile.TemporaryDirectory() as d:
        cfg = dataclasses.replace(cfg, save_dir=d)
        path = os.path.join(d, "run.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(generate_script(cfg))
        r = subprocess.run([sys.executable, path], cwd=d, capture_output=True, text=True, timeout=900)
        assert "NameError" not in r.stderr, r.stderr[-2000:]
        assert r.returncode == 0, r.stderr[-2000:]
        assert "Time step 2/2" in r.stdout, r.stdout[-2000:]


if __name__ == "__main__":
    test_growing_range_2d_runs_past_the_first_window()
    print("ok")
