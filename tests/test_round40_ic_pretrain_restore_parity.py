#!/usr/bin/env python3
"""
Round 40: codebase cleanup, phase 3 (audit §2c).

`ic_pretrain_restore`/`ic_pretrain_restore_path` (resume IC pre-training
from a checkpoint instead of training it fresh) existed in BOTH of
generate_script()'s IC Pre-Training copies (Standard and Time-Adaptive),
but had zero references in EITHER of generate_clean_script()'s own two
copies (its own Standard branch, and its separate Time-Adaptive per-step
loop branch) -- confirmed by grep across the whole file. Those two copies
also never saved a checkpoint after training (no `model_save_path=` at
all), so even if a user later tried to restore from one by hand there
would be nothing to restore from.

This test exercises both of generate_clean_script()'s IC Pre-Training
copies end to end, in a real round trip:

 1. Export + run with `ic_pretrain_restore=False` -- confirms a real
    checkpoint file now appears under `<save_dir>/ic_pretrain/`.
 2. Export + run again with `ic_pretrain_restore=True` pointed at that
    checkpoint -- confirms the script restores from it (prints the
    "restored -- skipping training" message) instead of training again,
    and still completes DONE/TA_DONE cleanly.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round40_ic_pretrain_restore_parity.py
"""
import os
import sys
import glob
import tempfile
import subprocess

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import generate_clean_script


def _run_script(script, tmpdir, tag, timeout_s=120):
    sp = os.path.join(tmpdir, f"{tag}_script.py")
    with open(sp, "w") as f:
        f.write(script)
    return subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=timeout_s)


def _transient_1d_cfg(save_dir, time_adaptive=False):
    cfg = PINNConfig()
    cfg.problem_dim = "1D"
    cfg.steady_state = False
    cfg.num_outputs = 1
    cfg.output_names = "u"
    cfg.x_min, cfg.x_max = 0.0, 1.0
    cfg.t_min, cfg.t_max = 0.0, 1.0
    cfg.num_domain = 20; cfg.num_boundary = 10; cfg.num_initial = 10; cfg.num_test = 15
    cfg.layers = [2, 8, 1]
    cfg.pde_expressions = "du_t - 0.1*du_xx"
    cfg.pde_expression = cfg.pde_expressions
    cfg.ic_expressions = "0.0"; cfg.ic_expression = cfg.ic_expressions
    cfg.custom_bc_json = "[]"
    cfg.adapt_method = "None"
    cfg.iterations = 2; cfg.optimizer_scheduler = False; cfg.iterations2 = 0
    cfg.plot_type = "Line (time steps)"
    cfg.plot_resolution = 12
    cfg.num_timesteps_line = 1
    cfg.save_dir = save_dir
    cfg.ic_pretrain = True
    cfg.ic_pretrain_iterations = 3
    cfg.ic_pretrain_num_initial = 10
    cfg.ic_pretrain_num_test = 10
    cfg.ic_pretrain_optimizer = "adam"
    cfg.ic_pretrain_lr = 0.001
    cfg.ic_pretrain_loss = "MSE"
    cfg.time_adaptive = time_adaptive
    if time_adaptive:
        cfg.ta_num_steps = 2
        cfg.layers = [2, 8, 1]
    return cfg


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
            print("FAIL:", msg)
        else:
            print("ok:", msg)

    try:
        import deepxde as _dde_probe  # noqa: F401
        import torch as _torch_probe  # noqa: F401
    except ImportError:
        print("SKIPPED: deepxde/torch not installed in this environment.")
        return failures

    # ── 1. Standard (non-TA) branch ───────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "run1")
        cfg = _transient_1d_cfg(save_dir, time_adaptive=False)
        clean = generate_clean_script(cfg)
        check("_ic_pretrain_save_dir = os.path.join" in clean and "skipping training" in clean,
              "generate_clean_script()'s Standard IC Pre-Training branch should embed the "
              "restore-from-checkpoint branch now")
        check("model_save_path=os.path.join(_ic_pretrain_save_dir" in clean,
              "generate_clean_script()'s Standard IC Pre-Training branch should save a checkpoint "
              "after training")

        proc = _run_script(clean, tmpdir, "std_train")
        check(proc.returncode == 0,
              f"Standard export with ic_pretrain (train mode) should run cleanly, got "
              f"{proc.returncode}:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        check("IC pre-train model saved to" in proc.stdout,
              "should report where the IC pre-train checkpoint was saved")

        ckpts = glob.glob(os.path.join(save_dir, "ic_pretrain", "ic_pretrain_model-*.pt"))
        check(len(ckpts) >= 1, f"a real IC pre-train checkpoint should exist under {save_dir}/ic_pretrain/")

        if ckpts:
            ckpt_path = sorted(ckpts)[-1]
            cfg2 = _transient_1d_cfg(os.path.join(tmpdir, "run2"), time_adaptive=False)
            cfg2.ic_pretrain_restore = True
            cfg2.ic_pretrain_restore_path = ckpt_path
            clean2 = generate_clean_script(cfg2)
            check(repr(ckpt_path) in clean2,
                  "the restore path should be embedded in the generated script")
            proc2 = _run_script(clean2, tmpdir, "std_restore")
            check(proc2.returncode == 0,
                  f"Standard export with ic_pretrain_restore=True should run cleanly, got "
                  f"{proc2.returncode}:\n{proc2.stdout[-1500:]}\n{proc2.stderr[-1500:]}")
            check("restored -- skipping training" in proc2.stdout,
                  "should report that the IC pre-train model was restored, not retrained")

    # ── 2. Time-Adaptive per-step-loop branch ─────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "ta_run1")
        cfg_ta = _transient_1d_cfg(save_dir, time_adaptive=True)
        clean_ta = generate_clean_script(cfg_ta)
        check("_ic_pretrain_save_dir_i = os.path.join" in clean_ta and "skipping training" in clean_ta,
              "generate_clean_script()'s Time-Adaptive IC Pre-Training branch should embed the "
              "restore-from-checkpoint branch now")
        check("model_save_path=os.path.join(_ic_pretrain_save_dir_i" in clean_ta,
              "generate_clean_script()'s Time-Adaptive IC Pre-Training branch should save a "
              "checkpoint after training")

        proc_ta = _run_script(clean_ta, tmpdir, "ta_train")
        check(proc_ta.returncode == 0,
              f"Time-Adaptive export with ic_pretrain (train mode) should run cleanly, got "
              f"{proc_ta.returncode}:\n{proc_ta.stdout[-2000:]}\n{proc_ta.stderr[-2000:]}")

        ckpts_ta = glob.glob(os.path.join(save_dir, "ic_pretrain", "ic_pretrain_model-*.pt"))
        check(len(ckpts_ta) >= 1,
              f"a real IC pre-train checkpoint should exist under {save_dir}/ic_pretrain/ for the "
              f"Time-Adaptive branch too")

        if ckpts_ta:
            ckpt_path_ta = sorted(ckpts_ta)[-1]
            cfg_ta2 = _transient_1d_cfg(os.path.join(tmpdir, "ta_run2"), time_adaptive=True)
            cfg_ta2.ic_pretrain_restore = True
            cfg_ta2.ic_pretrain_restore_path = ckpt_path_ta
            clean_ta2 = generate_clean_script(cfg_ta2)
            proc_ta2 = _run_script(clean_ta2, tmpdir, "ta_restore")
            check(proc_ta2.returncode == 0,
                  f"Time-Adaptive export with ic_pretrain_restore=True should run cleanly, got "
                  f"{proc_ta2.returncode}:\n{proc_ta2.stdout[-2000:]}\n{proc_ta2.stderr[-2000:]}")
            check("restored -- skipping training" in proc_ta2.stdout,
                  "should report that the Time-Adaptive IC pre-train model was restored, not "
                  "retrained")

    return failures


def main():
    failures = run()
    print()
    if failures:
        print(f"FAILURES: {len(failures)}")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("ALL ROUND 40 IC-PRETRAIN-RESTORE-PARITY TESTS PASSED")
        sys.exit(0)


def test_round40_ic_pretrain_restore_parity():
    """pytest entry point."""
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    main()
