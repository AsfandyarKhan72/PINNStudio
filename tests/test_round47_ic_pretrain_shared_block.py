#!/usr/bin/env python3
"""
Round 47 -- codebase cleanup, phase 10 (audit §2b/§4): the IC Pre-Training
"dummy-PDE dataset + compile + restore-or-train" block was hand-typed four
times -- generate_script()'s Standard and Time-Adaptive paths, and
generate_clean_script()'s Standard path and Time-Adaptive per-step loop --
under four temp-model names (_model_pre, _model_pt, _model_pre again,
_model_pre_i) with small drifts between them. All four now come from one
codegen.py helper, _ic_pretrain_train_code(...), with one set of names.

This is a consolidation that slightly changes generated text (unified
names and messages), not a byte-for-byte refactor, so the check here is
behavioral, on all four paths:
  1. run with ic_pretrain_restore=False -> IC pre-training actually trains
     and saves a real checkpoint under <save_dir>/ic_pretrain/;
  2. run again with ic_pretrain_restore=True pointed at that checkpoint ->
     "restored -- skipping training", clean exit;
  3. cross-generator: a checkpoint saved by the live Solve script
     (generate_script) restores in an exported script
     (generate_clean_script).
Plus a textual check that every generated script uses the unified names
and none of the old per-copy ones.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round47_ic_pretrain_shared_block.py
"""
import glob
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from pinnstudio.core.codegen import generate_clean_script, generate_script
from test_round40_ic_pretrain_restore_parity import _transient_1d_cfg

_OLD_NAMES = ("_model_pt", "_model_pre_i", "_ic_pre_save_dir", "_pde_dummy_ta", "_ic_only_w_ta",
              "_data_pt", "_ic_ckpt_ta", "_ic_ckpt_i", "_ic_pretrain_save_dir_i")


def _run(script, tmpdir, tag, timeout_s=240):
    sp = os.path.join(tmpdir, f"{tag}.py")
    with open(sp, "w", encoding="utf-8") as f:
        f.write(script)
    return subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=timeout_s)


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    try:
        import deepxde as _dde_probe  # noqa: F401
        import torch as _torch_probe  # noqa: F401
    except ImportError:
        print("SKIPPED: deepxde/torch not installed in this environment.")
        return failures

    live_ckpt = None
    for gen_name, gen in (("generate_script", generate_script), ("generate_clean_script", generate_clean_script)):
        for ta in (False, True):
            tag = f"{gen_name}_{'ta' if ta else 'std'}"
            with tempfile.TemporaryDirectory() as tmpdir:
                save_dir = os.path.join(tmpdir, "run1")
                os.makedirs(save_dir, exist_ok=True)
                script = gen(_transient_1d_cfg(save_dir, time_adaptive=ta))
                # generate_script() emits both its Standard and Time-Adaptive
                # paths and picks one at runtime (2 copies of the block);
                # generate_clean_script() emits only the configured one (1).
                expected = 2 if gen_name == "generate_script" else 1
                check(script.count("_model_pre = dde.Model(_data_pre,") == expected,
                      f"{tag}: the shared IC pre-training block should appear {expected}x")
                for old in _OLD_NAMES:
                    check(old not in script, f"{tag}: old per-copy name {old} should no longer be generated")

                proc = _run(script, tmpdir, "train")
                check(proc.returncode == 0,
                      f"{tag}: train-mode run should exit cleanly, got {proc.returncode}:\n"
                      f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
                check("IC pre-training done. Final IC loss:" in proc.stdout,
                      f"{tag}: train-mode run should report the final IC loss")
                ckpts = glob.glob(os.path.join(save_dir, "ic_pretrain", "ic_pretrain_model-*.pt"))
                check(len(ckpts) >= 1, f"{tag}: a real IC pre-train checkpoint should be saved")
                if not ckpts:
                    continue
                ckpt = os.path.join(tmpdir, "keep.pt")
                os.replace(ckpts[0], ckpt)

                cfg2 = _transient_1d_cfg(os.path.join(tmpdir, "run2"), time_adaptive=ta)
                cfg2.ic_pretrain_restore = True
                cfg2.ic_pretrain_restore_path = ckpt
                proc2 = _run(gen(cfg2), tmpdir, "restore")
                check(proc2.returncode == 0,
                      f"{tag}: restore-mode run should exit cleanly, got {proc2.returncode}:\n"
                      f"{proc2.stdout[-1500:]}\n{proc2.stderr[-1500:]}")
                check("restored -- skipping training" in proc2.stdout,
                      f"{tag}: restore-mode run should skip IC pre-training")

                if gen_name == "generate_script" and not ta:
                    # Cross-generator: live-Solve checkpoint -> exported script.
                    cfg3 = _transient_1d_cfg(os.path.join(tmpdir, "run3"), time_adaptive=False)
                    cfg3.ic_pretrain_restore = True
                    cfg3.ic_pretrain_restore_path = ckpt
                    proc3 = _run(generate_clean_script(cfg3), tmpdir, "cross")
                    check(proc3.returncode == 0 and "restored -- skipping training" in proc3.stdout,
                          f"a generate_script() IC checkpoint should restore in an exported script, got "
                          f"{proc3.returncode}:\n{proc3.stdout[-1500:]}\n{proc3.stderr[-1500:]}")
                    live_ckpt = True
    check(live_ckpt, "the cross-generator restore case should have run")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 47 IC PRE-TRAINING SHARED-BLOCK TESTS PASSED")
    return 0


def test_round47_ic_pretrain_shared_block():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
