"""Runs behind the v1.6.0 audit fixes (see test_audit_v160.py): the generated scripts are trained briefly.

* The exported script with Model Checkpoint on runs and writes its checkpoints.
* Time windows with an output that has no initial condition: Solve's and the exported script keep the same loss
  terms in every window (no weight warning) and print the same loss history.
* 'Scale by initial losses' with IC pre-training: both scripts compute the same weights and train the same model.
"""
import dataclasses
import json
import os
import re
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

LOSS = re.compile(r"^\d+\s+\[")


def _short(cfg, d, n_adam=40, n_lbfgs=10):
    import check_export_equivalence as ce
    return ce.shorten(cfg, n_adam, n_lbfgs, d)


def _run(script, d, name):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(script)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, name], cwd=d, env=env, capture_output=True, text=True, timeout=1200)
    return r.returncode, r.stdout, r.stderr


def _both(cfg, d):
    from pinnstudio.core import codegen
    out = {}
    for kind, gen in (("solve", codegen.generate_script), ("export", codegen.generate_clean_script)):
        rc, o, e = _run(gen(cfg), d, kind + ".py")
        assert rc == 0, (kind, e[-2000:])
        out[kind] = o
    return out


def test_export_with_model_checkpoint_runs():
    import tuning_study as ts
    from pinnstudio.core import codegen
    with tempfile.TemporaryDirectory() as d:
        cfg = dataclasses.replace(_short(ts.template_config("1D Burgers", False), d), cb_model_checkpoint=True,
                                  cb_checkpoint_period=20)
        rc, o, e = _run(codegen.generate_clean_script(cfg), d, "export.py")
        assert rc == 0, e[-2000:]
        assert any(f.startswith("checkpoint_best") for f in os.listdir(d)), os.listdir(d)


def test_time_windows_with_an_output_without_initial_condition():
    import tuning_study as ts
    with tempfile.TemporaryDirectory() as d:
        cfg = _short(ts.template_config("2D Cahn-Hilliard (Wight & Zhao)", False), d, 20, 0)
        ph = [dict(p, iterations=20) for p in json.loads(cfg.scheduler_phases) if p["optimizer"] != "lbfgs"]
        cfg = dataclasses.replace(cfg, scheduler_phases=json.dumps(ph), time_adaptive=True, ta_num_steps=2,
                                  ta_grid_size=21, ta_step_groups="", num_domain=200, num_boundary=40,
                                  num_initial=60, ic_hard="False,False", ic_active="True,False")
        out = _both(cfg, d)
        for o in out.values():
            assert "configured weights have" not in o
        assert ([ln for ln in out["solve"].splitlines() if LOSS.match(ln)]
                == [ln for ln in out["export"].splitlines() if LOSS.match(ln)])


def test_initial_loss_weights_with_ic_pretraining_agree():
    import tuning_study as ts
    with tempfile.TemporaryDirectory() as d:
        cfg = dataclasses.replace(_short(ts.template_config("1D Burgers", False), d), ic_pretrain=True,
                                  ic_pretrain_iterations=30, loss_weights_from_initial=5.0)
        out = _both(cfg, d)
        w = {k: [ln for ln in o.splitlines() if "Loss weights from the initial losses" in ln][-1]
             for k, o in out.items()}
        assert w["solve"] == w["export"], w
        assert ([ln for ln in out["solve"].splitlines() if LOSS.match(ln)][-3:]
                == [ln for ln in out["export"].splitlines() if LOSS.match(ln)][-3:])
