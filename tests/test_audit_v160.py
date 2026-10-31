"""Regression checks for the v1.6.0 pre-release audit (generated code and validation only; the runs that train are
in test_audit_v160_runs.py).

1. Export with Model Checkpoint: the exported script used `_sol_dir`, which only Solve's script defines (NameError).
2. RAR rounds took the loss function from the hidden single-optimizer field and the Loss Weights panel, not from
   the Training Phases (a phase with MAE trained its RAR rounds with MSE).
3. Time windows: initial-condition guided learning and resampling rounds used the hidden learning rate, not the
   Training Phases'.
4. Time windows: from the second window on every output is held to the previous window's solution, one loss term
   per output, where the first window had one per output with an initial condition; the weights did not fit and
   every weight (PDE, boundary, initial condition) was replaced by 1.
5. Causal training: residuals of any point set at least as large as the last one (RAR or resampling candidates)
   became the stored training weights, and the epsilon schedule started over at every Adam training call.
6. Per-point weights (RBA / SA-PINN) with a point resampler: the weights started over at every resampling.
7. 'Scale by initial losses' with IC pre-training: Solve's script took the initial losses before pre-training,
   the exported script after it.
8. Time windows on a non-rectangular domain: the grid carrying the solution between windows covered the whole
   bounding box.
9. IC pre-training with no initial condition as a loss term (all exact) trained on a loss that is identically zero.
"""
import dataclasses
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from pinnstudio.core import codegen  # noqa: E402
from pinnstudio.core import deepxde_examples as dx  # noqa: E402

REF = os.path.join(HERE, "..", "reference_data")


def _dx(name, **upd):
    c = dx.build_config(dx.BY_NAME[name], REF)
    return dataclasses.replace(c, **upd) if upd else c


def _template(key, inverse=False):
    import tuning_study as ts
    return ts.template_config(key, inverse)


def _phases(c):
    return json.loads(c.scheduler_phases)


def test_export_model_checkpoint_uses_the_export_folder():
    c = _dx("Burgers equation with RAR", cb_model_checkpoint=True, save_dir="")
    s = codegen.generate_clean_script(c)
    assert "_sol_dir" not in s
    assert "os.path.join((save_dir or '.'), \"checkpoint_best\")" in s
    compile(s, "export", "exec")


def test_rar_rounds_use_the_phase_loss_and_weights():
    c = _dx("Burgers equation with RAR", adapt_method="RAR", rar_cycles=1)
    ph = _phases(c)
    for p in ph:
        p["loss"] = "MAE"
    c = dataclasses.replace(c, scheduler_phases=json.dumps(ph), optimizer_scheduler=True, loss_type="MSE")
    solve, export = codegen.generate_script(c), codegen.generate_clean_script(c)
    assert "loss='MAE',\n                          loss_weights=_rar_w" in solve
    assert "model.compile(\"L-BFGS\", loss='MAE', loss_weights=_rar_wl" in solve
    rar_part = export[export.index("rar_cycle"):]
    assert "loss='MSE'" not in rar_part and "loss=\"MSE\"" not in rar_part
    assert rar_part.count("loss='MAE'") >= 1


def test_time_window_substeps_use_the_phase_learning_rate():
    c = _dx("Burgers equation with RAR")
    ph = [dict(_phases(c)[0], optimizer="adam", lr=0.0001)]
    c = dataclasses.replace(c, time_adaptive=True, ta_num_steps=2, ta_icgl_fraction=0.2, ta_transfer_learning=True,
                            ta_resample_rounds=1, optimizer_scheduler=True, scheduler_phases=json.dumps(ph),
                            learning_rate=0.001)
    s = codegen.generate_script(c)
    assert '_model_icgl.compile("adam", lr=0.0001,' in s
    assert 'model_i.compile("adam", lr=0.0001, loss=' in s


def test_later_windows_keep_the_users_weights():
    c = _template("2D Cahn-Hilliard (Wight & Zhao)")
    assert c.num_outputs == 2
    c = dataclasses.replace(c, time_adaptive=True, ta_num_steps=2, ic_hard="False,False", ic_active="True,False")
    export = codegen.generate_clean_script(c)
    assert "def _ta_weights(w, constraints, step_i):" in export and "ic_outs = [0]" in export
    assert "prev_u[:, 1:2], component=1" in export          # every output is still held to the previous window
    solve = codegen.generate_script(c)
    assert "_sp_weights = _ta_w(" in solve and "_wx(_ta_w(_multi_weights))" in solve


def test_causal_weights_follow_the_training_points_only():
    c = _dx("Burgers equation with RAR", causal_training=True, causal_anneal=True)
    s = codegen.generate_script(c)
    assert 'if _CAUSAL["n"] and x.shape[0] != _CAUSAL["n"]:' in s
    assert 'x.shape[0] >= _CAUSAL["n"]' not in s
    assert 'if _rng != _CAUSAL.get("rng"):' in s


def test_per_point_weights_with_a_point_resampler_are_refused():
    base = _dx("Burgers equation with RAR")
    for pw in ("RBA", "SA-PINN"):
        errs = dataclasses.replace(base, point_weights=pw, cb_point_resampler=True).validate()
        assert any("Point Resampler" in e for e in errs), errs
    batched = _dx("Inverse: Poisson with unknown forcing field (batches)", point_weights="RBA").validate()
    assert any("batch size" in e for e in batched), batched
    assert not any("Point Resampler" in e or "batch size" in e
                   for e in dataclasses.replace(base, point_weights="RBA").validate())


def test_initial_loss_weights_follow_ic_pretraining_in_solve():
    c = _template("1D Burgers")
    c = dataclasses.replace(c, ic_pretrain=True, loss_weights_from_initial=5.0)
    s = codegen.generate_script(c)
    i_pre = s.index("=== Starting Main Training ===")
    assert s.find("Loss weights from the initial losses", i_pre) > i_pre
    # Solve's model already holds the entry from before pre-training: the weights must use the latest one
    assert "_lh0.loss_train[0]" not in s and "_lh0.loss_train[-1]" in s


def test_time_windows_on_a_disk_keep_only_grid_points_in_the_domain():
    c = _template("2D Heat")
    c = dataclasses.replace(c, time_adaptive=True, ta_num_steps=2, geometry_type="Disk")
    for s in (codegen.generate_script(c), codegen.generate_clean_script(c)):
        assert "_ta_in = np.asarray(_ta_gm.inside(x_grid))" in s


def test_ic_pretraining_without_any_initial_condition_loss_is_refused():
    c = _template("1D Burgers")
    errs = dataclasses.replace(c, ic_pretrain=True, ic_active="False", ic_hard="True").validate()
    assert any("IC pre-training" in e for e in errs), errs
    assert not any("IC pre-training" in e for e in dataclasses.replace(c, ic_pretrain=True).validate())
