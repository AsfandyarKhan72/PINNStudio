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
10. Output names that clash with the generated code (y, np, lambda, "my u", a name used twice) were accepted.
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


def test_output_names_that_clash_with_the_generated_code_are_refused():
    """10. Output names: 'y' (the array the outputs are read from) made Solve's script train a different problem
    and the exported script crash; np, dde, lambda, "my u", 2u and a name used twice crashed or shadowed."""
    c = _dx("ODE system")
    assert c.output_names == "y1,y2" and not c.validate()
    for names in ("y,v", "v,y", "np,v", "dde,v", "sin,v", "lambda,v", "my u,v", "2u,v", "u,u", "du_t,v"):
        pde = "|".join(f"d{n}_t" for n in names.split(","))
        errs = dataclasses.replace(c, output_names=names, pde_expressions=pde).validate()
        assert any("Output name" in e for e in errs), (names, errs)
    for names in ("u,v", "h,mu", "y1,y2", "u_re,u_im"):
        errs = dataclasses.replace(c, output_names=names).validate()
        assert not any("Output name" in e for e in errs), (names, errs)


def test_an_initial_condition_with_t_means_t_min():
    """11. An initial condition written with t (an exact solution typed as the IC) crashed both scripts with
    NameError: t, with or without time windows. t now means t_min there."""
    c = _template("1D Burgers")
    c = dataclasses.replace(c, ic_expressions="exp(-t)*sin(pi*x)", ic_expression="exp(-t)*sin(pi*x)",
                            ic_hard="False", ic_active="True")
    t0 = repr(float(c.t_min))
    for s in (codegen.generate_script(c), codegen.generate_clean_script(c)):
        assert f"np.exp(-({t0}))" in s and "np.exp(-t)" not in s
    ta = dataclasses.replace(c, time_adaptive=True, ta_num_steps=2)
    for s in (codegen.generate_script(ta), codegen.generate_clean_script(ta)):
        assert "np.exp(-t)" not in s


def test_ic_guided_learning_fits_only_outputs_with_an_initial_condition():
    """12. Time windows, IC guided learning: the first window fitted every output to its IC formula (an output
    without an initial condition, Cahn-Hilliard's mu, to the formula box's placeholder) and ignored an IC file."""
    c = _template("2D Cahn-Hilliard (Wight & Zhao)")
    c = dataclasses.replace(c, time_adaptive=True, ta_num_steps=2, ta_icgl_fraction=0.2, ic_hard="False,False",
                            ic_active="True,False")
    assert codegen._icgl_outputs(c) == [0]
    s = codegen.generate_script(c)
    assert "for _k in [0]:" in s and "[1.0] * len(_cons_icgl)" in s
    compile(s, "solve", "exec")
    f = dataclasses.replace(c, forward_ic_from_file=True, forward_ic_file="ic.txt", ic_active="False,False")
    s = codegen.generate_script(f)
    assert codegen._icgl_outputs(f) == [0] and "_load_ic_from_file('ic.txt')" in s


def test_time_window_error_analysis_reads_the_window_bounds():
    """13. Time-window Error Analysis took each window's bounds from its folder name (4 decimals)."""
    c = _template("1D Burgers")
    s = codegen.generate_script(dataclasses.replace(c, time_adaptive=True, ta_num_steps=2))
    assert s.count('open(_os.path.join(_sd, "step_config.json")) as _scj_f') >= 2


def test_export_keeps_the_gpu_choice_and_notes_export_data():
    """14. The exported script ignored the Hardware settings' GPU index and memory fraction, and said nothing
    about 'Export solution data' (written by Solve only)."""
    c = _dx("ODE system")
    plain = codegen.generate_clean_script(c)
    assert "set_device" not in plain and "Export solution data" not in plain
    s = codegen.generate_clean_script(dataclasses.replace(c, gpu_device_index=1, gpu_memory_fraction=0.5,
                                                          export_enabled=True))
    assert "torch.cuda.set_device(1)" in s and "set_per_process_memory_fraction(0.5, device=1)" in s
    assert "'Export solution data'" in s
    compile(s, "export", "exec")


def test_stmsffn_with_an_input_formula_must_end_with_t():
    """15. STMsFFN takes the last network input as time; an input-feature formula not ending with t was accepted."""
    c = dataclasses.replace(_template("1D Burgers"), network_type="STMsFFN", input_transform_enabled=True,
                            input_transform_mode="formula")
    bad = dataclasses.replace(c, input_feature_formula="t; cos(pi*x); sin(pi*x)").validate()
    assert any("STMsFFN takes" in e for e in bad), bad
    ok = dataclasses.replace(c, input_feature_formula="cos(pi*x); sin(pi*x); t").validate()
    assert not any("STMsFFN takes" in e for e in ok), ok


def test_error_analysis_names_the_field_a_plain_reference_file_is_compared_with():
    """16. A reference file without an output name (t_<time>.txt) is compared with the plotted field; on a
    problem with more than one output the log now says which."""
    s = codegen.generate_script(_template("2D Cahn-Hilliard (Wight & Zhao)"))
    assert "Reference files without an output name are compared with the plotted field" in s


def test_exact_periodicity_needs_a_periodic_condition_for_every_output():
    """17. Exact periodicity makes every output periodic; an output without a periodic condition on that axis
    was silently made periodic too."""
    import json as _j
    c = _dx("ODE system")
    c = dataclasses.replace(c, problem_dim="1D", output_names="u,v", num_outputs=2, periodic_inputs=True,
                            pde_expressions="du_t - du_xx|dv_t - dv_xx", ic_expressions="sin(pi*x)|0",
                            ic_active="True,True", t_min=0.0, t_max=1.0, x_min=0.0, x_max=2.0)
    both = _j.dumps([{"type": "periodic", "component": 0, "axis": "x", "location": "True", "value": "0"},
                     {"type": "periodic", "component": 1, "axis": "x", "location": "True", "value": "0"}])
    one = _j.dumps([{"type": "periodic", "component": 0, "axis": "x", "location": "True", "value": "0"},
                    {"type": "dirichlet", "component": 1, "location": "True", "value": "0"}])
    assert not any("makes every output periodic" in e for e in dataclasses.replace(c, custom_bc_json=both).validate())
    errs = dataclasses.replace(c, custom_bc_json=one).validate()
    assert any("v has no periodic boundary condition in x" in e for e in errs), errs


def test_bc_pinn_stored_points_follow_the_run_seed():
    """18. bc-PINN chose its stored points with np.random.default_rng(1234) whatever the seed setting. They now
    follow the run's seed (unseeded with 'Use random seed' off); the benchmark configurations keep 1234
    explicitly, the seed their recorded runs used."""
    from pinnstudio.core import paper_settings as ps
    sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
    import study_runner as sr
    c = dataclasses.replace(_template("1D Allen-Cahn (whole interval)"), time_adaptive=True, ta_num_steps=2,
                            ta_backward=True, use_random_seed=True, random_seed=7)
    assert c.ta_backward_seed == -1
    assert "np.random.default_rng(7).choice(len(_sp_all)" in codegen.generate_script(c)
    assert "np.random.default_rng().choice(len(_sp_all)" in codegen.generate_script(
        dataclasses.replace(c, use_random_seed=False))
    assert "1234" not in codegen.generate_script(c)
    mg = ps.apply(_template("1D Cahn-Hilliard (Mattey & Ghosh)"), "paper_mg", "1D Cahn-Hilliard (Mattey & Ghosh)")
    assert mg.ta_backward and mg.ta_backward_seed == 1234
    assert "np.random.default_rng(1234).choice(len(_sp_all)" in codegen.generate_script(mg)
    st = dataclasses.replace(c, ta_backward=False)
    sr.apply_setup(st, "Time-adaptive + backward", {})
    assert st.ta_backward and st.ta_backward_seed == 1234


def test_time_windows_on_a_disk_keep_the_domain_grid_after_the_first_window():
    """19. Solve's script rebuilt the whole bounding-box grid at the end of every window, so from the second
    window on points outside a disk/polygon were held to the network's extrapolation again (the export kept the
    domain grid). Both now predict on the window grid itself."""
    c = dataclasses.replace(_template("2D Heat"), time_adaptive=True, ta_num_steps=3, geometry_type="Disk")
    s = codegen.generate_script(c)
    assert "_xg_now = np.asarray(x_grid).reshape(-1, 3 if _is_3d else 2)" in s
    assert "x_grid  = _xyt_pred[:, :2]" not in s and "x_grid  = _xyzt_pred[:, :3]" not in s


def test_time_window_ic_pretraining_fits_only_outputs_with_an_ic_term():
    """20. With time windows, Solve's IC pre-training fitted every output to its IC formula (Cahn-Hilliard's mu,
    whose IC is off, to the placeholder 0); the export and the whole-interval script fit only the active ICs."""
    c = dataclasses.replace(_template("2D Cahn-Hilliard (Wight & Zhao)"), time_adaptive=True, ta_num_steps=2,
                            ic_hard="False,False", ic_active="True,False", ic_pretrain=True)
    s = codegen.generate_script(c)
    assert "for _oi_pt in [0]:" in s and "for _oi_pt in range(2):" not in s
    compile(s, "solve", "exec")


def test_named_constants_can_be_swept():
    """21. A sweep could not change a named constant (a PDE coefficient), although the README says almost any
    setting can be swept. Each constant is now a sweep parameter; constants defined from it follow."""
    import copy
    from pinnstudio.core import sweep_registry as sr
    c = _dx(next(e.name for e in dx.EXAMPLES if "Helmholtz" in e.name and "Dirichlet" in e.name))
    ids = [p.id for p in sr.available_params(c) if p.category == "Constants"]
    assert ids == ["const_n", "const_k0"], ids
    d = copy.deepcopy(c)
    sr.get_param("const_n", d).set_value(d, 3)
    assert abs(sr.get_param("const_k0", d).get_value(d) - 6 * 3.141592653589793) < 1e-9
    assert "(18.84955592153876)" in codegen.generate_script(d)
    assert not d.validate()


def test_rounds_skip_nncg_and_guided_learning_counts_adamw():
    """22. RAR and time-window resampling rounds compiled "nncg" with a learning rate when the last first-order
    phase was NNCG (DeepXDE raises NotImplementedError); IC guided learning ignored AdamW phases (hidden lr
    0.001 and 15000 iterations instead of the phase's)."""
    c = _dx("Burgers equation with RAR", adapt_method="RAR", rar_cycles=1)
    ph = [{"optimizer": "adam", "iterations": 100, "lr": 0.002, "loss": "MSE"},
          {"optimizer": "nncg", "iterations": 50, "lr": 1.0, "loss": "MSE"}]
    c = dataclasses.replace(c, optimizer_scheduler=True, scheduler_phases=json.dumps(ph))
    assert codegen._rar_optimizer(c) == ("adam", 0.002)
    for s in (codegen.generate_script(c), codegen.generate_clean_script(c)):
        assert 'compile("nncg"' not in s
    w = dataclasses.replace(_template("1D Burgers"), time_adaptive=True, ta_num_steps=2, ta_icgl_fraction=0.25,
                            optimizer_scheduler=True, scheduler_phases=json.dumps(
                                [{"optimizer": "adamw", "iterations": 400, "lr": 0.0005, "loss": "MSE"}]))
    assert codegen._ta_icgl_iterations(w) == 100 and codegen._first_adam_lr(w) == 0.0005


def test_initial_loss_weights_use_the_first_phase_loss():
    """23. 'Scale by initial losses' measured L_i(0) with the hidden loss function (MSE), not the first phase's."""
    c = _template("1D Burgers")
    ph = json.loads(c.scheduler_phases)
    ph[0]["loss"] = "MAE"
    c = dataclasses.replace(c, scheduler_phases=json.dumps(ph), loss_weights_from_initial=2.5, loss_type="MSE")
    for s in (codegen.generate_script(c), codegen.generate_clean_script(c)):
        i = s.index("Loss weights from the initial losses: w_i")
        assert 'loss="MAE"' in s[i:i + 300], s[i:i + 300]


def test_a_condition_with_no_training_point_is_reported():
    """24. A boundary condition selecting no point (the 2D Heat template's 'left'/'right' rows after switching
    the domain to a disk) had a NaN loss: training went on without it and L-BFGS stopped with an IndexError in
    PyTorch's line search. Both scripts now print which condition is empty."""
    c = dataclasses.replace(_template("2D Heat"), geometry_type="Disk")
    for s in (codegen.generate_script(c), codegen.generate_clean_script(c)):
        assert "_warn_empty_conditions(data)" in s and "def _warn_empty_conditions(_d):" in s
        compile(s, "script", "exec")


def test_data_files_with_semicolons_missing_values_or_odd_rows_load_the_same_everywhere(tmp_path):
    """P2: the measured-data and IC-file readers of both scripts took only commas and spaces, kept rows with a
    missing value (NaN in the data loss) and failed on a row with an extra column, while the window's reader
    accepted those files. All of them now read ';' too, skip rows with a missing value or an odd number of
    columns (the most common row length wins) and say how many rows they skipped."""
    import re
    import textwrap
    import numpy as np
    from pinnstudio.core import obs_reduce
    c = _dx("Inverse: diffusion equation")
    srcs = {"export": (codegen.generate_clean_script(c), "_load_obs_data"),
            "solve": (codegen.generate_script(c), "_load_data")}
    loaders = {}
    for k, (src, name) in srcs.items():
        m = re.search(r"^( *)def " + name + r"\(.*?(?=^\1\S|\Z)", src, re.S | re.M)
        ns = {"np": np}
        exec(textwrap.dedent(m.group(0)), ns)
        loaders[k] = ns[name]
    files = {"semi.txt": "x;t;u\n0.1;0.5;1.0\n0.2;0.5;2.0\n0.3;0.5;3.0\n",
             "nan.txt": "0.1 0.5 1.0\n0.2 0.5 nan\n0.3 0.5 3.0\n",
             "odd.txt": "0.1,0.5,1.0\n0.2,0.5,2.0,9\n0.3,0.5,3.0\n0.4,0.5,4.0\n"}
    for name, text in files.items():
        p = tmp_path / name
        p.write_text(text)
        shapes = {k: f(str(p)).shape for k, f in loaders.items()}
        shapes["window"] = obs_reduce.read_points(str(p)).shape
        assert len(set(shapes.values())) == 1, (name, shapes)
        assert all(np.isfinite(loaders["export"](str(p))).all() for _ in [0])
    assert loaders["export"](str(tmp_path / "odd.txt")).shape == (3, 3)


def test_model_checkpoint_with_time_windows_saves_each_window():
    """P2: with time windows, Solve's checkpoints of every window had the same name (each window's model counts
    from step 0 again, so later windows overwrote earlier ones), and the export shared one ModelCheckpoint across
    windows (its 'best' loss carried over). Both now save checkpoint_stepNNN_best-<step>.pt per window."""
    c = dataclasses.replace(_template("1D Burgers"), time_adaptive=True, ta_num_steps=2, cb_model_checkpoint=True,
                            cb_checkpoint_period=10, cb_checkpoint_save_better_only=True)
    solve, export = codegen.generate_script(c), codegen.generate_clean_script(c)
    assert 'f"checkpoint_step{step_i + 1:03d}_best"' in solve
    assert 'f"checkpoint_step{step_i + 1:03d}_best"' in export and "_cb.best, _cb.epochs_since_last_save = np.inf, 0" in export
    whole = codegen.generate_clean_script(dataclasses.replace(c, time_adaptive=False))   # whole interval: unchanged
    assert '"checkpoint_best"' in whole and "checkpoint_step" not in whole


def test_ic_pretraining_restore_needs_an_existing_model(tmp_path):
    """P2: 'Restore' with a wrong or empty path silently ran the whole IC pre-training instead."""
    c = dataclasses.replace(_template("1D Burgers"), ic_pretrain=True, ic_pretrain_restore=True)
    for path in ("", str(tmp_path / "missing.pt")):
        errs = dataclasses.replace(c, ic_pretrain_restore_path=path).validate()
        assert any("IC pre-training" in e and ("no saved" in e or "does not exist" in e) for e in errs), errs
    ok = tmp_path / "ic_pretrain_model-100.pt"
    ok.write_bytes(b"x")
    assert not any("saved pre-training model" in e
                   for e in dataclasses.replace(c, ic_pretrain_restore_path=str(ok)).validate())
    for s in (codegen.generate_script(c), codegen.generate_clean_script(c)):
        assert "IC pre-train model to restore not found" in s


def test_solution_data_export_marks_points_outside_the_domain():
    """P2: 'Export solution data' wrote the whole bounding-box grid for a disk or polygon (extrapolated values
    outside the domain, unmarked); those points are now NaN. An ODE's file header named its variable x."""
    c = dataclasses.replace(_template("2D Heat"), geometry_type="Disk", export_enabled=True)
    s = codegen.generate_script(c)
    assert "_u_exp = _nan_outside(_XYT_exp[:, :2], model.predict(_XYT_exp))" in s
    o = codegen.generate_script(dataclasses.replace(_dx("ODE system"), export_enabled=True))
    assert '_header_cols = "t," + ",".join(_out_names_list)' in o


def test_loss_functions_that_do_not_work_for_a_pinn_are_refused():
    """P1 (found while writing the documentation): three loss functions offered in the window did not work:
    'mean l2 relative error' and 'mean absolute percentage error' divide by the residual target, which is 0
    (infinite loss), and the latter and 'softplus' fail inside DeepXDE with PyTorch. The window offers MSE and
    MAE only; a saved problem naming one of the others is refused."""
    c = _template("1D Burgers")
    assert not any("Loss function" in e for e in c.validate())
    for bad in ("mean l2 relative error", "mean absolute percentage error", "softplus"):
        ph = json.loads(c.scheduler_phases)
        ph[0]["loss"] = bad
        errs = dataclasses.replace(c, scheduler_phases=json.dumps(ph)).validate()
        assert any("Loss function" in e and bad in e for e in errs), errs
        assert any("Loss function" in e for e in dataclasses.replace(c, ic_pretrain_loss=bad).validate())


def test_ic_pretraining_needs_a_time_dependent_pde():
    """IC pre-training could be ticked for an ODE (its initial values are conditions at t0, so Solve pre-trained
    on nothing while the export left it out) or a steady problem."""
    errs = dataclasses.replace(_dx("ODE system"), ic_pretrain=True).validate()
    assert any("IC pre-training needs a time-dependent PDE" in e for e in errs), errs


def test_tuned_settings_of_field_identification_examples_are_offered():
    """DeepXDE's inverse examples that identify an unknown field as a network output load as Forward problems;
    their Tuned settings (stored under 'inverse' by the tuning study) were looked up under 'forward', so Tuned
    was greyed out."""
    from pinnstudio.core import example_settings as es
    for name in ("Inverse: Poisson with unknown forcing field", "Inverse: Poisson with unknown forcing field (batches)",
                 "Inverse: diffusion-reaction with space-dependent rate"):
        c = _dx(name)
        assert c.problem_type == "Forward" and es.has_tuned(name, False), name
        out, msg = es.settings_for(c, name, False, es.TUNED)
        assert msg.startswith("Tuned settings: ") and "none for this example" not in msg, msg
    assert es._mode_key("1D Burgers", False) == "forward" and es._mode_key("1D Burgers", True) == "inverse"
