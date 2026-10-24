"""Time-adaptive training with one network through all windows: Wight &
Zhao's growing time range, Mattey & Ghosh's backward compatibility (bc-PINN)
and initial-condition guided learning (ICGL)."""
import dataclasses
import json

from pinnstudio.core.codegen import _ta_icgl_iterations, generate_script
from pinnstudio.core.config import PINNConfig

BASE = dict(problem_dim="1D", time_adaptive=True, ta_num_steps=4)


def _cfg(**kw):
    return dataclasses.replace(PINNConfig(), **{**BASE, **kw})


def _errors(c):
    r = c.validate()
    return r if isinstance(r, list) else list(r or [])


def test_one_network_options_need_transfer_learning():
    for kw in (dict(ta_backward=True), dict(ta_growing=True)):
        assert any("transfer learning" in e for e in _errors(_cfg(**kw)))
        assert not any("transfer learning" in e for e in _errors(_cfg(ta_transfer_learning=True, **kw)))


def test_icgl_fraction_range():
    assert any("guided learning" in e for e in _errors(_cfg(ta_icgl_fraction=1.0)))
    assert not any("guided learning" in e for e in _errors(_cfg(ta_icgl_fraction=0.2)))


def test_icgl_iterations_follow_the_first_adam_phase():
    phases = [{"optimizer": "adam", "iterations": 15000}, {"optimizer": "lbfgs", "iterations": 10000}]
    c = _cfg(ta_icgl_fraction=0.2, optimizer_scheduler=True, scheduler_phases=json.dumps(phases))
    assert _ta_icgl_iterations(c) == 3000
    assert _ta_icgl_iterations(_cfg(ta_icgl_fraction=0.0)) == 0


def test_script_settings_and_compiles():
    s = generate_script(_cfg(ta_transfer_learning=True, ta_backward=True, ta_backward_weight=64.0,
                             ta_backward_times=7, ta_backward_points=128, ta_icgl_fraction=0.1, iterations=1000))
    for want in ("_ta_bc = True", "_ta_bc_w = 64.0", "_ta_bc_nt = 7", "_ta_bc_np = 128", "_ta_icgl_iters = 100",
                 "_ta_growing = False", "loss_weights=_wx(_multi_weights)"):
        assert want in s, want
    compile(s, "<script>", "exec")
    off = generate_script(_cfg())
    assert "_ta_bc = False" in off and "_ta_icgl_iters = 0" in off
    compile(off, "<script>", "exec")


def test_resampling_in_each_window():
    """Time-adaptive resampling (Wight & Zhao's approach I + resampling): the script adds the
    largest-residual candidates and trains on; bad settings are refused before training."""
    import dataclasses
    from pinnstudio.core.config import PINNConfig
    from pinnstudio.core.codegen import generate_script
    c = PINNConfig(time_adaptive=True, adapt_method="Time Adaptive", ta_resample_rounds=2)
    assert not [m for m in c.validate() if "resampling" in m]
    src = generate_script(c)
    assert "Resampling round" in src and "data_i.add_anchors(_rs_new)" in src and "_ta_rs_rounds = 2" in src
    bad = dataclasses.replace(c, ta_resample_candidates=10, ta_resample_points=50)
    assert any("resampling needs" in m for m in bad.validate())
    bad = dataclasses.replace(c, point_weights="RBA")
    assert any("per-point weights" in m for m in bad.validate())
