#!/usr/bin/env python3
"""
Regression check for pinnstudio/core/sweep_registry.py -- the Parameter
Sweep feature's Phase 1 (config fields) + Phase 2 (registry) foundation.
No GUI sweep tab exists yet (that's Phase 3); this exercises the
registry and PINNConfig.validate()'s new sweep checks directly against
real configs built the same way the GUI does (MainWindow._build_config()
after selecting a template), across 1D/2D/3D and Forward/Inverse.

Checks, for a representative spread of templates:
 - available_params() returns unique ids, with the expected static
   entries always present (and num_initial correctly excluded for
   steady-state templates).
 - Phase-scoped entries (learning rate/iterations/optimizer) appear once
   per configured scheduler phase, with learning rate specifically
   omitted for any phase using an optimizer that has no meaningful LR
   (L-BFGS/NNCG) -- and that the values they report match the phase's
   real, already-configured values (not the legacy single-phase fields,
   which is exactly the class of bug this registry exists to avoid).
 - Every available parameter's get_value()/set_value() round-trips
   correctly (set to the value just read back, read again, must match).
 - Setting hidden_layers/neurons_per_layer actually reshapes
   config.layers as expected, preserving input/output size.
 - PINNConfig.validate()'s new sweep_enabled checks catch each class of
   malformed sweep_parameters JSON, and accept well-formed ones.
 - Loss weight (PDE/BC/IC) sweep entries ("SCOPE (v2)" in
   sweep_registry.py): slot order/keys match config.custom_bc_json/
   ic_active exactly across every CASE below, including a 3D template
   with several Boundary Conditions panel rows and a steady-state
   template with none; setting one slot resyncs EVERY scheduler phase's
   own "weights" string at that index without disturbing any other
   slot; the swept value is actually present in the generated training
   script's embedded per-phase weights (the specific class of "shared
   value looks right, actual training phase is still stale" bug this
   feature's resync logic exists to prevent); the entries disappear
   entirely when per-phase weights are allowed to diverge
   (scheduler_same_weights=False, where "the PDE weight" would be
   ambiguous) but remain available with the scheduler off entirely
   (still exactly one unambiguous copy); an Inverse problem's
   observation-file weight tail is left untouched by a PDE/BC/IC sweep.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_sweep_registry.py
"""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core import sweep_registry as reg

# A spread across every dimension, steady vs time-dependent, and Forward
# vs Inverse -- not the full 24-combination matrix test_templates.py
# already covers, just enough to exercise every code path in the
# registry (scheduler phases of varying optimizer, steady-state's
# num_initial exclusion, each dimension's own template set).
CASES = [
    ("1d", "1D Heat", False),
    ("1d", "1D Heat", True),
    ("2d", "2D Poisson (Disk)", False),   # steady-state 2D
    ("3d", "3D Poisson (Sphere)", False), # steady-state 3D
    ("3d", "3D Heat", False),             # time-dependent 3D
]


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    _app.processEvents()

    for dim, name, inverse in CASES:
        {"1d": win.radio_1d, "2d": win.radio_2d, "3d": win.radio_3d}[dim].setChecked(True)
        win.quick_examples_combo.setCurrentText(name)
        win.radio_forward.setChecked(not inverse)
        win.radio_inverse.setChecked(inverse)
        config = win._build_config()
        label = f"{name} inv={inverse}"

        params = reg.available_params(config)
        ids = [p.id for p in params]
        check(len(ids) == len(set(ids)), f"[{label}] duplicate ids in available_params(): {ids}")

        static_ids = {"weight_decay", "hidden_layers", "neurons_per_layer",
                      "num_domain", "num_boundary", "num_test"}
        check(static_ids.issubset(set(ids)),
              f"[{label}] missing expected static entries: {static_ids - set(ids)}")
        check(("num_initial" in ids) == (not config.steady_state),
              f"[{label}] num_initial availability ({'num_initial' in ids}) doesn't match "
              f"steady_state={config.steady_state}")

        n_phases = reg.num_phases(config)
        check(n_phases > 0, f"[{label}] expected the default Optimizer Scheduler to be active "
                             f"with >=1 phase, got {n_phases}")
        for i in range(n_phases):
            check(f"phase{i}_iterations" in ids, f"[{label}] missing phase{i}_iterations")
            check(f"phase{i}_optimizer" in ids, f"[{label}] missing phase{i}_optimizer")
            opt_param = reg.get_param(f"phase{i}_optimizer", config)
            real_opt = opt_param.get_value(config)
            lr_available = f"phase{i}_lr" in ids
            check(lr_available == (real_opt in ("adam", "adamw")),
                  f"[{label}] phase{i}_lr availability ({lr_available}) doesn't match "
                  f"this phase's real optimizer ({real_opt!r})")

        # Round-trip every available parameter: read, write back the same
        # value, read again, must match. Catches accessor bugs (wrong
        # index, wrong JSON key, stale cached state) without needing to
        # know each parameter's "correct" value in advance.
        for p in params:
            try:
                v1 = p.get_value(config)
                p.set_value(config, v1)
                v2 = p.get_value(config)
            except Exception as e:
                failures.append(f"[{label}] {p.id} round-trip raised: {e}")
                continue
            check(v1 == v2, f"[{label}] {p.id} round-trip mismatch: {v1!r} -> set -> {v2!r}")

        # hidden_layers / neurons_per_layer actually reshape config.layers,
        # preserving input/output size.
        in_sz, out_sz = config.layers[0], config.layers[-1]
        hl_param = reg.get_param("hidden_layers", config)
        hl_param.set_value(config, 5)
        check(len(config.layers) == 7,
              f"[{label}] hidden_layers=5 should give a 7-entry layers list, got {config.layers}")
        check(config.layers[0] == in_sz and config.layers[-1] == out_sz,
              f"[{label}] hidden_layers set changed input/output size: {config.layers}")
        np_param = reg.get_param("neurons_per_layer", config)
        np_param.set_value(config, 128)
        check(all(w == 128 for w in config.layers[1:-1]),
              f"[{label}] neurons_per_layer=128 didn't reach every hidden layer: {config.layers}")

    # ── PINNConfig.validate() sweep checks, against one simple config ──
    win.radio_1d.setChecked(True)
    win.quick_examples_combo.setCurrentText("1D Heat")
    win.radio_forward.setChecked(True)
    base_config = win._build_config()
    available = reg.available_params(base_config)
    some_id = available[0].id if available else None
    check(some_id is not None, "no available sweep params on 1D Heat -- can't test validate()")

    def errs_for(sweep_params_obj, enabled=True, mode="oat"):
        c = win._build_config()
        c.sweep_enabled = enabled
        c.sweep_mode = mode
        c.sweep_parameters = json.dumps(sweep_params_obj)
        return c.validate()

    check(errs_for([], enabled=False) == [], "sweep disabled with empty params should validate clean")
    check(any("no parameters" in e for e in errs_for([])),
          "sweep enabled with empty params list should error")
    check(any("not valid JSON" in e or "not a sweepable parameter" in e
               for e in errs_for("not a list")),  # type: ignore[arg-type]
          "non-list sweep_parameters should error")
    if some_id:
        ok_list = [{"id": some_id, "mode": "list", "values": [1, 2, 3]}]
        check(errs_for(ok_list) == [], f"well-formed list-mode sweep should validate clean, got {errs_for(ok_list)}")
        ok_range = [{"id": some_id, "mode": "linear", "min": 1.0, "max": 10.0, "n": 5}]
        check(errs_for(ok_range) == [], f"well-formed linear-range sweep should validate clean, got {errs_for(ok_range)}")
        bad_range = [{"id": some_id, "mode": "linear", "min": 10.0, "max": 1.0, "n": 5}]
        check(any("min < max" in e for e in errs_for(bad_range)),
              "min >= max should be flagged")
        bad_n = [{"id": some_id, "mode": "linear", "min": 1.0, "max": 10.0, "n": 1}]
        check(any("at least 2 steps" in e for e in errs_for(bad_n)),
              "n < 2 should be flagged")
        bad_log = [{"id": some_id, "mode": "log", "min": -1.0, "max": 10.0, "n": 5}]
        check(any("log spacing" in e for e in errs_for(bad_log)),
              "non-positive min/max with log mode should be flagged")
        bad_id = [{"id": "not_a_real_param_xyz", "mode": "list", "values": [1]}]
        check(any("not a sweepable parameter" in e for e in errs_for(bad_id)),
              "unknown parameter id should be flagged")
        bad_mode = [{"id": some_id, "mode": "oat"}]
        check(errs_for([{"id": "bad", "mode": "oat"}]) != [], "")  # sanity: bad id + bad mode both flagged
        unrecognized_mode = [{"id": some_id, "mode": "wat"}]
        check(any("unrecognized mode" in e for e in errs_for(unrecognized_mode)),
              "unrecognized sweep mode should be flagged")
        check(errs_for(ok_list, mode="zip") == [],
              f"a single-parameter zip-mode sweep should validate clean (no length "
              f"mismatch possible with only 1 param), got {errs_for(ok_list, mode='zip')}")

    # ── "zip" (Specified Combinations) sweep mode -- requires every ───
    # swept parameter to carry the same number of values; mirrors
    # COMSOL's "Specified combinations" semantics (run i of every
    # parameter's list together, instead of a full cross product).
    if len(available) >= 2:
        id_a, id_b = available[0].id, available[1].id
        zip_ok = [
            {"id": id_a, "mode": "list", "values": [1, 2, 3]},
            {"id": id_b, "mode": "list", "values": [10, 20, 30]},
        ]
        check(errs_for(zip_ok, mode="zip") == [],
              f"equal-length zip-mode sweep should validate clean, got {errs_for(zip_ok, mode='zip')}")
        zip_mismatch = [
            {"id": id_a, "mode": "list", "values": [1, 2, 3]},
            {"id": id_b, "mode": "list", "values": [10, 20]},
        ]
        check(any("Specified Combinations" in e for e in errs_for(zip_mismatch, mode="zip")),
              "mismatched value-list lengths in zip mode should be flagged")
        zip_mixed_range = [
            {"id": id_a, "mode": "linear", "min": 1.0, "max": 10.0, "n": 4},
            {"id": id_b, "mode": "list", "values": [10, 20, 30]},
        ]
        check(any("Specified Combinations" in e for e in errs_for(zip_mixed_range, mode="zip")),
              "mismatched lengths between a linear range (n=4) and a list (3 values) "
              "in zip mode should be flagged")
        # Same sweep_parameters is fine under "oat"/"grid" (length check is zip-only).
        check(errs_for(zip_mismatch, mode="oat") == [],
              "the zip-only length check must not fire for oat mode")
        check(errs_for(zip_mismatch, mode="grid") == [],
              "the zip-only length check must not fire for grid mode")

    # ── Loss weight (PDE/BC/IC) sweep entries -- "SCOPE (v2)" ──────────
    # Exercises sweep_registry.py's _weight_* helpers directly against
    # real GUI-built configs across the same CASES spread above, plus
    # the scheduler-phase resync these entries specifically depend on
    # (the round-trip loop above already covers plain get/set for these
    # the same way it does every other entry -- this section checks the
    # things unique to loss weights: slot count/order, the ambiguous
    # per-phase-weights gate, resync into every phase, and that the
    # swept value actually reaches the generated training script).
    for dim, name, inverse in CASES:
        {"1d": win.radio_1d, "2d": win.radio_2d, "3d": win.radio_3d}[dim].setChecked(True)
        win.quick_examples_combo.setCurrentText(name)
        win.radio_forward.setChecked(not inverse)
        win.radio_inverse.setChecked(inverse)
        config = win._build_config()
        label = f"{name} inv={inverse}"

        n_out = config.num_outputs
        bc_entries = reg._weight_bc_entries(config)
        active_ic = [] if config.steady_state else reg._weight_active_ic_outputs(config, n_out)
        slots = reg._weight_slot_descriptors(config)
        expected_keys = (
            [f"pde_{i}" for i in range(n_out)]
            + [f"bc_{j}" for j in range(len(bc_entries))]
            + [f"ic_{i}" for i in active_ic]
        )
        check([s["key"] for s in slots] == expected_keys,
              f"[{label}] weight slot order/keys mismatch: got {[s['key'] for s in slots]}, "
              f"expected {expected_keys}")
        check(len(slots) == len(reg._weight_flat_list(config)),
              f"[{label}] _weight_flat_list() length doesn't match slot count")
        check(("ic_0" not in [s["key"] for s in slots]) if config.steady_state else True,
              f"[{label}] steady-state config should have NO IC weight slot at all")

        weight_ids = {p.id for p in reg.available_params(config) if p.category == "Loss Weights"}
        expected_weight_ids = {f"weight_{k}" for k in expected_keys}
        check(weight_ids == expected_weight_ids,
              f"[{label}] Loss Weights category ids don't match slot descriptors: "
              f"{weight_ids} vs {expected_weight_ids}")

        # Per-phase resync: setting one slot must update EVERY scheduler
        # phase's own "weights" string at that same index, not just the
        # shared config.loss_weights_multi -- codegen.py reads each
        # phase's own copy once the scheduler is active (the default),
        # so a resync bug here would silently train with the old value.
        if expected_keys:
            target_key = expected_keys[-1]
            probe_value = 12.5
            probe_config = win._build_config()
            reg._weight_set(probe_config, target_key, probe_value)
            idx = expected_keys.index(target_key)
            flat = [float(v) for v in probe_config.loss_weights_multi.split(",")]
            check(abs(flat[idx] - probe_value) < 1e-9,
                  f"[{label}] loss_weights_multi wasn't updated at slot {idx} for {target_key}")
            phases = json.loads(probe_config.scheduler_phases) if probe_config.scheduler_phases else []
            check(bool(phases), f"[{label}] expected scheduler phases to resync against")
            for pi, ph in enumerate(phases):
                ph_w = [float(x) for x in ph["weights"].split(",")]
                check(idx < len(ph_w) and abs(ph_w[idx] - probe_value) < 1e-9,
                      f"[{label}] phase {pi} weights not resynced for {target_key}: {ph['weights']}")
                # Everything BEFORE the swept slot must be untouched.
                for other_idx in range(idx):
                    check(abs(ph_w[other_idx] - flat[other_idx]) < 1e-9,
                          f"[{label}] phase {pi} slot {other_idx} changed unexpectedly "
                          f"when only slot {idx} ({target_key}) was set")

            # Static regression check for the exact bug class this
            # module's docstring warns about: generate the real training
            # script and confirm the swept value is present in EVERY
            # phase's embedded weights string, not just the shared
            # loss_weights_multi (which would look right even if the
            # per-phase resync were silently broken).
            from pinnstudio.core.codegen import generate_script
            script = generate_script(probe_config)
            for ph in phases:
                check(json.dumps(ph["weights"]) in script,
                      f"[{label}] generated script is missing phase weights string "
                      f"{ph['weights']!r} for the swept config -- a phase may be "
                      "training with a stale value")

        # Ambiguous case: per-phase weights allowed to diverge -- loss
        # weight sweeping must be withdrawn entirely rather than silently
        # sweeping only one phase's copy.
        ambiguous_config = win._build_config()
        ambiguous_config.scheduler_same_weights = False
        check(not any(p.category == "Loss Weights" for p in reg.available_params(ambiguous_config)),
              f"[{label}] Loss Weights entries should disappear when scheduler_same_weights=False")

        # Scheduler off entirely: still exactly one unambiguous set of
        # weights (the synthesized single phase), so still available.
        no_sched_config = win._build_config()
        no_sched_config.optimizer_scheduler = False
        no_sched_ids = {p.id for p in reg.available_params(no_sched_config) if p.category == "Loss Weights"}
        check(no_sched_ids == expected_weight_ids,
              f"[{label}] Loss Weights entries should still be offered with the scheduler off")

    # 3D Heat specifically: 6 BC panel rows per output (box faces) --
    # confirms the slot count genuinely tracks a config with several BC
    # rows, not just the 1-2 row cases above.
    win.radio_3d.setChecked(True)
    win.quick_examples_combo.setCurrentText("3D Heat")
    win.radio_forward.setChecked(True)
    heat3d_config = win._build_config()
    heat3d_slots = reg._weight_slot_descriptors(heat3d_config)
    n_out_3d = heat3d_config.num_outputs
    bc_slots_3d = [s for s in heat3d_slots if s["key"].startswith("bc_")]
    check(len(bc_slots_3d) == 6 * n_out_3d,
          f"3D Heat should have 6 BC weight slots per output, got {len(bc_slots_3d)} "
          f"for {n_out_3d} output(s)")

    # Inverse: the observation-file weight tail (inverse_obs_files_json)
    # must be left completely untouched by a PDE/BC/IC weight sweep --
    # it's a separate mechanism (see module docstring).
    win.radio_1d.setChecked(True)
    win.quick_examples_combo.setCurrentText("1D Heat")
    win.radio_inverse.setChecked(True)
    inv_config = win._build_config()
    if reg.sched_active(inv_config) and json.loads(inv_config.scheduler_phases):
        n_slots_inv = len(reg._weight_slot_descriptors(inv_config))
        before_phases = json.loads(inv_config.scheduler_phases)
        before_tail = [ph["weights"].split(",")[n_slots_inv:] for ph in before_phases]
        reg._weight_set(inv_config, "pde_0", 42.0)
        after_phases = json.loads(inv_config.scheduler_phases)
        after_tail = [ph["weights"].split(",")[n_slots_inv:] for ph in after_phases]
        check(before_tail == after_tail,
              f"Inverse observation-weight tail changed after a PDE weight sweep set: "
              f"{before_tail} -> {after_tail}")
    win.radio_inverse.setChecked(False)
    win.radio_forward.setChecked(True)

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL SWEEP REGISTRY TESTS PASSED")
    return failures


def test_sweep_registry():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
