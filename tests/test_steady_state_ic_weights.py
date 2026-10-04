#!/usr/bin/env python3
"""
Steady-state problems have no time axis, so there's no Initial Condition
at all -- codegen.py already reflects this correctly at training time
(it builds an empty IC-weights list for a steady-state config, confirmed
by the "IC weights: []" line a real training run prints), and the
Initial Condition panel itself (ic_type/ic_expression) is already hidden
by _on_steady_state_changed() when Steady-state is checked.

But the separate Loss Weights panel's own per-output "IC {n} (...)"
weight rows were built by _build_weight_inputs() based only on each
output's ic_active checkbox, with no steady_state check at all -- so
those rows (and their weight fields) stayed visible and editable even
for a steady-state problem, even though the values were silently never
used. Found and reported by a user after loading a steady-state Custom-
geometry problem and asking why IC weights still showed.

Fix: _build_weight_inputs() now skips every "IC ..." row (both the
shared-weights block and each per-phase block) whenever Steady-state is
checked, and _on_steady_state_changed() now refreshes the Loss Weights
panel on toggle (previously it only updated the Initial Condition panel,
Adaptive Training, IC Pre-Training, and the "t:" row -- the Loss Weights
panel only got refreshed as a side effect of something else, e.g.
changing the output count, so toggling Steady-state on its own could
leave stale IC rows showing until an unrelated change happened to
rebuild the panel).

Checks:
 - A non-steady config's Loss Weights panel still shows one "IC {n}"
   row per active output -- this fix must not remove anything that
   should still be there.
 - Checking Steady-state removes every "IC ..." row from the panel,
   live (no other change needed to trigger the rebuild).
 - Unchecking Steady-state again restores them.
 - The same holds with per-phase (not shared) weights enabled, where
   each phase has its own "IC ... P{n}" rows.
 - _flat_loss_weights_string() (what actually gets saved/trained on)
   naturally emits no IC terms once the rows are gone, since it already
   only reads keys present in weight_widgets.
 - This is purely a Loss Weights GUI change: PDE and BC rows, and the
   values of any currently-visible weight widgets, are unaffected.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_steady_state_ic_weights.py
"""
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


def _ic_keys(win):
    return sorted(k for k in win.weight_widgets if k.startswith("ic_"))


def _pde_keys(win):
    return sorted(k for k in win.weight_widgets if k.startswith("pde_"))


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── Baseline: non-steady, shared weights -- IC rows present ────
    win = MainWindow()
    win.radio_2d.setChecked(True)
    win._on_dim_changed()
    win.num_outputs_spin.setValue(2)
    win._build_weight_inputs(2)
    check(not win.steady_state_check.isChecked(), "sanity: steady-state should start unchecked")
    check(_ic_keys(win) == ["ic_0", "ic_1"],
          f"a non-steady 2-output config should show IC rows for both outputs, got {_ic_keys(win)}")
    check(_pde_keys(win) == ["pde_0", "pde_1"], f"PDE rows should be unaffected, got {_pde_keys(win)}")

    # ── Checking Steady-state live-removes the IC rows ─────────────
    win.steady_state_check.setChecked(True)
    check(_ic_keys(win) == [],
          f"checking Steady-state should remove every IC weight row immediately, got {_ic_keys(win)}")
    check(_pde_keys(win) == ["pde_0", "pde_1"],
          "PDE rows should still be present after switching to steady-state")

    # ── Unchecking restores them ────────────────────────────────────
    win.steady_state_check.setChecked(False)
    check(_ic_keys(win) == ["ic_0", "ic_1"],
          f"unchecking Steady-state should restore the IC weight rows, got {_ic_keys(win)}")

    # ── Building fresh with Steady-state already checked ───────────
    # (not just toggling -- also covers _apply_config()'s own call order:
    # steady_state_check.setChecked() + _on_steady_state_changed() happen
    # before/independently of whatever next calls _build_weight_inputs().)
    win2 = MainWindow()
    win2.radio_2d.setChecked(True)
    win2._on_dim_changed()
    win2.num_outputs_spin.setValue(3)
    win2.steady_state_check.setChecked(True)
    win2._build_weight_inputs(3)
    check(_ic_keys(win2) == [], f"building weight inputs while steady-state is checked should yield no IC rows, got {_ic_keys(win2)}")

    # ── Per-phase (not shared) weights: "IC ... P{n}" rows too ──────
    win3 = MainWindow()
    win3.radio_2d.setChecked(True)
    win3._on_dim_changed()
    win3.num_outputs_spin.setValue(2)
    if hasattr(win3, 'sched_same_weights_cb'):
        win3.sched_same_weights_cb.setChecked(False)
    win3._build_weight_inputs(2)
    _pre_phase_ic = [k for k in win3.weight_widgets if k.startswith("ic_")]
    check(len(_pre_phase_ic) > 0,
          f"a non-steady config with per-phase weights should show per-phase IC rows, got {list(win3.weight_widgets)}")
    win3.steady_state_check.setChecked(True)
    _post_phase_ic = [k for k in win3.weight_widgets if k.startswith("ic_")]
    check(_post_phase_ic == [],
          f"checking Steady-state should also remove per-phase IC rows, got {_post_phase_ic}")
    _post_phase_pde = [k for k in win3.weight_widgets if k.startswith("pde_")]
    check(len(_post_phase_pde) > 0, "per-phase PDE rows should still be present after switching to steady-state")

    # ── _flat_loss_weights_string() emits no IC terms once hidden ──
    win4 = MainWindow()
    win4.radio_2d.setChecked(True)
    win4._on_dim_changed()
    win4.num_outputs_spin.setValue(2)
    win4.steady_state_check.setChecked(True)
    win4._build_weight_inputs(2)
    flat = win4._flat_loss_weights_string(2)
    parts = flat.split(",") if flat else []
    check(len(parts) == 2, f"a steady-state 2-output config with no BC rows should emit exactly 2 weights (PDE only), got {parts!r}")

    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL STEADY-STATE IC-WEIGHTS TESTS PASSED")
    return 0


def test_steady_state_ic_weights():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
