#!/usr/bin/env python3
"""
Run this on the `param-sweep` branch, on top of commit `9d47de1` (the
Phase 5 results-plot patch). Touches pinnstudio/core/sweep_registry.py,
tests/test_sweep_registry.py, and tests/test_sweep_execution.py.

WHAT THIS ADDS -- loss weight (PDE/BC/IC) sweeping

Adds Loss Weights to the Parameter Sweep tab's parameter list -- one
entry per PDE term, per Boundary Conditions panel row, and per active
Initial Condition, exactly matching what shows up (and in what order) in
the Setup tab's own "Loss Weights" panel. This was deliberately deferred
out of the original Phase 1+2 patch: these weights live in a single
flat, comma-separated string (config.loss_weights_multi) whose column
order depends on the number of outputs, how many rows are in the
Boundary Conditions panel, and which outputs currently have an Initial
Condition active -- exactly the kind of config-dependent indexing that
needed its own careful, separately-tested pass rather than a guess.

No GUI changes were needed for this -- the Parameter Sweep tab (Phase 3)
already builds its dropdown and its results plot entirely from
sweep_registry.available_params(), so a new registry entry just shows
up, with zero changes to pinnstudio/ui/main_window.py.

THE ACTUAL HARD PART, SOLVED HERE: once the Optimizer Scheduler is
active (the default for every template), codegen.py reads EACH
TRAINING PHASE'S OWN "weights" string, not config.loss_weights_multi
directly. In the GUI those never diverge, because _build_scheduler_
phases_json() rebuilds every phase's weights from the exact same
widgets loss_weights_multi comes from, every single time a config is
built. But a swept config is produced by mutating a deep copy directly
(sweep_runner.build_runs()), bypassing that GUI rebuild entirely -- so
setting ONLY loss_weights_multi would have silently left every phase
training with its old, stale weight. Fixed by having the new weight
parameters' set_value() also re-sync every scheduler phase's own
"weights" string at the same index. Verified two independent ways: (1)
a static check that generates the real training script and confirms
the swept value is present in EVERY phase's embedded weights string,
not just the shared loss_weights_multi (which would look right even if
the per-phase resync were silently broken); (2) an actual real-training
run (1D Heat, Initial Condition weight pushed from 100 to 100000 for a
handful of iterations) confirming the real, parsed final_loss actually
changes -- if the sweep were silently using the stale weight, baseline
and swept would come back statistically identical.

SCOPE: PDE/BC/IC weights only (config.loss_weights_multi). Inverse
observation-file weights (inverse_obs_files_json) are a separate
mechanism with their own per-file spinbox already and are NOT included
here -- a PDE/BC/IC weight sweep leaves that tail completely untouched
(verified in tests/test_sweep_registry.py). Sweeping a loss weight is
only offered when there's exactly one unambiguous copy of the weights
for the whole run -- the Optimizer Scheduler off, or "Same weights (all
phases)" checked (the default for every template). When per-phase
weights are allowed to diverge (scheduler_same_weights=False), "the PDE
weight" is ambiguous -- which phase's copy? -- so these entries
disappear entirely in that case rather than silently sweeping only one
phase's copy while the others train with whatever they already had.

VERIFICATION DONE

 - New tests/test_sweep_registry.py checks, across the same 1D/2D/3D x
   steady/time-dependent x Forward/Inverse spread the file already
   covered: weight slot order/keys match config.custom_bc_json/
   ic_active exactly (including a 3D template with 6 Boundary
   Conditions panel rows per output, and steady-state templates with no
   Initial Condition slot at all); setting one slot resyncs EVERY
   scheduler phase's own weights string at that index without
   disturbing any other slot; the swept value is actually present in
   the generated training script's embedded per-phase weights (the
   specific "shared value looks right, phase is still stale" bug class
   described above); the entries disappear when per-phase weights are
   allowed to diverge but remain available with the scheduler off
   entirely; an Inverse problem's observation-file weight tail is left
   untouched by a PDE/BC/IC sweep.
 - New tests/test_sweep_execution.py check (real training, not CI-wired
   -- see that file's own docstring): a real sweep over the Initial
   Condition weight on 1D Heat, confirming the real final_loss actually
   differs from baseline.
 - Full existing regression suite (test_templates.py 24/24,
   test_config_options.py, test_export_parity.py,
   test_v56_no_template_solve.py, test_sweep_tab.py) all still pass --
   this patch only adds new registry entries, nothing existing changed
   behavior. pinnstudio/ui/main_window.py, pinnstudio/core/config.py,
   pinnstudio/core/codegen.py, and pinnstudio/core/sweep_runner.py are
   completely untouched by this patch.
 - Patch verified to apply cleanly, byte-identical result confirmed, in
   a fresh independent clone of param-sweep at commit 9d47de1, with the
   full test suite (including the new checks) re-run there and passing.

Applies cleanly to a clean checkout of the `param-sweep` branch at
commit 9d47de1.

Run from the repo root:
    python3 apply_v67_fixes.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "v67_loss_weight_sweep_delta.patch"
MARKER = "_weight_slot_descriptors"


def find_repo_root():
    here = Path(__file__).resolve().parent
    for candidate in (Path.cwd(), here):
        if (candidate / "pinnstudio" / "core" / "codegen.py").is_file():
            return candidate
    return None


def git_apply_check(root, patch_path, reverse=False):
    cmd = ["git", "apply", "--check", str(patch_path)]
    if reverse:
        cmd.insert(2, "--reverse")
    return subprocess.run(cmd, cwd=root, capture_output=True, text=True)


def main():
    root = find_repo_root()
    if root is None:
        print("Could not find 'pinnstudio/core/codegen.py' under the current "
              "directory or this script's directory.\n"
              "cd into your PINNStudio repo root (the folder containing the "
              "'pinnstudio' folder) and run this script from there.")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    patch_path = script_dir / PATCH_FILE
    if not patch_path.is_file():
        print(f"Expected to find '{PATCH_FILE}' next to this script (looked "
              f"in {script_dir}), but it's missing.\nMake sure you "
              "downloaded both files into the same folder.")
        sys.exit(1)

    if not (root / ".git").is_dir():
        print(f"'{root}' doesn't look like a git repository (no .git folder) "
              "-- this patch is applied with 'git apply', which needs one.")
        sys.exit(1)

    registry_path = root / "pinnstudio" / "core" / "sweep_registry.py"
    text = registry_path.read_text() if registry_path.is_file() else ""

    if MARKER in text:
        print("Already applied -- this fix is already present in this "
              "checkout. Nothing to do.")
        sys.exit(0)

    check = git_apply_check(root, patch_path)
    if check.returncode != 0:
        if git_apply_check(root, patch_path, reverse=True).returncode == 0:
            print("Already applied -- this fix is already present in this "
                  "checkout. Nothing to do.")
            sys.exit(0)
        print("This patch doesn't apply cleanly here. Most likely the "
              "checkout has diverged from the expected base (local edits, "
              "or you're not on the param-sweep branch at commit 9d47de1 -- "
              "run the earlier patches first if you haven't already).\n\n"
              "git apply's error:\n" + check.stderr)
        sys.exit(1)

    apply = subprocess.run(
        ["git", "apply", str(patch_path)], cwd=root,
        capture_output=True, text=True,
    )
    if apply.returncode != 0:
        print("git apply failed unexpectedly after passing --check:\n" + apply.stderr)
        sys.exit(1)

    print("Applied cleanly:")
    print("  - pinnstudio/core/sweep_registry.py")
    print("  - tests/test_sweep_registry.py")
    print("  - tests/test_sweep_execution.py")
    print()
    print("Next: review with 'git diff', then commit and push to")
    print("param-sweep (NOT main), e.g.")
    print("  git add pinnstudio/core/sweep_registry.py \\")
    print("          tests/test_sweep_registry.py tests/test_sweep_execution.py")
    print('  git commit -m "Parameter Sweep: loss weight (PDE/BC/IC) sweeping"')
    print("  git push")
    print()
    print("Then open the Parameter Sweep tab, hit Refresh if the dropdown")
    print("is already open, and you should see a new 'Loss Weights' group")
    print("of entries (PDE 1 (...), BC 1 (...), IC 1 (...), etc.) alongside")
    print("the existing Network/Collocation Points/Phase N entries.")
    print("IMPORTANT: remember you must `pip install -e .` from this repo's")
    print("root for the installed `pinnstudio` command to reflect this (or")
    print("any) change on this branch.")


if __name__ == "__main__":
    main()
