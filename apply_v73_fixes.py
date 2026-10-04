#!/usr/bin/env python3
"""
Run this on the `param-sweep` branch, on top of commit `01dea16` (the v72
pass that showed the last sweep run's figure, added Inverse initial-
guess/loss-weight sweep params, hard-stopped sweeps on Stop, and hid
Training Callbacks behind a checkbox). Touches
pinnstudio/core/sweep_registry.py, pinnstudio/ui/main_window.py,
tests/test_sweep_registry.py, tests/test_sweep_tab.py,
tests/test_sweep_execution.py, and adds a new
tests/test_sweep_param_expansion.py.

WHAT THIS CHANGES

1. FIXED: THE TRUE-VALUE DASHED LINE WAS MISSING FROM THE PARAMETER
   CONVERGENCE PLOT DURING AN INVERSE PARAMETER SWEEP. Every time a
   sweep row's set_value() touched a trainable variable's initial guess,
   it rebuilt the config's inverse_variables_json from
   sweep_registry.py's own _inv_vars_list() helper -- which only ever
   round-tripped each variable's "name" and "init" fields, silently
   dropping "true" even for variables the sweep never touched. That
   blanked out every variable's own true value the moment ANY sweep row
   existed, which is why the dashed reference line that a normal
   (non-swept) Inverse run already draws disappeared specifically during
   a sweep. Fixed by having _inv_vars_list() also preserve "true" when
   present, so it round-trips losslessly like every other field already
   did.

2. MOVED "SAVE PARAMETER" TO ITS OWN ROW, DIRECTLY BELOW PLOT OUTPUT/
   PLOT TYPE. It used to sit crammed in next to the Error Analysis/
   Export Solution buttons, far from the other plot-related controls it
   actually goes with. It's now its own row immediately under Plot
   output/Plot type (not merged into that same row -- this is
   deliberate: combining them risks reintroducing a previously-fixed bug
   where a template's Custom expression/label fields plus Save Parameter
   combined could force the left configuration panel's splitter down to
   its minimum width). Error Analysis/Export Solution keep their own row
   underneath, unchanged otherwise.

3. WIDENED THE PLOT OUTPUT / PLOT TYPE / SAVE PARAMETER DROPDOWNS SO
   THEIR FULL TEXT ACTUALLY SHOWS. All three used a fixed width that
   truncated longer entries ("Output 1 (u)", "Parameter Convergence",
   "Every 100 iters"). They now use the same width-fitting helper several
   other combos already use (sized to the widest entry actually in the
   list, recomputed every time "Parameter Convergence" or an output
   entry is dynamically added or removed), so nothing is ever clipped.

4. TIME ADAPTIVE "DISAPPEARING": CONFIRMED WORKING AS INTENDED, NOT A
   BUG -- no code change. The only thing that ever removes "Time
   Adaptive Training" from the Adaptive method dropdown is switching
   Problem type to Inverse (Time Adaptive doesn't yet wire the inferred
   variable into its per-step training loop, so it would silently fail
   to converge for Inverse problems); switching back to Forward always
   restores it. This matches what was being seen while testing Inverse
   sweeps.

5. RAR'S OWN PARAMETERS ARE NOW SWEEPABLE. When RAR (Residual-based
   Adaptive Refinement) is the selected adaptive method, the Parameter
   Sweep dropdown now also offers: Training rounds (rar_cycles),
   Residual sampling points (rar_candidates), Points added per cycle
   (rar_add_points), Adam iterations (rar_adam_iters), and L-BFGS
   iterations (rar_lbfgs_iters) -- gated so they only appear when RAR is
   actually selected.

6. TIME ADAPTIVE'S OWN PARAMETERS ARE NOW SWEEPABLE. When Time Adaptive
   is selected, the dropdown now also offers, per existing time-step
   group: that group's own step count ("Time Adaptive: Group N steps
   (t_start->t_end)"), plus "Time Adaptive: IC grid resolution"
   (11/21/51/101). Sweeping a group's steps keeps every other group
   untouched and keeps the legacy derived ta_num_steps field resynced to
   the sum across all groups, the same way the Time Adaptive panel's own
   editing already does.

7. OUTPUT TRANSFORM AND INPUT TRANSFORM SCALE ARE NOW SWEEPABLE, ONE
   ENTRY PER OUTPUT/INPUT AXIS. The GUI already had the structured
   "y_raw * scale + shift" / per-axis scale+shift boxes this was asking
   for -- no GUI rework was needed. What was missing was exposing each
   one's own scale value to the sweep registry: with Output transform
   enabled, the dropdown now offers "Output transform: Output N scale"
   for each output; with Input transform enabled, "Input transform: x
   scale" / "t scale" (or y/z for 2D/3D) for each input axis. Each entry
   sweeps only that one scale value, leaving every other output/axis's
   scale and every shift value untouched.

8. THREE MORE GENERALLY-USEFUL PARAMETERS ADDED TO THE SWEEP LIST:
   Point distribution (Hammersley/uniform/Halton/LHS/Sobol/
   pseudorandom), Activation (tanh/relu/sigmoid/swish), and Kernel
   initializer (Glorot uniform/Glorot normal/He uniform/He normal/
   zeros) -- all always available, regardless of problem type or
   adaptive method.

SCOPE NOTE: config.py, codegen.py, and sweep_runner.py are all
untouched. Every new sweepable entry in items 5-8 reads/writes PINN
Config fields codegen.py already interpolates directly into the
generated script, so a deep-copied, swept config already produces
correctly different training code with no changes needed to how a
sweep actually runs a script. The only GUI change is item 2 (Save
Parameter's position) and item 3 (combo widths) -- items 5-8 are purely
sweep_registry.py wiring against existing config fields and existing
GUI controls.

VERIFICATION DONE

 - Full existing headless regression suite (test_config_options.py,
   test_export_parity.py, test_sweep_registry.py, test_sweep_tab.py,
   test_sweep_param_expansion.py, test_templates.py 24/24,
   test_v56_no_template_solve.py, test_training_callbacks_visibility.py)
   passes.
 - New tests/test_sweep_param_expansion.py: drives the real sweep-row
   GUI widgets and sweep_runner.build_runs() (no subprocess) for every
   new category, then inspects codegen.generate_script()'s actual output
   text and confirms it parses as valid Python for every run --
   specifically re-confirms the true-value fix all the way through
   codegen's live Solve/Sweep path (generate_script()'s own
   _inv_var_trues literal, not the separate Export-as-DeepXDE-Script
   path's _inv_true_vals), RAR's range(...) loop, Time Adaptive's
   grid_size/step-group JSON, transform scale lists, and the
   activation/point-distribution/kernel-initializer calls.
 - tests/test_sweep_registry.py extended: true-value preservation across
   multiple independent variables; RAR's 5 params gated on/off and
   round-tripping; Time Adaptive's per-group steps (including a 2-group
   independence check confirming ta_num_steps resyncs to the sum, not
   just the changed group) and IC grid resolution; Output/Input
   transform scale per-axis labels, gating, and round-tripping without
   disturbing shift; the 3 new static params' choices and round-trip.
 - tests/test_sweep_tab.py extended: a generic case-insensitive check
   that no sweep-dropdown entry's label is ever doubled with its own
   category prefix (built from a config with Inverse + transforms + RAR
   all enabled at once, to surface every category together); layout
   checks that Save Parameter sits below Plot type and above Error
   Analysis; combo minimumWidth() checks against each combo's actual
   widest-item text, including after "Parameter Convergence" is added
   dynamically.
 - tests/test_sweep_execution.py (real subprocess training, not CI-
   wired -- run locally wherever torch/deepxde are installed), re-run
   for real and passing with zero failures: a real RAR sweep
   (rar_cycles), a real Time Adaptive sweep (ta_grid_size, one group/one
   step for speed), and a combined real sweep over an output-transform
   scale and Activation, each running to completion with every run
   reporting status "done" and a real parsed final_loss.
 - Visually verified with a real virtual display (Xvfb + the actual
   "xcb" Qt platform): Save Parameter now sits on its own row directly
   under Plot output/Plot type; none of the three widened combos clip
   their text; the sweep Parameter dropdown correctly offers RAR's
   params with RAR selected, Time Adaptive's params with Time Adaptive
   selected, and transform-scale entries with Output/Input transform
   enabled -- with no doubled category prefixes anywhere.
 - Patch isolated (6 files, see stat below) and verified byte-identical
   in a fresh independent clone of `param-sweep` at `01dea16`; this
   script itself verified end-to-end (first-run + idempotent second-run
   + byte-identical + full regression suite, including the real-
   training checks, re-run there) in a third fresh clone before
   delivery.

Applies cleanly to a clean checkout of the `param-sweep` branch at
commit 01dea16.

Run from the repo root:
    python3 apply_v73_fixes.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "v73_sweep_param_expansion_delta.patch"
MARKER = "ctrl_row_save"

CHANGED_FILES = [
    "pinnstudio/core/sweep_registry.py",
    "pinnstudio/ui/main_window.py",
    "tests/test_sweep_registry.py",
    "tests/test_sweep_tab.py",
    "tests/test_sweep_execution.py",
    "tests/test_sweep_param_expansion.py",
]


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

    main_window_path = root / "pinnstudio" / "ui" / "main_window.py"
    text = main_window_path.read_text() if main_window_path.is_file() else ""

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
              "or you're not on the param-sweep branch at commit 01dea16 -- "
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
    for f in CHANGED_FILES:
        print(f"  - {f}")
    print()
    print("Next: review with 'git diff', then commit and push to")
    print("param-sweep (NOT main), e.g.")
    print("  git add " + " ".join(CHANGED_FILES))
    print('  git commit -m "Fix missing Inverse true-value line in sweeps, move '
          'Save parameter under Plot type, widen combos, add RAR/Time '
          'Adaptive/transform-scale/activation sweep params"')
    print("  git push")
    print()
    print("Then open the app:")
    print("  - Inverse + Parameter Sweep over an initial guess: the Parameter")
    print("    Convergence plot now shows every variable's own dashed true-")
    print("    value line again, same as a normal (non-swept) Inverse run.")
    print("  - 'Save parameter' now sits on its own row directly under Plot")
    print("    output/Plot type, and none of those three dropdowns truncate")
    print("    their text anymore.")
    print("  - Select RAR as the adaptive method, enable Parameter Sweep: the")
    print("    Parameter dropdown now offers RAR's own 5 knobs (training")
    print("    rounds, sampling points, points per cycle, Adam/L-BFGS iters).")
    print("  - Select Time Adaptive Training: the dropdown now offers each")
    print("    step group's own step count and the IC grid resolution.")
    print("  - Enable Output transform and/or Input transform: the dropdown")
    print("    now offers each output's/axis's own scale value.")
    print("  - Point distribution, Activation, and Kernel initializer are now")
    print("    always offered too.")
    print("IMPORTANT: remember you must `pip install -e .` from this repo's")
    print("root for the installed `pinnstudio` command to reflect this (or")
    print("any) change on this branch.")


if __name__ == "__main__":
    main()
