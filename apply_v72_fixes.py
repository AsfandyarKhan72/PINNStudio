#!/usr/bin/env python3
"""
Run this on the `param-sweep` branch, on top of commit `def9e27` (the v71
pass that embedded Parameter Sweep fully into the left training panel).
Touches pinnstudio/ui/main_window.py, pinnstudio/core/sweep_runner.py,
pinnstudio/core/sweep_registry.py, tests/test_sweep_tab.py,
tests/test_sweep_registry.py, tests/test_sweep_execution.py, and adds a
new tests/test_training_callbacks_visibility.py.

WHAT THIS CHANGES

1. SHOW THE LAST SWEEP RUN'S FIGURE ON THE RIGHT PANEL WHEN IT FINISHES.
   A sweep used to leave the right panel's Loss/Solution plots blank the
   whole time, even after every run completed -- a normal (non-sweep)
   Solve already fills them in once training finishes, but nothing did
   that for a sweep. Fixed by pulling the plot-loading logic that
   _on_done() already used (GIF-vs-PNG extension detection, the
   solution_results/ subfolder fallback, etc.) out into a shared
   _display_run_result_plots(config, save_dir) method, calling it from
   _on_done() exactly as before, and now ALSO calling it from
   _on_sweep_finished() with the LAST SUCCESSFULLY COMPLETED run's own
   save_dir (tracked incrementally as each run finishes, so a failed or
   stopped final run doesn't blank out the last good figure) and the
   sweep's own plot_type (Surface by default, or whatever was chosen --
   since v71 a sweep always reuses the main Setup tab's plot settings,
   so this is the same figure a plain Solve of that same run would have
   shown). A Training Log line announces which run's figure is showing.

2. ADD "INITIAL GUESS" AND "INVERSE LOSS WEIGHT" AS SWEEPABLE PARAMETERS
   FOR INVERSE PROBLEMS. When the Inverse problem case is selected, the
   Parameter Sweep's own parameter dropdown now also offers, for every
   trainable variable row and every observed-data file row already in
   the Inverse panel: that variable's own initial guess, and that
   observation file's own inverse loss weight -- sweepable exactly like
   any other parameter (loss weights, learning rate, etc. already were).
   This mirrors the same pattern already used for per-phase loss-weight
   sweeping in sweep_registry.py: new get/set helpers that read/write
   the canonical inverse_variables_json/inverse_obs_files_json (falling
   back to the legacy single-field inverse_param_init/loss_weight_obs
   exactly the way codegen.py's own parsers already do), gated so these
   entries only appear when problem_type == "Inverse". No GUI changes
   were needed beyond this -- the entries just appear in the existing
   parameter dropdown once a sweep row calls Refresh (or is reopened)
   with Inverse selected.

3. STOP NOW HARD-STOPS A SWEEP IMMEDIATELY, INSTEAD OF LETTING THE
   CURRENTLY-RUNNING TRAINING RUN KEEP GOING. Previously, clicking Stop
   during a sweep only set a flag that was checked BETWEEN runs -- the
   in-flight run kept printing iterations and training to completion
   regardless, which looked like Stop wasn't doing anything (exactly
   the behavior reported: iterations kept streaming in after "Cancelling
   -- finishing the current run, then stopping..."). Fixed the same way
   a normal single Solve's Stop already works: sweep_runner.run_sweep()
   now accepts and passes through a set_process callback to every
   run_pinn() call (previously hardcoded to None), SweepThread captures
   the live subprocess the same way SolverThread already does, and
   SweepThread.stop() now terminates (then kills, if it doesn't exit
   within 3s) that exact subprocess immediately, the same two-step
   terminate/kill SolverThread.stop() already used. _on_stop()'s sweep
   branch now mirrors the normal-run branch: it stops the thread, waits
   for it, and immediately reports "Stopped by user" with the Solve/Stop
   buttons back in their normal state, instead of the old "Cancelling..."
   message that implied (incorrectly) a graceful wind-down was happening.

4. TRAINING CALLBACKS (Early Stopping / Point Resampling / Model
   Checkpoint / Training Timer) ARE NOW HIDDEN BY DEFAULT BEHIND A "Show
   Training Callbacks" CHECKBOX, INSTEAD OF ALWAYS TAKING UP SPACE. The
   divider no longer says "(optional)" -- the checkbox makes that
   redundant. Unticked by default (matching how Adaptive Training and
   Parameter Sweep already start collapsed/off), ticking it reveals all
   four group boxes exactly as before; nothing about what each callback
   does, how it's configured, or what _build_config() reports changes --
   this is purely a visibility toggle. Restoring a saved config that
   already has any one of the four callbacks enabled auto-ticks the
   checkbox too, so a previously-configured callback is never silently
   hidden on load.

SCOPE NOTE: config.py is untouched -- no new PINNConfig fields were
needed anywhere. Item 2's new sweep parameters read/write the EXISTING
inverse_variables_json/inverse_obs_files_json (and their legacy
fallback fields) exactly as the Inverse panel itself already does;
item 1 reuses the existing save_dir/plot_type/solution_results/ layout
every run already writes; item 3 reuses the existing subprocess-handle
pattern SolverThread already had; item 4 reuses the existing cb_*
config fields untouched, only wrapping their widgets in a container.

VERIFICATION DONE

 - Full existing headless regression suite (test_config_options.py,
   test_export_parity.py, test_sweep_registry.py, test_sweep_tab.py,
   test_templates.py 24/24, test_v56_no_template_solve.py) passes.
 - New tests/test_training_callbacks_visibility.py: divider text no
   longer contains "(optional)"; the container (and all 4 group boxes
   inside it) is hidden and the checkbox unticked by default; ticking it
   shows the container and unticking hides it again; a callback's own
   checked state/field values and what _build_config() reports are
   identical whether the container is shown or hidden; restoring a
   config with any one callback enabled auto-ticks and reveals the
   container, restoring one with all four off leaves it unticked.
 - tests/test_sweep_registry.py extended: a Forward-problem sweep's
   available parameters have no Inverse entries at all; switching to
   Inverse surfaces inv_var_init_<n> and inv_obs_weight_<n> entries with
   the right category/labels; setting one through the sweep registry's
   own set_value correctly updates the canonical
   inverse_variables_json/inverse_obs_files_json (and keeps the legacy
   single-field mirrors in sync, same as the Inverse panel's own
   _build_inverse_variables_json()/_build_inverse_obs_files_json() do);
   multiple variable/observation rows are addressed independently
   without disturbing each other.
 - tests/test_sweep_tab.py: fixed the Stop-routing test for the new
   synchronous hard-stop flow (checks for "Stopped by user" and the
   Solve/Stop buttons' immediate state instead of the old
   "Cancelling..." text); added a synthetic-data test (real tiny PNGs
   via PIL) confirming _on_sweep_run_done()/_on_sweep_finished() show
   the LAST SUCCESSFULLY COMPLETED run's own figure, even when a later
   run in the same sweep errors out.
 - tests/test_sweep_execution.py (real subprocess training, not CI-
   wired -- run locally wherever torch/deepxde are installed), re-run
   for real and passing with zero failures:
     - Extended the existing per-run-save-folder sweep check to also
       feed its real results through _on_sweep_run_done()/
       _on_sweep_finished() and confirm loss_plot.png/solution_plot.png
       exist and the right panel's loss_label/solution_label
       _source_path point at the LAST run's own files.
     - New combined Inverse sweep (2D Poisson Disk, Inverse mode)
       sweeping both a trainable variable's initial guess AND an
       observation file's inverse loss weight in one run, confirming
       both swept entries measurably change final_loss relative to the
       baseline.
     - New hard-stop test: starts a real SweepThread on a long-running
       config, waits for its subprocess to actually be spawned, calls
       .stop(), and confirms both the subprocess and the QThread exit
       quickly (well under the run's full iteration count) instead of
       continuing to train to completion.
 - Visually verified with a real virtual display (Xvfb + the actual
   "xcb" Qt platform): Training Callbacks collapsed by default showing
   only "Show Training Callbacks" with no "(optional)" text anywhere;
   ticking it reveals Early Stopping/Point Resampling/Model Checkpoint/
   Training Timer; with Inverse mode selected and Parameter Sweep
   enabled, the sweep row's Parameter dropdown offers and can select
   "Inverse: trainable_variable_1: initial guess".
 - Patch isolated (7 files, see stat below) and verified byte-identical
   in a fresh independent clone of `param-sweep` at `def9e27`; this
   script itself verified end-to-end (first-run + idempotent second-run
   + byte-identical + full regression suite, including the real-
   training checks, re-run there) in a third fresh clone before
   delivery.

Applies cleanly to a clean checkout of the `param-sweep` branch at
commit def9e27.

Run from the repo root:
    python3 apply_v72_fixes.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "v72_embed_plots_inverse_sweep_hardstop_callbacks_delta.patch"
MARKER = "_display_run_result_plots"

CHANGED_FILES = [
    "pinnstudio/ui/main_window.py",
    "pinnstudio/core/sweep_runner.py",
    "pinnstudio/core/sweep_registry.py",
    "tests/test_sweep_tab.py",
    "tests/test_sweep_registry.py",
    "tests/test_sweep_execution.py",
    "tests/test_training_callbacks_visibility.py",
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
              "or you're not on the param-sweep branch at commit def9e27 -- "
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
    print('  git commit -m "Show last sweep run figure on finish, add inverse '
          'initial-guess/loss-weight sweep params, hard-stop sweeps on Stop, '
          'hide Training Callbacks behind a Show checkbox"')
    print("  git push")
    print()
    print("Then open the app:")
    print("  - Run a sweep: once it finishes, the right panel's Loss/")
    print("    Solution plots now fill in with the last completed run's own")
    print("    figure (same plot type as a normal Solve would show), and the")
    print("    Training Log says which run it's showing.")
    print("  - Switch Problem type to Inverse, add a trainable variable and")
    print("    an observed-data file, enable Parameter Sweep: the parameter")
    print("    dropdown now also offers that variable's initial guess and")
    print("    that file's inverse loss weight.")
    print("  - Click Stop during a running sweep: it now stops immediately,")
    print("    the same way Stop already works for a normal Solve, instead")
    print("    of letting the in-flight run keep training to completion.")
    print("  - Scroll to 'Training Callbacks': it's now collapsed by default")
    print("    behind a 'Show Training Callbacks' checkbox -- tick it to see")
    print("    Early Stopping / Point Resampling / Model Checkpoint /")
    print("    Training Timer.")
    print("IMPORTANT: remember you must `pip install -e .` from this repo's")
    print("root for the installed `pinnstudio` command to reflect this (or")
    print("any) change on this branch.")


if __name__ == "__main__":
    main()
