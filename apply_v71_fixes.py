#!/usr/bin/env python3
"""
Run this on the `param-sweep` branch, on top of commit `6beee6f` (the
v70 pass that put Parameter Sweep's Enable toggle in the left panel and
its full configuration/results panel in the right one). Touches
pinnstudio/ui/main_window.py, tests/test_sweep_tab.py and
tests/test_sweep_execution.py only.

WHAT THIS CHANGES -- embed Parameter Sweep fully into the training panel

After trying v70, the feedback was that Parameter Sweep still felt like
its own separate thing taking over the right panel (its own save
location, its own plot/export settings, its own Run/Cancel/Export
buttons, its own results table) instead of feeling like a normal part
of training -- the way Adaptive Training and the Training Phases list
already do, configured entirely in the left panel while the main
Solve/Stop buttons and the Training Log do the actual work.

This patch:

1. REMOVES THE EXPLANATORY HINT TEXT right after the Mode picker ("Sweep
   one or more of the problem's own settings... on the right") -- that
   kind of explanation is going into the GitHub README instead.

2. MOVES THE "SWEEP PARAMETERS" ROWS (the parameter list, add/remove
   rows, Refresh) OUT OF THE RIGHT PANEL AND INTO THE SAME LEFT-PANEL
   "Parameter Sweep" GROUP, directly below Mode -- so Enable, Mode, and
   every parameter row all live together in one place, the same way
   Training Phases already work. The rows are rebuilt as a stacked
   block (Parameter / Sweep as / Values, or Min/Max/Steps for a range)
   instead of v69/v70's single wide line per row, since this narrow
   column has nowhere near the ~700px that needed.

3. REMOVES THE DEDICATED "Save Sweep Results To" FIELD. A sweep now
   saves under the same "Save to:" folder as a normal run (the right
   panel's save_dir_input) -- sweep_runner.py already builds a
   `sweep_<timestamp>/run_NNN_.../` structure under whatever directory
   it's given, so pointing it at the same folder as normal runs needed
   no changes there at all.

4. REMOVES THE DEDICATED "Results To Save" EXPORT OVERRIDE SECTION
   (plot type / output / time steps, previously only used when you
   explicitly chose "custom" there). A sweep now always reuses the main
   Setup tab's own plot/export settings, unconditionally --
   sweep_export_mode is now always "same_as_setup" from the GUI's side
   (sweep_runner.py's "custom" branch is untouched and still covered by
   its own direct unit test, in case a future caller needs it again).

5. REMOVES THE DEDICATED Run Sweep / Cancel / Export CSV BUTTONS. The
   main Solve button now runs a sweep too (it just relabels itself "Run
   Sweep" while the Enable checkbox is on), and the main Stop button
   cancels one the same way it stops a normal run (gracefully --
   finishing the in-flight run, then stopping, exactly as Cancel used
   to work).

6. REMOVES THE DEDICATED RUN/STATUS/FINAL LOSS/L2 RESULTS TABLE AND THE
   RESULTS COMPARISON PLOT. Each run's progress and result now streams
   into the normal Training Log, the same text every normal run already
   produces -- "Run 1/3: hidden_layers=2", then that run's own training
   output, then a final-loss/L2 summary line once it's done, and so on
   for the next run. The per-run folders, sweep_manifest.json and
   sweep_summary.csv are still written automatically exactly as before
   (sweep_runner.py is unchanged), so nothing about what gets SAVED
   changes -- only the dedicated in-GUI table/plot are gone, since the
   log already shows the same information as it streams, and the saved
   CSV/manifest capture it afterwards.

7. Consequently, the right panel is now ALWAYS just the normal Training
   Log + loss/solution plots + Save to/Error Analysis/Export Solution
   row, whether or not a sweep is enabled -- there is no more "right
   panel swaps to a different view" behavior from v70. Only the left
   panel's "Parameter Sweep" group (and the Solve/Stop buttons' label)
   reacts to the Enable checkbox now.

SCOPE NOTE: sweep_runner.py, sweep_registry.py and config.py are
completely untouched by this patch -- every PINNConfig sweep_* field
(sweep_enabled, sweep_mode, sweep_parameters, sweep_save_dir,
sweep_export_mode, sweep_plot_type, sweep_plot_output_idx,
sweep_plot_custom_expr, sweep_plot_custom_label, sweep_export_t_steps)
still exists and is still consumed exactly as before; only how
_build_config() populates them changed (sweep_save_dir now reads the
shared save_dir_input, and the 5 export-override fields are now
hardcoded constants matching "always same as setup", since their
dedicated widgets no longer exist).

VERIFICATION DONE

 - Full existing headless regression suite (test_config_options.py,
   test_export_parity.py, test_sweep_registry.py, test_sweep_tab.py,
   test_templates.py 24/24, test_v56_no_template_solve.py) passes.
 - tests/test_sweep_tab.py rewritten for the new structure: confirms
   there's still just the one "Setup" tab with its bar hidden; confirms
   sweep_panel_widget/normal_results_widget/sweep_save_dir_input/
   sweep_export_mode_combo no longer exist at all; confirms Solve stays
   enabled and just relabels ("Solve" <-> "Run Sweep") instead of being
   disabled while a sweep is enabled; confirms _on_solve() routes to a
   sweep start when config.sweep_enabled is True, and _on_stop() cancels
   a running sweep thread the same graceful way Cancel used to; confirms
   _build_config().sweep_save_dir tracks the shared "Save to:" field and
   sweep_export_mode is always "same_as_setup"; every row-building/JSON/
   validate()/oat-grid-zip run-count/naming-helper check from before
   still passes against the relocated, restacked rows.
 - tests/test_sweep_execution.py (real subprocess training, not CI-
   wired -- run locally wherever torch/deepxde are installed) updated
   to match: drops the removed results-plot rendering check, and points
   its per-run-folder check at the shared save_dir_input instead of the
   removed sweep_save_dir_input. Re-run for real and still passes: 1D
   Heat and 3D Sphere sweeps both complete with correct final_loss/
   l2_relative values, the loss-weight sweep still measurably changes
   training behavior, and the per-run-folder/manifest/summary-CSV and
   zip-mode checks are unaffected, since sweep_runner.py itself didn't
   change.
 - Visually verified with a real virtual display (Xvfb + the actual
   "xcb" Qt platform): with the sweep off, the right panel is byte-for-
   byte the plain pre-sweep-feature view; with it on, the left panel
   shows Enable + Mode + the "Sweep Parameters" divider + Refresh +
   stacked parameter-row blocks + Add Parameter, directly followed by
   the same Solve/Stop buttons (now reading "Run Sweep"/"Stop"), with
   the right panel still showing nothing but the normal Training
   Log/plots.
 - Patch isolated (3 files, see stat below) and verified byte-identical
   in a fresh independent clone of `param-sweep` at `6beee6f`; this
   script itself verified end-to-end (first-run + idempotent second-run
   + byte-identical + full regression suite, including the real-
   training checks, re-run there) in a third fresh clone before
   delivery.

Applies cleanly to a clean checkout of the `param-sweep` branch at
commit 6beee6f.

Run from the repo root:
    python3 apply_v71_fixes.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "v71_embed_in_training_panel_delta.patch"
MARKER = "_start_sweep"


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
              "or you're not on the param-sweep branch at commit 6beee6f -- "
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
    print("  - pinnstudio/ui/main_window.py")
    print("  - tests/test_sweep_tab.py")
    print("  - tests/test_sweep_execution.py")
    print()
    print("Next: review with 'git diff', then commit and push to")
    print("param-sweep (NOT main), e.g.")
    print("  git add pinnstudio/ui/main_window.py tests/test_sweep_tab.py tests/test_sweep_execution.py")
    print('  git commit -m "Embed Parameter Sweep fully into the training panel: '
          'configure in the left column, run/log/save with the normal controls"')
    print("  git push")
    print()
    print("Then open the app: there's no more separate sweep view taking")
    print("over the right panel. Scroll the left training panel down past")
    print("Adaptive Training and you'll see the 'Parameter Sweep' group --")
    print("Enable checkbox, Mode picker, and now the parameter rows")
    print("themselves (Add Parameter etc.) right there too. Check Enable,")
    print("add a parameter, and the normal Solve button (now reading")
    print("'Run Sweep') and Stop button do the rest -- progress streams into")
    print("the same Training Log as always, and each run's results still")
    print("land in their own folder under your normal 'Save to:' location,")
    print("with a sweep_summary.csv/sweep_manifest.json written automatically.")
    print("IMPORTANT: remember you must `pip install -e .` from this repo's")
    print("root for the installed `pinnstudio` command to reflect this (or")
    print("any) change on this branch.")


if __name__ == "__main__":
    main()
