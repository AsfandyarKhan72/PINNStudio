#!/usr/bin/env python3
"""
Run this on the `param-sweep` branch, on top of commit `ddf6dc9` (the
loss weight sweep patch). Touches pinnstudio/core/config.py,
pinnstudio/core/sweep_runner.py, pinnstudio/ui/main_window.py,
tests/test_sweep_registry.py, tests/test_sweep_tab.py, and
tests/test_sweep_execution.py.

WHAT THIS ADDS -- the 5-part Parameter Sweep UX pass

1. ADJUSTABLE SPLIT between the sweep setup area and the results panel
   on the Parameter Sweep tab. A vertical QSplitter (same pattern as
   the Setup tab's own right_splitter: childrenCollapsible disabled so
   a drag can't snap a pane fully shut) replaces the old fixed-stack
   layout, defaulting to roughly half/half (setSizes([450, 450])) and
   freely draggable from there.

2. SEPARATE, PROPERLY-NAMED RESULT FOLDERS, one per sweep run, so you
   can go back later and see exactly which model/figures came from
   which run -- mirroring the app's existing adaptive-time-stepping
   convention (step_NNN_... subfolders). A new "Save Sweep Results To"
   field + Browse button on the Parameter Sweep tab (separate from the
   Setup tab's own Save Directory, so sweep runs never collide into
   one folder) controls where; it defaults (when left blank) to
   ~/PINNStudio_Results/parameter_sweep_results. Every run actually
   trained produces:
       <sweep_save_dir>/sweep_<YYYYmmdd_HHMMSS>/
           run_000_baseline/{solution_results/, error_analysis/}
           run_001_<slugified run label>/{...}
           ...
           sweep_manifest.json   (status/final_loss/l2/folder per run)
           sweep_summary.csv     (same, as a quick-glance spreadsheet)
   This reuses the EXISTING codegen.py save pipeline unchanged -- each
   run's own config.save_dir is simply pointed at its own folder, so
   the model checkpoint/loss plot/solution plot/error-analysis files
   that already get written today just land in the right place with
   zero new plotting code.

3. PER-SWEEP EXPORT/FIGURE OVERRIDE. A new "Results To Save (for each
   run in this sweep)" group on the Parameter Sweep tab lets you leave
   it as "Same as Setup tab" (today's exact behavior, and the default)
   or switch to "Custom for this sweep" to pick a different figure
   type (Surface / Line / Line Animation GIF / Surface Animation GIF),
   output (or a custom expression), and number of exported time steps
   that apply to every run in that sweep, independent of whatever the
   Setup tab itself is currently set to.

4. DESCRIPTIVE FOLDER NAMES THAT REFLECT WHAT WAS SWEPT. Each run
   folder's name is built from that run's own already-descriptive
   label (e.g. "BC 2 (dirichlet, Output 0)=7.5" from a loss-weight
   sweep becomes run_NNN_BC_2_dirichlet_Output_0_=7.5) -- so sweeping
   one weight component, one network setting, or any other parameter
   always shows up unambiguously in the folder name.

5. COMSOL-STYLE SWEEP MODES. The sweep-mode picker is now worded like
   COMSOL's own Parametric Sweep: "One-at-a-time" (unchanged), "All
   combinations" (the existing cross-product mode, formerly just
   called "Grid", relabeled for clarity), and a brand-new "Specified
   combinations" mode ("zip" internally) that runs every parameter's
   Nth value together instead of crossing them -- e.g. 2 parameters
   with 3 values each give 3 runs, not 9. config.validate() enforces
   that every swept parameter has the same number of values whenever
   this mode is selected, with a clear error naming the mismatched
   lengths if not.

SCOPE NOTE: none of this touches pinnstudio/core/codegen.py or
pinnstudio/core/sweep_registry.py -- the per-run folder/export
machinery lives entirely in sweep_runner.py (where it can reuse the
existing save/plot pipeline by just pointing each run's save_dir at
its own folder), and the new sweep modes/fields live in config.py and
main_window.py. The existing Loss Weight sweep entries from the
previous patch, and everything else about how individual parameters
are discovered/read/written, are completely unchanged.

VERIFICATION DONE

 - New tests/test_sweep_registry.py checks: "zip" mode validates
   cleanly when every swept parameter has the same number of values
   (list-mode and linear-range modes alike), is flagged with a clear
   "Specified Combinations" error on any length mismatch (including a
   numeric range vs. a list of a different length), and that same
   mismatched config still validates fine under "oat"/"grid" (the
   length check is zip-only).
 - New tests/test_sweep_tab.py checks: the splitter exists, is
   vertical, has childrenCollapsible disabled, defaults to roughly
   half/half, and responds directionally to setSizes(); the sweep-mode
   combo has exactly the 3 expected entries (oat/grid/zip) in order; a
   real "zip" build_runs() pairs values instead of crossing them; the
   new Save Sweep Results To field/Browse handler and its
   placeholder/default-blank state; the export-override group's
   default ("same_as_setup"), its visibility toggle, the plot-type
   combo matching the Setup tab's own 4 options, the output combo
   (refreshed per-template) and its Custom.../expr/label visibility
   toggle; _build_config() round-tripping all 7 new config fields,
   including blanking the custom expr/label once a real (non-Custom)
   output is chosen; the new sweep_runner helpers directly
   (sweep_root_dir's default-vs-custom-root behavior, run_folder_name's
   zero-padded+slugified naming including the fixed "run_000_baseline"
   case, _slugify's character handling) and _apply_run_output_settings
   leaving plot_type/output/expr/label/t_steps untouched in
   "same_as_setup" mode while overriding every one of them in "custom"
   mode, with save_dir always overridden either way.
 - New tests/test_sweep_execution.py checks (real training, not
   CI-wired -- see that file's own docstring): a real sweep with a
   custom sweep_save_dir produces the exact expected folder tree on
   disk (run_000_baseline/, run_001_<slug>/, each with a real saved
   model checkpoint under solution_results/), a real sweep_manifest.
   json and sweep_summary.csv at the sweep root with the real parsed
   results in them; a real "zip" mode sweep over 2 parameters with 2
   values each actually runs exactly 3 runs (not 5), correctly paired.
 - Full existing regression suite (test_templates.py 24/24,
   test_config_options.py, test_export_parity.py,
   test_v56_no_template_solve.py) still passes unchanged.
 - Patch verified to apply cleanly, byte-identical result confirmed, in
   a fresh independent clone of param-sweep at commit ddf6dc9, with the
   full test suite (including the new checks) re-run there and passing.

Applies cleanly to a clean checkout of the `param-sweep` branch at
commit ddf6dc9.

Run from the repo root:
    python3 apply_v68_fixes.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "v68_sweep_ux_delta.patch"
MARKER = "sweep_save_dir_input"


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
              "or you're not on the param-sweep branch at commit ddf6dc9 -- "
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
    print("  - pinnstudio/core/config.py")
    print("  - pinnstudio/core/sweep_runner.py")
    print("  - pinnstudio/ui/main_window.py")
    print("  - tests/test_sweep_registry.py")
    print("  - tests/test_sweep_tab.py")
    print("  - tests/test_sweep_execution.py")
    print()
    print("Next: review with 'git diff', then commit and push to")
    print("param-sweep (NOT main), e.g.")
    print("  git add pinnstudio/core/config.py pinnstudio/core/sweep_runner.py \\")
    print("          pinnstudio/ui/main_window.py tests/test_sweep_registry.py \\")
    print("          tests/test_sweep_tab.py tests/test_sweep_execution.py")
    print('  git commit -m "Parameter Sweep: adjustable split, per-run result '
          'folders, per-sweep export settings, descriptive naming, COMSOL-style '
          'sweep modes"')
    print("  git push")
    print()
    print("Then open the Parameter Sweep tab: you should see the setup area and")
    print("results panel split by a draggable divider (half/half by default), a")
    print("new 'Save Sweep Results To' field with a Browse button, a new")
    print("'Results To Save' group (Same as Setup tab / Custom for this sweep),")
    print("and the sweep-mode dropdown now reading One-at-a-time / All")
    print("combinations / Specified combinations. Running a sweep will create a")
    print("timestamped folder under your chosen save location with one")
    print("descriptively-named subfolder per run plus sweep_manifest.json and")
    print("sweep_summary.csv.")
    print("IMPORTANT: remember you must `pip install -e .` from this repo's")
    print("root for the installed `pinnstudio` command to reflect this (or")
    print("any) change on this branch.")


if __name__ == "__main__":
    main()
