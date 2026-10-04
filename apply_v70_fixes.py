#!/usr/bin/env python3
"""
Run this on the `param-sweep` branch, on top of commit `b0938dc` (the
v69 sweep tab styling pass: compact single-row parameters, own
"Sweep Parameters" panel, no nested height-capped scroll area). Touches
pinnstudio/ui/main_window.py and tests/test_sweep_tab.py only.

WHAT THIS CHANGES -- merge Parameter Sweep into the Setup tab

Until now, Parameter Sweep lived in its own top-level tab next to
"Setup". After trying v69 for real, the feedback was to fold it into
the main training panel instead: an "Enable Parameter Sweep" toggle
right after Adaptive Training, off by default, so the whole training
setup -- normal or swept -- lives in one place instead of a separate
full-screen tab.

This patch:

1. REMOVES THE SEPARATE "PARAMETER SWEEP" TAB. The Setup tab's
   QTabWidget now has a single tab (its header is hidden, since a
   one-item tab bar looked unfinished) -- "Settings" (the app
   preferences tab) is unaffected.

2. ADDS AN "ENABLE PARAMETER SWEEP" TOGGLE TO THE LEFT TRAINING PANEL,
   right after the Adaptive Training section, off by default. It's a
   small group: the Enable checkbox, a Mode picker (One-at-a-time /
   All combinations / Specified combinations -- unchanged from before),
   and a one-line hint. This is deliberately kept in the narrow left
   column -- just the toggle and mode, not the parameter list itself,
   which needs real width (see point 3).

3. MOVES EVERYTHING ELSE (the parameter rows, "Save Sweep Results To",
   "Results To Save", Run Sweep/Cancel/Export CSV, the results table
   and plot) INTO THE RIGHT PANEL, where the normal single-run results
   view (Training Log, loss/solution plots, Save to/Export) used to be
   the only thing shown. Checking "Enable Parameter Sweep" swaps the
   right panel from the normal results view to the sweep panel; the
   compact one-line-per-parameter rows from v69 need that panel's
   width to render well, which the narrow left column could never have
   given them. Unchecking it swaps back. The main Solve button is
   disabled while a sweep is enabled (Run Sweep/Cancel/Export CSV,
   inside the sweep panel, are kept as their own dedicated controls
   rather than merged into Solve/Stop).

SCOPE NOTE: this is purely a widget-relocation/wiring change in
main_window.py -- every sweep-related field still uses the exact same
self.attribute_name as before (self.sweep_enable_cb, self.sweep_mode_
combo, self.sweep_save_dir_input, etc.), so _build_config(),
pinnstudio/core/config.py, pinnstudio/core/sweep_runner.py, and
pinnstudio/core/sweep_registry.py are completely untouched. Running a
sweep behaves exactly as before; only where its controls live in the
window has changed.

VERIFICATION DONE

 - Full existing headless regression suite (test_config_options.py,
   test_export_parity.py, test_sweep_registry.py, test_sweep_tab.py,
   test_templates.py 24/24, test_v56_no_template_solve.py) passes.
 - tests/test_sweep_tab.py rewritten for the new structure: only one
   "Setup" tab exists (no more "Parameter Sweep" tab); the Enable
   checkbox/sweep panel/normal results view are off/hidden/shown (and
   vice versa) in the right combination by default and after toggling;
   Solve is enabled/disabled in step with the toggle; the "Sweep
   Parameters" panel and its rows are found inside the new sweep panel
   widget rather than a tab; every other existing check (add/remove
   rows, categorical vs. numeric widgets, _build_config()/validate()
   round-tripping, oat/grid/zip run counts, the stale-results guard,
   results-plot rendering, the adjustable splitter, save-dir/export
   override fields, naming helpers) still passes unchanged against the
   relocated panel.
 - Visually verified with a real virtual display (Xvfb + the actual
   "xcb" Qt platform, not "offscreen") rather than just headless
   grab(), per this project's own established practice for layout
   checks: the toggle group renders correctly right after Adaptive
   Training in the left panel; toggling it correctly swaps the right
   panel between the normal results view and the sweep panel; the
   compact parameter rows render correctly at the right panel's width;
   the tab bar stays hidden.
 - Full real-training suite (test_sweep_execution.py -- actual
   subprocess training, not mocked) re-run and still passes: a real
   sweep still produces the same per-run folders/manifest/summary.csv,
   since sweep_runner.py/config.py are untouched by this patch.
 - Patch isolated (2 files, 204 insertions / 83 deletions) and verified
   byte-identical in a fresh independent clone of `param-sweep` at
   `b0938dc`; apply_v70_fixes.py itself verified end-to-end (first-run
   + idempotent second-run + byte-identical + full regression suite,
   including the real-training checks, re-run there) in a third fresh
   clone before delivery.

Applies cleanly to a clean checkout of the `param-sweep` branch at
commit b0938dc.

Run from the repo root:
    python3 apply_v70_fixes.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "v70_sweep_panel_merge_delta.patch"
MARKER = "sweep_panel_widget"


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
              "or you're not on the param-sweep branch at commit b0938dc -- "
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
    print()
    print("Next: review with 'git diff', then commit and push to")
    print("param-sweep (NOT main), e.g.")
    print("  git add pinnstudio/ui/main_window.py tests/test_sweep_tab.py")
    print('  git commit -m "Merge Parameter Sweep into the Setup tab: enable '
          'toggle in the left panel, full config in the right panel"')
    print("  git push")
    print()
    print("Then open the app: the 'Parameter Sweep' tab is gone. Scroll the")
    print("left training panel down past Adaptive Training and you'll see a")
    print("new 'Parameter Sweep' group with an Enable checkbox and Mode")
    print("picker, off by default. Check it, and the right panel swaps from")
    print("the normal Training Log/plots view to the full sweep setup --")
    print("parameters, save location, export settings, Run Sweep, results --")
    print("with Solve disabled for as long as the sweep stays enabled.")
    print("Uncheck it to go back to normal single-run training.")
    print("IMPORTANT: remember you must `pip install -e .` from this repo's")
    print("root for the installed `pinnstudio` command to reflect this (or")
    print("any) change on this branch.")


if __name__ == "__main__":
    main()
