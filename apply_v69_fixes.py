#!/usr/bin/env python3
"""
Run this on the `param-sweep` branch, on top of commit `e794ca4` (the
v68 sweep UX patch: adjustable split, per-run result folders, per-sweep
export settings, descriptive naming, COMSOL-style sweep modes). Touches
pinnstudio/ui/main_window.py and tests/test_sweep_tab.py only.

WHAT THIS FIXES -- Parameter Sweep tab styling pass

Follow-up to v68 after trying it in the real GUI: the tab worked, but
didn't look or behave as cleanly as it should. Four things, all purely
visual/layout (no config fields, no sweep_runner.py behavior, no new
registry entries -- scope is entirely main_window.py's own widget
layout):

1. THE PARAMETER LIST WAS STUCK IN A SECOND, INNER, HEIGHT-CAPPED
   SCROLL BOX. Each parameter row previously lived inside its own
   QScrollArea, hard-capped at 260px, nested INSIDE the tab's outer
   scroll area. Adding more than ~3 parameters meant scrolling inside
   that tiny inner box -- dragging the splitter bigger (v68's own
   adjustable split) didn't help, since the inner box's height was
   fixed regardless. Fixed by removing that nested scroll area
   entirely: the parameter rows now sit directly in their panel, so
   they grow with their own content and the ONE outer scroll area (or
   dragging the splitter) handles overflow -- exactly the same pattern
   the Setup tab's own scheduler-phase list already uses successfully
   (sched_phases_widget is added directly to its layout the same way).

2. EACH PARAMETER TOOK 3-4 STACKED LINES WITH A HUGE GAP BETWEEN EACH
   LABEL AND ITS OWN FIELD. The old per-row layout put "Parameter:" on
   its own line with `addStretch()` BEFORE the combo box, which pushes
   the combo to the far-right edge of a full-width tab -- on a narrow
   panel (like the Setup tab's own left column) that's barely
   noticeable, but on this tab's full window width it left a huge,
   broken-looking gap between a label and its own field. Fixed by
   rebuilding each parameter as ONE compact line (number, Parameter
   combo, Sweep-as combo, Values/range fields, remove button, all
   adjacent with small fixed spacing) instead of 4 stacked blocks --
   closer to how COMSOL's own Parametric Sweep lists each parameter as
   a single table row rather than a separate labeled block per field.
   A thin bottom border per row stands in for a table's row separators.

3. THE TAB READ AS ONE LONG UNDIFFERENTIATED COLUMN. The "Save Sweep
   Results To" and "Results To Save" groups (from v68) were already
   their own bordered QGroupBox panels, but the parameter list and
   Enable/Mode controls above them weren't -- wrapped them in their own
   "Sweep Parameters" QGroupBox so the tab now reads as 3 clearly
   separated panels, consistent with the other two.

4. TWO SMALL BUGS CAUGHT WHILE REBUILDING THIS (both pre-existing,
   just far more visible in the new compact layout):
   - A freshly added parameter row showed BOTH the Values field AND
     the Min/Max/Steps range fields at once, until the user happened
     to touch the Parameter or Sweep-as combo. _refresh_sweep_param_
     choices() populates the Parameter combo with blockSignals(True)
     (so repopulating every row doesn't spam signals on every other
     row), which meant the function that hides one or the other
     (_update_value_widgets()) never actually ran for a brand new row.
     Fixed by calling it once explicitly right after a new row is
     added.
   - A phase-scoped sweep entry's dropdown text read "Phase 1: Phase
     1: Optimizer" -- its registry label already embeds "Phase N: "
     (used elsewhere for run folder naming, so that's left alone), and
     the dropdown was ALSO prepending the category on top of that.
     Fixed by only prepending the category when the label doesn't
     already start with it.

VERIFICATION DONE

 - The nested-scroll-area removal was first caught as a layout bug
   under headless/offscreen Qt (parameter rows rendering squeezed to a
   few px tall) -- traced to a known, pre-existing offscreen-Qt layout
   quirk (confirmed by reproducing the identical squeeze on the
   Setup tab's own, completely untouched scheduler-phase list), not a
   real bug. Re-verified visually with an actual virtual display
   (Xvfb + the real "xcb" Qt platform, not "offscreen") showing correct
   layout, confirming the earlier squeeze was a headless-rendering
   artifact only.
 - New tests/test_sweep_tab.py checks: exactly one "Sweep Parameters"
   QGroupBox panel exists and contains the parameter rows; sweep_rows_
   widget's direct parent is no longer a QScrollArea; a freshly added
   row shows exactly one of (Values field / range fields), never both
   or neither; each row's own layout is a single QHBoxLayout (one
   line); a phase-scoped entry's dropdown text contains "Phase 1:"
   exactly once, not doubled.
 - Full existing regression suite (test_templates.py 24/24,
   test_config_options.py, test_export_parity.py,
   test_v56_no_template_solve.py, test_sweep_registry.py) still passes
   unchanged -- nothing about sweep_runner.py, config.py, or
   sweep_registry.py changed in this patch, only main_window.py's own
   widget layout/wiring.
 - test_sweep_execution.py (real training, not CI-wired) re-run and
   still passes -- run labels/folder names are unaffected since this
   patch only changes the GUI's dropdown DISPLAY text, never
   SweepParam.label itself.
 - Patch isolated (2 files, 158 insertions / 31 deletions) and
   verified byte-identical in a fresh independent clone of
   `param-sweep` at `e794ca4`; `apply_v69_fixes.py` itself verified
   end-to-end (first-run + idempotent second-run + byte-identical +
   full regression suite, including the real-training checks, re-run
   there) in a third fresh clone before delivery.

Applies cleanly to a clean checkout of the `param-sweep` branch at
commit e794ca4.

Run from the repo root:
    python3 apply_v69_fixes.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "v69_sweep_tab_styling_delta.patch"
MARKER = "sweepParamRow"


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
              "or you're not on the param-sweep branch at commit e794ca4 -- "
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
    print('  git commit -m "Parameter Sweep tab: compact single-row '
          'parameters, own panel, remove nested scroll cap"')
    print("  git push")
    print()
    print("Then open the Parameter Sweep tab: parameters now live in their")
    print("own 'Sweep Parameters' panel (matching 'Save Sweep Results To'")
    print("and 'Results To Save' below it), each parameter is a single")
    print("compact line instead of a tall stacked block, and adding more")
    print("parameters grows that panel -- draggable via the splitter --")
    print("instead of being stuck scrolling inside a small fixed-size box.")
    print("IMPORTANT: remember you must `pip install -e .` from this repo's")
    print("root for the installed `pinnstudio` command to reflect this (or")
    print("any) change on this branch.")


if __name__ == "__main__":
    main()
