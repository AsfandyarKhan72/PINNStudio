#!/usr/bin/env python3
"""
Restore Model fix, part 2: model_config.json never saved "steady_state".

Builds on top of the already-pushed Restore Model fix (commit 830e211), on
the SAME `custom-geometry` branch -- no new branch.

STEP 1 -- MAKE SURE YOU'RE ON THE custom-geometry BRANCH, UP TO DATE

    git checkout custom-geometry
    git pull origin custom-geometry

STEP 2 -- DOUBLE-CHECK YOU'RE AT THE REPO ROOT (not inside the inner
`pinnstudio` package folder), THEN RUN THIS SCRIPT

    ls setup.py pinnstudio/core/codegen.py

If both print back cleanly, you're in the right place:

    python3 apply_restore_steady_state_config_fix.py

WHY YOU STILL SAW THE SAME CRASH AFTER THE LAST PATCH

The last fix (830e211) was correct but incomplete: _build_restore_script()/
_build_restore_ea_script() in main_window.py now correctly check
cfg.get("steady_state", False) to decide whether to drop the time column --
but your model_config.json never had a "steady_state" key in it at all
(confirmed directly: grep -o '"steady_state"[^,}]*' on your own
model_config.json came back empty). That's because codegen.py's
_model_config dict -- the ONE place that actually writes model_config.json
at the end of a training run -- never included that key in the first
place. So cfg.get("steady_state", False) was correctly falling back to its
default (False, i.e. "treat as transient") every single time, for every
restored steady-state model, regardless of the first fix.

WHAT THIS PATCH FIXES

1. pinnstudio/core/codegen.py -- the `_model_config` dict (around line
   670, the only write site for model_config.json -- the separate
   Time-Adaptive step_config.json writer already includes everything it
   needs and is unreachable for steady-state anyway, since Time Adaptive
   is hidden whenever Steady-state is checked) now includes:
     - "steady_state": config.steady_state -- the actual missing piece.
     - "z_min"/"z_max" -- also missing (only x_min/x_max/y_min/y_max were
       saved before), needed for a 3D model's restore to use the real
       domain instead of silently defaulting to [0,1]. The Time-Adaptive
       step writer already saves these; the main writer never did.
2. tests/test_restore_steady_state.py -- added a new end-to-end check,
   test_real_save_then_restore_steady_2d, that goes through the REAL
   pipeline this time: loads the GUI's own "2D Poisson (Disk)" steady-state
   Quick Example, builds a real config, runs the actual generated training
   script as a subprocess (so model_config.json is written by the real
   save code, not assembled by the test), reads that file back, and
   restores the real checkpoint it produced. The existing checks in this
   file all hand-built the cfg dict with "steady_state" already set, which
   is exactly why they didn't catch this gap -- verified this new check
   fails (reproducing your exact RuntimeError) without this patch, and
   passes with it.

VERIFICATION DONE

 - Full headless regression suite passes (test_config_options.py,
   test_export_parity.py, test_sweep_registry.py, test_sweep_tab.py,
   test_sweep_param_expansion.py, test_templates.py 24/24,
   test_v56_no_template_solve.py, test_training_callbacks_visibility.py,
   test_custom_geometry.py, test_steady_state_ic_weights.py, and the
   updated test_restore_steady_state.py).
 - This patch isolated and verified byte-identical in a fresh independent
   clone of `custom-geometry` at the first Restore-Model-fix tip (830e211),
   before delivery.
 - The new real-pipeline test (train via the real generated script,
   restore the real checkpoint) was run end to end, not just syntax-
   checked, and confirmed to fail without this patch and pass with it.

Applies cleanly on top of `custom-geometry` at commit 830e211.

Run from the repo root, after Step 1 above:
    python3 apply_restore_steady_state_config_fix.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "restore_steady_state_config_fix.patch"
MARKER_TEXT = '"steady_state": {config.steady_state},'

CHANGED_FILES = [
    "pinnstudio/core/codegen.py",
    "tests/test_restore_steady_state.py",
]


def find_repo_root():
    here = Path(__file__).resolve().parent
    for candidate in (Path.cwd(), here):
        if (candidate / "pinnstudio" / "core" / "codegen.py").is_file():
            return candidate
    return None


def looks_like_inner_package_folder(cwd):
    return (
        (cwd / "core").is_dir() and (cwd / "ui").is_dir() and (cwd / "main.py").is_file()
        and not (cwd / "setup.py").is_file()
    )


def git_apply_check(root, patch_path, reverse=False):
    cmd = ["git", "apply", "--check", str(patch_path)]
    if reverse:
        cmd.insert(2, "--reverse")
    return subprocess.run(cmd, cwd=root, capture_output=True, text=True)


def main():
    if looks_like_inner_package_folder(Path.cwd()):
        print("You're standing inside the 'pinnstudio' PACKAGE folder itself "
              "(it directly contains core/, ui/, main.py) -- one level too "
              "deep. The repo root is one level up (it has setup.py, README.md, "
              "tests/, and a 'pinnstudio' subfolder).\n\n"
              "Run:\n  cd ..\nthen re-run this script from there.")
        sys.exit(1)

    root = find_repo_root()
    if root is None:
        print("Could not find 'pinnstudio/core/codegen.py' under the current "
              "directory or this script's directory.\n"
              "cd into your PINNStudio repo root (check with: "
              "ls setup.py pinnstudio/core/codegen.py) and run this script from there.")
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

    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root,
        capture_output=True, text=True,
    ).stdout.strip()
    if branch != "custom-geometry":
        print(f"You're on branch '{branch}', not 'custom-geometry'.\n"
              "Run Step 1 first:\n"
              "  git checkout custom-geometry\n"
              "  git pull origin custom-geometry\n"
              "...then run this script again.")
        sys.exit(1)

    codegen_path = root / "pinnstudio" / "core" / "codegen.py"
    try:
        already_applied = MARKER_TEXT in codegen_path.read_text()
    except Exception:
        already_applied = False
    if already_applied:
        print("Already applied -- this fix is already present in this "
              "checkout. Nothing to do.")
        sys.exit(0)

    check = git_apply_check(root, patch_path)
    if check.returncode != 0:
        if git_apply_check(root, patch_path, reverse=True).returncode == 0:
            print("Already applied -- this fix is already present in this "
                  "checkout. Nothing to do.")
            sys.exit(0)
        print("This patch doesn't apply cleanly here. Most likely this "
              "branch has diverged from the `custom-geometry` commit it was "
              "built against (830e211, the first Restore-Model-fix tip), or "
              "Step 1 wasn't done yet (git pull origin custom-geometry).\n\n"
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
    print("Next: review with 'git diff', run the test suite, then commit "
          "and push, e.g.")
    print("  git add " + " ".join(CHANGED_FILES))
    print('  git commit -m "Save steady_state in model_config.json so Restore Model can see it"')
    print("  git push origin custom-geometry")
    print()
    print("IMPORTANT: this only fixes it for MODELS YOU TRAIN FROM NOW ON.")
    print("Your existing model_config.json files (already saved, including the")
    print("one you've been restoring) still don't have \"steady_state\" in them.")
    print("Two ways to get your current Soomro_Paper model restoring correctly")
    print("without retraining:")
    print('  1. Edit that file by hand and add a line:  "steady_state": true,')
    print("     right after any other top-level key (model_config.json is")
    print("     plain JSON -- open it in any text editor).")
    print("  2. Or just retrain once with this patch applied -- the new")
    print("     model_config.json will have it automatically from then on.")
    print()
    print("Quick way to run the test suite (needs PyQt6/numpy/matplotlib, and")
    print("deepxde/torch -- already required to run PINNStudio at all -- for")
    print("its real restore/Error-Analysis/training checks):")
    print("  QT_QPA_PLATFORM=offscreen python3 tests/test_restore_steady_state.py")


if __name__ == "__main__":
    main()
