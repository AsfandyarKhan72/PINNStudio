#!/usr/bin/env python3
"""
Restore Model fix: restoring a steady-state model no longer crashes.

Builds on top of the already-pushed IC-weights fix (commit 7fc3eaa), on
the SAME `custom-geometry` branch -- no new branch.

STEP 1 -- MAKE SURE YOU'RE ON THE custom-geometry BRANCH, UP TO DATE

    git checkout custom-geometry
    git pull origin custom-geometry

STEP 2 -- DOUBLE-CHECK YOU'RE AT THE REPO ROOT (not inside the inner
`pinnstudio` package folder), THEN RUN THIS SCRIPT

    ls setup.py pinnstudio/core/codegen.py

If both print back cleanly, you're in the right place:

    python3 apply_restore_steady_state_fix.py

WHAT THIS PATCH FIXES

The crash you hit: restoring your trained steady-state Soomro_Paper model
and generating a plot raised

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (40000x3 and 2x64)

Root cause: _build_restore_script() in main_window.py -- the script
builder behind the Restore panel's "Restore && Visualize" button -- never
checked whether the config being restored was steady-state. A steady-state
problem has no time axis at all (you already confirmed this is correct --
your training log showed "IC weights: []", and the network's first layer
is sized for 2 inputs, not 3, for your 2D case), but this function always
appended a time column to the x,y grid it fed into model.predict(...)
regardless -- so a 200x200 resolution grid (40000 points) with x,y,t (3
columns) was fed into a network whose first layer only accepts 2 inputs.
Reproduced exactly (same shapes, same error) against the pre-fix code
before building this patch.

1. pinnstudio/ui/main_window.py --
   - _build_restore_script() now checks cfg.get("steady_state", False)
     throughout: the restore scaffold builds a plain dde.data.PDE (no
     GeometryXTime/time axis) for a steady-state config, same convention
     generate_script() already uses correctly at training time, and every
     model.predict(...) input array in the Surface branch (1D/2D/3D --
     the actually reachable paths) drops the time column when steady.
     "Line (time steps)" and the two Animation types get a defensive
     steady-state fallback too (so they can't crash even if somehow
     reached), on top of being hidden from the dropdown per the next item.
   - _build_restore_ea_script() (the separate script-builder for Error
     Analysis run against a restored model) got the same treatment --
     reference-file loading and prediction arrays now match
     generate_script()'s own steady-state reference-file format (x,y,u /
     x,y,z,u, no time column), covering your Error-Analysis-on-restore
     workflow for the corrected Soomro_Paper problem too.
   - The Restore panel's "Visualization type" dropdown now hides "Line
     (time steps)", "Animation Line (GIF)", and "Animation Surface (GIF)"
     when the config about to be restored is steady-state -- there's no
     time axis to step through or animate over, same reasoning
     _on_steady_state_changed() already applies to Time Adaptive Training
     on the main Setup tab. Only "Surface" (plus, in Inverse mode, the two
     Parameter Convergence options, which never touch the model/time axis
     at all) stay available. Switching back to a non-steady model/config
     restores all four options.
2. tests/test_restore_steady_state.py (new) -- covers: the dropdown
   filtering for steady/non-steady/Inverse+steady; _build_restore_script's
   generated script actually restoring a real trained steady-state model
   (1D, 2D -- your exact reported case, and 3D) and producing a Surface
   plot without crashing; the non-steady 2D Surface path still working
   unchanged (regression check); and _build_restore_ea_script's generated
   script running cleanly against a steady 2D reference file and a
   restored steady 2D 4-output model, matching your actual Error Analysis
   use case. Verified this test actually catches the original bug (ran it
   against the pre-fix code -- it failed; reproduced your exact
   RuntimeError shape mismatch directly against the old function before
   writing the fix). The deepxde/torch-dependent checks skip cleanly (not
   a failure) in an environment without them installed, matching this
   repo's existing lightweight CI smoke-test install step -- every real
   environment that can run PINNStudio at all has them.
3. .github/workflows/smoke-test.yml -- added
   tests/test_restore_steady_state.py to the CI test list.

VERIFICATION DONE

 - Full headless regression suite passes (test_config_options.py,
   test_export_parity.py, test_sweep_registry.py, test_sweep_tab.py,
   test_sweep_param_expansion.py, test_templates.py 24/24,
   test_v56_no_template_solve.py, test_training_callbacks_visibility.py,
   test_custom_geometry.py, test_steady_state_ic_weights.py, and the new
   test_restore_steady_state.py).
 - This patch isolated and verified byte-identical in a fresh, independent
   clone of `custom-geometry` (at the IC-weights-fix tip, 7fc3eaa), before
   delivery.
 - The new test's real-restore checks (steady 1D/2D/3D Surface, steady 2D
   Error Analysis, plus the non-steady 2D Surface regression check) were
   run end-to-end against actually-trained-and-saved checkpoints, not just
   syntax-checked.

Applies cleanly on top of `custom-geometry` at commit 7fc3eaa.

Run from the repo root, after Step 1 above:
    python3 apply_restore_steady_state_fix.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "restore_model_steady_state_fix.patch"
MARKER_TEXT = "def _refresh_restore_viz_options"

CHANGED_FILES = [
    ".github/workflows/smoke-test.yml",
    "pinnstudio/ui/main_window.py",
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

    main_window_path = root / "pinnstudio" / "ui" / "main_window.py"
    try:
        already_applied = MARKER_TEXT in main_window_path.read_text()
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
              "built against (7fc3eaa, the IC-weights-fix tip), or Step 1 "
              "wasn't done yet (git pull origin custom-geometry).\n\n"
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
    print('  git commit -m "Fix Restore Model crash for steady-state problems"')
    print("  git push origin custom-geometry")
    print()
    print("Quick way to run the new test (needs PyQt6/numpy/matplotlib, and")
    print("deepxde/torch -- already required to run PINNStudio at all -- for")
    print("its real restore/Error-Analysis checks):")
    print("  QT_QPA_PLATFORM=offscreen python3 tests/test_restore_steady_state.py")
    print()
    print("After this, restoring a steady-state model (Surface plot, or Error")
    print("Analysis against it) will work correctly, and the Restore panel's")
    print("Visualization type dropdown will only offer 'Surface' for a")
    print("steady-state config -- the three time-based options don't apply")
    print("and are hidden rather than crashing if picked.")


if __name__ == "__main__":
    main()
