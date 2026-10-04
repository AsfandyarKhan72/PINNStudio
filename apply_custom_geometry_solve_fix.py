#!/usr/bin/env python3
"""
Custom Geometry -- Solve-button fix (training was never actually blocked
at the codegen level, just at a separate, stale GUI gate).

Builds on top of the already-pushed 3D extension (commit 2566af9), on
the SAME `custom-geometry` branch -- no new branch.

STEP 1 -- MAKE SURE YOU'RE ON THE custom-geometry BRANCH, UP TO DATE

    git checkout custom-geometry
    git pull origin custom-geometry

STEP 2 -- DOUBLE-CHECK YOU'RE AT THE REPO ROOT (not inside the inner
`pinnstudio` package folder), THEN RUN THIS SCRIPT

    ls setup.py pinnstudio/core/codegen.py

If both print back cleanly, you're in the right place:

    python3 apply_custom_geometry_solve_fix.py

WHAT THIS PATCH FIXES

This is the bug behind the message you hit: "Training for 'Custom'
geometry isn't wired up yet -- switch Geometry Type to one of Interval,
Rectangle, Disk, Ellipse, Triangle, Polygon, Cuboid, Sphere to train."

The actual training machinery (codegen.py's generate_script(), which
Solve runs under the hood via runner.py's run_pinn()) has fully
supported Custom geometry since Phase 1 (2D) and the 3D extension
(Cuboid/Sphere) -- both were verified end-to-end with real DeepXDE
training runs before each was delivered. But a SEPARATE, older gate --
main_window.py's _geometry_supported_for_training(), checked first thing
when you click Solve -- had its own hardcoded list of trainable geometry
types that was simply never updated to include "Custom" when Custom
geometry was added. So Solve always refused Custom outright, regardless
of what shapes were in it, even though the exact same config trains
correctly when run directly (which is how every previous round's
end-to-end verification tested it -- never through the actual Solve
button itself, which is why this slipped through until you hit it in
your own GUI session).

1. pinnstudio/ui/main_window.py -- `"Custom"` added to
   _geometry_supported_for_training()'s `_supported` tuple. Nothing else
   about that gate changes: an empty or malformed Custom shape list is
   still caught separately and clearly by config.validate() right after
   this gate (in _on_solve()), and a genuinely unsupported geometry type
   is still refused by this gate exactly as before.
2. tests/test_custom_geometry.py (extended) -- a regression test that
   calls _geometry_supported_for_training() directly for Custom in both
   2D and 3D (confirming it now returns allowed), re-checks every other
   geometry type is still allowed (so this fix can't itself regress
   something that already worked), and confirms a genuinely unsupported
   type is still refused with a clear message. This check was verified
   to actually catch the original bug (temporarily reverted the fix,
   confirmed the test fails with the exact message you saw, then
   restored the fix) before this patch was finalized.

VERIFICATION DONE

 - Confirmed the fix is the complete story: runner.py's run_pinn() (what
   SolverThread actually runs when you click Solve) calls
   codegen.generate_script(config) directly -- the exact same function
   every prior round's "real end-to-end DeepXDE training" verification
   already exercised for Custom geometry (2D: Triangle minus Disk; 3D:
   Cuboid minus Sphere). There's no other geometry-building logic in the
   Solve path, so this one-line gate was the only thing standing between
   a working Custom-geometry config and an actual training run.
 - Full headless regression suite passes (test_config_options.py,
   test_export_parity.py, test_sweep_registry.py, test_sweep_tab.py,
   test_sweep_param_expansion.py, test_templates.py 24/24,
   test_v56_no_template_solve.py, test_training_callbacks_visibility.py,
   and the extended test_custom_geometry.py).
 - This patch isolated and verified byte-identical in a fresh,
   independent clone of `custom-geometry` (at the 3D-extension tip,
   2566af9), before delivery.

Applies cleanly on top of `custom-geometry` at commit 2566af9.

Run from the repo root, after Step 1 above:
    python3 apply_custom_geometry_solve_fix.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "custom_geometry_solve_gate_fix.patch"
# Must be unique to THIS fix. Note: "Polygon", "Cuboid", "Sphere", "Custom")"
# on its own is NOT unique -- _preview_domain()'s own geometry-type check
# already contains that exact substring (it's included Custom since the 3D
# round), so using it as a marker would cause a false "already applied" on
# every checkout, patched or not. This phrase is from this fix's own new
# docstring text and appears nowhere else in the file.
MARKER = "Custom in both, via"

CHANGED_FILES = [
    "pinnstudio/ui/main_window.py",
    "tests/test_custom_geometry.py",
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

    marker_in_main = root / "pinnstudio" / "ui" / "main_window.py"
    if marker_in_main.is_file() and MARKER in marker_in_main.read_text():
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
              "built against (2566af9, the 3D-extension tip), or Step 1 "
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
    print('  git commit -m "Fix Solve-button gate blocking Custom geometry from training"')
    print("  git push origin custom-geometry")
    print()
    print("Quick way to run the test suite (needs PyQt6/numpy/matplotlib):")
    print("  QT_QPA_PLATFORM=offscreen python3 tests/test_custom_geometry.py")
    print()
    print("After this, Custom geometry (2D and 3D) should actually start "
          "training when you click Solve, the same way it already does "
          "when you Export as DeepXDE Script and run it directly.")


if __name__ == "__main__":
    main()
