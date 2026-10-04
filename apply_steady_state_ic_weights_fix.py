#!/usr/bin/env python3
"""
Steady-state fix: Loss Weights panel no longer shows IC weight rows.

Builds on top of the already-pushed Solve-gate fix (commit 557025f), on
the SAME `custom-geometry` branch -- no new branch.

STEP 1 -- MAKE SURE YOU'RE ON THE custom-geometry BRANCH, UP TO DATE

    git checkout custom-geometry
    git pull origin custom-geometry

STEP 2 -- DOUBLE-CHECK YOU'RE AT THE REPO ROOT (not inside the inner
`pinnstudio` package folder), THEN RUN THIS SCRIPT

    ls setup.py pinnstudio/core/codegen.py

If both print back cleanly, you're in the right place:

    python3 apply_steady_state_ic_weights_fix.py

WHAT THIS PATCH FIXES

The question you asked: a steady-state problem has no time axis, so
there's no Initial Condition at all -- confirmed by codegen.py already
building an empty IC-weights list for a steady-state config (the
training log's own "IC weights: []" line), and the Initial Condition
panel itself is already correctly hidden once you check Steady-state.

But the separate Loss Weights panel's own per-output "IC {n} (...)"
weight rows were built with no steady_state check at all -- only each
output's own IC-active checkbox decided whether its row showed. So those
rows (and their weight fields) stayed visible and editable for a
steady-state problem even though the values were silently never used in
training. Purely cosmetic (nothing was being miscomputed -- your training
log already proved that), but confusing, and worth cleaning up now that
you noticed it.

1. pinnstudio/ui/main_window.py --
   - _build_weight_inputs() now skips every "IC ..." row (both the
     shared-weights block and each per-phase block) whenever Steady-state
     is checked.
   - _on_steady_state_changed() now also refreshes the Loss Weights panel
     when you toggle the checkbox, so this takes effect immediately --
     previously that function updated the Initial Condition panel,
     Adaptive Training, IC Pre-Training, and the "t:" row, but never the
     Loss Weights panel itself, so toggling Steady-state on its own could
     leave stale IC rows showing until something unrelated (e.g. changing
     the output count) happened to rebuild the panel.
2. tests/test_steady_state_ic_weights.py (new) -- covers: a non-steady
   config still shows its IC rows (this fix must not remove anything
   that should stay); checking Steady-state removes them live, with no
   other change needed; unchecking restores them; the same for per-phase
   (not shared) weights; and _flat_loss_weights_string() (what actually
   gets saved/trained on) naturally emits no IC terms once the rows are
   gone. Verified this test actually catches the original bug (reverted
   the fix, confirmed every one of these checks fails with the stale
   rows still present, then restored the fix and re-confirmed green).
   Wired into .github/workflows/smoke-test.yml alongside the existing
   suite.

VERIFICATION DONE

 - Full headless regression suite passes (test_config_options.py,
   test_export_parity.py, test_sweep_registry.py, test_sweep_tab.py,
   test_sweep_param_expansion.py, test_templates.py 24/24,
   test_v56_no_template_solve.py, test_training_callbacks_visibility.py,
   test_custom_geometry.py, and the new test_steady_state_ic_weights.py).
 - This patch isolated and verified byte-identical in a fresh,
   independent clone of `custom-geometry` (at the Solve-gate-fix tip,
   557025f), before delivery.

Applies cleanly on top of `custom-geometry` at commit 557025f.

Run from the repo root, after Step 1 above:
    python3 apply_steady_state_ic_weights_fix.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "steady_state_ic_weights_fix.patch"
MARKER = "tests/test_steady_state_ic_weights.py"

CHANGED_FILES = [
    ".github/workflows/smoke-test.yml",
    "pinnstudio/ui/main_window.py",
    "tests/test_steady_state_ic_weights.py",
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

    marker_path = root / MARKER
    if marker_path.is_file():
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
              "built against (557025f, the Solve-gate-fix tip), or Step 1 "
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
    print('  git commit -m "Hide IC loss-weight rows for steady-state problems"')
    print("  git push origin custom-geometry")
    print()
    print("Quick way to run the test suite (needs PyQt6/numpy/matplotlib):")
    print("  QT_QPA_PLATFORM=offscreen python3 tests/test_steady_state_ic_weights.py")
    print()
    print("After this, checking Steady-state in the GUI will immediately drop "
          "the 'IC ...' rows from the Loss Weights panel, matching what was "
          "already true at training time.")


if __name__ == "__main__":
    main()
