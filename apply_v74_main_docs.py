#!/usr/bin/env python3
"""
Run this on the `main` branch, AFTER fast-forwarding it to merge in
`param-sweep` (see the merge step below -- that part is just plain git
commands, not a patch, since `main` hasn't diverged from `param-sweep`'s
own base at all: merging it is a pure fast-forward with zero conflicts).

This script's own patch only touches documentation/CI plumbing around
that already-merged code: README.md, CHANGELOG.md, setup.py, a new
screenshot, and the CI workflow file. No application code changes here
-- pinnstudio/ itself is untouched by this patch (it was already merged
in by the fast-forward step before this script runs).

STEP 1 -- MERGE param-sweep INTO main (fast-forward, no conflicts)

    git checkout main
    git pull origin main
    git merge --ff-only param-sweep

If that fails with "not possible to fast-forward", main has moved since
this was last checked -- stop and let me know rather than forcing a
merge commit, since that changes how this script's patch needs to be
built.

STEP 2 -- RUN THIS SCRIPT (applies the documentation patch on top)

    python3 apply_v74_main_docs.py

WHAT THIS PATCH CHANGES

1. README.md: new "## Parameter Sweep" section under Features (right
   after "Live parameter convergence (Inverse mode)", before "Analysis &
   Output") documenting the whole feature for anyone installing
   PINNStudio fresh from `main` -- what it is, where it lives in the GUI,
   the three combination modes (one-at-a-time / all combinations / zip),
   the full current list of sweepable parameter categories (Network,
   Collocation Points, Training Phases, Loss Weights, Inverse, RAR, Time
   Adaptive, Input/Output Transform), what each run produces (its own
   results folder plus the sweep-wide manifest/summary CSV), and how the
   right panel behaves once a sweep finishes. Includes a real screenshot
   of the panel configured with a 2-parameter grid sweep
   (assets/Images/Parameter_Sweep/sweep_panel.png, new file, captured
   from this exact merged code with a real virtual display). Repository
   Structure's file tree also now lists the two new core files
   (sweep_registry.py, sweep_runner.py) that a reader browsing the repo
   would otherwise wonder about.
2. CHANGELOG.md: new "## [1.4.0]" entry (Keep a Changelog format,
   matching every prior release entry) summarizing Parameter Sweep as a
   finished feature for anyone reading the changelog, not a round-by-
   round development history.
3. setup.py: version bumped 1.3.0 -> 1.4.0, matching the CHANGELOG entry
   -- this does NOT publish anything to PyPI by itself; that still needs
   your own `python -m build` + `twine upload` (or however you normally
   cut a release) whenever you're ready to actually publish 1.4.0.
4. .github/workflows/smoke-test.yml: CI now also runs
   tests/test_sweep_param_expansion.py and
   tests/test_training_callbacks_visibility.py -- both already existed
   and already passed without torch/deepxde installed (confirmed again
   before adding this), but were never wired into CI in the rounds that
   added them. Now every headless test in the repo actually runs on
   every push/PR to main, not just most of them.

VERIFICATION DONE

 - Fast-forward merge confirmed conflict-free: `main`'s HEAD before this
   round was exactly `param-sweep`'s own merge-base, so
   `git merge --ff-only param-sweep` requires no conflict resolution at
   all, by construction.
 - Full headless regression suite (test_config_options.py,
   test_export_parity.py, test_sweep_registry.py, test_sweep_tab.py,
   test_sweep_param_expansion.py, test_templates.py 24/24,
   test_v56_no_template_solve.py, test_training_callbacks_visibility.py)
   passes on the merged + documented `main`.
 - This patch isolated and verified byte-identical (including the binary
   PNG) in a fresh independent clone of `main`, fast-forwarded the same
   way, before delivery.

Applies cleanly on top of `main` fast-forwarded to `param-sweep`'s tip
(commit 4292089).

Run from the repo root, after Step 1 above:
    python3 apply_v74_main_docs.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "v74_main_docs_delta.patch"
MARKER = "sweep_panel.png"

CHANGED_FILES = [
    ".github/workflows/smoke-test.yml",
    "CHANGELOG.md",
    "README.md",
    "assets/Images/Parameter_Sweep/sweep_panel.png",
    "setup.py",
]


def find_repo_root():
    here = Path(__file__).resolve().parent
    for candidate in (Path.cwd(), here):
        if (candidate / "pinnstudio" / "core" / "codegen.py").is_file():
            return candidate
    return None


def git_apply_check(root, patch_path, reverse=False):
    cmd = ["git", "apply", "--check", "--binary", str(patch_path)]
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

    marker_path = root / "assets" / "Images" / "Parameter_Sweep" / "sweep_panel.png"
    if marker_path.is_file():
        print("Already applied -- this fix is already present in this "
              "checkout. Nothing to do.")
        sys.exit(0)

    sweep_registry_path = root / "pinnstudio" / "core" / "sweep_registry.py"
    if not sweep_registry_path.is_file():
        print("This doesn't look like it has Parameter Sweep merged in yet "
              "(pinnstudio/core/sweep_registry.py is missing).\n"
              "Run Step 1 first:\n"
              "  git checkout main\n"
              "  git pull origin main\n"
              "  git merge --ff-only param-sweep\n"
              "...then run this script again.")
        sys.exit(1)

    check = git_apply_check(root, patch_path)
    if check.returncode != 0:
        if git_apply_check(root, patch_path, reverse=True).returncode == 0:
            print("Already applied -- this fix is already present in this "
                  "checkout. Nothing to do.")
            sys.exit(0)
        print("This patch doesn't apply cleanly here. Most likely `main` "
              "has moved since this was built, or Step 1's merge wasn't "
              "done yet.\n\ngit apply's error:\n" + check.stderr)
        sys.exit(1)

    apply = subprocess.run(
        ["git", "apply", "--binary", str(patch_path)], cwd=root,
        capture_output=True, text=True,
    )
    if apply.returncode != 0:
        print("git apply failed unexpectedly after passing --check:\n" + apply.stderr)
        sys.exit(1)

    print("Applied cleanly:")
    for f in CHANGED_FILES:
        print(f"  - {f}")
    print()
    print("Next: review with 'git diff', then commit and push to main, e.g.")
    print("  git add " + " ".join(CHANGED_FILES))
    print('  git commit -m "Document Parameter Sweep in README, bump to 1.4.0, '
          'wire remaining sweep tests into CI"')
    print("  git push")
    print()
    print("Remember: this does NOT publish to PyPI. When you're ready to")
    print("actually release 1.4.0, build and upload it the way you normally")
    print("do (python -m build && twine upload, or your usual process).")


if __name__ == "__main__":
    main()
