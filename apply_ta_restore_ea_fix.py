#!/usr/bin/env python3
"""
Fixes a crash reported after applying the Time-Adaptive loss-history
patch: restoring a Time-Adaptive run's COMBINED models (>=2 step
checkpoints auto-detected and stitched together, used whenever a
Restore config has 2+ time_adaptive_steps/step_*/ siblings) with Error
Analysis reference files configured always crashed, for every viz type:

    File "<script>", line NNN, in <module>
      _ea_u_pinns.append(_extract_restore_field(model.predict(_xt)).flatten())
    NameError: name 'model' is not defined
    Restore failed -- check architecture matches saved model.

The restore itself (the actual plot/gif, RESTORE_DONE) always finished
fine -- only the separate Error-Analysis-on-restore pass that runs right
after it crashed.

Root cause: pinnstudio/ui/main_window.py's _build_restore_ea_script() --
the shared Error-Analysis-on-restore script builder _on_restore() always
appends after EITHER _build_restore_script() (single model, binds a
plain `model`) OR _build_restore_script_ta() (Time-Adaptive combined,
multiple per-step models, NEVER a single `model`) -- unconditionally
generated `model.predict(...)` in its 3 prediction call sites, assuming
whichever restore script ran before it always left a plain `model`
bound. It never does for the Time-Adaptive combined case: that script
restores each detected step's own model lazily and routes by time range
through `_ta_model_for_t(tv)` instead (already defined, unconditionally,
near the top of that generated script).

What this patch changes (pinnstudio/ui/main_window.py only):

  - _build_restore_ea_script() takes a new `is_ta` parameter (default
    False). When True, all 3 prediction call sites use
    `_ta_model_for_t(_tv).predict(...)` instead of a bare
    `model.predict(...)` -- each site is already inside a
    `for _i, _tv in enumerate(_ea_times):` loop, so `_tv` is in scope.
    This is also the CORRECT behavior, not just a crash fix: the
    reference times being compared can span more than one Time-Adaptive
    step's own time window, so no single model would even be valid for
    every one of them.
  - _on_restore() now passes `is_ta=_use_ta_restore` to that call -- the
    same flag it already computes to decide which restore-script builder
    to call in the first place.
  - The single-model case (is_ta omitted/False) renders byte-identical
    text to before this fix, so nothing else about Restore && Visualize
    changes.

Adds tests/test_restore_ta_error_analysis.py (registered in CI) --
trains 2 tiny real Time-Adaptive step models, auto-detects them via the
real _detect_restore_ta_steps(), and runs a real combined restore +
Error Analysis with reference times spanning both steps' windows. A
built-in negative control first confirms the OLD call convention
(is_ta omitted) reproduces the user's exact NameError on this exact
script, before confirming the fix resolves it. Also re-checks
test_export_parity.py's existing single-model EA-script assertions still
hold unchanged.

Usage:
    python3 apply_ta_restore_ea_fix.py [path-to-pinnstudio-repo]

If no path is given, the current directory is used. Safe to re-run --
if the fix is already present, this exits without changing anything.
"""
import os
import subprocess
import sys

# A short, uniquely-identifying string this patch adds to main_window.py
# -- verified absent before this patch and appearing exactly once after.
MARKER = '_predict_call = "_ta_model_for_t(_tv).predict" if is_ta else "model.predict"'

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PATCH_PATH = os.path.join(_SCRIPT_DIR, "ta_restore_ea_fix.patch")


def looks_like_inner_package_folder(path):
    """True if `path` is the INNER pinnstudio/ package folder itself
    (containing ui/, core/, ...) rather than the repo root that CONTAINS
    that folder."""
    return (
        os.path.basename(os.path.normpath(path)) == "pinnstudio"
        and os.path.isdir(os.path.join(path, "ui"))
        and os.path.isdir(os.path.join(path, "core"))
        and not os.path.isdir(os.path.join(path, "pinnstudio"))
    )


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    repo = os.path.abspath(repo)

    if looks_like_inner_package_folder(repo):
        parent = os.path.dirname(repo)
        print(f"❌ {repo!r} looks like the INNER pinnstudio package folder, not the repo root.")
        print(f"   Run this from the repo root instead (the folder that CONTAINS pinnstudio/):")
        print(f"     cd {parent!r} && python3 {os.path.basename(__file__)}")
        return 1

    main_window_path = os.path.join(repo, "pinnstudio", "ui", "main_window.py")
    if not os.path.isfile(main_window_path):
        print(f"❌ Could not find pinnstudio/ui/main_window.py under {repo!r}.")
        print(f"   Run this from your pinnstudio repo root, or pass its path as an argument:")
        print(f"     python3 {os.path.basename(__file__)} /path/to/pinnstudio")
        return 1

    with open(main_window_path, "r") as f:
        main_window_content = f.read()
    if MARKER in main_window_content:
        print("✅ Already applied -- main_window.py already routes Restore Error Analysis through "
              "_ta_model_for_t() for Time-Adaptive combined restores. Nothing to do.")
        return 0

    if not os.path.isfile(_PATCH_PATH):
        print(f"❌ Could not find ta_restore_ea_fix.patch -- it must be in the same folder as this script "
              f"({_SCRIPT_DIR!r}).")
        return 1

    print(f"Applying ta_restore_ea_fix.patch to {repo!r} ...")
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "--binary", _PATCH_PATH],
        cwd=repo, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("❌ git apply failed:")
        if result.stdout.strip():
            print(result.stdout)
        if result.stderr.strip():
            print(result.stderr)
        print()
        print("This usually means your working tree has local changes that overlap with this patch's lines. "
              "Commit or stash any local changes and re-run, or apply ta_restore_ea_fix.patch by hand and "
              "resolve the conflicts.")
        return 1

    print("✅ Patch applied successfully.")
    print("   Changed: pinnstudio/ui/main_window.py, .github/workflows/smoke-test.yml")
    print("   Added:   tests/test_restore_ta_error_analysis.py")
    print()
    print("Next steps:")
    print("  1. Review the changes:  git diff")
    print("  2. Run the test (needs PyQt6 + torch + deepxde installed, same as the app itself --")
    print("     this one actually trains 2 tiny real models, so it takes a minute or two):")
    print("     QT_QPA_PLATFORM=offscreen python3 tests/test_restore_ta_error_analysis.py")
    print("  3. git add -A && git commit -m \"...\" && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
