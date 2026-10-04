#!/usr/bin/env python3
"""
Applies the Restore panel display-refresh fix: after Restore && Visualize
finished, the right-side results panel sometimes showed a stale leftover
plot/animation from a COMPLETELY DIFFERENT, earlier restore run in the
same save folder, instead of what the just-finished run actually
produced -- e.g. a steady-state Surface restore correctly saving
restored_plot.png, but the right panel showing an unrelated animated GIF
left over from an earlier Animation-type restore in that same folder.

Root cause: _on_restore_done() in main_window.py decided what to show
purely by os.path.exists(restored_plot.png) / os.path.exists(
restored_animation.gif) -- so a stale file with the right name, left
over from any earlier restore, was indistinguishable from this run's own
fresh output.

Fix: _on_restore() now deletes any pre-existing restored_plot.png,
restored_animation.gif, and the three Error-Analysis-on-restore output
files in save_dir BEFORE launching the new restore subprocess, so the
only files that can exist by the time _on_restore_done() checks are ones
THIS run actually just wrote.

Adds tests/test_restore_display_refresh.py (registered in the CI
smoke-test workflow) -- trains two tiny real models, pre-plants a stale
leftover file of the "other" type in save_dir, runs a real restore
through _on_restore()'s actual QThread+subprocess path, and confirms the
stale file is gone and the right panel shows this run's own output.

See restore_display_refresh_fix.patch (same folder as this script) for
the full diff.

Usage:
    python3 apply_restore_display_refresh_fix.py [path-to-pinnstudio-repo]

If no path is given, the current directory is used. Safe to re-run --
if the fix is already present, this exits without changing anything.

NOTE: this patch is built on top of the geometry-masking fix delivered
earlier (geometry_masking_fix.patch / commit "Restore Model: mask
prediction to the real domain, not the bounding box"). Apply it to a
repo that already has that fix -- which yours does, since you already
applied, committed, and pushed it.
"""
import os
import subprocess
import sys

# A short, uniquely-identifying string this patch adds to main_window.py
# -- verified absent from the pre-patch tip this was built against, and
# appearing exactly once after.
MARKER = "Clear any restored_plot.png/restored_animation.gif"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PATCH_PATH = os.path.join(_SCRIPT_DIR, "restore_display_refresh_fix.patch")


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
        print("✅ Already applied -- main_window.py already has the display-refresh fix. Nothing to do.")
        return 0

    if not os.path.isfile(_PATCH_PATH):
        print(f"❌ Could not find restore_display_refresh_fix.patch -- it must be in the same folder as "
              f"this script ({_SCRIPT_DIR!r}).")
        return 1

    print(f"Applying restore_display_refresh_fix.patch to {repo!r} ...")
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", _PATCH_PATH],
        cwd=repo, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("❌ git apply failed:")
        if result.stdout.strip():
            print(result.stdout)
        if result.stderr.strip():
            print(result.stderr)
        print()
        print("This patch assumes the geometry-masking fix (geometry_masking_fix.patch, "
              "\"Restore Model: mask prediction to the real domain...\") is already applied -- "
              "if your main_window.py doesn't have that yet, apply that one first. Otherwise this "
              "usually means your working tree has local changes that overlap with this patch's "
              "lines. Commit or stash any local changes and re-run, or apply "
              "restore_display_refresh_fix.patch by hand and resolve the conflicts.")
        return 1

    print("✅ Patch applied successfully.")
    print("   Changed: pinnstudio/ui/main_window.py, .github/workflows/smoke-test.yml")
    print("   Added:   tests/test_restore_display_refresh.py")
    print()
    print("Next steps:")
    print("  1. Review the changes:  git diff")
    print("  2. Run the new test (needs deepxde + torch installed, same as the app itself):")
    print("     QT_QPA_PLATFORM=offscreen python3 tests/test_restore_display_refresh.py")
    print("  3. git add -A && git commit -m \"...\" && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
