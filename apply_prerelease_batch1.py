#!/usr/bin/env python3
"""
Applies the first batch of pre-release fixes/tweaks from the chat -- the
5 clear, unambiguous ones (the other 3 raised need a decision first, see
the chat: 1D Surface Animation's meaningless t-axis, a Plot Settings size
option, and the Time-Adaptive loss plot showing only the last sub-domain).

What this patch changes:

  1. Restore panel's "Save visualization to:" field now defaults to
     ~/PINNStudio_Results (same default as the main Solve panel's
     "Save to:" field), instead of starting empty.

  2. The 1D "Animation Line (GIF)" time label -- in the main Results
     panel (codegen.py) AND both Restore-panel script builders
     (main_window.py, standard + Time-Adaptive) -- moves from inside the
     plot in red (ax.text at (0.02, 0.95), color #ff8787) to just above
     the plot in black, centered, matching how 2D/3D's own Surface
     Animation already shows "t = ..." via ax.set_title(). A real
     set_title() isn't used because blit=True animations don't redraw
     the title each frame; a Text artist positioned just above the axes
     (y=1.02 in axes coordinates) renders in the same place while still
     being blit-compatible. Rendered and visually checked before
     shipping -- see the chat.

  3. The Custom-geometry README image/example changed from a Triangle
     with a Disk subtracted (the same shape as this project's own
     MHD-validation case, which traces back to a colleague's paper) to a
     generic "plate with a hole" Rectangle-minus-Disk example, so the
     README doesn't read as reusing someone else's figure. Same
     descriptive text, different (newly generated) image.

  4. The 8 domain spinboxes (x_min/x_max, y_min/y_max, z_min/z_max,
     t_min/t_max) now share one fixed width (85px) so they render the
     same size regardless of decimal count. t_min/t_max keep their
     4-decimal precision (needed so e.g. 1D Schrodinger's t_max = pi/2
     can actually be set exactly -- 2 decimals would silently round it to
     1.57); the fix is a shared width, not reduced precision.

  5. Boundary Conditions panel: the "📖 Show location examples" hint
     (a few lines explaining the location-expression convention) used to
     repeat on every single BC row, each with its own collapsed toggle.
     It's now a single toggle + hint block at the top of the whole panel.
     Each row's "Where:" field keeps its own short inline placeholder
     example. Also: each row's "Type:"/"Output #:" controls, previously
     on two separate lines with the control pushed to the far right by an
     addStretch() before it (leaving a wide empty gap after the label),
     are now on one line, controls right next to their own labels.

Adds tests/test_prerelease_gui_tweaks.py (registered in the CI
smoke-test workflow) covering all of 1/4/5 above as pure widget-state
checks (no training needed).

Usage:
    python3 apply_prerelease_batch1.py [path-to-pinnstudio-repo]

If no path is given, the current directory is used. Safe to re-run --
if the fix is already present, this exits without changing anything.
"""
import os
import subprocess
import sys

# A short, uniquely-identifying string this patch adds to main_window.py
# -- verified absent before this patch and appearing exactly once after.
MARKER = "self.bc_loc_hint_toggle = QCheckBox"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PATCH_PATH = os.path.join(_SCRIPT_DIR, "prerelease_batch1.patch")


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
        print("✅ Already applied -- main_window.py already has the panel-level BC location hint. Nothing to do.")
        return 0

    if not os.path.isfile(_PATCH_PATH):
        print(f"❌ Could not find prerelease_batch1.patch -- it must be in the same folder as this script "
              f"({_SCRIPT_DIR!r}).")
        return 1

    print(f"Applying prerelease_batch1.patch to {repo!r} ...")
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
              "Commit or stash any local changes and re-run, or apply prerelease_batch1.patch by hand and "
              "resolve the conflicts.")
        return 1

    print("✅ Patch applied successfully.")
    print("   Changed: README.md, pinnstudio/core/codegen.py, pinnstudio/ui/main_window.py,")
    print("            .github/workflows/smoke-test.yml")
    print("   Added:   tests/test_prerelease_gui_tweaks.py,")
    print("            assets/Images/View_Domain_Pictures/Custom_Geometry_Rectangle_minus_Disk.png")
    print("   Removed: assets/Images/View_Domain_Pictures/Custom_Geometry_Triangle_minus_Disk.png")
    print()
    print("Next steps:")
    print("  1. Review the changes:  git diff")
    print("  2. Run the new test (needs PyQt6 installed, same as the app itself):")
    print("     QT_QPA_PLATFORM=offscreen python3 tests/test_prerelease_gui_tweaks.py")
    print("  3. git add -A && git commit -m \"...\" && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
