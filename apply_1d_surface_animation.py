#!/usr/bin/env python3
"""
Applies this round's fix for discussion item #3 from the chat: 1D's
"Surface Animation (GIF)" plot type didn't make sense -- a 1D animation
frame has only one real spatial axis (x), so the GIF code faked a second
axis by stretching u(x) sideways across a cosmetic range rather than
showing real data varying across it. "Line Animation (GIF)" already
covers 1D's real time-animated case correctly, and "Surface" (static)
still shows the full, real x-t field in one image.

What this patch changes (pinnstudio/ui/main_window.py only):

  1. "Surface Animation (GIF)" is removed from plot_type_combo's initial
     item list (the app starts in 1D), and _on_dim_changed() now adds it
     back the moment the dimension switches to 2D/3D, and removes it
     again switching back to 1D -- falling back to "Line Animation (GIF)"
     first if it had been selected (done with blockSignals so that
     programmatic fallback doesn't pop the real "Line Plot Settings"
     dialog that a *user* picking a new plot type normally gets).

  2. The Restore panel's own, separately-filtered viz dropdown
     (_restore_forward_viz_items) applies the same exclusion -- "Animation
     Surface (GIF)" is left out when the config being restored has
     "problem_dim": "1D". A config missing that field entirely (e.g. one
     saved before it existed) is NOT assumed to be 1D -- its real
     dimension is simply unknown, and defaulting it to 1D would silently
     hide this option for an old 2D/3D config too. This matches the
     existing test tests/test_restore_steady_state.py, which restores a
     config with no "problem_dim" key and expects all four options.

  3. codegen.py's actual GENERATION code for 1D's "Surface Animation
     (GIF)" is deliberately left in place, unreachable through the GUI
     from here on but still working for anyone who already has a saved
     script or config referencing it.

Extends tests/test_prerelease_gui_tweaks.py (already in CI) with checks
for all of the above -- the dimension-switch add/remove/fallback behavior,
and _restore_forward_viz_items()'s filtering for 1D/2D/3D/steady-state
configs with and without "problem_dim" set.

Usage:
    python3 apply_1d_surface_animation.py [path-to-pinnstudio-repo]

If no path is given, the current directory is used. Safe to re-run --
if the fix is already present, this exits without changing anything.
"""
import os
import subprocess
import sys

# A short, uniquely-identifying string this patch adds to main_window.py
# -- verified absent before this patch and appearing exactly once after.
MARKER = '_sa_idx = self.plot_type_combo.findText("Surface Animation (GIF)")'

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PATCH_PATH = os.path.join(_SCRIPT_DIR, "surface_animation_1d.patch")


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
        print("✅ Already applied -- main_window.py already excludes 1D's Surface Animation (GIF). Nothing to do.")
        return 0

    if not os.path.isfile(_PATCH_PATH):
        print(f"❌ Could not find surface_animation_1d.patch -- it must be in the same folder as this script "
              f"({_SCRIPT_DIR!r}).")
        return 1

    print(f"Applying surface_animation_1d.patch to {repo!r} ...")
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
              "Commit or stash any local changes and re-run, or apply surface_animation_1d.patch by hand and "
              "resolve the conflicts.")
        return 1

    print("✅ Patch applied successfully.")
    print("   Changed: pinnstudio/ui/main_window.py, tests/test_prerelease_gui_tweaks.py")
    print()
    print("Next steps:")
    print("  1. Review the changes:  git diff")
    print("  2. Run the test (needs PyQt6 installed, same as the app itself):")
    print("     QT_QPA_PLATFORM=offscreen python3 tests/test_prerelease_gui_tweaks.py")
    print("  3. git add -A && git commit -m \"...\" && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
