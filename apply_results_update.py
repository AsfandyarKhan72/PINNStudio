#!/usr/bin/env python3
"""
Applies this round's README/results update:

  - Fisher-KPP ("Getting Started with Your Own PDE" section): replaces the
    single old assets/Images/Fisher-KPP.png with a 3-up table (Line
    comparison, Surface, Animation) at assets/Images/1D-Fisher-KPP/*. The
    old orphaned Fisher-KPP.png is removed from the repo.
  - 3D Poisson (Sphere) template: adds an *Inverse mode* paragraph +
    convergence image (recovering the source term, fixed at 1 in Forward
    mode) -- this template previously had no Inverse-mode example, unlike
    every other template.
  - GUI dimension selector (pinnstudio/ui/main_window.py): the "1D (x, t)"
    / "2D (x, y, t)" / "3D (x, y, z, t)" radio-button labels always showed
    a time axis even for steady-state problems -- same issue the README
    fix addressed last round. Changed to "1D (x)" / "2D (x, y)" /
    "3D (x, y, z)", which is accurate either way. Pure label strings, no
    behavior change.

IMPORTANT -- this patch is TEXT-ONLY for the new images: it does NOT ship
the new PNG/GIF bytes (Fisher-KPP_line_plot.png, Fisher-KPP_surface_plot.png,
Fisher-KPP_animation.gif, sourceterm_inverse_plot.png, and the regenerated
surface_comparison.png). Copy those into the repo BEFORE applying this
patch -- see the exact `cp` commands in the chat. Without them, the patch
still applies cleanly (it only touches README.md, main_window.py, and
deletes the old Fisher-KPP.png), but the README will reference image paths
that don't exist yet until you've copied the files over.

Usage:
    python3 apply_results_update.py [path-to-pinnstudio-repo]

If no path is given, the current directory is used. Safe to re-run -- if
the update is already present, this exits without changing anything.
"""
import os
import subprocess
import sys

# A short, uniquely-identifying string this patch adds to README.md --
# verified absent before this patch and appearing exactly once after.
MARKER = "recovering the source term (fixed at"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PATCH_PATH = os.path.join(_SCRIPT_DIR, "results_update.patch")


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

    readme_path = os.path.join(repo, "README.md")
    if not os.path.isfile(readme_path):
        print(f"❌ Could not find README.md under {repo!r}.")
        print(f"   Run this from your pinnstudio repo root, or pass its path as an argument:")
        print(f"     python3 {os.path.basename(__file__)} /path/to/pinnstudio")
        return 1

    with open(readme_path, "r") as f:
        readme_content = f.read()
    if MARKER in readme_content:
        print("✅ Already applied -- README.md already has the 3D Poisson (Sphere) Inverse-mode section. Nothing to do.")
        return 0

    if not os.path.isfile(_PATCH_PATH):
        print(f"❌ Could not find results_update.patch -- it must be in the same folder as this script "
              f"({_SCRIPT_DIR!r}).")
        return 1

    print(f"Applying results_update.patch to {repo!r} ...")
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
              "Commit or stash any local changes and re-run, or apply results_update.patch by hand and resolve "
              "the conflicts.")
        return 1

    print("✅ Patch applied successfully.")
    print("   Changed: README.md, pinnstudio/ui/main_window.py")
    print("   Removed: assets/Images/Fisher-KPP.png (replaced by the 1D-Fisher-KPP/ folder)")
    print()
    print("Reminder: if you haven't already, copy the new image files into the repo -- see the `cp` commands")
    print("in the chat -- BEFORE committing, so README.md doesn't point at missing files:")
    print("  assets/Images/1D-Fisher-KPP/Fisher-KPP_line_plot.png")
    print("  assets/Images/1D-Fisher-KPP/Fisher-KPP_surface_plot.png")
    print("  assets/Images/1D-Fisher-KPP/Fisher-KPP_animation.gif")
    print("  assets/Images/3D-Poisson-Sphere/sourceterm_inverse_plot.png")
    print("  assets/Images/3D-Poisson-Sphere/surface_comparison.png  (overwrite with the regenerated one)")
    print()
    print("Next steps:")
    print("  1. Review the changes:  git diff")
    print("  2. git add -A && git commit -m \"...\" && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
