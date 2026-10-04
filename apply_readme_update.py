#!/usr/bin/env python3
"""
Applies the README/docs-wording update described in the chat, on top of a
repo that has ALREADY merged `custom-geometry` into `main` (see the merge
instructions in the chat -- this script does not do the merge itself, only
the documentation changes on top of it).

Changes this patch makes:
  - Overview paragraph: dimension notation no longer implies every problem
    has a time axis ("1D (x), 2D (x, y), 3D (x, y, z) -- time-dependent or
    steady-state"); also now mentions Custom (CSG) geometry.
  - Features > Geometry & domains: new paragraph documenting the Custom
    geometry option (2D/3D, stack primitives + Union/Subtract/Intersect),
    with a domain-preview image (Triangle minus Disk) added at
    assets/Images/View_Domain_Pictures/Custom_Geometry_Triangle_minus_Disk.png.
  - Parameter Sweep's <details> block is no longer forced open by default
    (was the only feature entry using <details open>; now collapsed like
    every other one, consistent with the rest of Features).
  - The two "*The remaining templates are 2D/3D (x, y[, z], t) problems.*"
    dividers were factually wrong (some of those templates -- the Poisson
    ones -- are steady-state, no t axis at all); reworded to say so.
  - Replaced the stale "No-code GUI" wording in three places with the
    current project description (no-code scientific computing environment
    for forward/inverse PINNs): the BibTeX citation title in README.md,
    setup.py's PyPI `description`, and CITATION.cff's `abstract`.

Usage:
    python3 apply_readme_update.py [path-to-pinnstudio-repo]

If no path is given, the current directory is used. Safe to re-run -- if
the update is already present, this exits without changing anything.

IMPORTANT: run this AFTER merging custom-geometry into main (or onto any
checkout that already has both). custom-geometry itself never touches
README.md/setup.py/CITATION.cff, so this patch applies the same either way,
but it's meant to land together with that merge, not instead of it.
"""
import os
import subprocess
import sys

# A short, uniquely-identifying string this patch adds to README.md --
# verified absent before this patch and appearing exactly once after.
MARKER = "Need a shape that isn't one of those on its own?"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PATCH_PATH = os.path.join(_SCRIPT_DIR, "readme_update.patch")


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
        print("✅ Already applied -- README.md already has the Custom geometry section. Nothing to do.")
        return 0

    if not os.path.isfile(_PATCH_PATH):
        print(f"❌ Could not find readme_update.patch -- it must be in the same folder as this script "
              f"({_SCRIPT_DIR!r}).")
        return 1

    print(f"Applying readme_update.patch to {repo!r} ...")
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
        print("This usually means custom-geometry hasn't been merged into this checkout yet (merge it first -- "
              "see the chat for the exact commands), or your working tree has local changes that overlap with "
              "this patch's lines. Commit or stash any local changes and re-run, or apply readme_update.patch by "
              "hand and resolve the conflicts.")
        return 1

    print("✅ Patch applied successfully.")
    print("   Changed: README.md, setup.py, CITATION.cff")
    print("   Added:   assets/Images/View_Domain_Pictures/Custom_Geometry_Triangle_minus_Disk.png")
    print()
    print("Next steps:")
    print("  1. Review the changes:  git diff")
    print("  2. git add -A && git commit -m \"...\" && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
