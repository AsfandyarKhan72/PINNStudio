#!/usr/bin/env python3
"""
Applies the Restore Model domain-masking fix: restoring a model for a
non-rectangular problem (Disk/Ellipse/Triangle/Polygon/Sphere, or a
Custom CSG combination like a triangular cavity with a circular cutout)
previously always predicted and plotted over the model's full
rectangular/cuboid BOUNDING BOX instead of the real domain. This patch:

  - codegen.py's `_model_config` dict now also saves geometry_type and
    every geom_* shape parameter (center/radius/semi-axes/angle/
    vertices/geom_custom_shapes_json) into model_config.json -- only the
    bounding box was ever saved there before, for every geometry type.
  - main_window.py's _build_restore_script() now reconstructs the REAL
    geometry from those fields (reusing codegen.py's own
    _clean_geom_line()/_build_custom_geom_code()/_parse_vertex_list(),
    the same dispatcher "Export as DeepXDE Script" already uses) instead
    of always building a plain Rectangle/Cuboid/Interval, and masks the
    Surface plot (2D contourf + 3D 6-face plot) down to the real domain
    via geom.inside(...) -- same convention the app's own Results-panel
    Surface plots already use for non-rectangular geometries.
  - Adds tests/test_restore_geometry_masking.py and registers it in the
    CI smoke-test workflow.

See geometry_masking_fix.patch (same folder as this script) for the full
diff.

Usage:
    python3 apply_geometry_masking_fix.py [path-to-pinnstudio-repo]

If no path is given, the current directory is used. Safe to re-run --
if the fix is already present, this exits without changing anything.

IMPORTANT -- an existing checkpoint's saved model_config.json still won't
have the new geometry fields (only runs saved AFTER this patch will).
Restoring an old checkpoint for a non-rectangular geometry will fall back
to a plain Rectangle/Cuboid, same as before this patch -- see the two
options in the chat for fixing that for an *existing* save (hand-edit
model_config.json, or just retrain/re-save once).
"""
import os
import subprocess
import sys

# A short, uniquely-identifying string this patch adds to main_window.py
# -- NOT present anywhere in the file before this patch (verified via
# `git show HEAD:pinnstudio/ui/main_window.py | grep -c` against the
# pre-patch tip this was built from) and appearing exactly once after.
MARKER = "Reconstruct the ACTUAL problem geometry (Disk/Ellipse/Triangle/"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PATCH_PATH = os.path.join(_SCRIPT_DIR, "geometry_masking_fix.patch")


def looks_like_inner_package_folder(path):
    """True if `path` is the INNER pinnstudio/ package folder itself
    (containing ui/, core/, ...) rather than the repo root that CONTAINS
    that folder. Running this script one level too deep would silently
    try to patch a nonexistent pinnstudio/pinnstudio/ui/main_window.py
    and fail confusingly -- this catches that and tells the user to go
    up one directory instead."""
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
    codegen_path = os.path.join(repo, "pinnstudio", "core", "codegen.py")
    if not os.path.isfile(main_window_path) or not os.path.isfile(codegen_path):
        print(f"❌ Could not find pinnstudio/ui/main_window.py and pinnstudio/core/codegen.py under {repo!r}.")
        print(f"   Run this from your pinnstudio repo root, or pass its path as an argument:")
        print(f"     python3 {os.path.basename(__file__)} /path/to/pinnstudio")
        return 1

    with open(main_window_path, "r") as f:
        main_window_content = f.read()
    if MARKER in main_window_content:
        print("✅ Already applied -- main_window.py already has the geometry-masking fix. Nothing to do.")
        return 0

    if not os.path.isfile(_PATCH_PATH):
        print(f"❌ Could not find geometry_masking_fix.patch -- it must be in the same folder as this script "
              f"({_SCRIPT_DIR!r}).")
        return 1

    print(f"Applying geometry_masking_fix.patch to {repo!r} ...")
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
        print("This usually means your working tree has local changes that overlap with this patch's "
              "lines, or it's being applied to a different commit than it was built against. Commit or "
              "stash any local changes to pinnstudio/ui/main_window.py / pinnstudio/core/codegen.py and "
              "re-run, or apply geometry_masking_fix.patch by hand and resolve the conflicts.")
        return 1

    print("✅ Patch applied successfully.")
    print("   Changed: pinnstudio/core/codegen.py, pinnstudio/ui/main_window.py,")
    print("            .github/workflows/smoke-test.yml")
    print("   Added:   tests/test_restore_geometry_masking.py")
    print()
    print("Next steps:")
    print("  1. Review the changes:  git diff")
    print("  2. Run the new test (needs deepxde + torch installed, same as the app itself):")
    print("     QT_QPA_PLATFORM=offscreen python3 tests/test_restore_geometry_masking.py")
    print("  3. git add -A && git commit -m \"...\" && git push")
    print()
    print("Reminder: this only fixes restores of models saved AFTER this patch is applied and used for a "
          "new training run -- an already-saved model_config.json for a non-rectangular geometry won't "
          "retroactively gain the geometry fields it's missing. See the chat for how to fix an existing save.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
