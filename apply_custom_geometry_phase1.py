#!/usr/bin/env python3
"""
Custom Geometry -- Phase 1 (2D primitives + boolean ops).

This is the first round on a brand-new branch, so there's no remote
`custom-geometry` branch to fast-forward into yet (unlike the Parameter
Sweep -> main merge, which reused an already-pushed branch). Create the
branch locally off `main` first, THEN run this script on top of it.

STEP 1 -- CREATE THE custom-geometry BRANCH OFF main

    git checkout main
    git pull origin main
    git checkout -b custom-geometry

(If you already have a local `custom-geometry` branch from an earlier
round, just `git checkout custom-geometry` instead -- this script detects
whether Phase 1 is already applied and says so rather than double-
applying.)

STEP 2 -- RUN THIS SCRIPT

    python3 apply_custom_geometry_phase1.py

WHAT THIS PATCH ADDS

A "Custom" geometry type (2D only this round -- 3D is a later phase), for
domains that don't fit any of the fixed shapes (Rectangle/Disk/Ellipse/
Triangle/Polygon): build one out of those same primitives combined with
boolean operations (Union/Subtract/Intersect), e.g. a triangle with a
circular hole cut out of it, matching the reference image this feature
was requested from.

1. pinnstudio/core/config.py: a new `geom_custom_shapes_json` field
   holding the shape list (type + params + combine-op per shape, in
   build order), plus a validate() check that a Custom geometry has at
   least one shape with a recognized type (light-touch, matching how
   Disk/Ellipse/etc.'s own numeric fields already aren't deeply
   validated).
2. pinnstudio/ui/main_window.py: "Custom" is now a Geometry Type choice
   in 2D. Selecting it shows a shape-builder panel -- Add Shape, each
   shape with its own type (Rectangle/Disk/Ellipse/Triangle/Polygon) and
   parameter fields, a Combine op (Union/Subtract/Intersect) for every
   shape after the first, plus remove and reorder (up/down) buttons,
   since CSG ops aren't commutative and getting the order right matters.
   The domain preview (the "Preview Domain" checkbox's plot) now handles
   Custom geometries too: it builds the real CSG-chained geometry,
   samples it with DeepXDE the same way every other shape's preview
   already does, and draws each leaf shape's own outline overlaid as a
   visual reference (the sampled point cloud's correctness comes
   entirely from DeepXDE's own on_boundary() test, which already works
   for any CSG combination unchanged).
3. pinnstudio/core/codegen.py: both script-generation paths now build
   the matching dde.geometry.CSGUnion/CSGDifference/CSGIntersection
   chain for a Custom geometry -- the live Solve/Sweep path
   (_build_geom(), which Time-Adaptive/RAR/IC pre-training/Error
   Analysis all already share, so they get Custom-geometry support for
   free) and the separate "Export as DeepXDE Script" path
   (_clean_geom_line()). A pre-existing dtype workaround
   (_DTypeSafeGeom, wrapping Disk/Ellipse/Triangle/Polygon/Sphere's
   point sampling since this installed DeepXDE version can return the
   wrong float dtype for them) is applied correctly for Custom too: only
   for the degenerate single-shape case with a known-buggy leaf type --
   verified empirically that any 2+-shape CSG combination already
   normalizes to the right dtype on its own, regardless of which leaf
   types feed into it.
4. tests/test_custom_geometry.py (new): config validation, the shape-
   builder panel (add/remove/reorder, JSON round-trip, the empty-list
   validation error surfacing through the GUI), the domain-preview
   script builder, and both codegen paths' CSG-chain + dtype-wrap logic.
   Wired into .github/workflows/smoke-test.yml alongside the existing
   suite.

Not in this round (tracked as later phases, per the plan this was built
against): a freehand sketch tool, auto-derived per-shape BC boundary
regions (the existing Boundary Conditions panel already works against
any geometry via manual location expressions -- e.g. "on_boundary"-style
coverage is just the default/empty location, which already matches every
boundary point DeepXDE finds for the CSG result), and 3D Custom geometry.

VERIFICATION DONE

 - Full headless regression suite (test_config_options.py,
   test_export_parity.py, test_sweep_registry.py, test_sweep_tab.py,
   test_sweep_param_expansion.py, test_templates.py 24/24,
   test_v56_no_template_solve.py, test_training_callbacks_visibility.py,
   and the new test_custom_geometry.py) passes on this branch.
 - A real end-to-end DeepXDE training run (not part of the headless
   suite, which runs in CI without torch/deepxde installed) on a
   triangle-minus-disk Custom geometry -- the same shape as the
   reference image -- was run manually in an environment with DeepXDE
   installed: it compiled, trained, and converged (final loss ~2e-3)
   with no errors.
 - This patch isolated and verified byte-identical in a fresh,
   independent clone of `main`, on a freshly created `custom-geometry`
   branch, before delivery.

Applies cleanly on top of `main` at commit 492936f (where this branch
was created from).

Run from the repo root, after Step 1 above:
    python3 apply_custom_geometry_phase1.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "custom_geometry_phase1.patch"
MARKER = "tests/test_custom_geometry.py"

CHANGED_FILES = [
    ".github/workflows/smoke-test.yml",
    "pinnstudio/core/codegen.py",
    "pinnstudio/core/config.py",
    "pinnstudio/ui/main_window.py",
    "tests/test_custom_geometry.py",
]


def find_repo_root():
    here = Path(__file__).resolve().parent
    for candidate in (Path.cwd(), here):
        if (candidate / "pinnstudio" / "core" / "codegen.py").is_file():
            return candidate
    return None


def git_apply_check(root, patch_path, reverse=False):
    cmd = ["git", "apply", "--check", str(patch_path)]
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

    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root,
        capture_output=True, text=True,
    ).stdout.strip()
    if branch != "custom-geometry":
        print(f"You're on branch '{branch}', not 'custom-geometry'.\n"
              "Run Step 1 first (create the branch off main):\n"
              "  git checkout main\n"
              "  git pull origin main\n"
              "  git checkout -b custom-geometry\n"
              "...then run this script again. (Already have a local "
              "custom-geometry branch from an earlier round? Just "
              "'git checkout custom-geometry' instead.)")
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
              "branch has diverged from the `main` commit it was built "
              "against (492936f), or Step 1 wasn't done yet.\n\n"
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
          "and push the branch, e.g.")
    print("  git add " + " ".join(CHANGED_FILES))
    print('  git commit -m "Add Custom geometry (Phase 1): 2D primitives + boolean ops"')
    print("  git push -u origin custom-geometry")
    print()
    print("Quick way to run the test suite (needs PyQt6/numpy/matplotlib):")
    print("  QT_QPA_PLATFORM=offscreen python3 tests/test_custom_geometry.py")
    print()
    print("This is still on its own branch, not main -- same plan as "
          "Parameter Sweep: keep building on custom-geometry until it's "
          "fully ready (3D, more polish, whatever else gets bundled in), "
          "then merge to main in one combined release.")


if __name__ == "__main__":
    main()
