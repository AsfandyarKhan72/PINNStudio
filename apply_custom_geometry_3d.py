#!/usr/bin/env python3
"""
Custom Geometry -- 3D extension (Cuboid/Sphere primitives + CSG in 3D).

This builds ON TOP of the already-pushed Phase 1 work (2D Custom
geometry: Rectangle/Disk/Ellipse/Triangle/Polygon + CSG), on the SAME
`custom-geometry` branch -- no new branch this round, since Phase 1 is
already on `custom-geometry` and pushed to origin.

STEP 1 -- MAKE SURE YOU'RE ON THE custom-geometry BRANCH, UP TO DATE

    git checkout custom-geometry
    git pull origin custom-geometry

STEP 2 -- DOUBLE-CHECK YOU'RE AT THE REPO ROOT (not inside the inner
`pinnstudio` package folder -- see note below), THEN RUN THIS SCRIPT

    ls setup.py pinnstudio/core/codegen.py

If both of those print back without "No such file or directory", you're
in the right place:

    python3 apply_custom_geometry_3d.py

(A past round's troubleshooting found that this checkout has the repo
root and the `pinnstudio` Python package folder nested -- both can look
like "the pinnstudio folder" at a glance. The repo root contains
setup.py, README.md, tests/, .github/, AND a `pinnstudio/` subfolder;
that subfolder itself contains core/, ui/, main.py -- no further nesting.
Run the `ls` check above first if you're ever unsure; this script also
checks for you and tells you plainly if you're one level too deep.)

WHAT THIS PATCH ADDS

Extends the Phase 1 Custom-geometry shape-builder to 3D: the same
shape-builder idea (Add Shape, each shape with its own type + params, a
Combine op for every shape after the first, remove/reorder), now also
usable in 3D with Cuboid and Sphere as the available primitives (matching
the fixed Cuboid/Sphere geometry types already offered outside Custom).

1. pinnstudio/core/config.py: validate()'s recognized-shape-type check
   now also accepts "Cuboid" and "Sphere".
2. pinnstudio/ui/main_window.py: "Custom" is now also offered in 3D.
   Each shape row's type dropdown offers the dimension-appropriate
   primitives only (2D: Rectangle/Disk/Ellipse/Triangle/Polygon; 3D:
   Cuboid/Sphere) -- new Cuboid (x/y/z range) and Sphere (center +
   radius) parameter panels per row, mirroring the existing top-level
   Cuboid/Sphere panels. Switching problem dimension now clears the
   Custom shape list (a 2D shape isn't valid in 3D and vice versa),
   same reasoning already used for the Boundary Conditions panel's own
   dimension-staleness handling -- Custom auto-reseeds one dimension-
   appropriate starting shape if it's still selected afterward. The
   domain preview now handles 3D Custom geometries too: builds the real
   CSG-chained 3D geometry, samples it with DeepXDE the same way every
   other 3D shape's preview already does, and draws each leaf's own 3D
   outline (box edges for Cuboid, a wireframe sphere for Sphere)
   overlaid as a visual reference. The existing single-shape Cuboid/
   Sphere 3D preview code is untouched logic-wise -- it was refactored
   to share its rendering template with the new Custom-3D path, and
   this patch includes a direct check that both still produce correct,
   unchanged output after that refactor.
3. pinnstudio/core/codegen.py: both script-generation paths now build
   the matching CSG chain for a 3D Custom geometry too. The pre-existing
   dtype workaround (_DTypeSafeGeom) is applied correctly for the 3D
   leaf types: Cuboid is never dtype-buggy (like Rectangle), Sphere
   alone is buggy (like Disk/Ellipse/Triangle/Polygon), and -- verified
   empirically the same way the 2D leaf types were -- any 2+-shape CSG
   combination of Cuboid/Sphere, in any order/operation, already
   normalizes to the correct dtype on its own and never needs wrapping.
   The empty-shape-list fallback is now dimension-aware too (falls back
   to a unit Cuboid for a 3D problem, a unit Rectangle for 2D, instead
   of always defaulting to 2D).
4. tests/test_custom_geometry.py (extended, not replaced): adds 3D
   coverage for everything Phase 1's test file already covers for 2D --
   config validation, the dimension-aware shape-builder panel (type-list
   switching, dimension-change clearing, Cuboid/Sphere row widgets, JSON
   round-trip), the refactored 3D preview script builder (both the
   unchanged Cuboid/Sphere path and the new Custom-3D CSG path), and
   both codegen paths' CSG-chain + dtype-wrap logic for 3D. Already
   wired into .github/workflows/smoke-test.yml from Phase 1 (same file,
   same CI entry) -- nothing new to add there.

Not in this round (tracked as later phases, unchanged from the Phase 1
plan): a freehand sketch tool and auto-derived per-shape BC boundary
regions.

VERIFICATION DONE

 - Full headless regression suite (test_config_options.py,
   test_export_parity.py, test_sweep_registry.py, test_sweep_tab.py,
   test_sweep_param_expansion.py, test_templates.py 24/24,
   test_v56_no_template_solve.py, test_training_callbacks_visibility.py,
   and the extended test_custom_geometry.py, now covering 2D and 3D)
   passes on this branch.
 - A real end-to-end DeepXDE training run (not part of the headless
   suite, which runs in CI without torch/deepxde installed) on a
   Cuboid-minus-Sphere Custom 3D geometry was run manually in an
   environment with DeepXDE installed: it compiled, trained, and
   converged (final loss ~8.8e-3) with no errors.
 - The 3D Custom domain-preview script was generated and actually
   executed (not just syntax-checked) against real DeepXDE/PyTorch for
   the same Cuboid-minus-Sphere geometry, and the resulting plot was
   visually confirmed correct: the cuboid and sphere outlines render,
   domain points correctly avoid the sphere's interior, and boundary
   points correctly cover both the cuboid's outer faces and the
   sphere's surface.
 - This patch isolated and verified byte-identical in a fresh,
   independent clone of `custom-geometry` (at the Phase 1 tip,
   e344721), before delivery.

Applies cleanly on top of `custom-geometry` at commit e344721 (the
pushed Phase 1 tip).

Run from the repo root, after Step 1 above:
    python3 apply_custom_geometry_3d.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "custom_geometry_3d.patch"
MARKER = "dde.geometry.Cuboid([{x_min}, {y_min}, {z_min}]"

CHANGED_FILES = [
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


def looks_like_inner_package_folder(cwd):
    """Catches the exact trap from last round: standing inside the
    `pinnstudio` package folder itself (which directly contains core/,
    ui/, main.py) instead of the repo root one level up (which contains
    setup.py and a `pinnstudio/` subfolder)."""
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
              "cd into your PINNStudio repo root (the folder containing the "
              "'pinnstudio' folder -- check with: ls setup.py pinnstudio/core/codegen.py) "
              "and run this script from there.")
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
              "This round builds on top of the Phase 1 work, which is already "
              "on that branch and pushed to origin. Run Step 1 first:\n"
              "  git checkout custom-geometry\n"
              "  git pull origin custom-geometry\n"
              "...then run this script again.")
        sys.exit(1)

    marker_in_codegen = root / "pinnstudio" / "core" / "codegen.py"
    if marker_in_codegen.is_file() and MARKER in marker_in_codegen.read_text():
        print("Already applied -- this 3D extension is already present in "
              "this checkout. Nothing to do.")
        sys.exit(0)

    check = git_apply_check(root, patch_path)
    if check.returncode != 0:
        if git_apply_check(root, patch_path, reverse=True).returncode == 0:
            print("Already applied -- this 3D extension is already present "
                  "in this checkout. Nothing to do.")
            sys.exit(0)
        print("This patch doesn't apply cleanly here. Most likely this "
              "branch has diverged from the `custom-geometry` commit it was "
              "built against (e344721, the Phase 1 tip), or Step 1 wasn't "
              "done yet (git pull origin custom-geometry).\n\n"
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
    print('  git commit -m "Add 3D Custom geometry: Cuboid/Sphere primitives + CSG in 3D"')
    print("  git push origin custom-geometry")
    print()
    print("Quick way to run the test suite (needs PyQt6/numpy/matplotlib):")
    print("  QT_QPA_PLATFORM=offscreen python3 tests/test_custom_geometry.py")
    print()
    print("Still on custom-geometry, not main -- same plan as before: keep "
          "building on this branch until everything's fully ready, then "
          "merge to main in one combined release.")


if __name__ == "__main__":
    main()
