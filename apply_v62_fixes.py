#!/usr/bin/env python3
"""
Run this on top of your current `main` (commit `00d0646`, the v1.3.0
release merge). This patch touches only pinnstudio/core/codegen.py.

WHAT THIS FIXES

The 3D Poisson (Sphere) "Surface comparison" plot (and any other curved
3D geometry, should one get added later) was showing only a handful of
scattered points instead of a full comparison, even though the Line
comparison, error metrics, and accuracy were all fine.

ROOT CAUSE

The 3D "surface comparison" scatter plot filters your reference data down
to only the points that land exactly on the geometry's boundary
(geom.on_boundary(...)) before plotting anything. That works fine for a
box-shaped geometry (3D Heat's Cuboid) -- a flat face is just a 2D slice
of an axis-aligned grid, so a regular Cartesian reference grid naturally
has thousands of points sitting exactly on each face. It badly breaks for
a CURVED boundary like the Sphere: an axis-aligned grid essentially never
lands exactly on a curved surface, so on_boundary() comes back with only
a handful of coincidental exact matches (for a unit sphere, literally
just the 6 axis points: (+-1,0,0), (0,+-1,0), (0,0,+-1)) even though the
reference file has thousands of valid interior points. There was already
a "fall back to all points if fewer than 4 matched" safety net, but 6
matches is >= 4, so it never triggered -- the plot silently rendered
those 6 stray points and nothing else.

This exists in THREE separate places in codegen.py (each script
generator keeps its own copy of this plotting logic): generate_script()'s
Standard-training path, generate_script()'s Time-Adaptive-training path,
and generate_clean_script() (the "Export as DeepXDE Script" feature).

THE FIX

Changed the fallback condition in all three places from "fewer than 4
boundary matches" to "fewer than 4 matches OR less than 5% of the
reference data" -- catches the Sphere case (6 out of ~33,000 points is
nowhere near 5%) while leaving box geometries alone (a Cuboid's face
points are normally a large share of its grid, well above 5%). When the
fallback triggers, the full interior+boundary point cloud is used
instead -- capped at a random 3,000-point subsample (fixed seed, so it's
reproducible) to keep the plot fast to render and legible rather than
visually saturated.

VERIFICATION DONE

 - Ran a real, real end-to-end Solve on 3D Poisson (Sphere) (reduced to
   300+300 iterations for a fast check, not for plot quality) before and
   after this fix. Before: the "Surface comparison" scatter showed ~6
   isolated points. After: a full, dense spherical point cloud (3,000
   points) that clearly shows the solution's radial structure (peak near
   the center, tapering to zero at the boundary) and lines up visually
   between the PINN and Ground Truth panels, with errors concentrated in
   a believable pattern.
 - Confirmed the actual generated script (via generate_script(), the same
   path the Solve button uses) contains the fix and produces this result
   -- not just a theoretical code read.
 - Full regression suite (4/4) passes.
 - Patch verified to apply cleanly, byte-identical output confirmed, in a
   fresh independent clone at the exact base commit (`00d0646`, the
   v1.3.0 release merge).

Applies cleanly to a clean checkout of `main` at commit 00d0646 (your
current v1.3.0 release). Since it only touches script-generation logic
(what gets written into the temp file that actually trains/plots), it
doesn't change anything about the app itself -- no version bump needed,
this only affects future Solve/Export runs.

Run from the repo root:
    python3 apply_v62_fixes.py
"""
import subprocess
import sys
from pathlib import Path

PATCH_FILE = "v62_sphere_surface_scatter_delta.patch"
MARKER = "less than 5% of the data"


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

    codegen_path = root / "pinnstudio" / "core" / "codegen.py"
    text = codegen_path.read_text() if codegen_path.is_file() else ""

    if MARKER in text:
        print("Already applied -- this fix is already present in this "
              "checkout. Nothing to do.")
        sys.exit(0)

    check = git_apply_check(root, patch_path)
    if check.returncode != 0:
        if git_apply_check(root, patch_path, reverse=True).returncode == 0:
            print("Already applied -- this fix is already present in this "
                  "checkout. Nothing to do.")
            sys.exit(0)
        print("This patch doesn't apply cleanly here. Most likely the "
              "checkout has diverged from the expected base (local edits, "
              "or you're not on a clean `main` at the v1.3.0 release).\n\n"
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
    print("  - pinnstudio/core/codegen.py")
    print()
    print("Next: review with 'git diff', then commit and push, e.g.")
    print('  git add pinnstudio/core/codegen.py')
    print('  git commit -m "Fix 3D surface comparison scatter showing only a few points for curved geometries (e.g. Sphere)"')
    print('  git push')
    print()
    print("Then re-run the 3D Poisson (Sphere) template's Solve + Error")
    print("Analysis and check the surface_comparison.png -- it should now")
    print("show a full, dense point cloud instead of a handful of points.")


if __name__ == "__main__":
    main()
