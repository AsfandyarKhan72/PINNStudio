#!/usr/bin/env python3
"""
Item #4 from the pre-release request: "general figure/plot dimension-size
standardization -- add a 'Plot Settings' option where the user can choose
plot dimensions (square, current default, and something more useful)."

What this patch adds:

  - PINNConfig (pinnstudio/core/config.py) gets 3 new fields:
    plot_figsize_mode ("Default" | "Square" | "Wide" | "Custom", default
    "Default"), plot_figsize_w (default 7.0), plot_figsize_h (default 5.0).

  - The Setup tab's existing "Plot Settings" dialog (_on_plot_settings)
    and the Restore tab's own viz-settings dialog
    (_on_restore_viz_settings) both get a new "Figure size:" control --
    a Default/Square/Wide/Custom combo, plus Width/Height spinboxes that
    only show up for Custom.

  - Every generated script (the main Setup/Run script from codegen.py's
    generate_script(), and all 3 Restore script builders --
    _build_restore_script, _build_restore_script_ta,
    _build_restore_param_script -- in main_window.py) gets a
    `_plot_figsize(default_w, default_h)` helper near its top, and every
    SINGLE-PANEL results plot's figsize=(...) call (Loss, Line, Surface,
    both Animation GIF types, Parameter Convergence, the Parametric-sweep
    summary bar chart, and all of their Time-Adaptive equivalents) is
    changed to figsize=_plot_figsize(w, h) in its place.

    "Default" mode makes every one of those calls behave *exactly* as it
    did before this patch -- this is purely additive, nothing about plot
    sizing changes unless you actually open Plot Settings and pick
    something else. "Square" forces a 1:1 aspect ratio (using whichever
    side was already larger for that particular plot). "Wide" keeps that
    plot's own default height but widens it to a fixed 1.8x ratio (handy
    for slides). "Custom" uses your own Width/Height for every one of
    those plots, regardless of what each one's own default used to be.

  Deliberately NOT touched, on purpose: the multi-column Error Analysis
  comparison grids (both the standalone Error Analysis tab's and the
  Restore tab's "Error Analysis on restore" add-on) -- those size
  themselves from however many reference files/columns are being
  compared, and forcing a fixed aspect ratio on a grid like that would
  make it look wrong, not better. Also untouched: the Setup tab's
  domain-sampling preview plots (sized to the problem's own geometry, not
  a "results" plot), and generate_clean_script (a separate, more minor
  exporter) -- consistent with the same scoping decision made for the
  earlier Time-Adaptive loss-history fix.

Adds tests/test_plot_figsize_settings.py (registered in CI) -- checks the
new config defaults, that the right call sites get wrapped and the EA/
domain-preview ones don't, that the real _plot_figsize code (extracted
from an actual generated script and executed, not reimplemented in the
test) returns the right numbers for every mode in every one of the 4
script builders, that the new dialog controls exist, and -- the strongest
check -- one real, tiny, actual training run per mode whose saved
loss_plot.png's real pixel dimensions are checked against figsize*dpi.

Usage:
    python3 apply_plot_figsize_settings.py [path-to-pinnstudio-repo]

If no path is given, the current directory is used. Safe to re-run -- if
the fix is already present, this exits without changing anything.
"""
import os
import subprocess
import sys

# A short, uniquely-identifying string this patch adds to config.py --
# verified absent before this patch and appearing exactly once after.
MARKER = "plot_figsize_mode: str = \"Default\""

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PATCH_PATH = os.path.join(_SCRIPT_DIR, "plot_figsize_settings.patch")


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

    config_path = os.path.join(repo, "pinnstudio", "core", "config.py")
    if not os.path.isfile(config_path):
        print(f"❌ Could not find pinnstudio/core/config.py under {repo!r}.")
        print(f"   Run this from your pinnstudio repo root, or pass its path as an argument:")
        print(f"     python3 {os.path.basename(__file__)} /path/to/pinnstudio")
        return 1

    with open(config_path, "r") as f:
        config_content = f.read()
    if MARKER in config_content:
        print("✅ Already applied -- pinnstudio/core/config.py already has plot_figsize_mode. Nothing to do.")
        return 0

    if not os.path.isfile(_PATCH_PATH):
        print(f"❌ Could not find plot_figsize_settings.patch -- it must be in the same folder as this script "
              f"({_SCRIPT_DIR!r}).")
        return 1

    print(f"Applying plot_figsize_settings.patch to {repo!r} ...")
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
              "Commit or stash any local changes and re-run, or apply plot_figsize_settings.patch by hand and "
              "resolve the conflicts.")
        return 1

    print("✅ Patch applied successfully.")
    print("   Changed: pinnstudio/core/config.py, pinnstudio/core/codegen.py, pinnstudio/ui/main_window.py, "
          ".github/workflows/smoke-test.yml")
    print("   Added:   tests/test_plot_figsize_settings.py")
    print()
    print("Next steps:")
    print("  1. Review the changes:  git diff")
    print("  2. Run the test (the config/script/dialog checks run instantly; the real-run pixel checks need "
          "PyQt6 + torch + deepxde installed, same as the app itself, and take under a minute):")
    print("     QT_QPA_PLATFORM=offscreen python3 tests/test_plot_figsize_settings.py")
    print("  3. git add -A && git commit -m \"...\" && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
