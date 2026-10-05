#!/usr/bin/env python3
"""
Applies this round's fix for discussion item #5 from the chat: the
Time-Adaptive training loss plot only ever showed the LAST time
sub-domain's loss curve, titled "Loss — Last Time Sub-domain" -- every
earlier sub-domain's (and, within a sub-domain, every earlier optimizer
phase's) loss history was silently discarded.

Root cause: codegen.py's Time-Adaptive loop re-binds `lh_i`/`ts_i` on
EVERY `model_i.train(...)` call -- there can be several per sub-domain
(one per optimizer-scheduler phase), plus a separate one for the legacy
Adam-then-L-BFGS path -- and the loss plot read `lh_i.loss_train`/
`lh_i.loss_test` directly after the whole loop finished, which by then
only ever held whichever single train() call happened to run LAST.

What this patch changes (pinnstudio/core/codegen.py only):

  - A new `_ta_accumulate_loss(lh)` helper (defined once, before the
    per-sub-domain loop) appends each phase's loss_train/loss_test/steps
    into running `_ta_all_*` lists, with a cumulative iteration offset
    added to `steps` so the x-axis keeps increasing across phases and
    sub-domains instead of restarting at 0 each time (DeepXDE's
    LossHistory numbers every train() call's own steps from scratch).
    Called after each of the 5 model_i.train(...) call sites in the
    Time-Adaptive loop.
  - The loss plot now reads `_ta_all_train_loss`/`_ta_all_test_loss`/
    `_ta_all_steps` instead of the bare last `lh_i`, with light vertical
    markers at each sub-domain boundary and a "Loss — All N Time
    Sub-domain(s)" title (was "Loss — Last Time Sub-domain").

NOT changed this round (flagged in the chat as a related, lower-priority
gap): the separate "Export as DeepXDE Script" clean-script generator
(generate_clean_script, used only by that standalone export feature, not
the main Results panel) has its own, simpler Time-Adaptive loop with the
exact same underlying limitation (its own loss_comment explicitly says
"last Time-Adaptive step's loss curve") -- left as-is for now since it's
a separate, less central code path and fixing it would mean touching
_clean_phase_train_lines(), which is shared with the Standard
(non-adaptive) clean-script path too.

Adds tests/test_ta_loss_history.py (registered in CI) -- verified to
fail against the pre-fix code (NameError on the new accumulator
variables once the diagnostic side-channel this test injects tries to
read them, plus the old "last sub-domain" markers still present) and
pass with the fix, via a real tiny 2-sub-domain Time-Adaptive training
run.

Usage:
    python3 apply_ta_loss_history.py [path-to-pinnstudio-repo]

If no path is given, the current directory is used. Safe to re-run --
if the fix is already present, this exits without changing anything.
"""
import os
import subprocess
import sys

# A short, uniquely-identifying string this patch adds to codegen.py --
# verified absent before this patch and appearing exactly once after.
MARKER = "def _ta_accumulate_loss(_lh):"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PATCH_PATH = os.path.join(_SCRIPT_DIR, "ta_loss_history.patch")


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

    codegen_path = os.path.join(repo, "pinnstudio", "core", "codegen.py")
    if not os.path.isfile(codegen_path):
        print(f"❌ Could not find pinnstudio/core/codegen.py under {repo!r}.")
        print(f"   Run this from your pinnstudio repo root, or pass its path as an argument:")
        print(f"     python3 {os.path.basename(__file__)} /path/to/pinnstudio")
        return 1

    with open(codegen_path, "r") as f:
        codegen_content = f.read()
    if MARKER in codegen_content:
        print("✅ Already applied -- codegen.py already accumulates the Time-Adaptive loss history. Nothing to do.")
        return 0

    if not os.path.isfile(_PATCH_PATH):
        print(f"❌ Could not find ta_loss_history.patch -- it must be in the same folder as this script "
              f"({_SCRIPT_DIR!r}).")
        return 1

    print(f"Applying ta_loss_history.patch to {repo!r} ...")
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
              "Commit or stash any local changes and re-run, or apply ta_loss_history.patch by hand and "
              "resolve the conflicts.")
        return 1

    print("✅ Patch applied successfully.")
    print("   Changed: pinnstudio/core/codegen.py, .github/workflows/smoke-test.yml")
    print("   Added:   tests/test_ta_loss_history.py")
    print()
    print("Next steps:")
    print("  1. Review the changes:  git diff")
    print("  2. Run the test (needs PyQt6 + torch + deepxde installed, same as the app itself --")
    print("     this one actually trains a tiny real Time-Adaptive model, so it takes a few minutes):")
    print("     QT_QPA_PLATFORM=offscreen python3 tests/test_ta_loss_history.py")
    print("  3. git add -A && git commit -m \"...\" && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
