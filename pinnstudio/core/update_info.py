"""How to update this PINNStudio, for the "newer version available" notice.

A copy installed with the README's one-command installer lives in a uv tool
environment, and running the installer again updates it. Telling that user
to run `pip install --upgrade pinnstudio` would update (or create) a second
copy in whatever Python pip belongs to, and leave theirs old. A copy
installed with pip is updated with the pip of the Python it runs in.
"""
import os
import sys

INSTALL_SH = "curl -LsSf https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/main/get-pinnstudio.sh | sh"
INSTALL_PS1 = ('powershell -ExecutionPolicy ByPass -c "irm '
               'https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/main/get-pinnstudio.ps1 | iex"')


def is_uv_tool_install(prefix, env=None):
    """True when `prefix` (sys.prefix) is the uv tool environment of pinnstudio."""
    env = os.environ if env is None else env
    p = os.path.normcase(os.path.normpath(str(prefix))).replace("\\", "/")
    if p.endswith("/uv/tools/pinnstudio"):
        return True
    tool_dir = env.get("UV_TOOL_DIR")
    if tool_dir:
        t = os.path.normcase(os.path.normpath(tool_dir)).replace("\\", "/")
        if p == t + "/pinnstudio":
            return True
    return False


def update_command(prefix=None, executable=None, platform=None, env=None):
    """The command that updates this copy of PINNStudio."""
    prefix = sys.prefix if prefix is None else prefix
    executable = sys.executable if executable is None else executable
    platform = sys.platform if platform is None else platform
    if is_uv_tool_install(prefix, env):
        return INSTALL_PS1 if platform.startswith("win") else INSTALL_SH
    return f'"{executable}" -m pip install --upgrade pinnstudio'
