"""Home folder for the screenshot tools.

Every screenshot run needs its own home folder: PINNStudio keeps its settings there and Solve writes its
results to ~/PINNStudio_Results. Two runs that shared one folder could overwrite or clean up each other's
runs, so a run never reuses a folder that already holds files.

* PINNSTUDIO_DOCS_HOME set: that folder is used. It must not exist yet or must be empty, otherwise the run
  stops. For the published screenshots use a neutral path, so the images show no build-machine paths:
  ``PINNSTUDIO_DOCS_HOME=/home/researcher`` (see tools/docs/README.md).
* not set: a new private folder is made with tempfile.mkdtemp (safe for tests and local previews; its
  /tmp path then shows in the status bar, so such images are not for publication).

Nothing is ever deleted by these tools.
"""
import os
import sys
import tempfile


def private_home():
    home = os.environ.get("PINNSTUDIO_DOCS_HOME")
    if home:
        home = os.path.abspath(home)
        if os.path.isdir(home) and os.listdir(home):
            sys.exit(f"PINNSTUDIO_DOCS_HOME={home} already contains files; give a new or empty folder, "
                     "so this run cannot mix with another run's results.")
    else:
        home = os.path.join(tempfile.mkdtemp(prefix="pinnstudio_docs_"), "user")
    os.makedirs(home, exist_ok=True)
    os.environ["HOME"] = home
    return home
