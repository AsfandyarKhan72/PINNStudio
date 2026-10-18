"""PINNStudio templates defined by their settings.

The older templates are set up widget by widget in the main window; the
ones here are plain PINNConfig settings, loaded the same way as DeepXDE's
examples (deepxde_examples.build_config, then MainWindow._apply_config), so
a template is one dictionary. They are listed as PINNStudio templates in the
Examples lists (example_catalog), with the paper they follow.

Templates you keep for yourself go in pinnstudio/core/local_examples.py
(ignored by git), with their reference data under reference_data/local/:
the file defines EXAMPLES, CATALOG ({name: (shown name, paper)}) and PAPERS
({paper: link}) in the same form, and is added to the lists when present.
"""
from pinnstudio.core import deepxde_examples as _dx

PAPERS = {}
EXAMPLES = []
# shown name and paper of each, for the example catalogue
CATALOG = {}

try:                                    # local templates, if there are any
    from pinnstudio.core import local_examples as _local
except ImportError:
    _local = None
if _local is not None:
    PAPERS.update(getattr(_local, "PAPERS", {}))
    EXAMPLES.extend(getattr(_local, "EXAMPLES", []))
    CATALOG.update(getattr(_local, "CATALOG", {}))

BY_NAME = {e.name: e for e in EXAMPLES}


def build_config(example, ref_root, keep=None, inverse=False):
    return _dx.build_config(example, ref_root, keep=keep, inverse=inverse)
