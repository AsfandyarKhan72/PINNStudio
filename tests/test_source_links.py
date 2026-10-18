"""The README's "Source code" links (tools/update_source_links.py): every
feature names code that still exists. Line numbers are rewritten by the
script before a merge into main, so a moved function is not an error here;
a renamed or deleted one is."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import update_source_links as usl


def _specs():
    with open(usl.README, encoding="utf-8") as f:
        return [m.group(1) for m in map(usl.MARKER.match, f.read().split("\n")) if m]


def test_every_target_exists():
    specs = _specs()
    assert len(specs) >= 20
    for spec in specs:
        for path, name, label in usl.parse_targets(spec):
            assert label, spec
            assert usl.link(path, name, label).startswith(f"[{label}]({usl.REPO_URL}/blob/main/{path}")


def test_every_feature_has_a_source_line():
    with open(usl.README, encoding="utf-8") as f:
        text = f.read()
    feats = text[text.index("## Features"):text.index("## Solving Your Own Problem")]
    summaries = re.findall(r"<summary><strong>(.*?)</strong></summary>\n\n<!-- source: ", feats)
    assert len(summaries) == feats.count("<summary>"), "a feature without a <!-- source: --> line"


def test_rendering():
    a, b = usl.find("pinnstudio/core/codegen.py", "_training_methods_defs")
    assert 0 < a < b
    line = usl.render("pinnstudio/core/codegen.py::_training_methods_defs | methods ; pinnstudio/ui/results.py | results")
    assert line.startswith(usl.PREFIX) and f"#L{a}-L{b}" in line and line.endswith("results.py) (`results.py`)")
