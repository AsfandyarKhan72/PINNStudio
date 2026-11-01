"""The "Source code" links of the documentation's feature map (website/developer/structure.md, kept by
tools/update_source_links.py): every feature names code that still exists. Line numbers are rewritten by the
script before a merge into main, so a moved function is not an error here; a renamed or deleted one is.
The page is on the documentation branch; without it the page tests are skipped."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import update_source_links as usl

needs_page = pytest.mark.skipif(not os.path.isfile(usl.DOC), reason=f"{usl.DOC_NAME} is not in this checkout")

SECTION = "## Where each feature is in the code"


def _text():
    with open(usl.DOC, encoding="utf-8") as f:
        return f.read()


def _specs():
    return [m.group(1) for m in map(usl.MARKER.match, _text().split("\n")) if m]


@needs_page
def test_every_target_exists():
    specs = _specs()
    assert len(specs) >= 20
    for spec in specs:
        for path, name, label in usl.parse_targets(spec):
            assert label, spec
            assert usl.link(path, name, label).startswith(f"[{label}]({usl.REPO_URL}/blob/main/{path}")


@needs_page
def test_every_feature_has_a_source_line():
    text = _text()
    feats = text[text.index(SECTION) + len(SECTION):]
    end = re.search(r"(?m)^## ", feats)
    feats = feats[:end.start()] if end else feats
    heads = re.findall(r"(?m)^#### .+$", feats)
    with_source = re.findall(r"(?m)^#### .+\n\n<!-- source: ", feats)
    assert heads and len(with_source) == len(heads), "a feature without a <!-- source: --> line"


def test_rendering():
    a, b = usl.find("pinnstudio/core/codegen.py", "_training_methods_defs")
    assert 0 < a < b
    line = usl.render("pinnstudio/core/codegen.py::_training_methods_defs | methods ; pinnstudio/ui/results.py | results")
    assert line.startswith(usl.PREFIX) and f"#L{a}-L{b}" in line and line.endswith("results.py) (`results.py`)")


def test_without_the_page(monkeypatch, capsys):
    monkeypatch.setattr(usl, "DOC", os.path.join(usl.ROOT, "no-such-page.md"))
    assert usl.main(["--check"]) == 0
    assert "not in this checkout" in capsys.readouterr().out
