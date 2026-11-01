"""Links into the documents that took over the README's long sections: GUIDE.md (features, the walkthrough,
how PINNStudio works) and benchmarks/examples.md (every example, written by tools/make_benchmarks_readme.py).
The model tree's "?" opens a section of GUIDE.md, and the benchmark pages link to the examples by anchor, so a
renamed heading or a missing example section would leave a dead link."""
import ast
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return f.read()


def _anchors(md):
    """GitHub's heading anchors, and explicit <details id=...> anchors."""
    out = set()
    for h in re.findall(r"(?m)^#{1,6} (.+)$", md):
        h = re.sub(r"[`*]", "", h).strip().lower()
        out.add("".join(c for c in h if c.isalnum() or c in "-_ ").replace(" ", "-"))
    return out | set(re.findall(r'<details id="([^"]+)"', md))


def _class_attr(name):
    tree = ast.parse(_read("pinnstudio", "ui", "main_window.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise LookupError(name)


def test_help_buttons_open_guide_sections():
    assert _class_attr("DOCS_URL").endswith("/blob/main/GUIDE.md")
    guide = _anchors(_read("GUIDE.md"))
    for key, anchor in _class_attr("_HELP_ANCHORS").items():
        assert anchor.lstrip("#") in guide, (key, anchor)


def test_every_example_has_its_section():
    ex = _anchors(_read("benchmarks", "examples.md"))
    for ent in json.loads(_read("benchmarks", "problems", "index.json")):
        assert ent["anchor"] in ex, ent["name"]


def test_no_links_to_removed_readme_sections():
    """README.md keeps only Installation, Quick start, Examples (a short table), Documentation, Reproducibility,
    Citation, Acknowledgment, Contributing, Contact and License."""
    readme = _anchors(_read("README.md"))
    pages = {"benchmarks/README.md": "../README.md#", "benchmarks/paper.md": "../README.md#",
             "benchmarks/tuning_study.md": "../README.md#", "benchmarks/examples.md": "../README.md#",
             "GUIDE.md": "README.md#", "CONTRIBUTING.md": "README.md#", "THIRD_PARTY_LICENSES.md": "README.md#"}
    for page, prefix in pages.items():
        text = _read(*page.split("/"))
        for a in re.findall(r"\]\(" + re.escape(prefix) + r"([^)]+)\)", text):
            assert a in readme, (page, a)


if __name__ == "__main__":
    test_help_buttons_open_guide_sections()
    test_every_example_has_its_section()
    test_no_links_to_removed_readme_sections()
    print("ok")
