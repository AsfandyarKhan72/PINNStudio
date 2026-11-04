"""Links into the pages that took over the README's long sections: the documentation website (the user
guide, which the model tree's "?" opens) and benchmarks/examples.md (every example, written by
tools/make_benchmarks_readme.py). The benchmark pages link to the examples by anchor, so a renamed heading or a
missing example section would leave a dead link.

The website's sources (website/, mkdocs.yml) are on the docs branch, not necessarily in this checkout: the "?"
pages are checked against a fixed list of the user-guide pages, and against website/ itself when it is here."""
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


# The user guide's pages on the website (website/guide/<name>.md, served at guide/<name>/).
SITE_URL = "https://asfandyarkhan72.github.io/PINNStudio/"
GUIDE_PAGES = {"problem", "equations", "inverse", "geometry", "points", "initial-conditions",
               "boundary-conditions", "network", "training", "loss-weights", "adaptive", "sweeps", "results",
               "restore", "solve", "error-analysis", "export", "settings", "interface", "choosing-a-strategy"}


def test_help_buttons_open_guide_pages():
    assert _class_attr("DOCS_URL") == SITE_URL
    pages = _class_attr("_HELP_ANCHORS")
    assert {"problem", "equations", "geometry", "bcs", "training", "results", "restore"} <= set(pages)
    for key, path in pages.items():
        m = re.fullmatch(r"guide/([a-z0-9-]+)/", path)
        assert m and m.group(1) in GUIDE_PAGES, (key, path)
        if os.path.isdir(os.path.join(ROOT, "website", "guide")):
            assert os.path.isfile(os.path.join(ROOT, "website", "guide", m.group(1) + ".md")), (key, path)


def test_every_example_has_its_section():
    ex = _anchors(_read("benchmarks", "examples.md"))
    for ent in json.loads(_read("benchmarks", "problems", "index.json")):
        assert ent["anchor"] in ex, ent["name"]


def test_no_links_to_removed_readme_sections():
    """README.md keeps only Installation, Quick start, Examples (a short table), Documentation, Reproducibility,
    Citation, Acknowledgment, Contributing, Contact and License."""
    readme = _anchors(_read("README.md"))
    pages = {"benchmarks/README.md": "../README.md#",
             "benchmarks/tuning_study.md": "../README.md#", "benchmarks/examples.md": "../README.md#",
             "CONTRIBUTING.md": "README.md#", "THIRD_PARTY_LICENSES.md": "README.md#"}
    for page, prefix in pages.items():
        text = _read(*page.split("/"))
        for a in re.findall(r"\]\(" + re.escape(prefix) + r"([^)]+)\)", text):
            assert a in readme, (page, a)


if __name__ == "__main__":
    test_help_buttons_open_guide_pages()
    test_every_example_has_its_section()
    test_no_links_to_removed_readme_sections()
    print("ok")
