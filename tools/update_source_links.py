"""Keep the README's "Source code" links pointing at the right lines.

Each feature in the README names the code it links to once, in a comment:

    <!-- source: pinnstudio/core/codegen.py::_training_methods_defs | causal training, RBA and SA-PINN ;
         pinnstudio/ui/results.py | the whole file -->

A target is a file, or a file and a function or class (Class.method for a
method). This script finds each one with Python's own parser and writes the
line under the comment:

    **Source code:** [causal training, RBA and SA-PINN](https://github.com/.../codegen.py#L712-L919) · ...

so the line numbers follow the code. Run it after changing the code and
before merging into main (the links point at main):

    python tools/update_source_links.py           # rewrite README.md
    python tools/update_source_links.py --check   # exit 1 if a link is out of date or a target is gone
"""
import argparse
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
REPO_URL = "https://github.com/AsfandyarKhan72/PINNStudio"
BRANCH = "main"

MARKER = re.compile(r"^<!-- source: (.*?) -->$")
PREFIX = "**Source code:** "

_trees = {}


def _tree(path):
    if path not in _trees:
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            _trees[path] = ast.parse(f.read())
    return _trees[path]


def find(path, qualname):
    """(first line, last line) of a function or class, by its dotted name."""
    nodes = _tree(path).body
    node = None
    for part in qualname.split("."):
        node = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            raise LookupError(f"{path}: no {qualname!r}")
        nodes = node.body
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return first, node.end_lineno


def parse_targets(spec):
    """'path::name | label ; path | label' -> [(path, name or None, label)]."""
    out = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        target, _, label = item.partition("|")
        path, _, name = target.strip().partition("::")
        out.append((path.strip(), name.strip() or None, label.strip()))
    return out


def link(path, name, label):
    url = f"{REPO_URL}/blob/{BRANCH}/{path}"
    if not os.path.isfile(os.path.join(ROOT, path)):
        raise LookupError(f"no file {path}")
    if name:
        a, b = find(path, name)
        url += f"#L{a}-L{b}"
    return f"[{label}]({url}) (`{os.path.basename(path)}`)"


def render(spec):
    return PREFIX + " · ".join(link(*t) for t in parse_targets(spec))


def update(text):
    lines = text.split("\n")
    out, i, n = [], 0, 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = MARKER.match(line)
        if m:
            n += 1
            out.append(render(m.group(1)))
            if i + 1 < len(lines) and lines[i + 1].startswith(PREFIX):
                i += 1                         # replace the old line
            if i + 1 < len(lines) and lines[i + 1].strip():
                out.append("")                 # its own paragraph
        i += 1
    return "\n".join(out), n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="only check; exit 1 if README.md would change")
    args = ap.parse_args(argv)
    with open(README, encoding="utf-8") as f:
        old = f.read()
    try:
        new, n = update(old)
    except LookupError as e:
        print(f"source link target not found: {e}")
        return 1
    if args.check:
        if new != old:
            print("README source links are out of date: run python tools/update_source_links.py")
            return 1
        print(f"{n} source links up to date")
        return 0
    if new != old:
        with open(README, "w", encoding="utf-8") as f:
            f.write(new)
    print(f"{n} source links written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
