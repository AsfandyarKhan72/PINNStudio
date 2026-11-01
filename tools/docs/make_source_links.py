#!/usr/bin/env python3
"""Write the documentation's "Implementation and source code" snippets from tools/docs/source_map.py.

    python tools/docs/make_source_links.py           # write website/_snippets/source/*.md
    python tools/docs/make_source_links.py --check   # exit 1 if a file or name is missing or a snippet is stale
    python tools/docs/make_source_links.py --table   # print the coverage table (feature, code, tests, pages)

For each feature of the mapping it checks that every file exists and that every named function, class, method
or assignment is defined in it (Python's own parser: `MainWindow._build_config` is the method `_build_config` of
the class `MainWindow`; for a file that is not Python, the name must appear in its text), then writes one snippet,
website/_snippets/source/<id>.md, which the feature's pages include with

    --8<-- "_snippets/source/<id>.md"

and website/_snippets/source/_overview.md (every feature, for the developer page "Repository structure").
The links are `extra.repo_blob` + `extra.source_ref` + path, both read from mkdocs.yml, so a release changes
one value there (the branch to the release tag) and this script is run again. Links name the file and the
function, not line numbers: a branch moves, and line numbers would be wrong after the next change.

--check also fails when a page listed for a feature does not exist, when a feature's reference/ page does not
include its snippet, when developer/structure.md does not include the overview, or when a page includes a snippet
the mapping does not define.
"""
import argparse
import ast
import importlib.util
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB = os.path.join(ROOT, "website")
OUT = os.path.join(WEB, "_snippets", "source")
OUT_NAME = "website/_snippets/source"
MKDOCS = os.path.join(ROOT, "mkdocs.yml")
MAP = os.path.join(ROOT, "tools", "docs", "source_map.py")
OVERVIEW = "_overview"
OVERVIEW_PAGE = "developer/structure.md"
ROLES = [("entry point", "Entry points"), ("helper", "Helpers"), ("test", "Tests")]
INCLUDE = re.compile(r'--8<--\s+"_snippets/source/([A-Za-z0-9_-]+)\.md"')


def load_map(path=MAP):
    spec = importlib.util.spec_from_file_location("pinnstudio_docs_source_map", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FEATURES


def mkdocs_extra(path=MKDOCS):
    """(repo_blob, source_ref) from the `extra:` block of mkdocs.yml (read as text: the file has Python tags)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"(?ms)^extra:\n(.*?)(?=^\S)", text + "\nEOF")
    block = m.group(1) if m else ""
    vals = {}
    for key in ("repo_blob", "source_ref"):
        k = re.search(rf'(?m)^\s+{key}:\s*"?([^"#\n]+?)"?\s*(?:#.*)?$', block)
        if not k:
            raise LookupError(f"mkdocs.yml has no extra.{key}")
        vals[key] = k.group(1).strip()
    blob = vals["repo_blob"]
    if not blob.endswith("/"):
        blob += "/"
    return blob, vals["source_ref"].strip("/")


# ----------------------------------------------------------------- checking names
_trees = {}


def _tree(path):
    if path not in _trees:
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            _trees[path] = ast.parse(f.read(), filename=path)
    return _trees[path]


def _names(body):
    """{name: node} of the defs, classes and assigned names directly in `body`."""
    out = {}
    for n in body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[n.name] = n
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = n
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out[n.target.id] = n
    return out


def find(path, symbol):
    """Raise LookupError unless `symbol` (dotted) is defined in the file `path`."""
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        raise LookupError(f"no file {path}")
    if not symbol:
        return
    if not path.endswith(".py"):
        with open(full, encoding="utf-8") as f:
            if symbol not in f.read():
                raise LookupError(f"{path}: does not contain {symbol!r}")
        return
    body = _tree(path).body
    for part in symbol.split("."):
        node = _names(body).get(part)
        if node is None:
            raise LookupError(f"{path}: no {symbol!r}")
        body = getattr(node, "body", [])


def validate(features):
    """List of problems in the mapping (empty: every file and name exists)."""
    errors, ids = [], set()
    for feat in features:
        fid = feat.get("id", "")
        if not re.fullmatch(r"[a-z0-9-]+", fid) or fid in ids:
            errors.append(f"feature id {fid!r}: lower case, digits and '-', unique")
        ids.add(fid)
        if not feat.get("title"):
            errors.append(f"{fid}: no title")
        for page in feat.get("pages", []):
            if not os.path.isfile(os.path.join(WEB, page)):
                errors.append(f"{fid}: page {page} does not exist")
        roles = {r for r, _ in ROLES}
        if not any(x["role"] == "entry point" for x in feat.get("entries", [])):
            errors.append(f"{fid}: no entry point")
        for x in feat.get("entries", []):
            if x["role"] not in roles:
                errors.append(f"{fid}: role {x['role']!r} is not one of {sorted(roles)}")
            try:
                find(x["path"], x.get("symbol"))
            except (LookupError, SyntaxError) as exc:
                errors.append(f"{fid}: {exc}")
    return errors


# ----------------------------------------------------------------- rendering
def _ref_words(ref):
    return f"release `{ref}`" if re.fullmatch(r"v\d+(\.\d+)*", ref) else f"the `{ref}` branch"


def _entry_line(x, blob, ref):
    url = f"{blob}{ref}/{x['path']}"
    link = f"[`{x['path']}`]({url})"
    head = f"`{x['symbol']}` in {link}" if x.get("symbol") else link
    return f"- {head}" + (f": {x['note']}" if x.get("note") else "")


def _lists(feat, blob, ref):
    out = []
    for role, label in ROLES:
        items = [x for x in feat["entries"] if x["role"] == role]
        if items:
            out += [f"**{label}**", ""] + [_entry_line(x, blob, ref) for x in items] + [""]
    return out


HEADER = "<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->"


def render(feat, blob, ref):
    lines = [HEADER, ""] + _lists(feat, blob, ref)
    lines.append(f"Files on {_ref_words(ref)} of the repository; each link names the function or class to look "
                 "for in the file.")
    return "\n".join(lines) + "\n"


def _title(page):
    with open(os.path.join(WEB, page), encoding="utf-8") as f:
        for line in f:
            if line.startswith("# "):
                return line[2:].strip()
    return page


def render_overview(features, blob, ref):
    """Every feature, for website/developer/structure.md (links relative to a page one folder deep)."""
    lines = [HEADER, ""]
    for feat in features:
        lines += [f"#### {feat['title']}", ""] + _lists(feat, blob, ref)
        pages = [f"[{_title(p)}](../{p})" for p in feat.get("pages", [])]
        if pages:
            lines += ["Described in " + ", ".join(pages) + ".", ""]
    lines.append(f"Files on {_ref_words(ref)} of the repository.")
    return "\n".join(lines) + "\n"


def wanted(features, blob, ref):
    out = {f"{feat['id']}.md": render(feat, blob, ref) for feat in features}
    out[f"{OVERVIEW}.md"] = render_overview(features, blob, ref)
    return out


def includes():
    """{snippet id: [pages that include it]} over the site's Markdown files."""
    found = {}
    for d, dirs, files in os.walk(WEB):
        dirs[:] = [x for x in dirs if x not in ("_snippets", "assets", "overrides")]
        for name in files:
            if name.endswith(".md"):
                p = os.path.join(d, name)
                with open(p, encoding="utf-8") as f:
                    for m in INCLUDE.finditer(f.read()):
                        found.setdefault(m.group(1), []).append(os.path.relpath(p, WEB).replace(os.sep, "/"))
    return {k: sorted(set(v)) for k, v in found.items()}


def table(features, blob, ref):
    inc = includes()
    rows = ["| Feature | Entry points | Helpers | Tests | Pages that include it |", "|---|---|---|---|---|"]
    for feat in features:
        def cell(role):
            return "<br>".join(f"`{x['path'].rsplit('/', 1)[-1]}`" + (f"::`{x['symbol']}`" if x.get("symbol") else "")
                               for x in feat["entries"] if x["role"] == role)
        rows.append(f"| {feat['title']} (`{feat['id']}`) | {cell('entry point')} | {cell('helper')} | {cell('test')} | "
                    + "<br>".join(inc.get(feat["id"], [])) + " |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 on a missing name or a stale snippet")
    ap.add_argument("--table", action="store_true", help="print the coverage table and exit")
    args = ap.parse_args(argv)
    if not os.path.isdir(WEB):
        print("website/ is not in this checkout (the site is on the documentation branch): nothing to do")
        return 0
    features = load_map()
    try:
        blob, ref = mkdocs_extra()
    except LookupError as exc:
        print(exc)
        return 1
    errors = validate(features)
    if args.table:
        print(table(features, blob, ref))
        return 1 if errors else 0
    if errors:
        print("source map: " + "\n            ".join(errors))
        return 1
    want = wanted(features, blob, ref)
    have = {n for n in os.listdir(OUT) if n.endswith(".md")} if os.path.isdir(OUT) else set()
    stale = sorted(n for n, text in want.items()
                   if n not in have or open(os.path.join(OUT, n), encoding="utf-8").read() != text)
    extra = sorted(have - set(want))
    if args.check:
        problems = [f"{OUT_NAME}/{n} is out of date" for n in stale]
        problems += [f"{OUT_NAME}/{n} is not in the source map" for n in extra]
        inc = includes()
        for f in features:          # a feature's own reference page ends with its snippet
            problems += [f"{p} does not include _snippets/source/{f['id']}.md" for p in f.get("pages", [])
                         if p.startswith("reference/") and p not in inc.get(f["id"], [])]
        if OVERVIEW_PAGE not in inc.get(OVERVIEW, []):
            problems.append(f"{OVERVIEW_PAGE} does not include _snippets/source/{OVERVIEW}.md")
        problems += [f"a page includes _snippets/source/{k}.md, which the source map does not define"
                     for k in inc if f"{k}.md" not in want]
        if problems:
            print("\n".join(problems))
            print("run: python tools/docs/make_source_links.py")
            return 1
        n = sum(len(f["entries"]) for f in features)
        print(f"{len(features)} features, {n} source links (on {ref}): up to date")
        return 0
    os.makedirs(OUT, exist_ok=True)
    for n in stale:
        with open(os.path.join(OUT, n), "w", encoding="utf-8", newline="\n") as f:
            f.write(want[n])
    for n in extra:
        os.remove(os.path.join(OUT, n))
    print(f"{len(stale)} snippets written, {len(extra)} removed, {len(want) - len(stale)} unchanged ({OUT_NAME}, on {ref})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
