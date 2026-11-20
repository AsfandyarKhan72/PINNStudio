"""PINNStudio's model, views and controllers: the tiers and the rules, checked on the source.

See website/developer/mvc.md. Every file under pinnstudio/ has a tier by its path:

| Tier | Paths |
|---|---|
| launcher | pinnstudio/__init__.py, __main__.py, main.py, errors.py, shortcut.py (outside the triad) |
| domain (model) | pinnstudio/core/ |
| state (model) | pinnstudio/ui/state/ |
| controller | pinnstudio/ui/controllers/ |
| view | every other file under pinnstudio/ui/ |

Rules:
- R1 the domain is UI-free: no PySide6, shiboken6, Qt matplotlib backends or pinnstudio.ui, at any depth;
- R2 the state tier imports from Qt only PySide6.QtCore, and from pinnstudio.ui only the state tier;
- R3 controllers import from Qt only PySide6.QtCore, and from pinnstudio.ui only state and controllers;
- R4 views import from pinnstudio.core only VIEW_SAFE modules (imports under TYPE_CHECKING are exempt);
- R5 views never call or hand on a state mutator (MUTATORS) on a state object: they call a controller;
- R7 inside pinnstudio/ui, no `x._name` unless x is self, cls or super(), and no `from M import _name`;
- R8 only main_window.py builds the state and the controllers; among the views only main_window.py imports
  pinnstudio.ui.controllers or pinnstudio.ui.state at run time.

Legacy code from before the rules is listed in tests/mvc_baseline.json. A violation not in the baseline
fails; so does a baseline entry that no longer occurs (remove it): the list can only shrink.

    python tests/test_mvc.py                    # print the violations by rule and file
    python tests/test_mvc.py --write-baseline   # only when legacy code is moved out, never to admit new code
"""
import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "pinnstudio")
BASELINE = os.path.join(ROOT, "tests", "mvc_baseline.json")

LAUNCHER = {"pinnstudio/__init__.py", "pinnstudio/__main__.py", "pinnstudio/main.py", "pinnstudio/errors.py",
            "pinnstudio/shortcut.py"}
# Domain modules and names a view may import: constants, types, and pure functions on small values that
# do no I/O and fit nothing (website/developer/mvc.md, VIEW_SAFE).
VIEW_SAFE = {"pinnstudio.core.netdesign.design", "pinnstudio.core.netdesign.params",
             "pinnstudio.core.netdesign.layers", "pinnstudio.core.netdesign.presets.PRESETS"}
MUTATORS = ("set_design", "reset", "undo", "redo", "set_path", "select", "set_io", "set_message")
QT = ("PySide6", "shiboken6")
QT_MPL = ("matplotlib.backends.backend_qt", "matplotlib.backends.backend_qtagg", "matplotlib.backends.qt_compat")
COMPOSER = "pinnstudio/ui/main_window.py"
PRIVATE_OK = {"_replace", "_asdict", "_fields", "_make", "_exit", "_getframe", "_MEIPASS"}


def tier(rel: str) -> str:
    if rel in LAUNCHER:
        return "launcher"
    if rel.startswith("pinnstudio/core/"):
        return "domain"
    if rel.startswith("pinnstudio/ui/state/"):
        return "state"
    if rel.startswith("pinnstudio/ui/controllers/"):
        return "controller"
    if rel.startswith("pinnstudio/ui/"):
        return "view"
    return ""


def _files():
    for base, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(base, f)
                yield os.path.relpath(path, ROOT).replace(os.sep, "/"), path


def _type_checking_nodes(tree):
    """Nodes inside `if TYPE_CHECKING:` blocks."""
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            t = node.test
            name = t.id if isinstance(t, ast.Name) else (t.attr if isinstance(t, ast.Attribute) else "")
            if name == "TYPE_CHECKING":
                for sub in node.body:
                    for n in ast.walk(sub):
                        inside.add(id(n))
    return inside


def _imports(tree, skip):
    """(module, names, node) of every import, at any depth, outside `skip`."""
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, [], node
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module, [a.name for a in node.names], node


def _under(module, prefixes):
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _receiver_text(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


STATE_NAMES = ("state", "_state", "design_state", "s")


def _is_state(expr, aliases) -> bool:
    """Is `expr` a state object: a name or attribute chain ending in a state name, or a local alias of one."""
    text = _receiver_text(expr)
    if not text:
        return False
    if isinstance(expr, ast.Name) and expr.id in aliases:
        return True
    last = text.split(".")[-1]
    return last in STATE_NAMES and (last != "s" or "." in text)      # a bare local `s` only as an alias


def _r5(tree) -> list:
    """Mutator calls (or mutators handed on) on a state object, following simple local aliases
    (`st = self.design_state`, `s, t = self.state, 1`, `for s in [self.state]` is not followed)."""
    out = []
    scopes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                     ast.Lambda))]
    seen = set()
    for scope in scopes:
        aliases = set()
        body = list(ast.walk(scope))
        for n in body:
            if isinstance(n, ast.Assign):
                pairs = []
                for tgt in n.targets:
                    if isinstance(tgt, ast.Name):
                        pairs.append((tgt, n.value))
                    elif isinstance(tgt, ast.Tuple) and isinstance(n.value, ast.Tuple):
                        pairs += list(zip(tgt.elts, n.value.elts))
                for tgt, val in pairs:
                    if isinstance(tgt, ast.Name) and _is_state(val, aliases):
                        aliases.add(tgt.id)
            elif isinstance(n, ast.NamedExpr) and _is_state(n.value, aliases):
                aliases.add(n.target.id)
        for n in body:
            if isinstance(n, ast.Attribute) and n.attr in MUTATORS and _is_state(n.value, aliases):
                key = (n.lineno, n.col_offset)
                if key not in seen:
                    seen.add(key)
                    out.append(f"{_receiver_text(n.value)}.{n.attr}")
    return out


def check_file(rel, path):
    t = tier(rel)
    out = []
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, rel)
    skip = _type_checking_nodes(tree)
    if not t:
        return [("R0", rel, "the file has no tier: put it in a tier's folder")]
    for module, names, _node in _imports(tree, skip if t == "view" else set()):
        if t == "domain":
            if _under(module, QT) or _under(module, ("pinnstudio.ui",)) or _under(module, QT_MPL):
                out.append(("R1", rel, f"imports {module}"))
        elif t == "state":
            if _under(module, QT) and module != "PySide6.QtCore" and not module.startswith("PySide6.QtCore."):
                out.append(("R2", rel, f"imports {module}"))
            if _under(module, ("pinnstudio.ui",)) and not _under(module, ("pinnstudio.ui.state",)):
                out.append(("R2", rel, f"imports {module}"))
        elif t == "controller":
            if _under(module, QT) and module != "PySide6.QtCore":
                out.append(("R3", rel, f"imports {module}"))
            if _under(module, ("pinnstudio.ui",)) and not _under(module, ("pinnstudio.ui.state",
                                                                           "pinnstudio.ui.controllers")):
                out.append(("R3", rel, f"imports {module}"))
        elif t == "view":
            if _under(module, ("pinnstudio.core",)) and not _under(module, VIEW_SAFE):
                if names:      # from pinnstudio.core.netdesign import design: the name may be a safe module
                    for n in names:
                        if f"{module}.{n}" not in VIEW_SAFE:
                            out.append(("R4", rel, f"imports {module}.{n}"))
                else:
                    out.append(("R4", rel, f"imports {module}"))
            if rel != COMPOSER and _under(module, ("pinnstudio.ui.controllers", "pinnstudio.ui.state")):
                out.append(("R8", rel, f"imports {module} at run time"))
    if t == "view":
        out += [("R5", rel, d) for d in _r5(tree)]
    for node in ast.walk(tree):
        if rel.startswith("pinnstudio/ui/") and isinstance(node, ast.Attribute) and node.attr.startswith("_") \
                and not node.attr.startswith("__") and node.attr not in PRIVATE_OK:
            v = node.value
            ok = (isinstance(v, ast.Name) and v.id in ("self", "cls")) or \
                 (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super")
            if not ok:
                out.append(("R7", rel, f"{_receiver_text(v) or '(expr)'}.{node.attr}"))
        if rel.startswith("pinnstudio/ui/") and isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith("pinnstudio"):
            own = rel.rsplit("/", 1)[0].replace("/", ".")
            for a in node.names:
                if a.name.startswith("_") and not node.module.startswith(own):
                    out.append(("R7", rel, f"from {node.module} import {a.name}"))
        if rel != COMPOSER and t != "state" and isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("DesignState", "DesignerController"):
            out.append(("R8", rel, f"builds {node.func.id}"))
    return out


def violations():
    out = []
    for rel, path in _files():
        out += check_file(rel, path)
    return sorted({f"{r}|{f}|{d}" for r, f, d in out})


def _baseline():
    try:
        with open(BASELINE, encoding="utf-8") as fh:
            return set(json.load(fh))
    except (OSError, ValueError):
        return set()


def test_mvc_rules():
    now = set(violations())
    base = _baseline()
    new = sorted(now - base)
    stale = sorted(base - now)
    assert not new, ("MVC rules broken (website/developer/mvc.md):\n  " + "\n  ".join(new))
    assert not stale, ("These legacy violations are gone: remove them from tests/mvc_baseline.json:\n  "
                       + "\n  ".join(stale))


def test_new_code_has_no_baseline_entries():
    """The designer's packages and the domain start clean: nothing of theirs may be grandfathered."""
    clean = ("pinnstudio/core/", "pinnstudio/ui/state/", "pinnstudio/ui/controllers/", "pinnstudio/ui/designer/")
    grand = [e for e in _baseline() if e.split("|")[1].startswith(clean)]
    assert not grand, grand


def test_the_checker_catches_each_rule(tmp_path):
    cases = {
        "pinnstudio/core/x.py": ("from PySide6.QtWidgets import QWidget\n", "R1"),
        "pinnstudio/ui/state/x.py": ("from PySide6.QtWidgets import QWidget\n", "R2"),
        "pinnstudio/ui/controllers/x.py": ("from PySide6.QtGui import QColor\n", "R3"),
        "pinnstudio/ui/designer/x.py": ("from pinnstudio.core import codegen\n", "R4"),
        "pinnstudio/ui/designer/y.py": ("def f(self):\n    self.state.set_design({})\n", "R5"),
        "pinnstudio/ui/designer/y2.py": ("def f(self):\n    st = self.design_state\n    st.undo()\n", "R5"),
        "pinnstudio/ui/designer/y3.py": ("def f(self, b):\n    b.clicked.connect(self.state.redo)\n", "R5"),
        "pinnstudio/ui/designer/z.py": ("def f(c):\n    return c._fitted\n", "R7"),
        "pinnstudio/ui/designer/w.py": ("from pinnstudio.ui.controllers.designer import DesignerController\n", "R8"),
    }
    for rel, (src, rule) in cases.items():
        p = tmp_path / rel.replace("/", "_")
        p.write_text(src, encoding="utf-8")
        found = {r for r, _f, _d in check_file(rel, str(p))}
        assert rule in found, (rel, found)
    ok = tmp_path / "ok.py"
    ok.write_text("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from pinnstudio.core import codegen\n"
                  "from pinnstudio.core.netdesign import design\n", encoding="utf-8")
    assert check_file("pinnstudio/ui/designer/ok.py", str(ok)) == []


if __name__ == "__main__":
    vs = violations()
    if "--write-baseline" in sys.argv:
        with open(BASELINE, "w", encoding="utf-8") as fh:
            json.dump(vs, fh, indent=1)
        print(f"wrote {len(vs)} entries to {BASELINE}")
    else:
        by = {}
        for v in vs:
            r, f, d = v.split("|", 2)
            by.setdefault((r, f), []).append(d)
        for (r, f), ds in sorted(by.items()):
            print(f"{r} {f}: {len(ds)}")
            for d in ds[:5]:
                print("    ", d)
        print(f"{len(vs)} violations; baseline has {len(_baseline())}")
